"""Continuous-batching scheduler over a paged KV cache.

Every call to ``step`` does two things. It admits waiting requests (one prefill forward
each) while memory allows, then decodes one token for every request that was already
running, in a single batched forward. Requests join and leave the batch at any step, so
short requests never wait behind long ones.

When memory runs out mid-decode, the most recently admitted request is preempted: its
blocks are freed and it goes back to the front of the queue. Later it is recomputed from
its prompt plus the tokens generated so far, so preemption costs time, not correctness.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import torch
from microllm.model import GPT
from microllm.sampling import logits_to_probs, sample

from nanoserve.forward import paged_forward
from nanoserve.paged import PagedKVCache


@dataclass
class Request:
    id: int
    prompt: list[int]
    max_new_tokens: int
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float | None = None
    eos_id: int | None = None
    seed: int = 0


@dataclass
class RequestState:
    request: Request
    generator: torch.Generator
    tokens: list[int]  # prompt followed by everything generated so far
    generated: int = 0
    table: list[int] = field(default_factory=list)
    cached: int = 0  # how many leading tokens have their keys and values in the cache
    preemptions: int = 0
    submitted_at: float = 0.0
    first_token_at: float | None = None
    finished_at: float | None = None
    finish_reason: str | None = None

    @property
    def output(self) -> list[int]:
        return self.tokens[len(self.request.prompt) :]


@dataclass
class EngineStats:
    steps: int = 0
    prefills: int = 0
    decode_tokens: int = 0
    preemptions: int = 0
    peak_blocks_used: int = 0
    peak_batch: int = 0


class Engine:
    def __init__(
        self,
        model: GPT,
        num_blocks: int = 256,
        block_size: int = 16,
        max_batch: int = 32,
    ) -> None:
        if max_batch < 1:
            raise ValueError("max_batch must be at least 1")
        self.model = model
        self.device = model.tok_emb.weight.device
        self.cache = PagedKVCache(model.cfg, num_blocks, block_size, self.device)
        self.max_batch = max_batch
        self.waiting: deque[RequestState] = deque()
        self.running: list[RequestState] = []
        self.stats = EngineStats()
        self._ids: set[int] = set()

    # ------------------------------------------------------------------ submission

    def submit(self, request: Request) -> RequestState:
        """Queue a request. Raises ``ValueError`` if it could never complete on its own."""
        if request.id in self._ids:
            raise ValueError(f"duplicate request id {request.id}")
        if not request.prompt:
            raise ValueError("prompt must contain at least one token")
        if request.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        vocab = self.model.cfg.vocab_size
        if min(request.prompt) < 0 or max(request.prompt) >= vocab:
            raise ValueError(f"prompt contains token ids outside [0, {vocab})")
        total = len(request.prompt) + request.max_new_tokens
        if total > self.model.cfg.block_size:
            raise ValueError(f"prompt plus max_new_tokens ({total}) exceeds the model context")
        if self.cache.blocks_for(total) > self.cache.allocator.num_blocks:
            raise ValueError("request needs more cache blocks than the engine has in total")
        logits_to_probs(torch.zeros(vocab), request.temperature, request.top_k, request.top_p)

        self._ids.add(request.id)
        state = RequestState(
            request=request,
            generator=torch.Generator(device=self.device).manual_seed(request.seed),
            tokens=list(request.prompt),
            submitted_at=time.perf_counter(),
        )
        self.waiting.append(state)
        return state

    @property
    def idle(self) -> bool:
        return not self.waiting and not self.running

    # ------------------------------------------------------------------ scheduling

    def step(self) -> list[RequestState]:
        """Advance every request by one token where possible. Returns requests that finished."""
        self.stats.steps += 1
        finished: list[RequestState] = []
        decoding = list(self.running)  # admitted before this step; they decode below

        self._admit(finished)
        if decoding:
            self._decode(decoding, finished)
        self.stats.peak_batch = max(self.stats.peak_batch, len(self.running))
        used = self.cache.allocator.num_blocks - self.cache.allocator.num_free
        self.stats.peak_blocks_used = max(self.stats.peak_blocks_used, used)
        return finished

    def run_until_idle(self, max_steps: int = 1_000_000) -> list[RequestState]:
        done: list[RequestState] = []
        for _ in range(max_steps):
            if self.idle:
                return done
            done.extend(self.step())
        raise RuntimeError("engine did not become idle; scheduler is not making progress")

    def _admit(self, finished: list[RequestState]) -> None:
        alloc = self.cache.allocator
        while self.waiting and len(self.running) < self.max_batch:
            state = self.waiting[0]
            need = self.cache.blocks_for(len(state.tokens) + 1)
            # Keep one spare block per running request so admission does not immediately
            # force a preemption on the next decode step.
            reserve = len(self.running)
            if need + reserve > alloc.num_free and (self.running or need > alloc.num_free):
                break
            self.waiting.popleft()
            self._prefill(state)
            if not self._maybe_finish(state):
                self.running.append(state)
            else:
                finished.append(state)

    def _prefill(self, state: RequestState) -> None:
        n = len(state.tokens)
        self.cache.grow(state.table, n)
        ids = torch.tensor([state.tokens], device=self.device)
        pos = torch.arange(n, device=self.device)[None]
        logits = paged_forward(self.model, self.cache, ids, pos, self._tables([state]))
        state.cached = n
        self.stats.prefills += 1
        self._append(state, logits[0, -1])

    def _decode(self, batch: list[RequestState], finished: list[RequestState]) -> None:
        batch = self._reserve_blocks(batch)
        if not batch:
            return
        ids = torch.tensor([[s.tokens[-1]] for s in batch], device=self.device)
        pos = torch.tensor([[s.cached] for s in batch], device=self.device)
        logits = paged_forward(self.model, self.cache, ids, pos, self._tables(batch))
        self.stats.decode_tokens += len(batch)
        for row, state in enumerate(batch):
            state.cached += 1
            self._append(state, logits[row, 0])
            if self._maybe_finish(state):
                self.running.remove(state)
                finished.append(state)

    def _reserve_blocks(self, batch: list[RequestState]) -> list[RequestState]:
        """Make room for each request's next token, preempting the newest ones if needed.

        Victims come from every running request, newest first, including ones admitted in
        this same step, because they hold blocks too. The oldest request is therefore only
        ever preempted when it is the sole one running, which cannot happen because
        ``submit`` guarantees a lone request fits. That is what guarantees progress.
        """
        batch = list(batch)
        while batch:
            need = sum(
                max(0, self.cache.blocks_for(s.cached + 1) - len(s.table)) for s in batch
            )
            if need <= self.cache.allocator.num_free:
                break
            if len(self.running) == 1:
                raise RuntimeError("a lone request does not fit; submit() should have refused it")
            victim = self.running[-1]
            if victim in batch:
                batch.remove(victim)
            self._preempt(victim)
        for state in batch:
            self.cache.grow(state.table, state.cached + 1)
        return batch

    def _preempt(self, state: RequestState) -> None:
        self.cache.free(state.table)
        state.cached = 0
        state.preemptions += 1
        self.stats.preemptions += 1
        self.running.remove(state)
        self.waiting.appendleft(state)

    # ------------------------------------------------------------------ per-request logic

    def _append(self, state: RequestState, logits: torch.Tensor) -> None:
        req = state.request
        probs = logits_to_probs(logits, req.temperature, req.top_k, req.top_p)
        token = int(sample(probs, state.generator))
        state.tokens.append(token)
        state.generated += 1
        if state.first_token_at is None:
            state.first_token_at = time.perf_counter()

    def _maybe_finish(self, state: RequestState) -> bool:
        req = state.request
        if req.eos_id is not None and state.tokens[-1] == req.eos_id:
            state.finish_reason = "eos"
        elif state.generated >= req.max_new_tokens:
            state.finish_reason = "length"
        else:
            return False
        state.finished_at = time.perf_counter()
        self.cache.free(state.table)
        return True

    def _tables(self, states: list[RequestState]) -> torch.Tensor:
        width = max(len(s.table) for s in states)
        rows = [s.table + [0] * (width - len(s.table)) for s in states]
        return torch.tensor(rows, device=self.device)
