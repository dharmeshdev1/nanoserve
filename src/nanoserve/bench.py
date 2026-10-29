"""Throughput and latency of continuous batching versus one-request-at-a-time serving."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from microllm.generate import generate
from microllm.model import GPT

from nanoserve.engine import Engine, Request


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile, ``q`` in [0, 100]."""
    if not values:
        raise ValueError("no values")
    if not 0 <= q <= 100:
        raise ValueError("q must be in [0, 100]")
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True)
class BenchResult:
    name: str
    tokens: int
    seconds: float
    latency_p50: float
    latency_p95: float
    ttft_p50: float
    ttft_p95: float
    peak_batch: int = 1
    preemptions: int = 0
    peak_kv_tokens: int = 0

    @property
    def tokens_per_sec(self) -> float:
        return self.tokens / self.seconds


def make_workload(
    tokens: np.ndarray, n: int, prompt_range: tuple[int, int], new_range: tuple[int, int],
    context: int, seed: int,
) -> list[Request]:
    """Prompts are real text windows. Output lengths vary, which is what batching exploits."""
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        plen = rng.randint(*prompt_range)
        new = min(rng.randint(*new_range), context - plen)
        start = rng.randrange(0, len(tokens) - plen)
        reqs.append(Request(i, [int(t) for t in tokens[start : start + plen]], new))
    return reqs


def run_sequential(model: GPT, reqs: list[Request]) -> tuple[BenchResult, dict[int, list[int]]]:
    """Baseline: serve requests one after another with a normal KV cache."""
    # Time to first token for each request, measured outside the timed run: a request that
    # starts at time s produces its first token at s plus the cost of prefill and one step.
    first_cost = []
    for r in reqs:
        t = time.perf_counter()
        generate(model, r.prompt, 1, temperature=0)
        first_cost.append(time.perf_counter() - t)

    outputs: dict[int, list[int]] = {}
    latencies, ttfts = [], []
    t0 = time.perf_counter()
    for r, cost in zip(reqs, first_cost, strict=True):
        started = time.perf_counter() - t0
        outputs[r.id] = generate(model, r.prompt, r.max_new_tokens, temperature=0)
        latencies.append(time.perf_counter() - t0)
        ttfts.append(started + cost)
    seconds = time.perf_counter() - t0
    total = sum(len(o) for o in outputs.values())
    return (
        BenchResult("sequential", total, seconds, percentile(latencies, 50),
                    percentile(latencies, 95), percentile(ttfts, 50), percentile(ttfts, 95)),
        outputs,
    )


def run_engine(
    model: GPT, reqs: list[Request], max_batch: int, num_blocks: int, block_size: int
) -> tuple[BenchResult, dict[int, list[int]]]:
    engine = Engine(model, num_blocks=num_blocks, block_size=block_size, max_batch=max_batch)
    t0 = time.perf_counter()
    states = [engine.submit(r) for r in reqs]
    engine.run_until_idle()
    seconds = time.perf_counter() - t0
    lat = [s.finished_at - t0 for s in states]
    ttft = [s.first_token_at - t0 for s in states]
    outputs = {s.request.id: s.output for s in states}
    return (
        BenchResult(
            f"engine batch<={max_batch}", sum(len(o) for o in outputs.values()), seconds,
            percentile(lat, 50), percentile(lat, 95), percentile(ttft, 50), percentile(ttft, 95),
            engine.stats.peak_batch, engine.stats.preemptions,
            engine.stats.peak_blocks_used * block_size,
        ),
        outputs,
    )


def benchmark(
    model: GPT, reqs: list[Request], batches: list[int], num_blocks: int, block_size: int
) -> list[BenchResult]:
    """Run every configuration on the same workload and verify the outputs agree.

    The warm-up pass keeps one-time kernel compilation out of the timings, and every engine
    output is checked against the sequential baseline so a speedup can never hide a change
    in what the model generates.
    """
    run_engine(model, reqs[:2], 2, num_blocks, block_size)
    base, reference = run_sequential(model, reqs)
    results = [base]
    for b in batches:
        result, outputs = run_engine(model, reqs, b, num_blocks, block_size)
        if outputs != reference:
            bad = [i for i in reference if outputs[i] != reference[i]]
            raise AssertionError(f"engine output differs from sequential for requests {bad[:5]}")
        results.append(result)
    return results


def format_table(results: list[BenchResult], contiguous_tokens: int | None = None) -> str:
    head = (
        f"{'configuration':<20}{'tok/s':>9}{'speedup':>9}{'lat p50':>9}{'lat p95':>9}"
        f"{'ttft p50':>10}{'ttft p95':>10}{'peak batch':>12}{'preempt':>9}{'KV tokens':>11}"
    )
    lines = [head, "-" * len(head)]
    base = results[0].tokens_per_sec
    for r in results:
        kv = str(r.peak_kv_tokens) if r.peak_kv_tokens else "-"
        lines.append(
            f"{r.name:<20}{r.tokens_per_sec:>9.1f}{r.tokens_per_sec / base:>8.2f}x"
            f"{r.latency_p50:>8.2f}s{r.latency_p95:>8.2f}s{r.ttft_p50:>9.2f}s{r.ttft_p95:>9.2f}s"
            f"{r.peak_batch:>12}{r.preemptions:>9}{kv:>11}"
        )
    if contiguous_tokens:
        lines.append(f"(a contiguous max-length cache for the largest batch would reserve "
                     f"{contiguous_tokens} KV tokens)")
    return "\n".join(lines)


def device_of(model: GPT) -> torch.device:
    return model.tok_emb.weight.device
