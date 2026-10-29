# nanoserve

[![ci](https://github.com/dharmeshdev1/nanoserve/actions/workflows/ci.yml/badge.svg)](https://github.com/dharmeshdev1/nanoserve/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![pytorch](https://img.shields.io/badge/pytorch-2.4%2B-ee4c2c)
![license](https://img.shields.io/badge/license-MIT-green)

A small LLM inference engine: a paged KV cache, continuous batching with preemption, and
an HTTP server. It serves models trained with [microllm](https://github.com/dharmeshdev1/microllm)
and has one defining property that the tests enforce: **a request's output is identical
to running it alone, no matter what else is in the batch, how memory is carved up, or
how often it is preempted.**

## Results

Real trained model (24.6M parameters, from microllm), Apple M5 laptop, PyTorch MPS.
48 requests with prompts of 8 to 48 tokens and outputs of 16 to 128 tokens, all submitted
at once. Greedy decoding. The benchmark asserts that every engine output equals the
sequential output token for token.

| Configuration | Tokens/s | Speedup | Latency p50 | Latency p95 | First token p50 | First token p95 | Peak KV tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sequential baseline | 425 | 1.00x | 4.18 s | 7.77 s | 4.02 s | 7.73 s | n/a |
| engine, batch up to 1 | 315 | 0.74x | 5.64 s | 10.46 s | 5.39 s | 10.37 s | 176 |
| engine, batch up to 4 | 794 | 1.87x | 2.30 s | 4.17 s | 2.01 s | 3.91 s | 512 |
| engine, batch up to 8 | 1,164 | 2.74x | 1.60 s | 2.85 s | 1.13 s | 2.44 s | 816 |
| engine, batch up to 16 | 1,349 | 3.17x | 1.34 s | 2.55 s | 0.64 s | 1.64 s | 1,440 |
| engine, batch up to 32 | 1,702 | 4.01x | 1.43 s | 1.96 s | 0.09 s | 0.98 s | 2,544 |

How to read this:

- **Batching is the win.** Throughput rises 4x at batch 32, and a request no longer
  waits behind others. Median time to first token falls from 4.0 s to 0.09 s, because new
  requests join the running batch instead of queueing for the whole workload.
- **Run-to-run variance.** A second run gave the same engine throughput to within about
  2% (1,166 tok/s at batch 8, 1,733 at batch 32) but a sequential baseline of 395 tok/s,
  so the speedups are better stated as ranges: 1.9x to 2.1x, 2.7x to 3.0x, 3.2x to 3.7x
  and 4.0x to 4.4x for batch limits 4, 8, 16 and 32.
- **Paging is not free.** With a batch of one the engine is 0.74x to 0.81x of the
  sequential baseline. Reassembling each sequence's blocks into a contiguous tensor
  every step costs roughly a fifth to a quarter of the speed when there is no batching to pay
  for it.
- **Memory.** At a batch limit of 32 the peak was 2,544 cached tokens. A contiguous cache
  that reserves the model's full 256-token context per sequence would hold 8,192
  tokens for the same batch, 3.2x more, most of it unused.

## Architecture

```
HTTP request --> tokenizer --> inbox queue --> [ engine thread ]
                                                   |
                    +------------------------------+----------------------+
                    v                                                     v
          admit waiting requests                              decode all running requests
          (one prefill each, while                            (one batched forward, one
           blocks allow)                                       token per request)
                    |                                                     |
                    +--------------- paged forward pass ------------------+
                                            |
                          block pool + per-sequence block tables
                          (alloc on growth, free on finish,
                           preempt newest request when full)
```

- **Paged KV cache** (`paged.py`). Memory is a pool of fixed-size blocks. A sequence owns a
  block table and grows one block at a time, so nothing is reserved for tokens that
  never get generated. A reference-counted allocator tracks ownership.
- **Paged attention forward** (`forward.py`). Reuses a trained microllm model's weights.
  Keys and values are written to scattered blocks and reassembled per batch with a
  position-based mask, so sequences of different lengths share one forward pass.
- **Continuous batching** (`engine.py`). Requests join and leave the batch at any step.
- **Preemption by recomputation.** When blocks run out the newest request is evicted and
  later recomputed from its prompt plus the tokens it already generated.
- **HTTP server** (`server.py`). One thread owns the engine, so scheduling needs no
  locks. Request threads hand work over through a queue and wait on a future.

## How correctness is established

The engine's contract is that batching never changes an answer, so the tests compare
against the unbatched reference instead of only checking that it runs.

- **Paged attention equals the reference model.** Logits match for block sizes of 1, 3,
  4 and 16, for batches of sequences with different lengths, and for blocks that are
  scattered and out of order. Two sequences never see each other's cache.
- **Every request equals its solo output** across batch limits, block sizes, concurrent
  load, requests arriving mid flight, EOS stopping, and randomised stress workloads.
- **Preemption changes nothing.** Pools too small for the workload force preemptions, and
  every output still equals the reference. All blocks are back in the pool afterwards.
- **The oldest request is never preempted**, which is what guarantees progress. Requests
  that could never fit are rejected at submission instead of deadlocking.
- **Sampling is reproducible.** Each request has its own seeded generator, so other
  requests cannot consume its randomness.
- **Allocator invariants** hold under 5,000 random operations: blocks are conserved,
  never double-allocated, and double frees are errors.
- **The HTTP layer is tested as a real server**: concurrent clients each get their own
  correct answer, malformed input returns 400 without hurting the engine, and oversized
  bodies are rejected.

I broke the engine on purpose in six ways (leaking a victim's blocks, forgetting the
generated tokens on recompute, an off-by-one attention mask, a wrong decode position, a
shared random generator, and preempting the oldest request instead of the newest) and
checked that the tests fail each time. The last one initially survived. Outputs stay
correct either way, so the output tests could not see it, and chasing it exposed a flaw
in how victims were chosen: only requests in the decode batch were considered, not ones
admitted in the same step. That is fixed and covered by a progress test.

Building and testing the server surfaced three more real bugs, all fixed: a listen backlog of 5 that
reset connections when a burst of clients arrived, unread request bodies that could be
misparsed as the next request on a keep-alive connection, and a malformed `top_k` field
that would have crashed the engine thread and taken the server down.

73 tests run in about 2 seconds on a CPU, and CI runs them on Python 3.10 to 3.12.

## Quickstart

Serve a model trained with microllm (see its `scripts/reproduce.sh`):

```bash
pip install git+https://github.com/dharmeshdev1/nanoserve.git
nanoserve serve --run path/to/microllm/runs/target --port 8000
```

```bash
curl -s localhost:8000/v1/completions \
  -d '{"prompt": "Once upon a time", "max_tokens": 60, "temperature": 0.8, "top_k": 40, "seed": 1}'
```

The response has `text`, `tokens`, `finish_reason`, `usage` and `timing_ms`. `GET /health`
and `GET /metrics` report liveness, queue depth and free blocks.

Reproduce the benchmark:

```bash
nanoserve bench --run path/to/microllm/runs/target --requests 48 --batch 1 4 8 16 32
```

Development:

```bash
pip install -e ".[dev]"
ruff check . && pytest -q
```

## Design decisions

- **Recompute, do not swap.** Preempted requests drop their blocks and are rebuilt later.
  That needs no second memory tier, and exactness is easy to verify.
- **Newest request is the victim.** The oldest request can then always finish, so the
  engine cannot livelock.
- **Admission keeps one spare block per running request**, so admitting a new request does
  not immediately force a preemption on the next step.
- **Reject at the door.** A request whose prompt plus maximum output exceeds the context,
  or needs more blocks than exist, is refused at submission.
- **One thread owns the engine.** Concurrency lives at the edge, in the queue and futures.
- **A bad client cannot stop the server.** Input is validated and coerced in the handler,
  and the engine thread survives any submission error.

## Limitations

- **Attention is gather based.** Blocks are copied into a contiguous tensor each step and
  then fed to PyTorch's fused attention. A kernel that reads blocks in place would remove
  that copy. I did not profile it, but it is the likely cause of the batch-of-one slowdown.
- **Prefill is one whole prompt at a time.** A very long prompt stalls decoding for that
  step. Chunked prefill would fix it.
- **No prefix sharing.** The allocator supports reference counts and this is tested, but
  nothing yet shares blocks between requests with a common prompt.
- **No streaming responses**, no speculative decoding, one device, fp32 only.
- **No authentication or rate limiting.** It uses the standard library HTTP server and is
  meant for experiments, not exposure to the internet.
- **Sampled outputs are reproducible in practice, not by proof.** Batch composition can
  change floating point results in the last digits, which could in principle flip a
  sampled token. Greedy outputs are tested for exact equality.
- **Small scale.** One 24.6M parameter model, one synthetic workload, one laptop.

## License

MIT
