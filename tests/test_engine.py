import random

import pytest
import torch
from microllm.generate import generate

from nanoserve.engine import Engine, Request
from tests.conftest import make_model

EOS = 7


def reference(model, req):
    return generate(
        model, req.prompt, req.max_new_tokens, temperature=0, eos_id=req.eos_id
    )


def requests(n, seed=0, vocab=40, max_prompt=12, max_new=14, eos=None):
    rng = random.Random(seed)
    return [
        Request(
            id=i,
            prompt=[rng.randrange(vocab) for _ in range(rng.randint(1, max_prompt))],
            max_new_tokens=rng.randint(1, max_new),
            eos_id=eos,
        )
        for i in range(n)
    ]


def run(engine, reqs):
    states = {r.id: engine.submit(r) for r in reqs}
    engine.run_until_idle()
    return states


def test_single_request_matches_reference():
    model = make_model()
    req = Request(0, [1, 2, 3], 10)
    state = run(Engine(model, num_blocks=16, block_size=4), [req])[0]
    assert state.output == reference(model, req) and state.finish_reason == "length"


@pytest.mark.parametrize("block_size", [1, 4, 16])
@pytest.mark.parametrize("max_batch", [1, 3, 32])
def test_many_concurrent_requests_each_match_their_solo_output(block_size, max_batch):
    model = make_model()
    reqs = requests(12, seed=block_size + max_batch)
    engine = Engine(model, num_blocks=256, block_size=block_size, max_batch=max_batch)
    states = run(engine, reqs)
    for req in reqs:
        assert states[req.id].output == reference(model, req), f"request {req.id} differs"
    assert engine.stats.peak_batch <= max_batch


def test_batching_actually_happens():
    model = make_model()
    engine = Engine(model, num_blocks=256, block_size=4, max_batch=8)
    run(engine, requests(8, seed=1, max_new=10))
    assert engine.stats.peak_batch > 1
    assert engine.stats.steps < sum(r.max_new_tokens for r in requests(8, seed=1, max_new=10))


def test_tight_memory_forces_preemption_without_changing_any_output():
    model = make_model()
    reqs = requests(10, seed=3, max_prompt=10, max_new=20)
    engine = Engine(model, num_blocks=14, block_size=4, max_batch=10)
    states = run(engine, reqs)
    assert engine.stats.preemptions > 0, "the pool was meant to be too small"
    assert any(s.preemptions > 0 for s in states.values())
    for req in reqs:
        assert states[req.id].output == reference(model, req), f"request {req.id} differs"
    assert engine.cache.allocator.num_free == 14  # everything returned to the pool


def test_every_block_is_returned_when_the_engine_goes_idle():
    model = make_model()
    engine = Engine(model, num_blocks=40, block_size=4, max_batch=6)
    run(engine, requests(15, seed=5))
    assert engine.idle and engine.cache.allocator.num_free == 40
    assert all(engine.cache.allocator.refcount(b) == 0 for b in range(40))


def test_eos_stops_a_request_early_and_is_included():
    model = make_model()
    reqs = requests(10, seed=7, max_new=30, eos=EOS)
    states = run(Engine(model, num_blocks=128, block_size=4), reqs)
    for req in reqs:
        assert states[req.id].output == reference(model, req)
    assert any(s.finish_reason == "eos" for s in states.values())


def test_a_request_arriving_mid_flight_joins_the_running_batch():
    model = make_model()
    first, late = Request(0, [1, 2, 3], 20), Request(1, [4, 5], 8)
    engine = Engine(model, num_blocks=64, block_size=4)
    s0 = engine.submit(first)
    for _ in range(5):
        engine.step()
    assert s0.generated == 5
    s1 = engine.submit(late)
    engine.run_until_idle()
    assert s0.output == reference(model, first) and s1.output == reference(model, late)


def test_short_requests_do_not_wait_for_long_ones():
    model = make_model()
    engine = Engine(model, num_blocks=64, block_size=4, max_batch=8)
    long_, short = Request(0, [1, 2], 40), Request(1, [3, 4], 2)
    s_long, s_short = engine.submit(long_), engine.submit(short)
    finished_at = {}
    for step in range(1, 100):
        for s in engine.step():
            finished_at[s.request.id] = step
        if engine.idle:
            break
    assert finished_at[1] < finished_at[0]
    assert s_short.output == reference(model, short)
    assert s_long.output == reference(model, long_)


def test_sampling_is_reproducible_and_independent_of_batch_composition():
    model = make_model()
    mk = lambda i, seed: Request(  # noqa: E731
        i, [1, 2, 3], 12, temperature=1.0, top_k=10, seed=seed
    )
    alone = run(Engine(model, num_blocks=64, block_size=4), [mk(0, 11)])[0].output
    crowd = [mk(0, 11)] + [mk(i, 100 + i) for i in range(1, 7)]
    states = run(Engine(model, num_blocks=64, block_size=4, max_batch=8), crowd)
    assert states[0].output == alone
    assert len({tuple(states[i].output) for i in range(1, 7)}) > 1  # different seeds differ


def test_different_seeds_give_different_samples():
    model = make_model(scale=1.0)
    out = set()
    for seed in range(8):
        req = Request(0, [1], 16, temperature=1.5, seed=seed)
        out.add(tuple(run(Engine(model, num_blocks=32, block_size=4), [req])[0].output))
    assert len(out) > 1


@pytest.mark.parametrize(
    "kw,match",
    [
        (dict(prompt=[]), "at least one token"),
        (dict(max_new_tokens=0), "at least 1"),
        (dict(prompt=[40]), "outside"),
        (dict(prompt=[-1]), "outside"),
        (dict(prompt=[1] * 60, max_new_tokens=10), "exceeds the model context"),
        (dict(temperature=-1.0), "temperature"),
        (dict(top_p=2.0), "top_p"),
    ],
)
def test_invalid_requests_are_rejected_at_submission(kw, match):
    base = dict(id=0, prompt=[1, 2], max_new_tokens=4)
    engine = Engine(make_model(), num_blocks=8, block_size=4)
    with pytest.raises(ValueError, match=match):
        engine.submit(Request(**(base | kw)))
    assert engine.idle


def test_request_larger_than_total_memory_is_rejected_not_deadlocked():
    engine = Engine(make_model(), num_blocks=2, block_size=4)
    with pytest.raises(ValueError, match="more cache blocks"):
        engine.submit(Request(0, [1] * 6, 10))  # needs 4 blocks, pool has 2


def test_duplicate_request_ids_are_rejected():
    engine = Engine(make_model(), num_blocks=8, block_size=4)
    engine.submit(Request(0, [1], 2))
    with pytest.raises(ValueError, match="duplicate"):
        engine.submit(Request(0, [2], 2))


def test_worst_case_pool_sized_for_exactly_one_request_still_completes_everything():
    model = make_model()
    reqs = requests(6, seed=9, max_prompt=6, max_new=10)
    biggest = max(Engine(model, num_blocks=64, block_size=4).cache.blocks_for(
        len(r.prompt) + r.max_new_tokens) for r in reqs)
    engine = Engine(model, num_blocks=biggest, block_size=4, max_batch=6)
    states = run(engine, reqs)
    for req in reqs:
        assert states[req.id].output == reference(model, req)


def test_engine_makes_progress_on_a_randomised_stress_workload():
    model = make_model(seed=2)
    rng = random.Random(42)
    completed = 0
    for trial in range(10):
        reqs = requests(rng.randint(5, 14), seed=trial, max_prompt=14, max_new=16, eos=EOS)
        engine = Engine(
            model, num_blocks=rng.randint(10, 30), block_size=rng.choice([2, 4, 8]),
            max_batch=rng.randint(1, 8),
        )
        try:
            states = {r.id: engine.submit(r) for r in reqs}
        except ValueError:
            continue  # a request legitimately too big for this random pool
        engine.run_until_idle()
        for req in reqs:
            assert states[req.id].output == reference(model, req)
        assert engine.cache.allocator.num_free == engine.cache.allocator.num_blocks
        completed += 1
    assert completed >= 5, "too many trials were skipped for the test to mean anything"


def test_max_batch_must_be_positive():
    with pytest.raises(ValueError):
        Engine(make_model(), max_batch=0)


def test_generator_is_per_request_so_other_requests_cannot_consume_its_randomness():
    model = make_model()
    solo = Request(0, [3, 1], 10, temperature=1.0, seed=5)
    a = run(Engine(model, num_blocks=64, block_size=4), [solo])[0].output
    noisy = [Request(0, [3, 1], 10, temperature=1.0, seed=5)] + [
        Request(i, [2, 2], 10, temperature=1.0, seed=i) for i in range(1, 5)
    ]
    b = run(Engine(model, num_blocks=64, block_size=4, max_batch=5), noisy)[0].output
    assert a == b
    assert isinstance(torch.Generator().manual_seed(1), torch.Generator)


@pytest.mark.parametrize("num_blocks,block_size", [(12, 4), (14, 4), (20, 2), (9, 8)])
def test_the_oldest_request_is_never_preempted_so_the_engine_always_progresses(
    num_blocks, block_size
):
    model = make_model()
    reqs = requests(10, seed=num_blocks, max_prompt=8, max_new=16)
    engine = Engine(model, num_blocks=num_blocks, block_size=block_size, max_batch=10)
    try:
        states = {r.id: engine.submit(r) for r in reqs}
    except ValueError:
        pytest.skip("a request was too large for this pool")
    engine.run_until_idle()
    assert states[0].preemptions == 0
    assert all(states[r.id].output == reference(model, r) for r in reqs)


def test_preemption_workloads_actually_preempt_something_across_the_parametrisation():
    model = make_model()
    total = 0
    for num_blocks, block_size in [(12, 4), (14, 4), (20, 2), (9, 8)]:
        engine = Engine(model, num_blocks=num_blocks, block_size=block_size, max_batch=10)
        try:
            for r in requests(10, seed=num_blocks, max_prompt=8, max_new=16):
                engine.submit(r)
        except ValueError:
            continue
        engine.run_until_idle()
        total += engine.stats.preemptions
    assert total > 0
