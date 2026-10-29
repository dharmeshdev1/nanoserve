import random

import pytest
import torch
from microllm.model import ModelConfig

from nanoserve.forward import paged_forward
from nanoserve.paged import BlockAllocator, OutOfBlocks, PagedKVCache
from tests.conftest import make_model


def test_allocator_hands_out_every_block_once_then_runs_dry():
    alloc = BlockAllocator(4)
    got = sorted(alloc.allocate() for _ in range(4))
    assert got == [0, 1, 2, 3] and alloc.num_free == 0
    with pytest.raises(OutOfBlocks):
        alloc.allocate()


def test_released_blocks_are_reusable_and_double_free_is_an_error():
    alloc = BlockAllocator(2)
    a = alloc.allocate()
    alloc.release(a)
    assert alloc.num_free == 2
    with pytest.raises(ValueError, match="double free"):
        alloc.release(a)


def test_shared_block_returns_to_the_pool_only_after_the_last_owner_releases():
    alloc = BlockAllocator(1)
    b = alloc.allocate()
    alloc.retain(b)
    alloc.release(b)
    assert alloc.num_free == 0 and alloc.refcount(b) == 1
    alloc.release(b)
    assert alloc.num_free == 1
    with pytest.raises(ValueError):
        alloc.retain(b)


def test_allocator_conserves_blocks_under_random_operations():
    rng = random.Random(0)
    alloc, held = BlockAllocator(32), []
    for _ in range(5000):
        if held and (rng.random() < 0.5 or alloc.num_free == 0):
            alloc.release(held.pop(rng.randrange(len(held))))
        else:
            held.append(alloc.allocate())
        assert alloc.num_free + len(held) == 32
        assert len(set(held)) == len(held)


def test_allocator_rejects_empty_pool():
    with pytest.raises(ValueError):
        BlockAllocator(0)


def make_cache(model, num_blocks=16, block_size=4):
    return PagedKVCache(model.cfg, num_blocks, block_size, torch.device("cpu"))


def test_grow_is_all_or_nothing():
    cache = make_cache(make_model(), num_blocks=3, block_size=4)
    table: list[int] = []
    cache.grow(table, 8)
    assert len(table) == 2 and cache.allocator.num_free == 1
    with pytest.raises(OutOfBlocks):
        cache.grow(table, 16)  # needs 2 more, only 1 free
    assert len(table) == 2 and cache.allocator.num_free == 1  # nothing leaked
    cache.free(table)
    assert table == [] and cache.allocator.num_free == 3


def test_blocks_for_rounds_up():
    cache = make_cache(make_model(), block_size=4)
    assert [cache.blocks_for(n) for n in (0, 1, 4, 5, 8, 9)] == [0, 1, 1, 2, 2, 3]


def padded(tables):
    width = max(len(t) for t in tables)
    return torch.tensor([t + [0] * (width - len(t)) for t in tables])


def test_prefill_logits_match_the_reference_model_for_every_block_size():
    model = make_model()
    ids = torch.randint(0, 40, (1, 13))
    want, _ = model(ids)
    for block_size in (1, 3, 4, 16):
        cache = make_cache(model, num_blocks=32, block_size=block_size)
        table: list[int] = []
        cache.grow(table, 13)
        got = paged_forward(model, cache, ids, torch.arange(13)[None], padded([table]))
        torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_batched_decode_of_sequences_with_different_lengths_matches_reference():
    model = make_model()
    cache = make_cache(model, num_blocks=64, block_size=4)
    lengths = [3, 9, 14, 1]
    seqs = [torch.randint(0, 40, (1, n + 1)) for n in lengths]  # n cached tokens + 1 pending
    tables = []
    for ids, n in zip(seqs, lengths, strict=True):
        table: list[int] = []
        cache.grow(table, n + 1)
        if n:
            paged_forward(model, cache, ids[:, :n], torch.arange(n)[None], padded([table]))
        tables.append(table)

    batch = torch.cat([ids[:, -1:] for ids in seqs])
    pos = torch.tensor(lengths)[:, None]
    got = paged_forward(model, cache, batch, pos, padded(tables))
    for row, ids in enumerate(seqs):
        want, _ = model(ids)
        torch.testing.assert_close(got[row, 0], want[0, -1], atol=1e-4, rtol=1e-4)


def test_blocks_need_not_be_contiguous_or_ordered():
    model = make_model()
    cache = make_cache(model, num_blocks=16, block_size=4)
    cache.grow([], 0)
    scattered = [11, 3, 14, 6]  # arbitrary physical blocks for a 16 token sequence
    for b in scattered:
        cache.allocator._free.remove(b)
        cache.allocator._refs[b] = 1
    ids = torch.randint(0, 40, (1, 15))
    want, _ = model(ids)
    got = paged_forward(model, cache, ids, torch.arange(15)[None], torch.tensor([scattered]))
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_two_sequences_never_see_each_others_cache():
    model = make_model()
    cache = make_cache(model, num_blocks=32, block_size=4)
    a, b = torch.randint(0, 40, (1, 7)), torch.randint(0, 40, (1, 7))
    ta, tb = [], []
    cache.grow(ta, 7)
    cache.grow(tb, 7)
    out_a = paged_forward(model, cache, a, torch.arange(7)[None], padded([ta]))
    paged_forward(model, cache, b, torch.arange(7)[None], padded([tb]))
    again = paged_forward(model, cache, a, torch.arange(7)[None], padded([ta]))
    torch.testing.assert_close(out_a, again)


def test_cache_dtype_and_shapes():
    cfg = ModelConfig(vocab_size=10, block_size=16, n_layer=3, n_head=4, n_kv_head=2, n_embd=32)
    cache = PagedKVCache(cfg, 5, 8, torch.device("cpu"))
    assert len(cache.k) == 3 and cache.k[0].shape == (5, 2, 8, 8)
