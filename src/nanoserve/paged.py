"""Paged KV cache: fixed-size blocks, a free list with reference counts, and block tables.

Instead of reserving a contiguous ``max_len`` slab per sequence, memory is split into
blocks of ``block_size`` tokens. A sequence owns a list of block ids (its block table), so
memory is committed one block at a time as the sequence grows and returned the moment it
finishes. This removes the internal fragmentation of per-sequence slabs.
"""

from __future__ import annotations

import math
from collections import deque

import torch
from microllm.model import ModelConfig
from torch import Tensor


class OutOfBlocks(RuntimeError):
    """Raised when an allocation cannot be satisfied."""


class BlockAllocator:
    """Free list with reference counts.

    Reference counts let several sequences share a block (for prefix sharing). The block
    returns to the free list only when its last owner releases it.
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")
        self.num_blocks = num_blocks
        self._free: deque[int] = deque(range(num_blocks))
        self._refs = [0] * num_blocks

    @property
    def num_free(self) -> int:
        return len(self._free)

    def refcount(self, block: int) -> int:
        return self._refs[block]

    def allocate(self) -> int:
        if not self._free:
            raise OutOfBlocks("no free blocks")
        block = self._free.popleft()
        self._refs[block] = 1
        return block

    def retain(self, block: int) -> None:
        if self._refs[block] <= 0:
            raise ValueError(f"cannot retain block {block}: it is not allocated")
        self._refs[block] += 1

    def release(self, block: int) -> None:
        if self._refs[block] <= 0:
            raise ValueError(f"double free of block {block}")
        self._refs[block] -= 1
        if self._refs[block] == 0:
            self._free.append(block)


class PagedKVCache:
    """Key/value storage for every layer, indexed by block id."""

    def __init__(
        self,
        cfg: ModelConfig,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        shape = (num_blocks, cfg.n_kv_head, block_size, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)

    def blocks_for(self, n_tokens: int) -> int:
        return math.ceil(n_tokens / self.block_size)

    def grow(self, table: list[int], n_tokens: int) -> None:
        """Extend ``table`` so it can hold ``n_tokens``. All or nothing: no partial allocation."""
        missing = self.blocks_for(n_tokens) - len(table)
        if missing <= 0:
            return
        if missing > self.allocator.num_free:
            raise OutOfBlocks(f"need {missing} blocks, only {self.allocator.num_free} free")
        table.extend(self.allocator.allocate() for _ in range(missing))

    def free(self, table: list[int]) -> None:
        for block in table:
            self.allocator.release(block)
        table.clear()

    def write(self, layer: int, tables: Tensor, positions: Tensor, k: Tensor, v: Tensor) -> None:
        """Store ``k`` and ``v`` (``[B, n_kv, T, D]``) at absolute ``positions`` (``[B, T]``)."""
        blocks = tables.gather(1, positions // self.block_size)  # [B, T]
        offsets = positions % self.block_size
        self.k[layer][blocks, :, offsets] = k.transpose(1, 2)
        self.v[layer][blocks, :, offsets] = v.transpose(1, 2)

    def gather(self, layer: int, tables: Tensor) -> tuple[Tensor, Tensor]:
        """Reassemble each sequence's keys and values, ``[B, n_kv, max_blocks * block_size, D]``."""
        def collect(store: Tensor) -> Tensor:
            picked = store[tables]  # [B, max_blocks, n_kv, block_size, D]
            B, M, H, S, D = picked.shape
            return picked.permute(0, 2, 1, 3, 4).reshape(B, H, M * S, D)

        return collect(self.k[layer]), collect(self.v[layer])
