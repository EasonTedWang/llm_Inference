"""Minimal KV cache accounting primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class KVCacheEvent(str, Enum):
    ALLOCATED = "allocated"
    FREED = "freed"
    REUSED = "reused"
    EVICTED = "evicted"


@dataclass(frozen=True)
class KVCacheBlock:
    block_id: int
    owner_request_id: str


class KVCacheManager:
    def __init__(self, max_blocks: int) -> None:
        if max_blocks < 1:
            raise ValueError("max_blocks must be >= 1")
        self.max_blocks = max_blocks
        self._next_block_id = 0
        self._active: dict[int, KVCacheBlock] = {}

    @property
    def active_blocks(self) -> tuple[KVCacheBlock, ...]:
        return tuple(self._active.values())

    def allocate(self, request_id: str, count: int) -> tuple[KVCacheBlock, ...]:
        if count < 1:
            raise ValueError("count must be >= 1")
        if len(self._active) + count > self.max_blocks:
            raise MemoryError("not enough KV cache blocks")
        blocks = []
        for _ in range(count):
            block = KVCacheBlock(
                block_id=self._next_block_id, owner_request_id=request_id
            )
            self._active[block.block_id] = block
            self._next_block_id += 1
            blocks.append(block)
        return tuple(blocks)

    def free(self, block_ids: tuple[int, ...]) -> None:
        for block_id in block_ids:
            self._active.pop(block_id, None)

