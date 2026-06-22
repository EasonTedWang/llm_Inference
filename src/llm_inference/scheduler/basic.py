"""Initial FIFO scheduler boundary."""

from __future__ import annotations

from collections import deque

from llm_inference.scheduler.types import ScheduledRequest


class BasicScheduler:
    def __init__(self) -> None:
        self._queue: deque[ScheduledRequest] = deque()

    def enqueue(self, request: ScheduledRequest) -> None:
        self._queue.append(request)

    def dequeue_batch(self, max_batch_size: int) -> tuple[ScheduledRequest, ...]:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        batch = []
        while self._queue and len(batch) < max_batch_size:
            batch.append(self._queue.popleft())
        return tuple(batch)

    def __len__(self) -> int:
        return len(self._queue)

