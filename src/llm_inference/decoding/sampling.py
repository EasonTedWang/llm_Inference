"""Sampling interfaces for future logits-based decoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

Logits = Mapping[int, float] | Sequence[float]


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None


class GreedySampler:
    def select(self, logits: Logits) -> int:
        if not logits:
            raise ValueError("logits must not be empty")
        if isinstance(logits, Mapping):
            return max(logits, key=logits.__getitem__)
        return max(range(len(logits)), key=logits.__getitem__)
