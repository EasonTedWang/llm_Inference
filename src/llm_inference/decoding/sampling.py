"""Sampling interfaces for future logits-based decoding."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None


class GreedySampler:
    def select(self, logits: tuple[float, ...]) -> int:
        if not logits:
            raise ValueError("logits must not be empty")
        return max(range(len(logits)), key=logits.__getitem__)

