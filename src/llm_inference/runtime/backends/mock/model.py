"""Deterministic mock model used for M3 golden tests."""

from __future__ import annotations

from llm_inference.decoding.sampling import Logits


class DeterministicMockModel:
    def __init__(self, token_base: int) -> None:
        self.token_base = token_base

    def next_logits(self, generated_tokens: int) -> Logits:
        token_id = self.token_base + generated_tokens
        logits = [-1.0] * (token_id + 1)
        logits[token_id] = 1.0
        return tuple(logits)
