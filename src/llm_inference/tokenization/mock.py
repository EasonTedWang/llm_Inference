"""Deterministic tokenizer for M3 golden tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MockTokenizer:
    """Small deterministic tokenizer shared by mock and CPU reference backends."""

    eos_token_id: int = 0
    unk_token_id: int = 1
    mock_token_base: int = 10
    cpu_token_base: int = 100
    prompt_token_base: int = 1000

    def encode(self, text: str) -> tuple[int, ...]:
        pieces = text.split()
        return tuple(self.prompt_token_base + index for index, _ in enumerate(pieces))

    def decode_token(self, token_id: int) -> str:
        if token_id == self.eos_token_id:
            return ""
        if self.mock_token_base <= token_id < self.cpu_token_base:
            return f"mock_{token_id - self.mock_token_base}"
        if self.cpu_token_base <= token_id < self.prompt_token_base:
            return f"cpu_{token_id - self.cpu_token_base}"
        return "<unk>"

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(self.decode_token(token_id) for token_id in token_ids)

    def is_special_token(self, token_id: int) -> bool:
        return token_id in {self.eos_token_id, self.unk_token_id}

