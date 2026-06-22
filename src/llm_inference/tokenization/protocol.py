"""Tokenizer protocol used by runtime backends and the engine."""

from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    eos_token_id: int
    unk_token_id: int

    def encode(self, text: str) -> tuple[int, ...]:
        raise NotImplementedError

    def decode_token(self, token_id: int) -> str:
        raise NotImplementedError

    def decode(self, token_ids: tuple[int, ...]) -> str:
        raise NotImplementedError

    def is_special_token(self, token_id: int) -> bool:
        raise NotImplementedError
