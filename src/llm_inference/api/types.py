"""Public request and response types."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_new_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    stop: tuple[str, ...] = ()
    request_id: str = field(default_factory=lambda: str(uuid4()))

    def validate(self) -> None:
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1 when provided")


@dataclass(frozen=True)
class TokenChunk:
    token_id: int
    text: str


@dataclass(frozen=True)
class GenerationResult:
    request_id: str
    prompt: str
    output_text: str
    generated_tokens: tuple[TokenChunk, ...]
    finish_reason: str
    backend: str

