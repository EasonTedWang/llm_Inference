"""Backend contract shared by CUDA, CPU, and mock runtimes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from llm_inference.api.types import GenerationRequest
from llm_inference.config import BackendKind, EngineConfig


class BackendUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class KVAllocation:
    allocation_id: str
    request_id: str


@dataclass
class PrefillState:
    request_id: str
    prompt_tokens: int
    generated_tokens: int = 0


@dataclass(frozen=True)
class DecodeStep:
    token_id: int
    token_text: str
    finished: bool = False
    finish_reason: str | None = None


class Backend(ABC):
    kind: BackendKind

    @abstractmethod
    def load_model(self, config: EngineConfig) -> None:
        raise NotImplementedError

    @abstractmethod
    def allocate_kv(self, request: GenerationRequest) -> KVAllocation:
        raise NotImplementedError

    @abstractmethod
    def prefill(
        self, request: GenerationRequest, allocation: KVAllocation
    ) -> PrefillState:
        raise NotImplementedError

    @abstractmethod
    def decode_step(
        self, request: GenerationRequest, state: PrefillState
    ) -> DecodeStep:
        raise NotImplementedError

    @abstractmethod
    def free_kv(self, allocation: KVAllocation) -> None:
        raise NotImplementedError

