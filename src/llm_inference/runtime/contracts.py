"""Backend contract shared by CUDA, CPU, and mock runtimes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from llm_inference.api.types import GenerationRequest
from llm_inference.config import BackendKind, EngineConfig
from llm_inference.decoding.sampling import Logits
from llm_inference.tokenization import Tokenizer


class BackendUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class KVAllocation:
    allocation_id: str
    request_id: str


@dataclass
class PrefillState:
    request_id: str
    allocation_id: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: list[int]

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def generated_tokens(self) -> int:
        return len(self.generated_token_ids)


@dataclass(frozen=True)
class DecodeStep:
    logits: Logits
    finished: bool = False
    finish_reason: str | None = None


class Backend(ABC):
    kind: BackendKind

    @property
    @abstractmethod
    def tokenizer(self) -> Tokenizer:
        raise NotImplementedError

    @abstractmethod
    def load_model(self, config: EngineConfig) -> None:
        raise NotImplementedError

    @abstractmethod
    def allocate_kv(self, request: GenerationRequest) -> KVAllocation:
        raise NotImplementedError

    @abstractmethod
    def prefill(
        self,
        request: GenerationRequest,
        allocation: KVAllocation,
        prompt_token_ids: tuple[int, ...],
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
