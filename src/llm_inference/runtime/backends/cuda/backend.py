"""CUDA backend boundary.

The real GPU path lands here after M2. It remains independent from CpuBackend.
"""

from __future__ import annotations

from llm_inference.api.types import GenerationRequest
from llm_inference.config import BackendKind, EngineConfig
from llm_inference.runtime.contracts import (
    BackendUnavailable,
    DecodeStep,
    KVAllocation,
    PrefillState,
)


class CudaBackend:
    kind = BackendKind.CUDA

    def load_model(self, config: EngineConfig) -> None:
        raise BackendUnavailable(
            "CUDA backend skeleton is present, but GPU execution is not implemented yet"
        )

    def allocate_kv(self, request: GenerationRequest) -> KVAllocation:
        raise BackendUnavailable("CUDA KV allocation is not implemented yet")

    def prefill(
        self, request: GenerationRequest, allocation: KVAllocation
    ) -> PrefillState:
        raise BackendUnavailable("CUDA prefill is not implemented yet")

    def decode_step(
        self, request: GenerationRequest, state: PrefillState
    ) -> DecodeStep:
        raise BackendUnavailable("CUDA decode is not implemented yet")

    def free_kv(self, allocation: KVAllocation) -> None:
        return None

