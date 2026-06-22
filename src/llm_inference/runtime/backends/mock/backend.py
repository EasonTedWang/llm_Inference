"""Deterministic mock backend for tests."""

from __future__ import annotations

from llm_inference.api.types import GenerationRequest
from llm_inference.config import BackendKind, EngineConfig
from llm_inference.runtime.contracts import DecodeStep, KVAllocation, PrefillState


class MockBackend:
    kind = BackendKind.MOCK

    def __init__(self) -> None:
        self.loaded_model: str | None = None
        self._allocations: set[str] = set()

    def load_model(self, config: EngineConfig) -> None:
        self.loaded_model = config.model.model_name

    def allocate_kv(self, request: GenerationRequest) -> KVAllocation:
        allocation = KVAllocation(
            allocation_id=f"mock-kv-{request.request_id}",
            request_id=request.request_id,
        )
        self._allocations.add(allocation.allocation_id)
        return allocation

    def prefill(
        self, request: GenerationRequest, allocation: KVAllocation
    ) -> PrefillState:
        if allocation.allocation_id not in self._allocations:
            raise RuntimeError("unknown KV allocation")
        return PrefillState(
            request_id=request.request_id,
            prompt_tokens=len(request.prompt.split()),
        )

    def decode_step(
        self, request: GenerationRequest, state: PrefillState
    ) -> DecodeStep:
        token_index = state.generated_tokens
        state.generated_tokens += 1
        return DecodeStep(token_id=token_index, token_text=f"mock_{token_index}")

    def free_kv(self, allocation: KVAllocation) -> None:
        self._allocations.discard(allocation.allocation_id)

