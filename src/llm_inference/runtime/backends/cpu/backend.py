"""Independent CPU backend skeleton."""

from __future__ import annotations

from llm_inference.api.types import GenerationRequest
from llm_inference.config import BackendKind, EngineConfig
from llm_inference.runtime.contracts import DecodeStep, KVAllocation, PrefillState
from llm_inference.runtime.backends.cpu.model import CpuReferenceModel
from llm_inference.tokenization import MockTokenizer, Tokenizer


class CpuBackend:
    kind = BackendKind.CPU

    def __init__(self) -> None:
        self.loaded_model: str | None = None
        self._allocations: set[str] = set()
        self._tokenizer = MockTokenizer()
        self._model = CpuReferenceModel(token_base=self._tokenizer.cpu_token_base)

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    def load_model(self, config: EngineConfig) -> None:
        self.loaded_model = config.model.model_name

    def allocate_kv(self, request: GenerationRequest) -> KVAllocation:
        allocation = KVAllocation(
            allocation_id=f"cpu-kv-{request.request_id}",
            request_id=request.request_id,
        )
        self._allocations.add(allocation.allocation_id)
        return allocation

    def prefill(
        self,
        request: GenerationRequest,
        allocation: KVAllocation,
        prompt_token_ids: tuple[int, ...],
    ) -> PrefillState:
        if allocation.allocation_id not in self._allocations:
            raise RuntimeError("unknown KV allocation")
        return PrefillState(
            request_id=request.request_id,
            allocation_id=allocation.allocation_id,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=[],
        )

    def decode_step(
        self, request: GenerationRequest, state: PrefillState
    ) -> DecodeStep:
        return DecodeStep(logits=self._model.next_logits(state.generated_tokens))

    def free_kv(self, allocation: KVAllocation) -> None:
        self._allocations.discard(allocation.allocation_id)
