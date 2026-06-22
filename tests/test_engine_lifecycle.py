import unittest
from unittest.mock import patch

from llm_inference.api import GenerationRequest
from llm_inference.audit import AuditEventType, EventRecorder
from llm_inference.config import BackendKind, EngineConfig
from llm_inference.engine import InferenceEngine
from llm_inference.runtime.contracts import DecodeStep, KVAllocation, PrefillState
from llm_inference.tokenization import MockTokenizer


class FailingPrefillBackend:
    kind = BackendKind.MOCK

    def __init__(self) -> None:
        self._tokenizer = MockTokenizer()
        self.loaded_count = 0
        self.freed_allocations: list[str] = []

    @property
    def tokenizer(self) -> MockTokenizer:
        return self._tokenizer

    def load_model(self, config: EngineConfig) -> None:
        self.loaded_count += 1

    def allocate_kv(self, request: GenerationRequest) -> KVAllocation:
        return KVAllocation(allocation_id="failing-kv", request_id=request.request_id)

    def prefill(
        self,
        request: GenerationRequest,
        allocation: KVAllocation,
        prompt_token_ids: tuple[int, ...],
    ) -> PrefillState:
        raise RuntimeError("prefill failed")

    def decode_step(self, request: GenerationRequest, state: PrefillState) -> DecodeStep:
        raise AssertionError("decode should not be called")

    def free_kv(self, allocation: KVAllocation) -> None:
        self.freed_allocations.append(allocation.allocation_id)


class EngineLifecycleTests(unittest.TestCase):
    def test_invalid_request_fails_before_backend_starts(self) -> None:
        recorder = EventRecorder()
        engine = InferenceEngine(EngineConfig.for_backend("mock"), recorder=recorder)

        with self.assertRaises(ValueError):
            engine.generate(GenerationRequest(prompt="", max_new_tokens=1))

        self.assertEqual(recorder.events, ())

    def test_engine_records_failure_and_frees_kv(self) -> None:
        backend = FailingPrefillBackend()
        recorder = EventRecorder()

        with patch("llm_inference.engine.core.create_backend", return_value=backend):
            engine = InferenceEngine(EngineConfig.for_backend("mock"), recorder=recorder)
            with self.assertRaises(RuntimeError):
                engine.generate(GenerationRequest(prompt="fail", max_new_tokens=1))

        event_types = [event.event_type for event in recorder.events]
        self.assertIn(AuditEventType.REQUEST_FAILED, event_types)
        self.assertEqual(backend.freed_allocations, ["failing-kv"])

    def test_engine_start_loads_model_once(self) -> None:
        backend = FailingPrefillBackend()

        with patch("llm_inference.engine.core.create_backend", return_value=backend):
            engine = InferenceEngine(EngineConfig.for_backend("mock"))
            engine.start()
            engine.start()

        self.assertEqual(backend.loaded_count, 1)


if __name__ == "__main__":
    unittest.main()
