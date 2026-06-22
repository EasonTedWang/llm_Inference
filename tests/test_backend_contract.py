import unittest

from llm_inference.api import GenerationRequest
from llm_inference.config import BackendKind, EngineConfig
from llm_inference.engine import InferenceEngine
from llm_inference.runtime.backends.cpu import CpuBackend
from llm_inference.runtime.backends.mock import MockBackend
from llm_inference.runtime import BackendUnavailable
from llm_inference.decoding import GreedySampler


class BackendContractTests(unittest.TestCase):
    def test_mock_backend_generates_tokens(self) -> None:
        engine = InferenceEngine(EngineConfig.for_backend(BackendKind.MOCK))
        result = engine.generate(GenerationRequest(prompt="hello", max_new_tokens=3))

        self.assertEqual(result.backend, "mock")
        self.assertEqual(result.output_text, "mock_0mock_1mock_2")
        self.assertEqual(tuple(chunk.token_id for chunk in result.generated_tokens), (10, 11, 12))
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(len(result.generated_tokens), 3)

    def test_cpu_backend_is_independent_selectable_path(self) -> None:
        engine = InferenceEngine(EngineConfig.for_backend(BackendKind.CPU))
        result = engine.generate(GenerationRequest(prompt="hello cpu", max_new_tokens=2))

        self.assertEqual(result.backend, "cpu")
        self.assertEqual(result.output_text, "cpu_0cpu_1")
        self.assertEqual(tuple(chunk.token_id for chunk in result.generated_tokens), (100, 101))
        self.assertEqual(len(result.generated_tokens), 2)

    def test_cuda_backend_boundary_is_explicit(self) -> None:
        engine = InferenceEngine(EngineConfig.for_backend(BackendKind.CUDA))

        with self.assertRaises(BackendUnavailable):
            engine.generate(GenerationRequest(prompt="hello", max_new_tokens=1))

    def test_mock_backend_direct_contract_returns_logits(self) -> None:
        backend = MockBackend()
        request = GenerationRequest(prompt="direct")
        allocation = backend.allocate_kv(request)
        state = backend.prefill(request, allocation, backend.tokenizer.encode(request.prompt))
        step = backend.decode_step(request, state)

        self.assertEqual(state.request_id, request.request_id)
        self.assertEqual(state.prompt_token_ids, (1000,))
        self.assertEqual(GreedySampler().select(step.logits), 10)

    def test_cpu_backend_direct_contract_returns_logits(self) -> None:
        backend = CpuBackend()
        request = GenerationRequest(prompt="direct cpu")
        allocation = backend.allocate_kv(request)
        state = backend.prefill(request, allocation, backend.tokenizer.encode(request.prompt))
        step = backend.decode_step(request, state)

        self.assertEqual(state.prompt_token_ids, (1000, 1001))
        self.assertEqual(GreedySampler().select(step.logits), 100)

    def test_backend_rejects_unknown_kv_allocation_after_free(self) -> None:
        backend = MockBackend()
        request = GenerationRequest(prompt="cleanup")
        allocation = backend.allocate_kv(request)
        backend.free_kv(allocation)

        with self.assertRaises(RuntimeError):
            backend.prefill(request, allocation, backend.tokenizer.encode(request.prompt))


if __name__ == "__main__":
    unittest.main()
