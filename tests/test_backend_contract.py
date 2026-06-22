import unittest

from llm_inference.api import GenerationRequest
from llm_inference.config import BackendKind, EngineConfig
from llm_inference.engine import InferenceEngine
from llm_inference.runtime import BackendUnavailable


class BackendContractTests(unittest.TestCase):
    def test_mock_backend_generates_tokens(self) -> None:
        engine = InferenceEngine(EngineConfig.for_backend(BackendKind.MOCK))
        result = engine.generate(GenerationRequest(prompt="hello", max_new_tokens=3))

        self.assertEqual(result.backend, "mock")
        self.assertEqual(result.output_text, "mock_0mock_1mock_2")
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(len(result.generated_tokens), 3)

    def test_cpu_backend_is_independent_selectable_path(self) -> None:
        engine = InferenceEngine(EngineConfig.for_backend(BackendKind.CPU))
        result = engine.generate(GenerationRequest(prompt="hello cpu", max_new_tokens=2))

        self.assertEqual(result.backend, "cpu")
        self.assertEqual(result.output_text, "cpu_0cpu_1")
        self.assertEqual(len(result.generated_tokens), 2)

    def test_cuda_backend_boundary_is_explicit(self) -> None:
        engine = InferenceEngine(EngineConfig.for_backend(BackendKind.CUDA))

        with self.assertRaises(BackendUnavailable):
            engine.generate(GenerationRequest(prompt="hello", max_new_tokens=1))


if __name__ == "__main__":
    unittest.main()

