import unittest

from llm_inference.api import GenerationRequest
from llm_inference.config import EngineConfig
from llm_inference.decoding import GreedySampler
from llm_inference.engine import InferenceEngine
from llm_inference.runtime.backends.mock import MockBackend
from llm_inference.tokenization import MockTokenizer


class TokenizationModelTests(unittest.TestCase):
    def test_mock_tokenizer_encodes_prompt_and_decodes_generated_tokens(self) -> None:
        tokenizer = MockTokenizer()

        self.assertEqual(tokenizer.encode("hello world"), (1000, 1001))
        self.assertEqual(tokenizer.decode_token(10), "mock_0")
        self.assertEqual(tokenizer.decode_token(100), "cpu_0")
        self.assertEqual(tokenizer.decode((10, 11)), "mock_0mock_1")

    def test_mock_backend_returns_logits_for_sampler(self) -> None:
        backend = MockBackend()
        request = GenerationRequest(prompt="hello")
        allocation = backend.allocate_kv(request)
        state = backend.prefill(request, allocation, backend.tokenizer.encode("hello"))

        step = backend.decode_step(request, state)
        token_id = GreedySampler().select(step.logits)

        self.assertEqual(token_id, 10)
        self.assertEqual(backend.tokenizer.decode_token(token_id), "mock_0")

    def test_engine_uses_tokenizer_sampler_and_logits_loop(self) -> None:
        engine = InferenceEngine(EngineConfig.for_backend("mock"))

        result = engine.generate(GenerationRequest(prompt="golden", max_new_tokens=3))

        self.assertEqual(result.output_text, "mock_0mock_1mock_2")
        self.assertEqual(
            tuple(chunk.token_id for chunk in result.generated_tokens),
            (10, 11, 12),
        )

    def test_engine_stops_on_decoded_stop_text(self) -> None:
        engine = InferenceEngine(EngineConfig.for_backend("mock"))

        result = engine.generate(
            GenerationRequest(prompt="stop test", max_new_tokens=5, stop=("mock_1",))
        )

        self.assertEqual(result.output_text, "mock_0mock_1")
        self.assertEqual(result.finish_reason, "stop")


if __name__ == "__main__":
    unittest.main()
