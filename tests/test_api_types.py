import unittest

from llm_inference.api import GenerationRequest, GenerationResult, TokenChunk


class ApiTypesTests(unittest.TestCase):
    def test_generation_request_defaults_are_valid(self) -> None:
        request = GenerationRequest(prompt="hello")

        request.validate()

        self.assertEqual(request.max_new_tokens, 16)
        self.assertEqual(request.temperature, 0.0)
        self.assertEqual(request.top_p, 1.0)
        self.assertIsInstance(request.request_id, str)

    def test_generation_request_ids_are_unique_by_default(self) -> None:
        first = GenerationRequest(prompt="hello")
        second = GenerationRequest(prompt="hello")

        self.assertNotEqual(first.request_id, second.request_id)

    def test_generation_request_rejects_empty_prompt(self) -> None:
        with self.assertRaises(ValueError):
            GenerationRequest(prompt="").validate()

    def test_generation_request_rejects_invalid_max_new_tokens(self) -> None:
        with self.assertRaises(ValueError):
            GenerationRequest(prompt="hello", max_new_tokens=0).validate()

    def test_generation_request_rejects_negative_temperature(self) -> None:
        with self.assertRaises(ValueError):
            GenerationRequest(prompt="hello", temperature=-0.1).validate()

    def test_generation_request_rejects_invalid_top_p(self) -> None:
        for value in (0.0, 1.1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    GenerationRequest(prompt="hello", top_p=value).validate()

    def test_generation_request_rejects_invalid_top_k(self) -> None:
        with self.assertRaises(ValueError):
            GenerationRequest(prompt="hello", top_k=0).validate()

    def test_generation_result_keeps_token_chunks(self) -> None:
        chunk = TokenChunk(token_id=10, text="mock_0")
        result = GenerationResult(
            request_id="request-1",
            prompt="hello",
            output_text="mock_0",
            generated_tokens=(chunk,),
            finish_reason="length",
            backend="mock",
        )

        self.assertEqual(result.generated_tokens, (chunk,))
        self.assertEqual(result.output_text, "mock_0")


if __name__ == "__main__":
    unittest.main()
