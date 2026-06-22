import unittest

from llm_inference.api import GenerationRequest, TokenChunk
from llm_inference.decoding import GreedySampler, StopController


class DecodingTests(unittest.TestCase):
    def test_greedy_sampler_selects_max_from_sequence_logits(self) -> None:
        self.assertEqual(GreedySampler().select((0.1, 0.5, 0.2)), 1)

    def test_greedy_sampler_selects_max_from_mapping_logits(self) -> None:
        self.assertEqual(GreedySampler().select({10: 0.1, 11: 1.0}), 11)

    def test_greedy_sampler_rejects_empty_logits(self) -> None:
        with self.assertRaises(ValueError):
            GreedySampler().select(())

    def test_stop_controller_stops_on_length(self) -> None:
        request = GenerationRequest(prompt="hello", max_new_tokens=1)
        chunks = (TokenChunk(token_id=10, text="mock_0"),)

        decision = StopController().should_stop(request, chunks)

        self.assertTrue(decision.stopped)
        self.assertEqual(decision.reason, "length")

    def test_stop_controller_stops_on_stop_text(self) -> None:
        request = GenerationRequest(prompt="hello", max_new_tokens=4, stop=("mock_1",))
        chunks = (
            TokenChunk(token_id=10, text="mock_0"),
            TokenChunk(token_id=11, text="mock_1"),
        )

        decision = StopController().should_stop(request, chunks)

        self.assertTrue(decision.stopped)
        self.assertEqual(decision.reason, "stop")

    def test_stop_controller_stops_on_eos_token(self) -> None:
        request = GenerationRequest(prompt="hello", max_new_tokens=4)
        chunks = (TokenChunk(token_id=0, text=""),)

        decision = StopController().should_stop(request, chunks, eos_token_id=0)

        self.assertTrue(decision.stopped)
        self.assertEqual(decision.reason, "eos")

    def test_stop_controller_continues_when_no_condition_matches(self) -> None:
        request = GenerationRequest(prompt="hello", max_new_tokens=4, stop=("done",))
        chunks = (TokenChunk(token_id=10, text="mock_0"),)

        decision = StopController().should_stop(request, chunks)

        self.assertFalse(decision.stopped)


if __name__ == "__main__":
    unittest.main()
