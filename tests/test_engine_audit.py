import unittest

from llm_inference.api import GenerationRequest
from llm_inference.audit import AuditEventType, EventRecorder
from llm_inference.config import EngineConfig
from llm_inference.engine import InferenceEngine


class EngineAuditTests(unittest.TestCase):
    def test_generate_records_structured_events(self) -> None:
        recorder = EventRecorder()
        engine = InferenceEngine(EngineConfig.for_backend("mock"), recorder=recorder)
        result = engine.generate(GenerationRequest(prompt="audit", max_new_tokens=1))

        event_types = [event.event_type for event in recorder.events]
        self.assertEqual(result.backend, "mock")
        self.assertIn(AuditEventType.REQUEST_RECEIVED, event_types)
        self.assertIn(AuditEventType.BACKEND_SELECTED, event_types)
        self.assertIn(AuditEventType.KV_ALLOCATED, event_types)
        self.assertIn(AuditEventType.PREFILL_COMPLETED, event_types)
        self.assertIn(AuditEventType.DECODE_STEP_COMPLETED, event_types)
        self.assertIn(AuditEventType.REQUEST_FINISHED, event_types)

        prefill_event = next(
            event
            for event in recorder.events
            if event.event_type is AuditEventType.PREFILL_COMPLETED
        )
        self.assertEqual(prefill_event.payload["prompt_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
