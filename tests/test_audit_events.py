import unittest

from llm_inference.audit import AuditEventType, EventRecorder


class AuditEventsTests(unittest.TestCase):
    def test_event_recorder_records_payload_and_order(self) -> None:
        recorder = EventRecorder()

        first = recorder.record(AuditEventType.REQUEST_RECEIVED, "request-1", backend="mock")
        second = recorder.record(AuditEventType.REQUEST_FINISHED, "request-1", generated_tokens=1)

        self.assertEqual(recorder.events, (first, second))
        self.assertEqual(first.payload["backend"], "mock")
        self.assertEqual(second.payload["generated_tokens"], 1)

    def test_event_recorder_clear_removes_events(self) -> None:
        recorder = EventRecorder()
        recorder.record(AuditEventType.REQUEST_RECEIVED, "request-1")

        recorder.clear()

        self.assertEqual(recorder.events, ())


if __name__ == "__main__":
    unittest.main()
