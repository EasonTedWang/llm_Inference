"""Structured audit event primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class AuditEventType(str, Enum):
    REQUEST_RECEIVED = "request_received"
    BACKEND_SELECTED = "backend_selected"
    KV_ALLOCATED = "kv_allocated"
    PREFILL_COMPLETED = "prefill_completed"
    DECODE_STEP_COMPLETED = "decode_step_completed"
    REQUEST_FINISHED = "request_finished"
    REQUEST_FAILED = "request_failed"


@dataclass(frozen=True)
class AuditEvent:
    event_type: AuditEventType
    request_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EventRecorder:
    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    @property
    def events(self) -> tuple[AuditEvent, ...]:
        return tuple(self._events)

    def record(
        self, event_type: AuditEventType, request_id: str, **payload: Any
    ) -> AuditEvent:
        event = AuditEvent(event_type=event_type, request_id=request_id, payload=payload)
        self._events.append(event)
        return event

    def clear(self) -> None:
        self._events.clear()

