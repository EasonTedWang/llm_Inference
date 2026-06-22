"""Scheduler state types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from llm_inference.api.types import GenerationRequest


class RequestState(str, Enum):
    QUEUED = "queued"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SchedulerTaskType(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class ScheduledRequest:
    request: GenerationRequest
    state: RequestState = RequestState.QUEUED
    priority: int = 0

