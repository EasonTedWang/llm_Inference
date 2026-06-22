"""Engine orchestration for the initial project skeleton."""

from __future__ import annotations

from llm_inference.api.types import GenerationRequest, GenerationResult, TokenChunk
from llm_inference.audit import AuditEventType, EventRecorder
from llm_inference.config import EngineConfig
from llm_inference.decoding import StopController
from llm_inference.observability import configure_logging
from llm_inference.runtime.registry import create_backend


class InferenceEngine:
    def __init__(
        self,
        config: EngineConfig | None = None,
        recorder: EventRecorder | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.config.validate()
        self.recorder = recorder or EventRecorder()
        self.backend = create_backend(self.config.backend)
        self.stop_controller = StopController()
        self._model_loaded = False
        configure_logging(self.config.observability.log_level)

    def start(self) -> None:
        if self._model_loaded:
            return
        self.backend.load_model(self.config)
        self._model_loaded = True

    def generate(self, request: GenerationRequest) -> GenerationResult:
        request.validate()
        self.start()
        self.recorder.record(
            AuditEventType.REQUEST_RECEIVED,
            request.request_id,
            backend=self.backend.kind.value,
        )
        self.recorder.record(
            AuditEventType.BACKEND_SELECTED,
            request.request_id,
            backend=self.backend.kind.value,
        )

        allocation = self.backend.allocate_kv(request)
        self.recorder.record(
            AuditEventType.KV_ALLOCATED,
            request.request_id,
            allocation_id=allocation.allocation_id,
            backend=self.backend.kind.value,
        )

        chunks: list[TokenChunk] = []
        finish_reason = "length"
        try:
            state = self.backend.prefill(request, allocation)
            self.recorder.record(
                AuditEventType.PREFILL_COMPLETED,
                request.request_id,
                prompt_tokens=state.prompt_tokens,
            )
            while True:
                step = self.backend.decode_step(request, state)
                chunk = TokenChunk(token_id=step.token_id, text=step.token_text)
                chunks.append(chunk)
                self.recorder.record(
                    AuditEventType.DECODE_STEP_COMPLETED,
                    request.request_id,
                    token_id=step.token_id,
                    token_text=step.token_text,
                )
                stop = self.stop_controller.should_stop(request, tuple(chunks))
                if step.finished:
                    finish_reason = step.finish_reason or "backend"
                    break
                if stop.stopped:
                    finish_reason = stop.reason or "stop"
                    break
            output_text = "".join(chunk.text for chunk in chunks)
            result = GenerationResult(
                request_id=request.request_id,
                prompt=request.prompt,
                output_text=output_text,
                generated_tokens=tuple(chunks),
                finish_reason=finish_reason,
                backend=self.backend.kind.value,
            )
            self.recorder.record(
                AuditEventType.REQUEST_FINISHED,
                request.request_id,
                finish_reason=finish_reason,
                generated_tokens=len(chunks),
            )
            return result
        except Exception as exc:
            self.recorder.record(
                AuditEventType.REQUEST_FAILED,
                request.request_id,
                error=type(exc).__name__,
            )
            raise
        finally:
            self.backend.free_kv(allocation)

