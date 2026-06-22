"""Stop condition handling."""

from __future__ import annotations

from dataclasses import dataclass

from llm_inference.api.types import GenerationRequest, TokenChunk


@dataclass(frozen=True)
class StopDecision:
    stopped: bool
    reason: str | None = None


class StopController:
    def should_stop(
        self, request: GenerationRequest, chunks: tuple[TokenChunk, ...]
    ) -> StopDecision:
        if len(chunks) >= request.max_new_tokens:
            return StopDecision(stopped=True, reason="length")
        output_text = "".join(chunk.text for chunk in chunks)
        for stop_text in request.stop:
            if stop_text and stop_text in output_text:
                return StopDecision(stopped=True, reason="stop")
        return StopDecision(stopped=False)

