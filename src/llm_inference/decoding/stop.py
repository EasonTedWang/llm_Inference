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
        self,
        request: GenerationRequest,
        chunks: tuple[TokenChunk, ...],
        eos_token_id: int | None = None,
    ) -> StopDecision:
        if len(chunks) >= request.max_new_tokens:
            return StopDecision(stopped=True, reason="length")
        if eos_token_id is not None and chunks and chunks[-1].token_id == eos_token_id:
            return StopDecision(stopped=True, reason="eos")
        output_text = "".join(chunk.text for chunk in chunks)
        for stop_text in request.stop:
            if stop_text and stop_text in output_text:
                return StopDecision(stopped=True, reason="stop")
        return StopDecision(stopped=False)
