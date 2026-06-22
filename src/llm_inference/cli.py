"""Command line entry point for local smoke tests."""

from __future__ import annotations

import argparse

from llm_inference.api.types import GenerationRequest
from llm_inference.config import BackendKind, EngineConfig
from llm_inference.engine import InferenceEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-inference")
    parser.add_argument("prompt", help="Prompt text to generate from.")
    parser.add_argument(
        "--backend",
        choices=[item.value for item in BackendKind],
        default=BackendKind.MOCK.value,
        help="Execution backend. Each request runs in exactly one selected backend.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = EngineConfig.for_backend(args.backend)
    engine = InferenceEngine(config)
    result = engine.generate(
        GenerationRequest(prompt=args.prompt, max_new_tokens=args.max_new_tokens)
    )
    print(result.output_text)
    return 0

