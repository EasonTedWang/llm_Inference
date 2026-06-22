"""Backend factory."""

from __future__ import annotations

from llm_inference.config import BackendKind
from llm_inference.runtime.backends.cpu import CpuBackend
from llm_inference.runtime.backends.cuda import CudaBackend
from llm_inference.runtime.backends.mock import MockBackend
from llm_inference.runtime.contracts import Backend


def create_backend(kind: str | BackendKind) -> Backend:
    backend_kind = BackendKind.parse(kind)
    if backend_kind is BackendKind.CPU:
        return CpuBackend()
    if backend_kind is BackendKind.CUDA:
        return CudaBackend()
    if backend_kind is BackendKind.MOCK:
        return MockBackend()
    raise ValueError(f"unsupported backend: {kind}")

