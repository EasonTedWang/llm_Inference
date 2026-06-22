"""CUDA kernel boundary markers.

Real CUDA, Triton, FlashInfer, and CUDA Graph integrations belong here in M5.
"""

from __future__ import annotations


class CudaKernelUnavailable(RuntimeError):
    pass

