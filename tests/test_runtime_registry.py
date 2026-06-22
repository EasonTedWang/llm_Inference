import unittest

from llm_inference.config import BackendKind
from llm_inference.runtime.backends.cpu import CpuBackend
from llm_inference.runtime.backends.cuda import CudaBackend
from llm_inference.runtime.backends.mock import MockBackend
from llm_inference.runtime.registry import create_backend


class RuntimeRegistryTests(unittest.TestCase):
    def test_create_backend_from_enum(self) -> None:
        self.assertIsInstance(create_backend(BackendKind.MOCK), MockBackend)

    def test_create_backend_from_string(self) -> None:
        self.assertIsInstance(create_backend("cpu"), CpuBackend)

    def test_create_cuda_backend_boundary(self) -> None:
        self.assertIsInstance(create_backend("cuda"), CudaBackend)

    def test_create_backend_rejects_unknown_backend(self) -> None:
        with self.assertRaises(ValueError):
            create_backend("gpu")


if __name__ == "__main__":
    unittest.main()
