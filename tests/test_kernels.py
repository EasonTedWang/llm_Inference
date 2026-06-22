import unittest

from llm_inference.kernels.cpu.reference import add_vectors
from llm_inference.kernels.cuda.placeholders import CudaKernelUnavailable


class KernelBoundaryTests(unittest.TestCase):
    def test_cpu_reference_add_vectors(self) -> None:
        self.assertEqual(add_vectors((1.0, 2.0), (3.0, 4.0)), (4.0, 6.0))

    def test_cpu_reference_add_vectors_rejects_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            add_vectors((1.0,), (1.0, 2.0))

    def test_cuda_kernel_unavailable_is_runtime_error(self) -> None:
        self.assertTrue(issubclass(CudaKernelUnavailable, RuntimeError))


if __name__ == "__main__":
    unittest.main()
