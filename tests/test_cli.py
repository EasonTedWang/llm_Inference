import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

from llm_inference.cli import build_parser, main


class CliTests(unittest.TestCase):
    def test_parser_defaults_to_mock_backend(self) -> None:
        args = build_parser().parse_args(["hello"])

        self.assertEqual(args.backend, "mock")
        self.assertEqual(args.max_new_tokens, 4)

    def test_main_prints_generated_output(self) -> None:
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = main(["hello", "--backend", "mock", "--max-new-tokens", "2"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "mock_0mock_1")

    def test_parser_rejects_unknown_backend(self) -> None:
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["hello", "--backend", "gpu"])


if __name__ == "__main__":
    unittest.main()
