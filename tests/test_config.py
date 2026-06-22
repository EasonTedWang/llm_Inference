import unittest

from llm_inference.config import BackendKind, EngineConfig, MemoryConfig, SchedulerConfig


class ConfigTests(unittest.TestCase):
    def test_backend_parse_accepts_supported_values(self) -> None:
        self.assertEqual(BackendKind.parse("mock"), BackendKind.MOCK)
        self.assertEqual(BackendKind.parse("CPU"), BackendKind.CPU)

    def test_backend_parse_rejects_unknown_value(self) -> None:
        with self.assertRaises(ValueError):
            BackendKind.parse("gpu")

    def test_engine_config_validates_nested_config(self) -> None:
        config = EngineConfig(
            scheduler=SchedulerConfig(max_batch_size=1),
            memory=MemoryConfig(kv_block_size=16, max_kv_blocks=2),
        )

        config.validate()

    def test_engine_config_rejects_invalid_memory(self) -> None:
        config = EngineConfig(memory=MemoryConfig(max_kv_blocks=0))

        with self.assertRaises(ValueError):
            config.validate()


if __name__ == "__main__":
    unittest.main()

