import unittest

from llm_inference.api import GenerationRequest
from llm_inference.memory import KVCacheManager
from llm_inference.scheduler import BasicScheduler, ScheduledRequest


class SchedulerMemoryTests(unittest.TestCase):
    def test_basic_scheduler_dequeues_fifo_batch(self) -> None:
        scheduler = BasicScheduler()
        first = ScheduledRequest(GenerationRequest(prompt="first"))
        second = ScheduledRequest(GenerationRequest(prompt="second"))
        scheduler.enqueue(first)
        scheduler.enqueue(second)

        batch = scheduler.dequeue_batch(max_batch_size=1)

        self.assertEqual(batch, (first,))
        self.assertEqual(len(scheduler), 1)

    def test_kv_cache_manager_allocates_and_frees_blocks(self) -> None:
        manager = KVCacheManager(max_blocks=2)
        blocks = manager.allocate("request-1", count=2)

        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(manager.active_blocks), 2)

        manager.free(tuple(block.block_id for block in blocks))

        self.assertEqual(manager.active_blocks, ())

    def test_kv_cache_manager_rejects_over_allocation(self) -> None:
        manager = KVCacheManager(max_blocks=1)

        with self.assertRaises(MemoryError):
            manager.allocate("request-1", count=2)


if __name__ == "__main__":
    unittest.main()

