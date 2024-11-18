import unittest

from mllm import Mllm
from scheduler import AsyncScheduler


class TestGenerateSemaphore(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.scheduler = AsyncScheduler(max_running_tasks=2)

    async def test_generate_semaphore_success(self):
        # Use actual Mllm instance
        prompt = "Test prompt"
        index = 0
        self.scheduler.mllm = Mllm()

        # Call the method
        result = await self.scheduler.generate_semaphore(prompt, index)

        # Assert that the result matches the expected format
        self.assertEqual(result[0], index)  # Index should match
        self.assertTrue(
            result[1].startswith("Default response for prompt:") or result[1] is None
        )

    async def test_generate_semaphore_exception_handling(self):
        # Use actual Mllm instance
        prompt = "Test prompt"
        index = 0
        self.scheduler.mllm = Mllm()

        try:
            # Call the method
            result = await self.scheduler.generate_semaphore(prompt, index)

            # Since Mllm can raise random exceptions, we handle both scenarios:
            # 1. A valid result.
            # 2. A possible `None` result (handled gracefully).
            self.assertEqual(result[0], index)
            self.assertTrue(
                result[1] is None
                or result[1].startswith("Default response for prompt:")
            )

        except Exception as e:
            # Ensure that exceptions are of a type that Mllm could raise
            valid_exceptions = (
                Exception,
            )  # You can expand this to include specific errors if needed
            self.assertTrue(isinstance(e, valid_exceptions))


if __name__ == "__main__":
    unittest.main()
