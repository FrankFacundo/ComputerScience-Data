import asyncio
import json
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List

from mllm import Mllm


@dataclass
class Task(object):
    task_awaitable: Any  # Change to available
    prompt: str
    index: int


class AsyncScheduler(object):
    def __init__(self, online_serving=False, max_running_tasks=5):
        self.queue = asyncio.Queue()
        self.running_tasks = set()
        self.max_running_tasks = max_running_tasks
        self.semaphore = asyncio.Semaphore(self.max_running_tasks)
        self.online_serving = online_serving
        self.datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.mllm = Mllm()
        self.results = {}

        if online_serving:
            self.loop_task = asyncio.create_task(self.serving_loop())

    async def add_to_queue(self, task: Task) -> None:
        await self.queue.put(task)

    async def _run_task(self, task: Task) -> Any:
        try:
            index, result = await task.task_awaitable
            self.results[index] = result
            if not result:
                task = Task(
                    task_awaitable=self.generate_semaphore(
                        prompt=task.prompt, index=task.index
                    ),
                    prompt=task.prompt,
                    index=task.index,
                )
                await self.add_to_queue(task)
        except Exception as e:
            print(e)
        finally:
            with open(f"results_{self.datetime}.jsonl", "a") as f:
                json_record = json.dumps(
                    {"index": index, "result": result}, ensure_ascii=False
                )
                f.write(json_record + "\n")
            self.running_tasks.remove(asyncio.current_task())

    async def _serving_loop(self) -> None:
        while True:
            if (
                len(self.running_tasks) < self.max_running_tasks
                and not self.queue.empty()
            ):
                task = await self.queue.get()
                task_wrapper = asyncio.create_task(self._run_task(task))
                self.running_tasks.add(task_wrapper)

            await asyncio.sleep(0.1)

    async def generate_batch(self, prompts: List[str]) -> List[str]:
        results_texts = {}
        prompts_dict = {i: prompt for i, prompt in enumerate(prompts)}

        tasks = [
            self.generate_semaphore(prompt, i) for i, prompt in prompts_dict.items()
        ]

        for task in asyncio.as_completed(tasks):
            index, result = await task
            results_texts[index] = result

        ordered_results = [results_texts[index] for index in prompts_dict]
        return ordered_results

    async def generate_batch_fault_tolerant(self, prompts: List[str]) -> List[str]:
        self.results = {}
        for i, prompt in enumerate(prompts):
            task = Task(
                task_awaitable=self.generate_semaphore(prompt=prompt, index=i),
                prompt=prompt,
                index=i,
            )
            await self.add_to_queue(task)

        while not (self.queue.empty() and len(self.running_tasks) == 0):
            if not self.queue.empty():
                task = await self.queue.get()
                task_wrapper = asyncio.create_task(self._run_task(task))
                self.running_tasks.add(task_wrapper)
            await asyncio.sleep(0.1)

        return [self.results[i] for i in range(len(prompts))]

    async def generate_semaphore(self, prompt: str, index: int) -> tuple:
        async with self.semaphore:
            result = await self.mllm.generate(prompt)
            return index, result

    async def random_time_execution_semaphore(self, prompt, index):
        async with self.semaphore:
            start_time = datetime.now()
            print(f"Begin of job {index}: {start_time}")
            time_to_wait = random.randint(0, 5)
            await asyncio.sleep(time_to_wait)
            end_time = datetime.now()
            print(
                f"End of job {index}: {datetime.now()} Total time for job {index}: {end_time - start_time}"
            )

            if index == 10:
                # This will cause infinite loop, since task 10 is always failing. This was done for testing the scheduler.
                return index, None
            return index, time_to_wait

    def __del__(self):
        if self.online_serving:
            self.loop_task.cancel()


async def main_online_serving():
    manager = AsyncScheduler(online_serving=True)
    mllm = Mllm()
    for _ in range(10):
        await manager.add_to_queue(mllm.generate("hello"))

    await asyncio.sleep(20)


async def main():
    manager = AsyncScheduler(max_running_tasks=5)
    prompts = [
        "What are the health benefits of a Mediterranean diet?",
        "Describe the concept of quantum computing.",
        "How do you make a traditional Italian pizza?",
        "What are the top tourist attractions in Japan?",
        "Explain the process of photosynthesis.",
        "What is the history of the Roman Empire?",
        "How can I improve my public speaking skills?",
        "What are some effective time management techniques?",
        "Explain the theory of relativity in simple terms.",
        "What are the most common programming languages used in web development?",
    ]

    results = await manager.generate_batch_fault_tolerant(prompts)
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
