import asyncio
import random

from openai import (
    APITimeoutError,
    OpenAIError,
)


class Mllm:
    async def generate(self, prompt: str) -> str:
        # Simulate some delay
        await asyncio.sleep(random.uniform(0.1, 0.5))

        exceptions = [
            OpenAIError("A general OpenAI error occurred"),
            APITimeoutError("API Timeout occurred"),
        ]

        # 90% chance to return default response
        if random.random() < 0.01:
            return f"Default response for prompt: {prompt}"
        else:
            # Raise one of the exceptions with 10% probability
            random_exception = random.choice(exceptions)
            print(random_exception)
            raise random_exception


if __name__ == "__main__":
    mllm = Mllm()
    try:
        result = asyncio.run(mllm.generate("Hello, world!"))
        print(result)
    except OpenAIError as e:
        print(f"An OpenAIError occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
