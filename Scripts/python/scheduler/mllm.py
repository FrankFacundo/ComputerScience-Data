import asyncio
import random

import httpx
from openai import (
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAIError,
    RateLimitError,
)


class Mllm:
    async def generate(self, prompt: str) -> str:
        # Simulate some delay
        await asyncio.sleep(random.uniform(0.1, 0.5))

        response = httpx.Response(
            status_code=400,
            headers={"x-request-id": "abc123"},
            content=b'{"error": "Invalid request parameters"}',
        )

        exceptions = [
            OpenAIError(),
            RateLimitError(
                message="Rate limit exceeded",
                response=response,
                body={"error": "Invalid request parameters"},
            ),
            APITimeoutError("API Timeout occurred"),
            BadRequestError("Bad request"),
            InternalServerError("Internal server error"),
        ]

        # 90% chance to return default response
        if random.random() < 0.9:
            return f"Default response for prompt: {prompt}"
        else:
            # Raise one of the exceptions with 10% probability
            raise random.choice(exceptions)
