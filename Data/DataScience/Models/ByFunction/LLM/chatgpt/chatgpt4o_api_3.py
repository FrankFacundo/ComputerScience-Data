import asyncio
import base64
import os

import httpx


# Function to make a request to OpenAI API
async def make_request(base64_image: str, prompt: str) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:  # Increase the timeout
        try:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json()
        except httpx.RequestError as exc:
            print(f"An error occurred while requesting {exc.request.url!r}.")
            return {"error": str(exc)}
        except httpx.HTTPStatusError as exc:
            print(
                f"Error response {exc.response.status_code} while requesting {exc.request.url!r}."
            )
            return {"error": str(exc)}
        except httpx.TimeoutException as exc:
            print(f"Request timed out while requesting {exc.request.url!r}.")
            return {"error": "timeout"}


# Function to launch multiple async tasks
async def launch_tasks(base64_image: str, prompts: list):
    tasks = [make_request(base64_image, prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


if __name__ == "__main__":
    # api_key = os.environ.get("OPENAI_API_KEY")
    prompts = [
        "What’s in this image?",
        "Why this image is considered art?",
        "In which year was made this image?",
    ]

    image_path = "el_jardin_de_las_delicias.jpg"
    base64_image = encode_image(image_path)

    results = asyncio.run(launch_tasks(base64_image, prompts))

    for result in results:
        print(result)
