import base64
import os

import requests


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


image_path = "el_jardin_de_las_delicias.jpg"
base64_image = encode_image(image_path)

api_key = os.environ.get("OPENAI_API_KEY")

headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}


payload = {
    "model": "gpt-4o",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ],
    "max_tokens": 300,
}

response = requests.post(
    "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
)


# print(response.choices[0])
print(response.json())
