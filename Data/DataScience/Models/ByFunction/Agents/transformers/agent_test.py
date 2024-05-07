import os

from transformers import OpenAiAgent

agent = OpenAiAgent(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))

text = "Hello everyone!"

result = agent.run("Read the following text out loud", text=text)

print(result)
