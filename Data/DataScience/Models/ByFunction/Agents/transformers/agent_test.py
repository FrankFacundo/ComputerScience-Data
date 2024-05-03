import os

from transformers import OpenAiAgent

agent = OpenAiAgent(model="text-davinci-003", api_key=os.getenv("OPENAI_API_KEY"))

text = "Hello everyone!"

result = agent.run("Read the following text out loud", text=text)

print(result)
