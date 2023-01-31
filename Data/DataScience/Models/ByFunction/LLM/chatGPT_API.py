import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  model="text-chat-davinci-002-20230126",
  prompt="Which optimization algorithm is the best to solve SVM?",
  max_tokens=4000,
)

print(response['choices'][0]['text'])
