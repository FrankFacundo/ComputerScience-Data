import os
import sys
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

s1 = sys.argv[1]
print(s1)

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": s1},
    ]
)

print(response['choices'][0]['message']['content'])
