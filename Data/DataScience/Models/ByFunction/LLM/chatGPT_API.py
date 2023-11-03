import os
import sys
import openai
import time

openai.api_key = os.getenv("OPENAI_API_KEY")

# s1 = sys.argv[1]
s1 = "quién eres?"
print(s1)

start = time.time()

response = openai.ChatCompletion.create(
  # max_tokens=85,
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": s1},
    ]
)

print(time.time() - start)

print(response['choices'][0]['message']['content'])

response2 = response.create(
  messages=[
        {"role": "user", "content": "Alguna informacion extra que quieras agregar?"}
    ]
)

print(response2['choices'][0]['message']['content'])


