import os
import time

from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)
# s1 = sys.argv[1]
s1 = "quién eres?"
print(s1)

start = time.time()

chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": s1},
    ],
    model="gpt-3.5-turbo",
)


print(time.time() - start)

print(chat_completion.choices[0].message.content)
