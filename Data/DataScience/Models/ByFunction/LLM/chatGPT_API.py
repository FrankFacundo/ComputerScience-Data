import os
import sys
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

s1 = sys.argv[1]
print(s1)

response = openai.Completion.create(
    engine="code-davinci-002",
    prompt=s1,
    temperature=1,
    max_tokens=50,
    top_p=1,
    # stop=["\n"]
)

print(response['choices'][0]['text'])
