import dspy
from dspy.datasets.gsm8k import GSM8K

# Set up the LM.
turbo = dspy.OpenAI(model="gpt-3.5-turbo-instruct", max_tokens=250)
dspy.settings.configure(lm=turbo)

# Load math questions from the GSM8K dataset.
gsm8k = GSM8K()
gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:10]

print(gsm8k_trainset)
