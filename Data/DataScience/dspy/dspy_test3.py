import dspy

turbo = dspy.OpenAI(model="gpt-3.5-turbo-instruct", max_tokens=250)
dspy.settings.configure(lm=turbo)

# Define signature
signature = "sentence -> sentiment"
classify = dspy.Predict(signature)

# Run
sentence = "it's a charming and often affecting journey."
response = classify(sentence=sentence)

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print(response.sentiment)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print(turbo.inspect_history(n=1))
