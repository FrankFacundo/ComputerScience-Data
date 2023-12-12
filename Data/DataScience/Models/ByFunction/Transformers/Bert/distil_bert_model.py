from transformers import DistilBertModel, DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

text = "Hello, my name is [MASK]."
encoded_input = tokenizer(text, return_tensors="pt")
output = model(**encoded_input)

print(output.last_hidden_state)
print(output.last_hidden_state.shape)
