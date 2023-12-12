from transformers import DistilBertModel, DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

text = "Hello, my name is [MASK]."
encoded_input = tokenizer(text, return_tensors="pt")
output = model(**encoded_input)

print(output.last_hidden_state)
print(output.last_hidden_state.shape)
# torch.Size([1, 9, 768])
# 9 tokens
# 768 is the dimention of bert model (d_model)
