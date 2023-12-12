import torch
from transformers import DistilBertForMaskedLM, DistilBertTokenizer

# Initialize tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

# Text with a masked token
text = "Hello, my name is [MASK]."

# Encode text
encoded_input = tokenizer(text, return_tensors="pt")
print(torch.where(encoded_input["input_ids"] == tokenizer.mask_token_id))
mask_token_index = torch.where(encoded_input["input_ids"] == tokenizer.mask_token_id)[1]

# Predict masked token
with torch.no_grad():
    output = model(**encoded_input)
    print(output)
    print(output.logits.shape)
    logits = output.logits

# Select predicted token
predicted_token_id = logits[0, mask_token_index, :].argmax(axis=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"Original text: {text}")
print(f"Predicted token: {predicted_token}")
