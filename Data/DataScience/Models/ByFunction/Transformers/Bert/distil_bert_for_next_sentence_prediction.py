from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

# Load pre-trained model tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

# Prepare the sentences
sentence_a = "The weather is nice today."
sentence_b = "I will go for a walk."

# Encode the inputs
encoded = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt')
tokens_tensor = encoded['input_ids']
segments_tensors = encoded['token_type_ids']

# Predict
with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    logits = outputs.logits
    print(logits)

# Interpret the result
# The output has two logits: the first for "not next sentence", and the second for "is next sentence"
softmax = torch.nn.Softmax(dim=1)
probs = softmax(logits)

print(probs)
print("Probability of 'Is Not Next Sentence':", probs[0][0].item())
print("Probability of 'Is Next Sentence':", probs[0][1].item())
