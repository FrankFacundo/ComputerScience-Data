from transformers import AutoConfig, AutoModel

model_name_or_path = "distilbert-base-uncased"

config = AutoConfig.from_pretrained(model_name_or_path)

auto_model = AutoModel.from_pretrained(model_name_or_path, config=config)

# DistilBertModel class
print(auto_model)
