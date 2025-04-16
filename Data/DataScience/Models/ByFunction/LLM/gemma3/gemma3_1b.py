import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM

model_id = "google/gemma-3-1b-it"


model = Gemma3ForCausalLM.from_pretrained(model_id).eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Write a poem on Hugging Face, the company"},
            ],
        },
    ],
]
inputs = (
    tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    .to(model.device)
    .to(torch.bfloat16)
)


with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=64)

outputs = tokenizer.batch_decode(outputs)
