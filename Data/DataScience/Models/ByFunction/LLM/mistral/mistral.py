"""
|Parameter|Supported values|Default|Use|
|**Temperature**|Floating-point number in the range 0.0 (same as greedy decoding) to 2.0 (maximum creativity)|0.7|Higher values lead to greater variability|
|**Top K**|Integer in the range 1 to 100|50|Higher values lead to greater variability|
|**Top P**|Floating-point number in the range 0.0 to 1.0|1.0|Higher values lead to greater variability|
"""

from threading import Thread

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

device = "cpu"  # the device to load the model onto

cache_dir = "."

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", cache_dir=cache_dir
)
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-v0.1", cache_dir=cache_dir
)
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

prompt = "My favourite condiment is"

model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
model.to(device)

kwargs = dict(
    **model_inputs,
    streamer=streamer,
    # max_length=4096,
    # temperature=1.0,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=200,
    do_sample=False,
)

thread = Thread(target=model.generate, kwargs=kwargs)
thread.start()

for token in streamer:
    print(token, end="", flush=True)
