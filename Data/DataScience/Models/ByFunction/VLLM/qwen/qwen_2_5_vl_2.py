from collections import OrderedDict

from accelerate import dispatch_model
from model import get_model
from processor import get_processor
from transformers.generation.configuration_utils import GenerationConfig
from vision_process import process_vision_info

path_model = "/home/frank/Datalake/models/Qwen/Qwen2.5-VL-3B-Instruct"

processor = get_processor()
model = get_model()


model.generation_config = GenerationConfig.from_pretrained(
    path_model,
    cache_dir=None,
    force_download=False,
    proxies=None,
    local_files_only=False,
    token=None,
    revision="main",
    subfolder="",
    _from_auto=False,
    _from_pipeline=None,
)

device_map_kwargs = {
    "device_map": OrderedDict([("", 0)]),
    "offload_dir": None,
    "offload_index": None,
    "offload_buffers": False,
    "skip_keys": "past_key_values",
}

dispatch_model(model, **device_map_kwargs)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
