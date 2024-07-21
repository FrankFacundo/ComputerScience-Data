import os

import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor

local_path = "your_path"
hf_path = "OpenGVLab/InternViT-6B-448px-V1-5"


model = (
    AutoModel.from_pretrained(
        local_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    .cuda()
    .eval()
)

filepath = os.path.join(os.path.dirname(__file__), "examples/image1.png")

image = Image.open(filepath).convert("RGB")

image_processor = CLIPImageProcessor.from_pretrained(local_path)

pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
pixel_values = pixel_values.to(torch.bfloat16).cuda()

outputs = model(pixel_values)
print(outputs)
