from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import GRPOConfig, GRPOTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
import torch
from PIL import Image
from io import BytesIO
###


def strip_exif_from_image(image: Image.Image) -> Image.Image:
    """
    Strips EXIF and other problematic metadata from a PIL Image object
    by rebuilding it from pixel data and carefully constructing a new info dictionary.
    """
    # 1. Get pixel data and basic properties from the original image
    pixel_data = list(image.getdata())  # This gets the raw pixel values
    mode = image.mode
    size = image.size

    # 2. Create a new image. Its .info dictionary will be initially empty.
    new_image = Image.new(mode, size)
    new_image.putdata(pixel_data)

    # 3. Construct a new 'info' dictionary for the new_image.
    #    Start with an empty dictionary and only add back known-safe, essential items.
    #    This prevents accidental copying of 'exif', 'xmp', or other metadata tags.
    new_info = {}

    # Example: If you need to preserve transparency for palettized images ('P' mode)
    # and it's stored as an integer index in info["transparency"]:
    if (
        mode == "P"
        and "transparency" in image.info
        and isinstance(image.info["transparency"], int)
    ):
        new_info["transparency"] = image.info["transparency"]

    # Example: If you are dealing with animated images (e.g., GIFs) and need to preserve duration/loop:
    if "duration" in image.info:  # Typically for animated formats
        new_info["duration"] = image.info["duration"]
    if "loop" in image.info:  # Typically for animated formats
        new_info["loop"] = image.info["loop"]

    # Set the new, clean info dictionary
    new_image.info = new_info

    # 4. As an extra precaution, try to clear PIL's internal EXIF cache if it exists.
    #    This attribute is where Pillow caches parsed EXIF data.
    if hasattr(new_image, "_exif"):
        new_image._exif = None

    return new_image


###

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)
tokenizer = AutoProcessor.from_pretrained(
    "/home/frank/Datalake/models/Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True
)
# use cuda device
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/home/frank/Datalake/models/Qwen/Qwen2.5-VL-3B-Instruct",
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=compute_dtype,
    quantization_config=bnb_config,
).eval()

###

# RUN
from datasets import load_dataset, Dataset
from PIL import Image
import base64
from io import BytesIO
import pandas as pd
from tqdm import tqdm

ds = load_dataset(
    "BUAADreamer/llava-med-zh-instruct-60k",
    split="train[0:200]",
    trust_remote_code=True,
)
print(ds[0])  # show content


def encode_image(image):
    """将 PIL 图片转换为 base64 编码字符串"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_prompt_rft(example):
    """
        input: dict example, including PIL image object
        output: multiple samples, within a dict format, like: [
        {'image': 'data:image;base64,/9j/...'},
        {'text': '这是什么'},
    ]
    """
    dialogue_num = len(example["messages"])
    i = 0
    results = []
    while i < dialogue_num:
        assert (
            example["messages"][i]["role"] == "user"
            and example["messages"][i + 1]["role"] == "assistant"
        )
        question_sample = example["messages"][i]["content"]
        answer_sample = example["messages"][i + 1]["content"]

        # Original image
        img_pil_original = example["images"][0]

        # Resize
        img_pil_resized = img_pil_original.resize((128, 128))  # reduce vRAM burden

        # Strip EXIF data from the resized image
        img_pil_no_exif = strip_exif_from_image(img_pil_resized)

        SYSTEM_PROMPT = r"""
        Below is an instruction that describes a task, paired with an input that provides further context.
        Write a response that appropriately completes the request.
        Before answering, think carefully about the question and create a step-by-step chain of 
        thoughts to ensure a logical and accurate response.
        
        ### Instruction:
        You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
        Please answer the following medical question based on the input image. Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.The output answer format should be as follows:
<think> ... </think> 除了特殊符号，请用中文回答
        """.strip()
        results.append(
            {
                "prompt": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",  # This is just a placeholder for the template
                            },
                            {"type": "text", "text": question_sample},
                        ],
                    },
                ],
                "image": img_pil_no_exif,  # Use the image without EXIF
                "solution": answer_sample,
            }
        )
        i += 2
    return results


# Apply the same fix to get_prompt_sft if you intend to use it for dataset generation
def get_prompt_sft(example):
    dialogue_num = len(example["messages"])
    i = 0
    results = []
    while i < dialogue_num:
        assert (
            example["messages"][i]["role"] == "user"
            and example["messages"][i + 1]["role"] == "assistant"
        )
        question_sample = example["messages"][i]["content"]
        answer_sample = example["messages"][i + 1]["content"]

        image_pil_original = example["images"][0]
        # You might want to resize here too, depending on your needs for SFT
        # image_pil_resized = image_pil_original.resize((desired_x, desired_y))
        image_pil_no_exif = strip_exif_from_image(
            image_pil_original
        )  # Or image_pil_resized

        SYSTEM_PROMPT = r"""
        Below is an instruction that describes a task, paired with an input that provides further context.
        Write a response that appropriately completes the request.
        Before answering, think carefully about the question and create a step-by-step chain of 
        thoughts to ensure a logical and accurate response.
        
        ### Instruction:
        You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
        Please answer the following medical question based on the input image. 
        """.strip()
        TEXT_FORMAT = f"""
        ### Answer:
        {answer_sample}
        """.strip()
        results.append(
            {
                "prompt": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                    },
                    {
                        "role": "user",
                        "content": [
                            # Note: the Qwen template expects the image to be handled by the processor,
                            # not directly in the content list as a PIL object for training.
                            # For training, the image is usually passed as a separate argument
                            # to the model or processor alongside the tokenized text.
                            # The structure here might be more for inference.
                            # Let's assume for now this structure is intended for how you pass it to Qwen2VLGRPOTrainer.
                            # If not, this part might need adjustment based on trainer expectations.
                            {
                                "type": "image",
                                "image": image_pil_no_exif,
                            },  # This part is unusual for training templates
                            {"type": "text", "text": question_sample},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": TEXT_FORMAT}],
                    },
                ],
                "solution": [
                    answer_sample
                ],  # Should this be a list or just the string?
            }
        )
        i += 2
    return results


###

# RUN
# FastVisionModel.for_inference(model) # Enable for inference!

image = ds[0]["images"][0]
instruction = (
    "You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. \
Please answer the following medical question based on the input image. 请用中文回答"
)
# for a different language, please change the last few words.
messages = [
    {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": instruction}],
    }
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

from transformers import TextStreamer

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=512,
    use_cache=True,
    temperature=1.5,
    min_p=0.1,
)


###

# RUN
# All reward functions for GRPO
import re
from Levenshtein import ratio as levenshtein_ratio


def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # print(completions) #debug
    pattern = r"^<think>.*?</think>.*?<answer>.*?</answer>$"
    matches = [
        re.match(pattern, content[0]["content"], re.DOTALL) for content in completions
    ]
    return [1.0 if match else 0.0 for match in matches]


def levenshtein_reward_func(completions, solution, **kwargs):
    """Reward function that checks if the completion get solutions correctly."""
    res = []
    for completion, sol in zip(completions, solution):
        completion = completion[0]["content"]
        if "</think>" in completion:
            t = completion.split("</think>")[-1]  # calculate result distance
            res.append(levenshtein_ratio(t, sol))
        else:
            res.append(0.0)
    return res


def dataset_gen():
    for items in ds:
        multiple_out = get_prompt_rft(items)
        for single_out in multiple_out:
            yield single_out


my_gen = dataset_gen()

dataset_train = Dataset.from_generator(dataset_gen)
print(dataset_train[-1])

###

output_dir = "./outputs/Qwevl-Instruct-GRPO"
run_name = "Qwen-vl-GRPO-medical"
# from unsloth import is_bfloat16_supported
from trl import GRPOConfig

from grpo_trainer import Qwen2VLGRPOTrainer  # third-party trainer from open-R1


###

model.train()
peft_config = LoraConfig(
    r=32,  # Rank
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        # "gate_proj",
        # "up_proj",
        # "down_proj"
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
)

training_args = GRPOConfig(
    # use_vllm = True, # use vLLM for fast inference!
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=1,
    bf16=False,
    fp16=True,
    per_device_train_batch_size=1,  # keep same with num_generations
    gradient_accumulation_steps=2,  # Increase to 4 for smoother training
    num_generations=2,  # Decrease if out of memory
    max_prompt_length=2048,
    max_completion_length=2048,
    num_train_epochs=1,  # Set to 1 for a full training run
    max_steps=100,
    save_steps=5,
    max_grad_norm=0.1,
    report_to="none",  # Can use Weights & Biases
    output_dir="outputs",
)
trainer = Qwen2VLGRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        format_reward_func,  # all reward functions
        levenshtein_reward_func,
    ],
    args=training_args,
    train_dataset=dataset_train,
    peft_config=peft_config,
)

trainer.train()

trainer.save_model(output_dir)

###

# Preparation for inference
model.eval()
message = dataset_train[0]["prompt"]
image = dataset_train[0]["image"]
input_text = tokenizer.apply_chat_template(message, add_generation_prompt=True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

from transformers import TextStreamer

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=512,
    use_cache=True,
    temperature=1.5,
    min_p=0.1,
)

###

del Qwen2VLGRPOTrainer
# del model,tokenizer
torch.cuda.empty_cache()
