from datasets import load_dataset, Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)  # PeftModel is not used in this script directly for loading
from trl import GRPOConfig  # GRPOTrainer is not directly used, Qwen2VLGRPOTrainer is
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TextStreamer,  # Added for convenience
)
import torch
from PIL import Image
from io import BytesIO
import base64  # Not used in the main new flow, but kept if strip_exif related parts need it (they don't)
import pandas as pd  # Not directly used for parquet loading with `datasets`
from tqdm import (
    tqdm,
)  # Not directly used in the new data prep, but could be useful for manual iteration
import re  # For reward function
from Levenshtein import ratio as levenshtein_ratio  # For reward function

# Import the custom trainer (ensure grpo_trainer.py is in your PYTHONPATH or same directory)
from grpo_trainer import Qwen2VLGRPOTrainer


### Helper function to strip EXIF data (remains unchanged)
def strip_exif_from_image(image: Image.Image) -> Image.Image:
    """
    Strips EXIF and other problematic metadata from a PIL Image object
    by rebuilding it from pixel data and carefully constructing a new info dictionary.
    """
    pixel_data = list(image.getdata())
    mode = image.mode
    size = image.size

    new_image = Image.new(mode, size)
    new_image.putdata(pixel_data)

    new_info = {}
    if (
        mode == "P"
        and "transparency" in image.info
        and isinstance(image.info["transparency"], int)
    ):
        new_info["transparency"] = image.info["transparency"]
    if "duration" in image.info:
        new_info["duration"] = image.info["duration"]
    if "loop" in image.info:
        new_info["loop"] = image.info["loop"]
    new_image.info = new_info

    if hasattr(new_image, "_exif"):
        new_image._exif = None
    return new_image


### --- Configuration ---
# !! IMPORTANT: Set the path to your Parquet file here !!
PARQUET_FILE_PATH = (
    "synthetic_mermaid_vqa_dataset.parquet"  # e.g., "data/my_diagrams.parquet"
)
# For testing with a dummy file if you don't have one yet:
# import pandas as pd
# dummy_data = {
#     'diagram_image': [b"\x89PNG...", b"\x89PNG..."], # Replace with actual minimal PNG binary data
#     'mermaid_code': ["classDiagram...", "sequenceDiagram..."],
#     'question': ["How many classes?", "Does Alice interact with John?"],
#     'answer': ["3", "Yes"]
# }
# # Create a dummy PNG (replace with actual image data if testing image loading)
# from PIL import Image
# import io
# dummy_img = Image.new('RGB', (60, 30), color = 'red')
# img_byte_arr = io.BytesIO()
# dummy_img.save(img_byte_arr, format='PNG')
# img_byte_arr = img_byte_arr.getvalue()
# dummy_data['diagram_image'] = [img_byte_arr, img_byte_arr]
#
# PARQUET_FILE_PATH = "dummy_diagram_dataset.parquet"
# pd.DataFrame(dummy_data).to_parquet(PARQUET_FILE_PATH)


# Model and Tokenizer configuration
MODEL_PATH = "/home/frank/Datalake/models/Qwen/Qwen2.5-VL-3B-Instruct"  # User's path
compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

# Load processor (includes tokenizer and image processor)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Load model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=compute_dtype,
    quantization_config=bnb_config,
).eval()  # Start in eval mode, trainer will switch to train mode

### --- Data Loading and Preprocessing for the new Parquet Dataset ---

# Load the Parquet dataset
# The user's data has 5 rows, which is small for training but okay for script testing.
# For a real run, use a larger dataset and potentially split it.
# Using "train" split by default; if your parquet isn't split, this will load the whole thing.
try:
    raw_diagram_dataset = load_dataset(
        "parquet", data_files=PARQUET_FILE_PATH, split="train"
    )
    print(f"Successfully loaded dataset from {PARQUET_FILE_PATH}")
    print("First example from the raw Parquet dataset:")
    print(raw_diagram_dataset[0])
except Exception as e:
    print(f"Failed to load dataset from {PARQUET_FILE_PATH}. Error: {e}")
    print("Please ensure PARQUET_FILE_PATH is correct and the file exists.")
    print(
        "If you are running this for the first time, create a dummy Parquet file (see comments above) for testing."
    )
    exit()


def get_prompt_diagram_rft(example_row):
    """
    Processes a single row from the Parquet dataset for GRPO training.
    Input: dict example_row (a single row from the Parquet dataset)
    Output: dict suitable for the GRPO trainer, containing prompt, image, and solution.
    """
    question_sample = example_row["question"]
    answer_sample = str(example_row["answer"])  # Ensure answer is a string

    # Image handling
    image_bytes = example_row["diagram_image"]
    img_pil_original = Image.open(BytesIO(image_bytes))

    # Resize (Qwen-VL often uses 448x448, adjust as needed)
    img_pil_resized = img_pil_original.resize((448, 448))  # Adjusted size
    img_pil_no_exif = strip_exif_from_image(img_pil_resized)

    SYSTEM_PROMPT = r"""
    Below is an instruction that describes a task, paired with an input that provides further context.
    Write a response that appropriately completes the request.
    You are an AI assistant capable of understanding diagrams and answering questions about them.
    Before answering, think carefully about the question and the provided diagram to create a step-by-step chain of thoughts.
    Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.

    ### Instruction:
    Please answer the following question based on the input diagram.
    """.strip()

    prompt_messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image"
                },  # Placeholder for the image, handled by the processor
                {"type": "text", "text": question_sample},
            ],
        },
    ]

    return {
        "prompt": prompt_messages,  # List of message dicts for chat template
        "image": img_pil_no_exif,  # PIL Image object
        "solution": answer_sample,  # Ground truth answer string
    }


# This function is not directly used for GRPO training in this script,
# but adapted for consistency if SFT were to be performed.
def get_prompt_diagram_sft(example_row):
    """
    Processes a single row from the Parquet dataset for SFT (Supervised Fine-Tuning) data preparation.
    """
    question_sample = example_row["question"]
    answer_sample = str(example_row["answer"])

    image_bytes = example_row["diagram_image"]
    img_pil_original = Image.open(BytesIO(image_bytes))
    img_pil_resized = img_pil_original.resize((448, 448))
    img_pil_no_exif = strip_exif_from_image(img_pil_resized)

    SYSTEM_PROMPT = r"""
    Below is an instruction that describes a task, paired with an input that provides further context.
    Write a response that appropriately completes the request.
    You are an AI assistant capable of understanding diagrams and answering questions about them.

    ### Instruction:
    Please answer the following question based on the input diagram.
    """.strip()

    # For SFT, the answer is part of the assistant's turn
    assistant_response_text = f"<think>The user is asking: {question_sample}. I need to analyze the diagram to answer this.</think><answer>{answer_sample}</answer>"
    # Or simply:
    # assistant_response_text = answer_sample
    # Depending on whether you want the SFT to learn the think/answer tags too.
    # For this example, let's assume it should learn the full format.

    prompt_messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question_sample},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_response_text}],
        },
    ]

    # For typical SFTTrainer, you'd convert `prompt_messages` and `img_pil_no_exif`
    # into a single tokenized sequence with image tokens.
    # Here, we return a structure similar to GRPO for potential custom SFT processing.
    return {
        "prompt_for_sft": prompt_messages,  # The full conversation including the answer
        "image_for_sft": img_pil_no_exif,
        "raw_answer_for_sft": answer_sample,  # Raw answer for reference
    }


### --- Initial Inference Test (Optional) ---
# FastVisionModel.for_inference(model) # Enable for inference if this API exists and is needed.
# Qwen2.5-VL doesn't typically need this explicitly.

print("\n--- Running Initial Inference Test ---")
if len(raw_diagram_dataset) > 0:
    first_example_raw = raw_diagram_dataset[0]
    pil_image = Image.open(BytesIO(first_example_raw["diagram_image"]))
    # For inference, you might use original size or a consistent processed size
    pil_image_processed_for_test = strip_exif_from_image(pil_image.resize((448, 448)))

    test_instruction = (
        "You are an AI assistant skilled in interpreting diagrams. "
        "Please analyze the provided diagram and answer the question. "
        "Explain your reasoning before giving the final answer."
    )
    test_question = first_example_raw["question"]

    messages_for_test = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": f"{test_instruction}\nQuestion: {test_question}",
                },
            ],
        }
    ]

    # The processor.tokenizer is the actual tokenizer part
    input_text_test = processor.tokenizer.apply_chat_template(
        messages_for_test,
        add_generation_prompt=True,
        tokenize=False,  # Get text representation
    )

    # The processor itself handles image and text together
    inputs_test = processor(
        text=input_text_test,
        images=pil_image_processed_for_test,
        add_special_tokens=False,  # apply_chat_template should handle special tokens
        return_tensors="pt",
    ).to(model.device)  # Ensure inputs are on the same device as the model

    text_streamer = TextStreamer(processor.tokenizer, skip_prompt=True)
    print(f"Test Question: {test_question}")
    print("Model Generation (Initial Test):")
    with torch.cuda.amp.autocast(dtype=compute_dtype):  # If using mixed precision
        _ = model.generate(
            **inputs_test,
            streamer=text_streamer,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.7,  # Adjusted for potentially more factual answers
            top_p=0.9,  # Adjusted
        )
    print("\n--- End of Initial Inference Test ---\n")
else:
    print("Skipping initial inference test as the dataset is empty.")


### --- Reward Functions for GRPO (remain largely unchanged) ---
def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has the specific <think>...</think><answer>...</answer> format."""
    pattern = r"^<think>.*?</think>.*?<answer>.*?</answer>$"
    # completions is a list of lists of dicts: [[{'role': 'assistant', 'content': '...'}]]
    rewards = []
    for comp_list in completions:
        if comp_list and isinstance(comp_list[0], dict) and "content" in comp_list[0]:
            content = comp_list[0]["content"]
            match = re.match(pattern, content, re.DOTALL)
            rewards.append(1.0 if match else 0.0)
        else:
            rewards.append(0.0)  # Malformed completion
    return rewards


def levenshtein_reward_func(completions, solution, **kwargs):
    """Reward function that calculates Levenshtein ratio between generated answer and solution."""
    # solution is a list of strings (ground truth answers)
    res = []
    for comp_list, sol_str in zip(completions, solution):
        if comp_list and isinstance(comp_list[0], dict) and "content" in comp_list[0]:
            completion_content = comp_list[0]["content"]
            # Extract text after </think> and before </answer> if possible, or whole if tags not present
            answer_part = completion_content
            if "<answer>" in completion_content and "</answer>" in completion_content:
                answer_part = completion_content.split("<answer>", 1)[1].split(
                    "</answer>", 1
                )[0]
            elif (
                "</think>" in completion_content
            ):  # Fallback if no <answer> but <think> exists
                answer_part = completion_content.split("</think>", 1)[-1].strip()

            res.append(levenshtein_ratio(answer_part, sol_str))
        else:
            res.append(0.0)  # Malformed completion
    return res


### --- Dataset Preparation for Training ---
def diagram_dataset_generator_func():
    """Generator function to yield processed samples for training."""
    # raw_diagram_dataset should be in the global scope or passed appropriately
    for example_row in raw_diagram_dataset:
        yield get_prompt_diagram_rft(example_row)


# Create the training dataset using the generator
# Note: For very large datasets, consider using iterators or streaming with `load_dataset`
# if memory becomes an issue with loading the entire Parquet file at once.
processed_diagram_dataset_train = Dataset.from_generator(diagram_dataset_generator_func)

if len(processed_diagram_dataset_train) > 0:
    print("\nSample from the processed training dataset:")
    print(processed_diagram_dataset_train[-1])
else:
    print("Processed training dataset is empty. Check data loading and processing.")
    # exit() # Optionally exit if no data to train on

### --- Training Setup ---
output_dir = "./outputs/QwenVL_Diagram_GRPO"  # Updated output directory
run_name = "Qwen-VL-GRPO-DiagramQA"  # Updated run name

# PEFT Configuration
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],  # Common for Vision Transformers / Qwen-VL
    bias="none",
    lora_dropout=0.05,
)

# GRPO Training Arguments
training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",  # Ensure bitsandbytes is correctly installed for 8-bit optimizers
    logging_steps=1,
    bf16=False,  # Set to True if your GPU supports bfloat16 and you want to use it
    fp16=True,  # Use mixed precision training
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,  # Effective batch size = 2
    num_generations=2,  # Number of completions to generate per prompt for GRPO
    max_prompt_length=1024,  # Max length for tokenized prompts (adjust based on typical prompt length)
    max_completion_length=1024,  # Max length for generated completions (adjust)
    num_train_epochs=1,  # For a small dataset, 1 epoch might be enough or set max_steps
    max_steps=100,  # Max training steps (e.g., for quick test runs). For full run, set based on epochs.
    # If len(dataset) is 5, 1 epoch = ceil(5 / (1*2)) = 3 steps. So 100 steps = many epochs.
    save_steps=20,  # Save checkpoint every X steps
    max_grad_norm=0.1,  # Gradient clipping
    output_dir=output_dir,  # Base output directory
    # report_to="wandb", # Example: report to Weights & Biases, or "tensorboard", "none"
    report_to="none",
    # push_to_hub=False, # Set to True if you want to push to Hugging Face Hub
    # hub_model_id=f"your-username/{run_name}", # If pushing to hub
    remove_unused_columns=False,  # Important for custom dataset structures with GRPO trainer
)

# Initialize Trainer
# The model should be already prepared for k-bit training if bnb_config was used at load time.
# If not, and LoRA is applied to a non-kbit-trained model, prepare_model_for_kbit_training might be needed before get_peft_model.
# However, Qwen2VLGRPOTrainer might handle PEFT model creation internally if passed peft_config.
# The current script loads with bnb_config, so model is already quantized.
# Let's ensure model is in training mode before PEFT wrapping if not handled by trainer
model.train()
# model = prepare_model_for_kbit_training(model) # Usually done before from_pretrained if quantizing later
# model = get_peft_model(model, peft_config) # Trainer might do this. Check Qwen2VLGRPOTrainer docs.
# If Qwen2VLGRPOTrainer handles get_peft_model internally, don't do it here.

if len(processed_diagram_dataset_train) == 0:
    print("No data to train on. Exiting training phase.")
else:
    print("\n--- Starting GRPO Training ---")
    trainer = Qwen2VLGRPOTrainer(
        model=model,  # Pass the base model, peft_config will be applied by trainer
        processing_class=processor,  # Pass the AutoProcessor instance
        reward_funcs=[format_reward_func, levenshtein_reward_func],
        args=training_args,
        train_dataset=processed_diagram_dataset_train,
        peft_config=peft_config,  # Trainer will apply LoRA config
    )

    trainer.train()
    print("\n--- Training Finished ---")

    # Save the LoRA adapters
    final_save_path = f"{output_dir}/final_lora_adapters"
    trainer.save_model(final_save_path)  # Saves LoRA adapters
    print(f"LoRA adapters saved to {final_save_path}")

    # If you want to save the full model (merged)
    # print("Merging LoRA adapters and saving full model...")
    # merged_model = model.merge_and_unload() # If using PeftModel instance
    # merged_model.save_pretrained(f"{output_dir}/final_merged_model")
    # processor.save_pretrained(f"{output_dir}/final_merged_model")
    # print(f"Full merged model saved to {output_dir}/final_merged_model")


### --- Inference Test After Training (Optional) ---
print("\n--- Running Inference Test After Training ---")
if len(processed_diagram_dataset_train) > 0:
    model.eval()  # Set model to evaluation mode

    # Use an example from the processed dataset (which has prompt messages and PIL image)
    sample_for_inference = processed_diagram_dataset_train[0]
    inference_messages = sample_for_inference[
        "prompt"
    ]  # This is a list of message dicts
    inference_image = sample_for_inference["image"]  # This is a PIL Image

    # Prepare input for the model using the processor
    # The processor.tokenizer is the actual tokenizer part
    input_text_inference = processor.tokenizer.apply_chat_template(
        inference_messages,
        add_generation_prompt=True,
        tokenize=False,  # Get text representation
    )

    # The processor itself handles image and text together
    inputs_inference = processor(
        text=input_text_inference,
        images=inference_image,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    print(
        f"Test Question (Post-Training): {sample_for_inference['prompt'][-1]['content'][-1]['text']}"
    )  # Assuming user question is last
    print("Model Generation (Post-Training):")
    text_streamer_post = TextStreamer(processor.tokenizer, skip_prompt=True)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=compute_dtype):
        _ = model.generate(
            **inputs_inference,
            streamer=text_streamer_post,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.7,  # Consistent with earlier test or adjust as needed
            top_p=0.9,
        )
    print("\n--- End of Post-Training Inference Test ---")
else:
    print("Skipping post-training inference test as dataset was empty.")


### --- Cleanup (Optional) ---
# del model # If Qwen2VLGRPOTrainer holds a reference, this might not free all memory
# del processor
# del trainer
torch.cuda.empty_cache()
print("\nScript finished.")
