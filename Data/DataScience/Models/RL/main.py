from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
import torch

# dataset = load_dataset("trl-lib/tldr", split="train")
dataset = load_dataset("trl-lib/tldr", split="train[:1%]")


def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]


training_args = GRPOConfig(
    output_dir="test_model",
    logging_steps=10,
    per_device_train_batch_size=2,  # CRITICAL: Reduce batch size to save VRAM
    gradient_accumulation_steps=4,  # Accumulate gradients over 8 steps (effective batch size = 1 * 8 = 8)
    num_train_epochs=1,  # Keep short for testing
    remove_unused_columns=True,  # Important for custom datasets
    gradient_checkpointing=True,  # Explicitly enable here as well (redundant if model.gradient_checkpointing_enable() used, but safe)
    max_prompt_length=128,  # Max length of prompt
    max_completion_length=128,  # Max new tokens to generate for reward scoring
    num_generations=2,  # <--- ADD THIS LINE
    model_init_kwargs={
        "torch_dtype": torch.bfloat16
    },  # Use bfloat16 for better performance
)

trainer = GRPOTrainer(
    model="/home/frank/Datalake/models/Qwen/Qwen3-0.6B",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
