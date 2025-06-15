import os
import torch
from datasets import Dataset, load_dataset  # Added load_dataset for example
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,  # Base class we will inherit from
    DataCollatorForLanguageModeling,
)
from typing import List, Dict, Optional, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

# --- Configuration ---
DEFAULT_MAX_SEQ_LENGTH = 512


class QwenTrainer(Trainer):
    def __init__(
        self,
        model_name_or_path: Union[str, PreTrainedModel] = "Qwen/Qwen3-0.6B",
        tokenizer_name_or_path: Optional[Union[str, PreTrainedTokenizerBase]] = None,
        training_args: Optional[TrainingArguments] = None,
        train_dataset_raw: Optional[
            List[Dict]
        ] = None,  # Expects list of {"messages": [...]}
        eval_dataset_raw: Optional[
            List[Dict]
        ] = None,  # Expects list of {"messages": [...]}
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        data_collator_mlm: bool = False,  # For Causal LM, mlm should be False
        model_init_kwargs: Optional[Dict] = None,
        **kwargs,  # Pass any other Trainer args
    ):
        # --- 1. Handle Tokenizer ---
        if isinstance(tokenizer_name_or_path, str) or tokenizer_name_or_path is None:
            tokenizer_path = (
                tokenizer_name_or_path if tokenizer_name_or_path else model_name_or_path
            )
            print(f"Loading tokenizer from {tokenizer_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True
            )
        elif isinstance(tokenizer_name_or_path, PreTrainedTokenizerBase):
            self.tokenizer = tokenizer_name_or_path
        else:
            raise ValueError("Invalid tokenizer_name_or_path provided.")

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                print(
                    f"Set tokenizer.pad_token_id to eos_token_id: {self.tokenizer.eos_token_id}"
                )
            else:
                # Add a new pad token if EOS is also missing (unlikely for Qwen)
                print("Warning: EOS token not found. Adding a new pad token '[PAD]'.")
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # --- 2. Handle Model ---
        _model_init_kwargs = model_init_kwargs if model_init_kwargs is not None else {}
        _model_init_kwargs.setdefault("torch_dtype", "auto")
        _model_init_kwargs.setdefault("trust_remote_code", True)

        if isinstance(model_name_or_path, str):
            print(f"Loading model from {model_name_or_path}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, **_model_init_kwargs
            )
        elif isinstance(model_name_or_path, PreTrainedModel):
            self.model = model_name_or_path
        else:
            raise ValueError("Invalid model_name_or_path provided.")

        # Resize embeddings if a new pad token was added to tokenizer AND it was loaded by name
        if isinstance(tokenizer_name_or_path, str) or tokenizer_name_or_path is None:
            if (
                self.tokenizer.pad_token == "[PAD]"
                and len(self.tokenizer) > self.model.config.vocab_size
            ):
                print(
                    f"Resizing model token embeddings from {self.model.config.vocab_size} to {len(self.tokenizer)}"
                )
                self.model.resize_token_embeddings(len(self.tokenizer))

        # --- 3. Prepare Datasets ---
        self.max_seq_length = max_seq_length
        processed_train_dataset = None
        if train_dataset_raw:
            print("Processing raw training dataset...")
            processed_train_dataset = self._prepare_dataset(train_dataset_raw)

        processed_eval_dataset = None
        if eval_dataset_raw:
            print("Processing raw evaluation dataset...")
            processed_eval_dataset = self._prepare_dataset(eval_dataset_raw)

        # --- 4. Data Collator ---
        # If a data_collator is passed in kwargs, use that, otherwise create one
        if "data_collator" not in kwargs:
            print(f"Using DataCollatorForLanguageModeling with mlm={data_collator_mlm}")
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=data_collator_mlm
            )
        else:
            data_collator = kwargs.pop(
                "data_collator"
            )  # Remove it so it's not passed twice

        # --- 5. Initialize Parent Trainer ---
        super().__init__(
            model=self.model,
            args=training_args,
            train_dataset=processed_train_dataset,
            eval_dataset=processed_eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            **kwargs,  # Pass remaining kwargs to parent
        )

    def _apply_chat_template_and_tokenize(self, examples):
        """
        Applies chat template and tokenizes.
        Expects examples to have a "messages" key.
        """
        formatted_texts = []
        for conversation in examples["messages"]:
            try:
                # `add_generation_prompt=True` is crucial for training to predict assistant responses
                # It adds the tokens that signal the model to start generating (e.g., <|im_start|>assistant\n)
                text = self.tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
                formatted_texts.append(text)
            except Exception as e:
                print(f"Error applying chat template to: {conversation}. Error: {e}")
                formatted_texts.append(
                    ""
                )  # Append empty string or handle error as needed

        tokenized_inputs = self.tokenizer(
            formatted_texts,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,  # DataCollator will handle padding
        )
        # For Causal LM, labels are usually the input_ids themselves.
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs

    def _prepare_dataset(self, raw_dataset_list: List[Dict]):
        """
        Converts a list of raw conversation dicts into a tokenized Hugging Face Dataset.
        Each dict in raw_dataset_list should have a "messages" key.
        e.g., [{"messages": [{"role":"user", "content":"Hi"}, {"role":"assistant", "content":"Hello!"}]}]
        """
        if not raw_dataset_list:
            return None
        # Ensure the input is in the format map expects (a dict of lists)
        # Our input is a list of dicts, where each dict has "messages"
        # So we need to transform it to {"messages": [[conv1_msg_list], [conv2_msg_list], ...]}
        dataset_dict = {
            "messages": [example["messages"] for example in raw_dataset_list]
        }
        hf_dataset = Dataset.from_dict(dataset_dict)

        tokenized_dataset = hf_dataset.map(
            self._apply_chat_template_and_tokenize,
            batched=True,
            remove_columns=hf_dataset.column_names,  # Remove original 'messages' column
        )
        return tokenized_dataset


if __name__ == "__main__":
    # --- Configuration for the example ---
    model_name = "/home/frank/Datalake/models/Qwen/Qwen3-0.6B"
    output_dir = "./qwen3_finetuned_custom_trainer"
    logging_dir = f"{output_dir}/logs"
    final_model_dir = f"{output_dir}/final_model"

    # --- 1. Prepare a DUMMY Raw Dataset ---
    # (Replace with your actual data loading)
    # For example, using a small slice of tldr for structure demo
    # dataset_raw_train_full = load_dataset("trl-lib/tldr", split="train")
    # num_examples = 100 # Use a small number for quick test
    # dummy_train_data = []
    # for i in range(num_examples):
    #     prompt = dataset_raw_train_full[i]['prompt']
    #     label = dataset_raw_train_full[i]['label']
    #     # Convert to Qwen chat format
    #     dummy_train_data.append({
    #         "messages": [
    #             {"role": "user", "content": prompt + "\nTLDR;"}, # Adding a common instruction
    #             {"role": "assistant", "content": label}
    #         ]
    #     })

    # Or a simpler custom dataset:
    dummy_train_data = [
        {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant specialized in geography.",
                },
                {"role": "user", "content": "Which ocean is the largest?"},
                {
                    "role": "assistant",
                    "content": "The Pacific Ocean is the largest ocean on Earth.",
                },
            ]
        },
        {  # A slightly longer one
            "messages": [
                {
                    "role": "user",
                    "content": "Can you explain the concept of photosynthesis in simple terms?",
                },
                {
                    "role": "assistant",
                    "content": "Photosynthesis is how plants make their own food. They use sunlight, water from the soil, and a gas called carbon dioxide from the air. They turn these into sugar (their food) and release oxygen, which we breathe!",
                },
            ]
        },
    ] * 10  # Multiply to have a bit more data for the trainer to run a few steps

    dummy_eval_data = [
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
            ]
        }
    ] * 5

    # --- 2. Training Arguments ---
    print("Setting up Training Arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Keep it short for testing
        per_device_train_batch_size=1,  # Adjust based on GPU memory
        gradient_accumulation_steps=2,  # Effective batch size = 1 * 2 = 2
        learning_rate=5e-5,
        logging_dir=logging_dir,
        logging_steps=5,  # Log more frequently for small datasets
        save_strategy="epoch",
        # fp16=torch.cuda.is_available(), # Enable if desired and supported
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="tensorboard",
        remove_unused_columns=False,  # Important: QwenTrainer handles column removal
        # in _prepare_dataset. If True, might remove "messages" too early.
        # Base Trainer will remove columns not used by model's forward pass.
        # gradient_checkpointing=True, # Saves memory, slows down training
    )
    if not training_args.bf16 and not training_args.fp16 and torch.cuda.is_available():
        print(
            "Warning: Neither bfloat16 nor fp16 is enabled. Training will use float32."
        )
        training_args.fp16 = True  # Fallback to fp16
        if not training_args.fp16:
            print("FP16 not available either. Sticking to FP32.")

    # --- 3. Initialize QwenTrainer ---
    print("\nInitializing QwenTrainer...")
    trainer = QwenTrainer(
        model_name_or_path=model_name,
        training_args=training_args,
        train_dataset_raw=dummy_train_data,
        eval_dataset_raw=dummy_eval_data,  # Optional
        max_seq_length=256,  # Example max sequence length
        # model_init_kwargs={"torch_dtype": torch.bfloat16} # Already default in constructor
    )

    # Access the underlying model and tokenizer if needed
    # print(f"Trainer model: {trainer.model.config.model_type}")
    # print(f"Trainer tokenizer: {trainer.tokenizer.name_or_path}")

    # --- 4. Train ---
    print("\nStarting training...")
    try:
        trainer.train()
        print("Training completed.")

        # --- 5. Save ---
        print(f"\nSaving model to {final_model_dir}...")
        trainer.save_model(final_model_dir)
        # Tokenizer is also saved by save_model if it was loaded by the trainer
        # trainer.tokenizer.save_pretrained(final_model_dir) # usually not needed
        print(f"Model and tokenizer saved to {final_model_dir}")

        # --- 6. Test the Fine-tuned Model (Optional) ---
        print("\nTesting the fine-tuned model...")
        loaded_model = AutoModelForCausalLM.from_pretrained(
            final_model_dir, trust_remote_code=True, torch_dtype="auto"
        )
        loaded_tokenizer = AutoTokenizer.from_pretrained(
            final_model_dir, trust_remote_code=True
        )

        if torch.cuda.is_available():
            loaded_model.to("cuda")

        prompt = "What is the capital of France?"
        messages = [{"role": "user", "content": prompt}]
        text = loaded_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = loaded_tokenizer([text], return_tensors="pt").to(
            loaded_model.device
        )

        generated_ids = loaded_model.generate(**model_inputs, max_new_tokens=50)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        response = loaded_tokenizer.decode(output_ids, skip_special_tokens=True)

        print(f"\nPrompt: {prompt}")
        print(f"Fine-tuned Model Response: {response.strip()}")

    except Exception as e:
        print(f"An error occurred during training or testing: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nScript finished.")
