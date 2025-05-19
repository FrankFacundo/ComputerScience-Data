# qwen3.py
import json
import os
from collections import OrderedDict
# from contextlib import ExitStack # Not used after modifications
# from typing import ContextManager, Optional, Union # Optional, Union used
from typing import Optional, Union # Simplified imports

import torch
from accelerate import dispatch_model

from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from transformers.tokenization_utils_base import BatchEncoding # For type checking
from transformers.generation.configuration_utils import GenerationConfig

path_model = "/home/frank/Datalake/models/Qwen/Qwen3-0.6B"  # Example local path

# Helper function for sampling logits
def _sample_logits(logits: torch.Tensor, temperature: float, top_p: float, filter_value: float = -float("Inf")):
    """
    Sample from logits with temperature and top-p filtering.
    Args:
        logits: raw logits, shape (batch_size, vocab_size)
        temperature: temperature for sampling. If 0 or <=0, use greedy.
        top_p: top-p filtering. If >= 1.0 or <=0, no top-p.
    """
    if temperature <= 0: # Greedy decoding
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Apply temperature
    logits = logits / temperature

    if 0.0 < top_p < 1.0:
        # Sort logits to apply top-p filtering
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Identify tokens to remove
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep the first token that crosses the threshold and all tokens before it
        if sorted_indices_to_remove.shape[-1] > 1:
             sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        else: # vocab_size=1 case
            sorted_indices_to_remove[..., :] = False 
        
        sorted_indices_to_remove[..., 0] = False # Always keep the most probable token

        # Create a mask for tokens to remove and apply it
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        
        logits = logits.masked_fill(indices_to_remove, filter_value)

    # Sample from the (potentially filtered) distribution
    probabilities = torch.softmax(logits, dim=-1)
    
    # In case all probabilities became zero after filtering (e.g., if all logits were -inf initially or due to extreme filtering)
    # This should be rare with `sorted_indices_to_remove[..., 0] = False`
    # Add a small safeguard for multinomial if probs sum to 0 for any batch item
    sum_probs = torch.sum(probabilities, dim=-1, keepdim=True)
    probabilities = torch.where(sum_probs == 0, torch.ones_like(probabilities) / probabilities.shape[-1], probabilities / sum_probs)
    probabilities = torch.nan_to_num(probabilities, nan=1.0/probabilities.shape[-1]) # Handle potential NaNs if sum_probs was 0 for some reason

    next_tokens = torch.multinomial(probabilities, num_samples=1)
    return next_tokens


def get_model() -> Qwen3ForCausalLM:
    """Loads the Qwen2 text-only LLM.""" # User comment says Qwen2, code uses Qwen3
    print(f"Loading model config from: {path_model}")
    config, model_kwargs_unused = Qwen3Config.from_pretrained(
        path_model,
        cache_dir=None,
        return_unused_kwargs=True,
        force_download=False,
        proxies=None,
        local_files_only=False,
        token=None,
        revision="main",
        subfolder="",
        trust_remote_code=True,
    )

    def get_checkpoint_shard_files(pretrained_model_name_or_path, index_filename):
        with open(index_filename) as f:
            index = json.loads(f.read())
        shard_filenames = sorted(set(index["weight_map"].values()))
        sharded_metadata = index["metadata"]
        sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
        sharded_metadata["weight_map"] = index["weight_map"].copy()
        shard_filenames = [
            os.path.join(pretrained_model_name_or_path, f) for f in shard_filenames
        ]
        return shard_filenames, sharded_metadata

    def _get_resolved_checkpoint_files(
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
    ):
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        safetensors_index_file = os.path.join(
            pretrained_model_name_or_path, "model.safetensors.index.json"
        )
        pytorch_index_file = os.path.join(
            pretrained_model_name_or_path, "pytorch_model.bin.index.json"
        )

        if os.path.exists(safetensors_index_file):
            archive_file = safetensors_index_file
            print(f"Found safetensors index file: {archive_file}")
        elif os.path.exists(pytorch_index_file):
            archive_file = pytorch_index_file
            print(f"Found PyTorch bin index file: {archive_file}")
        else:
            safetensors_file = os.path.join(
                pretrained_model_name_or_path, "model.safetensors"
            )
            pytorch_file = os.path.join(
                pretrained_model_name_or_path, "pytorch_model.bin"
            )
            if os.path.exists(safetensors_file):
                print(f"Found single safetensors file: {safetensors_file}")
                return [safetensors_file], None
            elif os.path.exists(pytorch_file):
                print(f"Found single pytorch_model.bin file: {pytorch_file}")
                return [pytorch_file], None
            else:
                raise FileNotFoundError(
                    f"No model index file (model.safetensors.index.json or pytorch_model.bin.index.json) "
                    f"or single weight file (model.safetensors or pytorch_model.bin) found in {pretrained_model_name_or_path}."
                )
        return get_checkpoint_shard_files(pretrained_model_name_or_path, archive_file)

    checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(path_model)

    torch_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    print(f"Using torch_dtype: {torch_dtype}")

    config.torch_dtype = torch_dtype
    model = Qwen3ForCausalLM(config, **model_kwargs_unused)

    if torch.cuda.is_available():
        device_map_load = {"": 0}
        print("CUDA is available. Loading model to GPU 0.")
    else:
        device_map_load = {"": "cpu"}
        print("CUDA not available. Loading model to CPU.")

    (
        model,
        missing_keys,
        unexpected_keys,
        mismatched_keys,
        offload_index,
        error_msgs,
    ) = Qwen3ForCausalLM._load_pretrained_model(
        model,
        None, 
        checkpoint_files,
        path_model,
        ignore_mismatched_sizes=False,
        sharded_metadata=sharded_metadata,
        device_map=device_map_load,
        offload_state_dict=False,
        dtype=torch_dtype,
    )
    if missing_keys: print(f"Missing keys: {missing_keys}")
    if unexpected_keys: print(f"Unexpected keys: {unexpected_keys}")
    if mismatched_keys: print(f"Mismatched keys: {mismatched_keys}")
    if error_msgs: print(f"Error messages: {error_msgs}")

    model.tie_weights()
    model.eval()

    model.generation_config = GenerationConfig.from_pretrained(
        path_model, trust_remote_code=True
    )

    dispatch_model_kwargs = {
        "device_map": device_map_load,
        "offload_dir": None,
        "offload_buffers": False,
        "skip_keys": "past_key_values", # Common practice to skip KV cache during dispatch
    }
    if offload_index is not None: # offload_index can be returned by _load_pretrained_model
        dispatch_model_kwargs["offload_index"] = offload_index
    
    dispatch_model(model, **dispatch_model_kwargs)
    print(f"Model loaded and dispatched. Final model device: {model.device}")
    return model


def get_tokenizer(model_path_or_name: str):
    """Loads the tokenizer for the Qwen LLM."""
    print(f"Loading tokenizer from: {model_path_or_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path_or_name, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Tokenizer: Set pad_token to eos_token ('{tokenizer.eos_token}')")
        else:
            print("Warning: tokenizer.eos_token is None. Adding a default pad_token '<|pad|>'.")
            # Adding a pad token might change vocab size, ensure model is aware if resizing embeddings.
            # For inference, if pad_token_id is only used for padding and not generation, this is often fine.
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"}) 
            # If model embeddings are resized, weights need to be handled.
            # However, Qwen models usually have pad tokens defined.
    return tokenizer


if __name__ == "__main__":
    print(f"Using model path: {path_model}")

    tokenizer = get_tokenizer(path_model)
    model = get_model()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, can you write a short poem about the stars?"},
    ]

    print("\nPreparing inputs for inference...")
    input_ids_tensor = None
    attention_mask_tensor = None

    try:
        raw_tokenizer_output = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            padding=True, 
        )

        if isinstance(raw_tokenizer_output, torch.Tensor):
            print("tokenizer.apply_chat_template returned a Tensor directly. Assuming it's input_ids.")
            input_ids_unmoved = raw_tokenizer_output
            attention_mask_unmoved = (input_ids_unmoved != tokenizer.pad_token_id).long()
            
            input_ids_tensor = input_ids_unmoved.to(model.device)
            attention_mask_tensor = attention_mask_unmoved.to(model.device)

        elif isinstance(raw_tokenizer_output, (dict, BatchEncoding)):
            print("tokenizer.apply_chat_template returned a dictionary (BatchEncoding).")
            processed_inputs = raw_tokenizer_output.to(model.device)
            input_ids_tensor = processed_inputs['input_ids']
            if 'attention_mask' in processed_inputs:
                attention_mask_tensor = processed_inputs['attention_mask']
            else:
                print("Warning: 'attention_mask' not in tokenizer output dictionary. Creating manually.")
                attention_mask_tensor = (input_ids_tensor != tokenizer.pad_token_id).long().to(model.device)
        else:
            raise TypeError(
                f"Unexpected type from tokenizer.apply_chat_template: {type(raw_tokenizer_output)}. "
                "Expected torch.Tensor or a dictionary-like BatchEncoding."
            )

    except Exception as e:
        print(f"Error during input tokenization or processing: {e}")
        exit()

    if input_ids_tensor is None:
        print("Error: input_ids_tensor was not properly assigned.")
        exit()

    print(f"Inputs prepared. input_ids device: {input_ids_tensor.device}, attention_mask device: {attention_mask_tensor.device if attention_mask_tensor is not None else 'N/A'}")

    # --- Start of Token-by-Token Generation ---
    # Generation parameters from the original script (or defaults)
    # max_new_tokens = 128
    max_new_tokens = 4096
    temperature = 0.7 
    top_p = 0.9       

    eos_token_id = tokenizer.eos_token_id
    # Fallback for eos_token_id if not set on tokenizer directly
    if eos_token_id is None:
        if hasattr(model.config, "eos_token_id") and model.config.eos_token_id is not None:
            eos_token_id = model.config.eos_token_id
            print(f"Using eos_token_id from model.config: {eos_token_id}")
        elif tokenizer.special_tokens_map and "eos_token" in tokenizer.special_tokens_map:
             eos_token_val = tokenizer.special_tokens_map['eos_token']
             # Ensure the token is in the vocab before converting
             if eos_token_val in tokenizer.get_vocab():
                eos_token_id = tokenizer.convert_tokens_to_ids(eos_token_val)
                print(f"Using eos_token_id from special_tokens_map ('{eos_token_val}'): {eos_token_id}")
             else:
                print(f"Warning: EOS token '{eos_token_val}' from special_tokens_map not in tokenizer vocab.")


    if eos_token_id is None:
        print("Warning: EOS token ID not found or not in vocab. Generation will stop only at max_new_tokens.")


    # Prepare initial inputs for the generation loop
    current_loop_input_ids = input_ids_tensor.clone() 
    current_attention_mask = attention_mask_tensor.clone()
    past_key_values = None
    generated_token_ids_collector = [] 

    print("\nStarting token-by-token generation...")
    if input_ids_tensor.shape[0] == 1: # Assuming batch_size 1 for this detailed print
        # Decode carefully, special tokens might be part of the template
        prompt_text = tokenizer.decode(input_ids_tensor[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        print(f"Prompt (raw decoded): {prompt_text}")
    else:
        print(f"Prompt batch shape: {input_ids_tensor.shape}")
    print("------ Generated Text (token-by-token) ------")

    with torch.no_grad(): # Ensure no gradients are computed during generation
        for _ in range(max_new_tokens):
            model_outputs = model(
                input_ids=current_loop_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True, 
            )

            next_token_logits = model_outputs.logits[:, -1, :] 
            past_key_values = model_outputs.past_key_values    

            next_token_id_tensor = _sample_logits(next_token_logits, temperature=temperature, top_p=top_p)
            
            # Assuming batch_size = 1 for simplicity in checking/printing.
            # The input `messages` implies a single sequence, so batch_size=1.
            current_token_id_item = next_token_id_tensor[0, 0].item()
            
            token_text = tokenizer.decode([current_token_id_item], 
                                          skip_special_tokens=False, 
                                          clean_up_tokenization_spaces=False)
            
            print(token_text, end="", flush=True)

            generated_token_ids_collector.append(current_token_id_item)

            if eos_token_id is not None and current_token_id_item == eos_token_id:
                print("\n<EOS token generated>")
                break
            
            current_loop_input_ids = next_token_id_tensor
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones_like(next_token_id_tensor, dtype=torch.long, device=model.device)], 
                dim=1
            )
    
    print("\n------ End of Token-by-Token Generation ------")

    generated_ids_tensor_trimmed = torch.tensor([generated_token_ids_collector], dtype=torch.long, device=model.device)
    
    output_text = tokenizer.batch_decode(
        generated_ids_tensor_trimmed, # Contains only newly generated tokens
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    print("\nGenerated Output (decoded from collected new tokens):")
    for i, text in enumerate(output_text): # Will be one item due to batch_size=1
        print(f"Response {i + 1}: {text}")

    # --- End of Token-by-Token Generation ---