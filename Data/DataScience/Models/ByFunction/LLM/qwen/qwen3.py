import json
import os
from collections import OrderedDict
from contextlib import ExitStack
from typing import ContextManager, Optional, Union

import torch
from accelerate import dispatch_model

from transformers import AutoTokenizer, Qwen2Config as Qwen3Config, Qwen3ForCausalLM
from transformers.tokenization_utils_base import BatchEncoding # For type checking
from transformers.generation.configuration_utils import GenerationConfig

path_model = "/home/frank/Datalake/models/Qwen/Qwen3-0.6B"  # Example local path


def get_model() -> Qwen3ForCausalLM:
    """Loads the Qwen2 text-only LLM."""
    print(f"Loading model config from: {path_model}")
    # Note: User uses Qwen3Config and Qwen3ForCausalLM. Ensure these are correct for the model.
    # If Qwen3Config points to Qwen2Config in imports, this is just aliasing.
    # For this fix, we assume the class names used by the user are the intended ones.
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
        # _from_auto=False, # These _ prefixed args are often internal.
        # _from_pipeline=None,
        trust_remote_code=True,
    )
    # model_kwargs = {} # This was overriding the returned model_kwargs_unused

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
    # config.name_or_path = path_model # This is often set automatically or not needed for local

    # It seems Qwen3ForCausalLM.get_init_context might not exist or is an internal API.
    # Standard initialization usually doesn't require manual context management like this
    # unless dealing with very large models and specific initialization strategies (like DeepSpeed).
    # For typical Hugging Face models, direct instantiation is common.
    # However, preserving user's structure:
    # model_init_contexts = [] # Safely create an empty list
    # if hasattr(Qwen3ForCausalLM, 'get_init_context'):
    #     model_init_contexts = Qwen3ForCausalLM.get_init_context(is_quantized=False, _is_ds_init_called=False)
    # else:
    #     print("Warning: Qwen3ForCausalLM.get_init_context not found. Proceeding without it.")

    # Simplified initialization:
    # with ContextManagers(model_init_contexts):
    #     model = Qwen3ForCausalLM(config)
    # If get_init_context is not standard, direct instantiation is preferred
    model = Qwen3ForCausalLM(config, **model_kwargs_unused) # Pass unused kwargs here if relevant

    # model = model.to(torch_dtype) # This might be redundant if dtype is set in _load_pretrained_model
    # torch.set_default_dtype(torch_dtype) # Setting global default dtype can have side effects.

    if torch.cuda.is_available():
        device_map_load = {"": 0} # Load directly to GPU 0
        print("CUDA is available. Loading model to GPU 0.")
    else:
        device_map_load = {"": "cpu"} # Load to CPU
        print("CUDA not available. Loading model to CPU.")

    (
        model,
        missing_keys,
        unexpected_keys,
        mismatched_keys,
        offload_index,
        error_msgs,
    ) = Qwen3ForCausalLM._load_pretrained_model( # Using internal method, standard is from_pretrained
        model,
        None, # state_dict
        checkpoint_files,
        path_model, # pretrained_model_name_or_path
        ignore_mismatched_sizes=False,
        sharded_metadata=sharded_metadata,
        device_map=device_map_load, # Should make model.to(torch_dtype) potentially redundant
        # disk_offload_folder=None, # Deprecated, use offload_folder
        # offload_folder=None,
        offload_state_dict=False,
        dtype=torch_dtype, # Explicitly set dtype here
        # hf_quantizer=None,
        # keep_in_fp32_regex=None,
        # device_mesh=None,
        # key_mapping=None,
        # weights_only=True, # This might be for specific torch.load scenarios
    )
    if missing_keys: print(f"Missing keys: {missing_keys}")
    if unexpected_keys: print(f"Unexpected keys: {unexpected_keys}")
    if mismatched_keys: print(f"Mismatched keys: {mismatched_keys}")
    if error_msgs: print(f"Error messages: {error_msgs}")

    model.tie_weights()
    model.eval()

    model.generation_config = GenerationConfig.from_pretrained(
        path_model, trust_remote_code=True # Simplified
    )

    dispatch_model_kwargs = {
        "device_map": device_map_load, # Dispatch to the same device map
        "offload_dir": None, # keep if offload_index can be non-None
        "offload_buffers": False,
        "skip_keys": "past_key_values",
    }
    if offload_index is not None:
        dispatch_model_kwargs["offload_index"] = offload_index
    else: # If offload_index is None, remove it from kwargs for dispatch_model
        dispatch_model_kwargs.pop("offload_index", None)


    dispatch_model(model, **dispatch_model_kwargs)
    print(f"Model loaded and dispatched. Final model device: {model.device}")
    return model


def get_tokenizer(model_path_or_name: str):
    """Loads the tokenizer for the Qwen2 text-only LLM."""
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
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
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
        # Process with tokenizer.apply_chat_template
        # The key issue is that model_inputs was a tensor, not a dict for ** unpacking.
        # We need to get input_ids and attention_mask separately.
        
        # First, get the raw output from apply_chat_template
        # It might be a tensor or a BatchEncoding (dict-like)
        raw_tokenizer_output = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            padding=True, # Ensure padding creates attention_mask
        )

        # Determine if the output is a tensor or a dictionary
        if isinstance(raw_tokenizer_output, torch.Tensor):
            # This case matches the user's observed behavior and error
            print("tokenizer.apply_chat_template returned a Tensor directly. Assuming it's input_ids.")
            input_ids_unmoved = raw_tokenizer_output
            # Manually create attention_mask if tokenizer didn't provide it
            attention_mask_unmoved = (input_ids_unmoved != tokenizer.pad_token_id).long()
            
            input_ids_tensor = input_ids_unmoved.to(model.device)
            attention_mask_tensor = attention_mask_unmoved.to(model.device)

        elif isinstance(raw_tokenizer_output, (dict, BatchEncoding)):
            # This is the standard/expected behavior
            print("tokenizer.apply_chat_template returned a dictionary (BatchEncoding).")
            # Move all tensors in the dictionary to the model's device
            processed_inputs = raw_tokenizer_output.to(model.device)
            input_ids_tensor = processed_inputs['input_ids']
            if 'attention_mask' in processed_inputs:
                attention_mask_tensor = processed_inputs['attention_mask']
            else:
                # This case should be rare if padding=True
                print("Warning: 'attention_mask' not in tokenizer output dictionary. Creating manually.")
                attention_mask_tensor = (input_ids_tensor != tokenizer.pad_token_id).long().to(model.device)
        else:
            raise TypeError(
                f"Unexpected type from tokenizer.apply_chat_template: {type(raw_tokenizer_output)}. "
                "Expected torch.Tensor or a dictionary-like BatchEncoding."
            )

    except Exception as e:
        print(f"Error during input tokenization or processing: {e}")
        print("Make sure the tokenizer is correctly configured for chat, has a pad_token, and the model path is correct.")
        exit()

    if input_ids_tensor is None:
        print("Error: input_ids_tensor was not properly assigned.")
        exit()

    print(f"Inputs prepared. input_ids device: {input_ids_tensor.device}, attention_mask device: {attention_mask_tensor.device if attention_mask_tensor is not None else 'N/A'}")


    print("\nStarting generation...")
    # Inference: Generation of the output
    # Pass input_ids and attention_mask directly as keyword arguments
    generated_ids = model.generate(
        input_ids=input_ids_tensor,
        attention_mask=attention_mask_tensor, # Crucial for padded inputs
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )
    print("Generation complete.")

    # Trim the prompt from the generated IDs
    input_ids_length = input_ids_tensor.shape[1]
    generated_ids_trimmed = generated_ids[:, input_ids_length:]

    output_text = tokenizer.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    print("\nGenerated Output:")
    for i, text in enumerate(output_text):
        print(f"Response {i + 1}: {text}")
