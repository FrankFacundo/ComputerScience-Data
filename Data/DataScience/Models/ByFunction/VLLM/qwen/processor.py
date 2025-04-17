import json
import os
from typing import Dict

from tokenizers import AddedToken
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import (
    Qwen2VLImageProcessorFast,
)


def get_processor():
    path_model = "/home/frank/Datalake/models/Qwen/Qwen2.5-VL-3B-Instruct"

    with open(
        os.path.join(path_model, "preprocessor_config.json"), encoding="utf-8"
    ) as reader:
        text = reader.read()
    image_processor_dict = json.loads(text)

    image_processor = Qwen2VLImageProcessorFast(**image_processor_dict)

    with open(
        os.path.join(path_model, "tokenizer_config.json"), encoding="utf-8"
    ) as tokenizer_config_handle:
        init_kwargs = json.load(tokenizer_config_handle)

    added_tokens_decoder: Dict[int, AddedToken] = {}
    added_tokens_map: Dict[str, AddedToken] = {}
    for idx, token in init_kwargs["added_tokens_decoder"].items():
        if isinstance(token, dict):
            token = AddedToken(**token)
        if isinstance(token, AddedToken):
            added_tokens_decoder[int(idx)] = token
            added_tokens_map[str(token)] = token
        else:
            raise ValueError(
                f"Found a {token.__class__} in the saved `added_tokens_decoder`, should be a dictionary or an AddedToken instance"
            )
    init_kwargs["added_tokens_decoder"] = added_tokens_decoder
    init_kwargs["eos_token"] = added_tokens_map.get(str(init_kwargs["eos_token"]))
    init_kwargs["pad_token"] = added_tokens_map.get(str(init_kwargs["pad_token"]))

    tokenizer = Qwen2TokenizerFast(
        vocab_file=os.path.join(path_model, "vocab.json"),
        merges_file=os.path.join(path_model, "merges.txt"),
        tokenizer_file=os.path.join(path_model, "tokenizer.json"),
        name_or_path=path_model,
        use_fast=True,
        **init_kwargs,
    )
    processor = Qwen2_5_VLProcessor.from_args_and_dict(
        args=[image_processor, tokenizer],
        processor_dict={"chat_template": init_kwargs["chat_template"]},
    )
    return processor
