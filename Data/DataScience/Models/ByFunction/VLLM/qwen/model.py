import json
import os
from collections import OrderedDict
from contextlib import ExitStack
from typing import ContextManager, Dict, List, Optional, Tuple, Union

import torch
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)

path_model = "/home/frank/Datalake/models/Qwen/Qwen2.5-VL-3B-Instruct"


class ContextManagers:
    """
    Wrapper for `contextlib.ExitStack` which enters a collection of context managers. Adaptation of `ContextManagers`
    in the `fastcore` library.
    """

    def __init__(self, context_managers: list[ContextManager]):
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        for context_manager in self.context_managers:
            self.stack.enter_context(context_manager)

    def __exit__(self, *args, **kwargs):
        self.stack.__exit__(*args, **kwargs)


def get_model():
    config, model_kwargs = Qwen2_5_VLConfig.from_pretrained(
        path_model,
        cache_dir=None,
        return_unused_kwargs=True,
        force_download=False,
        proxies=None,
        local_files_only=False,
        token=None,
        revision="main",
        subfolder="",
        gguf_file=None,
        _from_auto=False,
        _from_pipeline=None,
    )
    model_kwargs = {}

    def get_checkpoint_shard_files(
        pretrained_model_name_or_path,
        index_filename,
    ):
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
    ) -> Tuple[Optional[List[str]], Optional[Dict]]:
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        # Load from a sharded safetensors checkpoint
        archive_file = os.path.join(
            pretrained_model_name_or_path,
            "model.safetensors.index.json",
        )

        print(f"loading weights file {archive_file}")
        resolved_archive_file = archive_file

        checkpoint_files, sharded_metadata = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            resolved_archive_file,
        )

        return checkpoint_files, sharded_metadata

    checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
        pretrained_model_name_or_path=path_model,
    )
    torch_dtype = torch.bfloat16
    dtype_orig = torch.bfloat16
    config.torch_dtype = torch_dtype
    config.name_or_path = path_model

    model_init_context = Qwen2_5_VLForConditionalGeneration.get_init_context(
        is_quantized=False, _is_ds_init_called=False
    )
    # config = copy.deepcopy(config)
    with ContextManagers(model_init_context):
        # Let's make sure we don't run the init function of buffer modules
        model = Qwen2_5_VLForConditionalGeneration(config)

    model = model.to(torch.bfloat16)
    model.tie_weights()
    config = model.config
    torch.set_default_dtype(torch_dtype)

    (
        model,
        missing_keys,
        unexpected_keys,
        mismatched_keys,
        offload_index,
        error_msgs,
    ) = Qwen2_5_VLForConditionalGeneration._load_pretrained_model(
        model,
        None,
        checkpoint_files,
        path_model,
        ignore_mismatched_sizes=False,
        sharded_metadata=sharded_metadata,
        device_map=OrderedDict([("", 0)]),
        # device_map="auto",
        disk_offload_folder=None,
        offload_state_dict=False,
        dtype=torch_dtype,
        hf_quantizer=None,
        keep_in_fp32_regex=None,
        device_mesh=None,
        key_mapping=None,
        weights_only=True,
    )
    model.tie_weights()
    model.eval()
    return model
