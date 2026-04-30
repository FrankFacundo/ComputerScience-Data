r"""Pure-torch SigLIP image encoder for the local SigLIP2 checkpoint.

Example::

    python siglip2.py

The folder is named ``siglip2``, but the local checkpoint declares
``model_type = "siglip"`` and Transformers loads it through ``SiglipModel``.
This file implements that checkpoint's vision tower without using
Transformers in the pure-torch inference path, then optionally compares the
result against Transformers' ``SiglipVisionModel``.

Pipeline
--------
1. Resize the image to the checkpoint input size, rescale to [0, 1], and
   normalize with mean/std 0.5.
2. Apply a Conv2d patch embedding with stride == kernel size.
3. Add learned absolute position embeddings.
4. Run the pre-norm ViT encoder stack:
   ``x += Attention(LN(x)); x += MLP(LN(x))``.
5. Apply the final layer norm.
6. Pool with a learned probe using PyTorch multi-head attention, then run the
   pooling MLP residual.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvF


DEFAULT_MODEL_PATH = "/Users/frankfacundo/Models/google/siglip2-so400m-patch14-384"
DEFAULT_IMAGE_PATH = Path(__file__).with_name("image.png")


@dataclass
class SiglipVisionConfig:
    hidden_size: int = 1152
    image_size: int = 384
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    num_channels: int = 3
    num_hidden_layers: int = 27
    patch_size: int = 14
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "SiglipVisionConfig":
        with open(Path(model_dir) / "config.json") as f:
            data = json.load(f)

        vision_data = dict(data["vision_config"])
        vision_data.pop("model_type", None)
        vision_data.pop("transformers_version", None)

        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in vision_data.items() if k in valid_keys})


@dataclass
class SiglipImageProcessorConfig:
    height: int = 384
    width: int = 384
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    rescale_factor: float = 1.0 / 255.0
    resample: int = 2

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "SiglipImageProcessorConfig":
        path = Path(model_dir) / "preprocessor_config.json"
        if not path.exists():
            return cls()

        with open(path) as f:
            data = json.load(f)

        size = data.get("size", {})
        return cls(
            height=int(size.get("height", 384)),
            width=int(size.get("width", 384)),
            image_mean=tuple(data.get("image_mean", [0.5, 0.5, 0.5])),
            image_std=tuple(data.get("image_std", [0.5, 0.5, 0.5])),
            rescale_factor=float(data.get("rescale_factor", 1.0 / 255.0)),
            resample=int(data.get("resample", 2)),
        )


class SiglipImageProcessorTorch:
    """Small PIL + torch image processor matching the local SigLIP processor."""

    def __init__(self, config: SiglipImageProcessorConfig):
        self.config = config

    def __call__(self, image_path: str | Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        pixel_values = tvF.pil_to_tensor(image).unsqueeze(0)
        pixel_values = tvF.resize(
            pixel_values,
            [self.config.height, self.config.width],
            interpolation=pil_resample_to_torch(self.config.resample),
            antialias=True,
        )

        # Transformers' torchvision backend fuses rescale + normalize as
        # normalize(uint8_image, mean / scale, std / scale).
        scale = 1.0 / self.config.rescale_factor
        mean = tuple(value * scale for value in self.config.image_mean)
        std = tuple(value * scale for value in self.config.image_std)
        return tvF.normalize(pixel_values.to(dtype=torch.float32), mean, std)


def pil_resample_to_torch(resample: int) -> InterpolationMode:
    mapping = {
        Image.Resampling.NEAREST: InterpolationMode.NEAREST,
        Image.Resampling.BILINEAR: InterpolationMode.BILINEAR,
        Image.Resampling.BICUBIC: InterpolationMode.BICUBIC,
        Image.Resampling.BOX: InterpolationMode.BOX,
        Image.Resampling.HAMMING: InterpolationMode.HAMMING,
        Image.Resampling.LANCZOS: InterpolationMode.BICUBIC,
    }
    return mapping[Image.Resampling(resample)]


class SiglipVisionEmbeddings(nn.Module):
    r"""Conv2d patch embedding plus learned absolute positions."""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid",
        )
        self.position_embedding = nn.Embedding(config.num_patches, config.hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(config.num_patches).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        return embeddings + self.position_embedding(self.position_ids)


class SiglipAttention(nn.Module):
    r"""Eager multi-head self-attention matching Transformers' SigLIP path."""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_len, self.num_heads, self.head_dim)

        queries = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        keys = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        values = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attn_weights = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(queries.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        if config.hidden_act != "gelu_pytorch_tanh":
            raise ValueError(f"Unsupported SigLIP activation: {config.hidden_act}")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        return self.fc2(hidden_states)


class SiglipEncoderLayer(nn.Module):
    r"""Pre-norm ViT block: attention residual followed by MLP residual."""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    r"""Learned-query attention pooling head used by SigLIP."""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            batch_first=True,
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        probe = self.probe.repeat(hidden_state.shape[0], 1, 1)
        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]
        hidden_state = hidden_state + self.mlp(self.layernorm(hidden_state))
        return hidden_state[:, 0]


class SiglipVisionModelTorch(nn.Module):
    """Pure-torch SigLIP vision encoder."""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head = SiglipMultiheadAttentionPoolingHead(config)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        last_hidden_state = self.post_layernorm(hidden_states)
        pooler_output = self.head(last_hidden_state)
        return {
            "last_hidden_state": last_hidden_state,
            "pooler_output": pooler_output,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local SigLIP image encoder with a pure-torch model."
    )
    parser.add_argument("--image", default=str(DEFAULT_IMAGE_PATH), help="Path to the input image.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local checkpoint directory.")
    parser.add_argument("--device", default=None, help="cpu | cuda | mps; auto-detect if omitted.")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default=None,
        help="Parameter dtype; defaults to fp16 on MPS, bf16/fp16 on CUDA, fp32 on CPU.",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip the final comparison against Transformers.",
    )
    parser.add_argument("--atol", type=float, default=None, help="Absolute tolerance for comparison.")
    parser.add_argument("--rtol", type=float, default=None, help="Relative tolerance for comparison.")
    return parser.parse_args()


def resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(requested: str | None, device: torch.device) -> torch.dtype:
    if requested is not None:
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[requested]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def load_siglip_vision_weights(
    model: nn.Module,
    model_dir: str | Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Stream vision weights from safetensors into the pure-torch module."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required to load this checkpoint") from exc

    model_dir = Path(model_dir)
    model_path = model_dir / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"Expected checkpoint file: {model_path}")

    state = model.state_dict()
    missing = set(state)
    unexpected: list[str] = []
    loaded: list[str] = []

    with safe_open(str(model_path), framework="pt") as f, torch.no_grad():
        for key in f.keys():
            if not key.startswith("vision_model."):
                continue

            target = key.removeprefix("vision_model.")
            if target not in state:
                unexpected.append(target)
                continue

            tensor = f.get_tensor(key)
            dest = state[target]
            if tensor.shape != dest.shape:
                raise ValueError(
                    f"shape mismatch for {target}: checkpoint {tuple(tensor.shape)} "
                    f"vs model {tuple(dest.shape)}"
                )
            dest.copy_(tensor.to(device=dest.device, dtype=dest.dtype))
            missing.discard(target)
            loaded.append(target)

    if strict and missing:
        raise RuntimeError(f"Missing {len(missing)} parameters, first few: {sorted(missing)[:5]}")
    if strict and unexpected:
        raise RuntimeError(f"Unexpected {len(unexpected)} keys, first few: {unexpected[:5]}")

    return {"loaded": loaded, "missing": sorted(missing), "unexpected": unexpected}


def build_torch_model(
    model_path: str | Path,
    device: torch.device,
    dtype: torch.dtype,
) -> SiglipVisionModelTorch:
    config = SiglipVisionConfig.from_pretrained(model_path)
    model = SiglipVisionModelTorch(config).eval()
    model.to(device=device, dtype=dtype)
    load_siglip_vision_weights(model, model_path)
    return model


def prepare_pixel_values(
    image_path: str | Path,
    model_path: str | Path,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    processor_config = SiglipImageProcessorConfig.from_pretrained(model_path)
    processor = SiglipImageProcessorTorch(processor_config)
    pixel_values = processor(image_path)
    return pixel_values.to(device=device, dtype=dtype)


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def default_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-4, 1e-4
    return 5e-2, 5e-2


@torch.inference_mode()
def run_torch(
    image_path: Path,
    model_path: str | Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    pixel_values = prepare_pixel_values(image_path, model_path, device, dtype)
    model = build_torch_model(model_path, device, dtype)
    outputs = model(pixel_values)
    outputs_cpu = {name: value.detach().float().cpu() for name, value in outputs.items()}
    pixels_cpu = pixel_values.detach().float().cpu()
    del model, outputs, pixel_values
    cleanup_memory()
    return outputs_cpu, pixels_cpu


@torch.inference_mode()
def run_transformers(
    image_path: Path,
    model_path: str | Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    from transformers import SiglipImageProcessor, SiglipVisionModel
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    processor = SiglipImageProcessor.from_pretrained(model_path)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=[image], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)

    model = SiglipVisionModel.from_pretrained(
        model_path,
        attn_implementation="eager",
        dtype=dtype,
    ).eval()
    model.to(device=device)

    outputs = model(pixel_values=pixel_values)
    outputs_cpu = {
        "last_hidden_state": outputs.last_hidden_state.detach().float().cpu(),
        "pooler_output": outputs.pooler_output.detach().float().cpu(),
    }
    pixels_cpu = pixel_values.detach().float().cpu()
    del model, outputs, pixel_values
    cleanup_memory()
    return outputs_cpu, pixels_cpu


def compare_outputs(
    torch_outputs: dict[str, torch.Tensor],
    hf_outputs: dict[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> bool:
    ok = True
    for name in ("last_hidden_state", "pooler_output"):
        left = torch_outputs[name]
        right = hf_outputs[name]
        max_abs = (left - right).abs().max().item()
        mean_abs = (left - right).abs().mean().item()
        matches = torch.allclose(left, right, atol=atol, rtol=rtol)
        ok = ok and matches
        print(
            f"{name}: shape={tuple(left.shape)} "
            f"max_abs_diff={max_abs:.6g} mean_abs_diff={mean_abs:.6g} "
            f"allclose={matches}"
        )
    return ok


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    atol, rtol = default_tolerances(dtype)
    if args.atol is not None:
        atol = args.atol
    if args.rtol is not None:
        rtol = args.rtol

    print(f"device={device} dtype={dtype} image={image_path.name}")
    torch_outputs, torch_pixels = run_torch(image_path, model_path, device, dtype)
    print(f"torch pooler_output shape: {tuple(torch_outputs['pooler_output'].shape)}")

    if args.no_compare:
        return

    hf_outputs, hf_pixels = run_transformers(image_path, model_path, device, dtype)
    pixel_max_abs = (torch_pixels - hf_pixels).abs().max().item()
    print(f"processor pixel_values max_abs_diff={pixel_max_abs:.6g}")
    print("comparison against Transformers:")
    matches = compare_outputs(torch_outputs, hf_outputs, atol=atol, rtol=rtol)
    print(f"torch_matches_transformers={matches} atol={atol} rtol={rtol}")


if __name__ == "__main__":
    main()
