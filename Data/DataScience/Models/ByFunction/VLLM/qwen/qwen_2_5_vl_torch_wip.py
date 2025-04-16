import io
import math
import os
from typing import List, Optional, Tuple, Union

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn import CrossEntropyLoss

# Assume huggingface tokenizers is acceptable for loading the tokenizer itself
# Replicating the tokenizer from scratch is extremely complex.
from transformers import AutoTokenizer  # Using HF just for the tokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast

# ============================================================================
# 1. Model Configuration (Manually Defined or Loaded from config.json)
# ============================================================================
# You would typically load these values from the model's config.json file
# For Qwen/Qwen2.5-VL-3B-Instruct (example values, check actual config.json)
# Main LLM Config
LLM_CONFIG = {
    "vocab_size": 151936,  # Check exact value
    "hidden_size": 3072,
    "intermediate_size": 8960,
    "num_hidden_layers": 28,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,  # GQA
    "hidden_act": "silu",
    "max_position_embeddings": 32768,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-06,  # Check exact value
    "use_cache": True,
    "tie_word_embeddings": False,
    "rope_theta": 10000.0,  # Check exact value, Qwen2.5 might use 1M
    "use_sliding_window": False,  # Or True depending on config
    "sliding_window": 4096,  # If use_sliding_window=True
    "max_window_layers": 28,  # Check exact value
    "attention_dropout": 0.0,
    "rope_scaling": {
        "type": "default",
        "mrope_section": 128,
    },  # Simplified, check actual config
    "_attn_implementation": "flash_attention_2",  # or "sdpa" or "eager"
    "pad_token_id": 151643,  # Check exact value for <|endoftext|>
    "image_token_id": 151646,  # <|image_pad|> - Check exact value
    "video_token_id": 151647,  # <|video_pad|> - Check exact value
    "vision_start_token_id": 151648,  # <|vision_start|> - Check exact value
}

# Vision Encoder Config
VISION_CONFIG = {
    "hidden_size": 1152,  # Check exact value
    "intermediate_size": 4304,  # Check exact value
    "num_hidden_layers": 24,  # Check exact value ('depth')
    "num_attention_heads": 16,  # Check exact value ('num_heads')
    "image_size": 448,  # Typical value, check config
    "patch_size": 14,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "in_channels": 3,
    "window_size": 112,
    "fullatt_block_indexes": [5, 11, 17, 23],  # Check exact value
    "hidden_act": "gelu",  # Often GELU in ViTs, check config
    "layer_norm_eps": 1e-6,  # Typical ViT value
    "out_hidden_size": LLM_CONFIG["hidden_size"],  # Needs to match LLM hidden size
    "tokens_per_second": 4,  # Check actual value
    "_attn_implementation": "sdpa",  # or "flash_attention_2" or "eager"
}


# Combine into a main config class/dict if needed
class Qwen2_5_VLConfig:
    def __init__(self, llm_config, vision_config):
        self.llm_config = llm_config
        self.vision_config = vision_config
        # Add top-level attributes by merging or delegation
        for k, v in llm_config.items():
            setattr(self, k, v)
        # Store vision config separately or merge non-conflicting keys
        self.vision_config = type(
            "VisionConfig", (), vision_config
        )()  # Simple object from dict


# Instantiate the combined config
config = Qwen2_5_VLConfig(LLM_CONFIG, VISION_CONFIG)
# Set _attn_implementation on sub-configs too if needed by classes
config.vision_config._attn_implementation = VISION_CONFIG["_attn_implementation"]


# ============================================================================
# 2. Model Definition (Manually Copied/Adapted from HF Source)
# ============================================================================
# --- Qwen2RMSNorm ---
class Qwen2RMSNorm(nn.Module):
    # ... (Copy implementation from modeling_qwen2_5_vl.py) ...
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# --- ACT2FN ---
# Define or import necessary activation functions (e.g., SiLU, GELU)
ACT2FN = {
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
    # Add others if needed
}


# --- RoPE Helper Functions ---
def rotate_half(x):
    # ... (Copy implementation from modeling_qwen2_5_vl.py) ...
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    # ... (Copy implementation from modeling_qwen2_5_vl.py) ...
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    # ... (Copy implementation from modeling_qwen2_5_vl.py) ...
    mrope_section = mrope_section * 2
    cos = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)
    sin = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # ... (Copy implementation from modeling_qwen2_5_vl.py) ...
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# --- Vision Components ---
class Qwen2_5_VisionPatchEmbed(nn.Module):
    # ... (Copy implementation from modeling_qwen2_5_vl.py) ...
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        # Adjust view based on expected input shape (B, C, T, H, W) or just (C, T, H, W) if single
        # The HF code assumes a flattened input that gets reshaped. Adapt if necessary.
        # Assuming input is (B, C, T_total, H, W) where T_total is divisible by temporal_patch_size
        B, C, T_total, H, W = hidden_states.shape
        num_temp_patches = T_total // self.temporal_patch_size
        num_spat_patches_h = H // self.patch_size
        num_spat_patches_w = W // self.patch_size

        # Reshape for Conv3D: (B * num_temp_patches * num_spat_patches_h * num_spat_patches_w, C, T_patch, H_patch, W_patch)
        # Or simpler if Conv3D handles strides correctly: (B, C, T, H, W) -> (B, D, T', H', W')
        # Let's assume the HF reshape logic implies the input is pre-flattened per patch
        # Input shape in HF `process_vision_info` -> `image_processor` needs careful checking.
        # Assuming `hidden_states` is (N, C, T_patch, H_patch, W_patch) where N = B * num_patches
        # The HF code uses:
        # hidden_states = hidden_states.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        # This suggests input might be (TotalPatches, C*T_patch*H_patch*W_patch) or similar.
        # Let's assume input is (NumImagesOrVideos, Channels, TemporalSize, Height, Width)
        # This needs clarification based on how pixel_values is prepared.
        # Assuming (N, C, T_p, H_p, W_p) where N = total number of patches across batch/images
        # If input is (B, C, T, H, W), reshape/unfold first.
        # Simplified assumption: Input is ALREADY patched: (TotalPatches, C, T_patch, H_patch, W_patch)

        # Let's stick to the HF code's view logic for now, assuming input is structured correctly:
        # Input shape: (TotalElements, C * T_patch * H_patch * W_patch) ??? No, that doesn't match the view.
        # Let's assume input `pixel_values` is (NumImages, C, T, H, W) or (NumVideos, C, T, H, W)
        # The `visual` forward gets (TotalPatches, embed_dim) after patch_embed
        # How does the input become (N, C, T_patch, H_patch, W_patch)? -> Likely done in image processor

        # Simplification: Assume input `hidden_states` to this *module* has shape (NumPatches, C, T_patch, H_patch, W_patch)
        # This shape seems inconsistent with how nn.Conv3d is typically used (expects B,C,D,H,W).
        # Re-reading HF Conv3D call:
        # self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)
        # Input to Conv3d: hidden_states.to(dtype=target_dtype)
        # Shape before view: Needs clarification. Let's assume B, C, T, H, W input for now.

        # If input is (B, C, T, H, W)
        B, C, T, H, W = hidden_states.shape
        hidden_states = self.proj(
            hidden_states.to(dtype=target_dtype)
        )  # Output: (B, embed_dim, T', H', W')
        # Output needs to be (TotalPatches, embed_dim)
        # TotalPatches = B * T' * H' * W'
        hidden_states = hidden_states.flatten(2).transpose(
            1, 2
        )  # (B, T'*H'*W', embed_dim)
        hidden_states = hidden_states.reshape(
            -1, self.embed_dim
        )  # (B * T'*H'*W', embed_dim)
        return hidden_states


class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    # ... (Copy implementation from modeling_qwen2_5_vl.py) ...
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen2_5_VLMLP(nn.Module):
    # ... (Copy implementation from modeling_qwen2_5_vl.py) ...
    def __init__(self, config, bias: bool = False):
        super().__init__()
        # Use vision_config here if called from vision model
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]  # Make sure config has hidden_act

    def forward(self, hidden_state):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
        )


class Qwen2_5_VLPatchMerger(nn.Module):
    # ... (Copy implementation from modeling_qwen2_5_vl.py) ...
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)  # Norm context_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),  # Input merged dim
            nn.GELU(),  # Use GELU as in HF code
            nn.Linear(self.hidden_size, dim),  # Output final dim
        )
        self.spatial_merge_size_sq = spatial_merge_size**2
        self.context_dim = context_dim  # Store for reshape if needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (N, context_dim), where N = total patches after block processing
        # HF logic merges *after* windowing/attention.
        # The input `x` to merger comes from the last vision block.
        # It needs reshaping before MLP if spatial merge happens here.
        # The HF code ln_q(x).view(-1, self.hidden_size) implies x is already mergeable
        # Let's follow HF: input x has shape (N, context_dim)
        # N must be divisible by spatial_merge_size**2
        n_tokens, ctx_dim = x.shape
        x_norm = self.ln_q(x)
        # Reshape for merging: (N / merge_sq, merge_sq, ctx_dim) -> (N / merge_sq, merge_sq * ctx_dim)
        x_merged = x_norm.view(-1, self.spatial_merge_size_sq * self.context_dim)
        x_out = self.mlp(x_merged)  # Output: (N / merge_sq, dim)
        return x_out


# --- Vision Attention Classes (Select one based on config._attn_implementation) ---
# Option 1: Eager Attention
class Qwen2_5_VLVisionAttention(nn.Module):
    # ... (Copy Qwen2_5_VLVisionAttention implementation) ...
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,  # Indices marking start/end of sequences if batching images/videos
        rotary_pos_emb: Optional[torch.Tensor] = None,  # Deprecated in HF > 4.54
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]  # Assumes input (TotalSeqLen, Dim)
        q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)
        q = q.view(seq_length, self.num_heads, self.head_dim)
        k = k.view(seq_length, self.num_heads, self.head_dim)
        v = v.view(seq_length, self.num_heads, self.head_dim)

        if position_embeddings is None:
            # Fallback or error needed if rotary_pos_emb is removed
            raise ValueError("position_embeddings (cos, sin) are required.")
        else:
            cos, sin = position_embeddings  # Shape: (seq_length, head_dim_rope * 2)

        # Apply RoPE
        # Note: HF apply_rotary_pos_emb_vision expects B, H, S, D or S, H, D input shape
        # Here we have S, H, D. Need to adapt or use the HF function directly.
        # Let's assume apply_rotary_pos_emb_vision handles S, H, D input
        # It needs cos/sin reshaped correctly. Cos/sin is (S, D_rope*2)
        cos = cos.view(seq_length, -1)
        sin = sin.view(seq_length, -1)
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)  # Check shapes

        # Attention calculation (manual)
        # Input q, k, v: (SeqLen, NumHeads, HeadDim)
        # Transpose for matmul: (NumHeads, SeqLen, HeadDim)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # Build attention mask from cu_seqlens (block diagonal)
        # Mask should be (1, SeqLen, SeqLen) or (NumHeads, SeqLen, SeqLen) if head-specific
        attention_mask = torch.full(
            (1, seq_length, seq_length),
            torch.finfo(q.dtype).min,
            device=q.device,
            dtype=q.dtype,
        )
        # cu_seqlens: [0, len1, len1+len2, ...]
        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            attention_mask[..., start:end, start:end] = 0

        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q.dtype)
        # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training) # Add dropout if needed

        attn_output = torch.matmul(attn_weights, v)  # (NumHeads, SeqLen, HeadDim)

        # Transpose back and reshape: (SeqLen, NumHeads, HeadDim) -> (SeqLen, NumHeads * HeadDim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


# Option 2: SDPA Attention
class Qwen2_5_VLVisionSdpaAttention(nn.Module):
    # ... (Copy Qwen2_5_VLVisionSdpaAttention implementation) ...
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,  # Deprecated
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]  # Assumes input (TotalSeqLen, Dim)
        q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)
        # Reshape for SDPA: (Batch=1, SeqLen, NumHeads, HeadDim) - need batch dim
        # Or maybe SDPA handles (SeqLen, NumHeads, HeadDim)? Check docs.
        # F.scaled_dot_product_attention expects (..., SeqLen, NumHeads, HeadDim) or (..., NumHeads, SeqLen, HeadDim)
        # Let's reshape to (1, SeqLen, NumHeads, HeadDim)
        q = q.view(seq_length, self.num_heads, self.head_dim).unsqueeze(
            0
        )  # (1, S, H, D)
        k = k.view(seq_length, self.num_heads, self.head_dim).unsqueeze(
            0
        )  # (1, S, H, D)
        v = v.view(seq_length, self.num_heads, self.head_dim).unsqueeze(
            0
        )  # (1, S, H, D)

        if position_embeddings is None:
            raise ValueError("position_embeddings (cos, sin) are required.")
        else:
            cos, sin = position_embeddings  # Shape: (seq_length, head_dim_rope * 2)

        # Apply RoPE - needs adaptation for the added batch dim
        # apply_rotary_pos_emb_vision needs Q/K shape S,H,D or B,H,S,D
        # Let's adapt cos/sin and call it
        cos = cos.view(1, seq_length, -1)  # Add batch dim if needed by RoPE func
        sin = sin.view(1, seq_length, -1)

        # Need to reshape Q/K temporarily? Or adapt RoPE func.
        # Assuming RoPE func can handle (1, S, H, D) if cos/sin are (1, S, D_rope*2)
        # This needs verification.
        # Qwen HF RoPE functions are complex. Let's assume it works for now.
        # The apply_rotary_pos_emb_vision used in HF expects S,H,D. Let's use that.
        q_squeezed = q.squeeze(0)
        k_squeezed = k.squeeze(0)
        cos_squeezed = cos.squeeze(0)
        sin_squeezed = sin.squeeze(0)
        q_rot, k_rot = apply_rotary_pos_emb_vision(
            q_squeezed, k_squeezed, cos_squeezed, sin_squeezed
        )
        q = q_rot.unsqueeze(0)  # (1, S, H, D)
        k = k_rot.unsqueeze(0)  # (1, S, H, D)

        # Build SDPA attention mask from cu_seqlens (block diagonal boolean mask)
        # Mask shape (Batch=1, NumHeads=1, SeqLen, SeqLen) or (1, 1, S, S)
        attn_mask = torch.zeros(
            (1, 1, seq_length, seq_length), dtype=torch.bool, device=q.device
        )
        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            attn_mask[..., start:end, start:end] = True  # True means attend

        # SDPA expects attention_mask where True means *masked out* if not using is_causal=True
        # Let's invert the mask for SDPA: True means DO NOT attend
        sdpa_mask = ~attn_mask

        # Call SDPA
        # Need to reshape Q,K,V to (B, H, S, D) probably
        q = q.transpose(1, 2)  # (1, H, S, D)
        k = k.transpose(1, 2)  # (1, H, S, D)
        v = v.transpose(1, 2)  # (1, H, S, D)
        sdpa_mask = sdpa_mask.squeeze(
            1
        )  # Mask shape (B, S, S) or (B, H, S, S)? SDPA handles broadcasting. (1, S, S) is fine.

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,  # Use the inverted mask
            dropout_p=0.0,  # Add dropout if needed
            is_causal=False,  # Not causal unless specified
        )  # Output: (1, H, S, D)

        # Reshape back to (SeqLen, Dim)
        attn_output = attn_output.transpose(1, 2)  # (1, S, H, D)
        attn_output = attn_output.reshape(seq_length, -1)  # (S, H*D)
        attn_output = self.proj(attn_output)
        return attn_output


# Option 3: Flash Attention 2 (Requires installation and CUDA)
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    IS_FLASH_ATTN_AVAILABLE = True
except ImportError:
    print("Flash Attention 2 not available. Install with `pip install flash-attn`")
    IS_FLASH_ATTN_AVAILABLE = False

    # Define apply_rotary_emb if FA2 is not available but used in the code below
    def apply_rotary_emb(x, cos, sin, interleaved=False):
        # Simplified implementation for non-FA2 RoPE
        x_float = x.float()
        cos = cos.float()
        sin = sin.float()
        if not interleaved:
            x_embed = (x_float * cos) + (rotate_half(x_float) * sin)
        else:
            # Handle interleaved if needed
            raise NotImplementedError("Interleaved RoPE not implemented here")
        return x_embed.to(x.dtype)


if IS_FLASH_ATTN_AVAILABLE:
    # Define FA2 RoPE helper if needed (HF code has one)
    def apply_rotary_pos_emb_flashatt(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ... (Copy implementation from modeling_qwen2_5_vl.py) ...
        # Assumes cos/sin are already prepared correctly (maybe half dim)
        cos = cos.chunk(2, dim=-1)[0].contiguous()
        sin = sin.chunk(2, dim=-1)[0].contiguous()
        # apply_rotary_emb expects (..., SeqLen, Dim)
        # Input q, k are likely (Batch, SeqLen, NumHeads, HeadDim) or similar
        # Adjust apply_rotary_emb call based on actual shapes
        # Assuming q, k are (B, S, H, D)
        # Assuming cos, sin are (B, S, D_rope) or (S, D_rope)
        # Needs careful shape handling based on FA2's RoPE integration
        # Simplified call, assuming correct shapes:
        q_embed = apply_rotary_emb(q, cos, sin)
        k_embed = apply_rotary_emb(k, cos, sin)
        return q_embed, k_embed

    class Qwen2_5_VLVisionFlashAttention2(nn.Module):
        # ... (Copy Qwen2_5_VLVisionFlashAttention2 implementation) ...
        def __init__(self, dim: int, num_heads: int = 16) -> None:
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.qkv = nn.Linear(dim, dim * 3, bias=True)
            self.proj = nn.Linear(dim, dim)

        def forward(
            self,
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,  # FA2 uses this for batching var len seqs
            rotary_pos_emb: Optional[torch.Tensor] = None,  # Deprecated
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ) -> torch.Tensor:
            # Input: (TotalSeqLen, Dim)
            qkv = self.qkv(hidden_states)  # (TotalSeqLen, 3 * Dim)

            # Reshape for FA2: (TotalSeqLen, NumHeads, 3, HeadDim)
            qkv = qkv.view(hidden_states.shape[0], self.num_heads, 3, self.head_dim)

            if position_embeddings is None:
                raise ValueError("position_embeddings (cos, sin) are required.")
            else:
                cos, sin = (
                    position_embeddings  # Shape: (TotalSeqLen, head_dim_rope * 2)
                )

            # Apply RoPE using FA2's apply_rotary_emb
            # Requires q/k split and careful shape handling for apply_rotary_emb
            q = qkv[..., 0, :]  # (TotalSeqLen, NumHeads, HeadDim)
            k = qkv[..., 1, :]  # (TotalSeqLen, NumHeads, HeadDim)
            v = qkv[..., 2, :]  # (TotalSeqLen, NumHeads, HeadDim)

            # Prepare cos/sin for apply_rotary_emb - needs D_rope dimension
            rope_dim = cos.shape[-1] // 2
            cos = cos[..., :rope_dim]
            sin = sin[..., :rope_dim]

            # apply_rotary_emb expects (..., SeqLen, Dim)
            # We have (TotalSeqLen, NumHeads, HeadDim)
            # Reshape q, k for RoPE: (TotalSeqLen, NumHeads * HeadDim) ? No.
            # (TotalSeqLen, NumHeads, HeadDim) -> needs RoPE on HeadDim part
            # Let's apply RoPE per head:
            q_rot = apply_rotary_emb(q.view(-1, self.head_dim), cos, sin).view(q.shape)
            k_rot = apply_rotary_emb(k.view(-1, self.head_dim), cos, sin).view(k.shape)

            # FA2 expects input Q, K, V: (TotalSeqLen, NumHeads, HeadDim)
            # cu_seqlens: (batch_size + 1,) tensor of sequence start indices
            # max_seqlen: maximum sequence length in the batch

            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

            attn_output = flash_attn_varlen_func(
                q_rot,
                k_rot,
                v,  # (TotalSeqLen, NumHeads, HeadDim)
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0.0,
                causal=False,  # Not causal for ViT encoder
            )  # Output: (TotalSeqLen, NumHeads, HeadDim)

            # Reshape back to (TotalSeqLen, Dim)
            attn_output = attn_output.reshape(hidden_states.shape[0], -1)
            attn_output = self.proj(attn_output)
            return attn_output


# --- Vision Block ---
class Qwen2_5_VLVisionBlock(nn.Module):
    # ... (Copy implementation from modeling_qwen2_5_vl.py) ...
    def __init__(self, vision_config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = Qwen2RMSNorm(
            vision_config.hidden_size, eps=vision_config.layer_norm_eps
        )  # Use vision config eps
        self.norm2 = Qwen2RMSNorm(
            vision_config.hidden_size, eps=vision_config.layer_norm_eps
        )

        # Select Attention Class
        if attn_implementation == "flash_attention_2" and IS_FLASH_ATTN_AVAILABLE:
            ATTN_CLASS = Qwen2_5_VLVisionFlashAttention2
        elif attn_implementation == "sdpa":
            ATTN_CLASS = Qwen2_5_VLVisionSdpaAttention
        else:  # eager
            ATTN_CLASS = Qwen2_5_VLVisionAttention
        self.attn = ATTN_CLASS(
            vision_config.hidden_size,
            num_heads=vision_config.num_attention_heads,  # Use vision heads
        )
        # Use the specific Vision MLP if different, otherwise use general MLP
        # HF code uses Qwen2_5_VLMLP here. Make sure config matches.
        self.mlp = Qwen2_5_VLMLP(vision_config, bias=True)  # Pass vision_config

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,  # Deprecated
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Residual connection for attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attn_output = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            # rotary_pos_emb=rotary_pos_emb, # Deprecated
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + attn_output

        # Residual connection for MLP
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# --- Vision Transformer ---
class Qwen2_5_VisionTransformer(nn.Module):
    # ... (Adapt Qwen2_5_VisionTransformerPretrainedModel) ...
    def __init__(self, vision_config) -> None:
        super().__init__()
        self.config = vision_config  # Store config

        self.spatial_merge_size = vision_config.spatial_merge_size
        self.patch_size = vision_config.patch_size
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes
        self.window_size = vision_config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            in_channels=vision_config.in_channels,
            embed_dim=vision_config.hidden_size,
        )

        head_dim = vision_config.hidden_size // vision_config.num_attention_heads
        # RoPE for spatial H, W dimensions only in ViT? Check HF code.
        # HF code uses `Qwen2_5_VisionRotaryEmbedding(head_dim // 2)`
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(
            head_dim // 2
        )  # RoPE for 2D spatial

        self.blocks = nn.ModuleList(
            [
                Qwen2_5_VLVisionBlock(vision_config, vision_config._attn_implementation)
                for _ in range(vision_config.num_hidden_layers)
            ]  # Use num_hidden_layers from vision_config
        )
        self.merger = Qwen2_5_VLPatchMerger(
            dim=vision_config.out_hidden_size,  # Output dim should match LLM
            context_dim=vision_config.hidden_size,  # Input dim from blocks
            spatial_merge_size=vision_config.spatial_merge_size,
        )
        # self.gradient_checkpointing = False # Control externally if needed

    def rot_pos_emb(self, grid_thw):
        # ... (Copy implementation from modeling_qwen2_5_vl.py Qwen2_5_VisionTransformerPretrainedModel) ...
        # Calculates spatial RoPE embeddings based on grid layout
        pos_ids = []
        # grid_thw: (NumImagesOrVideos, 3) -> [T, H, W] for each
        for t, h, w in grid_thw:
            # Calculate H, W position IDs within each T slice
            hpos_ids = (
                torch.arange(h, device=self.rotary_pos_emb.inv_freq.device)
                .unsqueeze(1)
                .expand(-1, w)
            )
            # The HF code does complex reshaping/permuting for windowing? Let's simplify first.
            # Simpler: Assume simple grid H, W pos ids repeated over T
            hpos_ids_flat = hpos_ids.flatten()  # (H*W,)
            wpos_ids = (
                torch.arange(w, device=self.rotary_pos_emb.inv_freq.device)
                .unsqueeze(0)
                .expand(h, -1)
            )
            wpos_ids_flat = wpos_ids.flatten()  # (H*W,)

            # Stack H, W pos ids and repeat for T dimension
            # Shape: (H*W, 2) -> repeat T times -> (T*H*W, 2)
            pos_ids_single = torch.stack([hpos_ids_flat, wpos_ids_flat], dim=-1)
            pos_ids.append(pos_ids_single.repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)  # (TotalPatches = sum(T*H*W), 2)

        # Get RoPE frequencies
        # Max grid size for H or W? HF uses grid_thw[:, 1:].max() -> max(H, W)
        max_grid_size = grid_thw[:, 1:].max()  # Max H or W across all inputs
        rotary_pos_emb_full = self.rotary_pos_emb(
            max_grid_size
        )  # (max_grid_size, rope_head_dim)

        # Select frequencies based on calculated pos_ids
        # Need RoPE for H and W separately? HF uses flatten(1) after indexing.
        # rotary_pos_emb_full[pos_ids] -> selects based on H and W indices?
        # This part of HF code is complex, likely tied to their specific windowing/merging.
        # Let's assume we get cos/sin for H and W dimensions.
        # Need to combine them or apply sequentially? Qwen VL paper might clarify.
        # Simplified: Use flatten(1) like HF.
        # rotary_pos_emb shape needs checking: (max_grid, rope_head_dim/2)? Yes.
        # rotary_pos_emb_full[pos_ids] -> (TotalPatches, 2, rope_head_dim/2)
        # Flatten -> (TotalPatches, rope_head_dim)

        h_freqs = rotary_pos_emb_full[pos_ids[:, 0]]  # (TotalPatches, rope_head_dim/2)
        w_freqs = rotary_pos_emb_full[pos_ids[:, 1]]  # (TotalPatches, rope_head_dim/2)
        rotary_pos_emb = torch.cat(
            [h_freqs, w_freqs], dim=-1
        )  # (TotalPatches, rope_head_dim)

        return rotary_pos_emb  # This is freqs, not cos/sin yet

    def get_window_index(self, grid_thw):
        # ... (Copy implementation from modeling_qwen2_5_vl.py Qwen2_5_VisionTransformerPretrainedModel) ...
        # Calculates indices for windowed attention based on grid and window size config
        # This is complex and depends heavily on exact config values.
        # Returns indices to reorder tokens for windowing and cu_seqlens for window blocks.
        # For simplicity in this skeleton, we might skip windowing or assume full attention.
        # If skipping, return None or identity indices.
        # Let's assume full attention for now (or copy the complex logic if needed)
        # If full attention:
        num_patches = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum()
        window_index = torch.arange(num_patches, device=grid_thw.device)
        # cu_window_seqlens are the boundaries for full attention blocks (i.e., per image/video)
        lengths = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
        cu_window_seqlens = torch.cat(
            [torch.tensor([0], device=grid_thw.device), lengths.cumsum(0)]
        )
        return window_index, cu_window_seqlens.to(torch.int32)  # Return int32 like HF

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        # Input pixel_values: (B, C, T, H, W) or similar, needs patching
        # Input grid_thw: (NumImagesOrVideos, 3) with T, H, W *in patch units*
        # print(f"ViT Input pixel_values shape: {pixel_values.shape}") # Debug
        # print(f"ViT Input grid_thw: {grid_thw}") # Debug

        # 1. Patch Embedding
        # Assume patch_embed handles the input shape correctly now
        hidden_states = self.patch_embed(
            pixel_values
        )  # Output: (TotalPatches, embed_dim)
        # print(f"ViT after patch_embed shape: {hidden_states.shape}") # Debug

        # 2. Calculate Positional Embeddings (RoPE)
        # Grid needs to be in units of pixels initially for rot_pos_emb?
        # HF `rot_pos_emb` uses grid_thw which seems to be in patch units.
        rotary_emb_freqs = self.rot_pos_emb(grid_thw)  # (TotalPatches, rope_head_dim)
        # Convert freqs to cos/sin
        # HF code does `emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)`
        # Let's replicate that for now.
        emb = torch.cat(
            (rotary_emb_freqs, rotary_emb_freqs), dim=-1
        )  # (TotalPatches, rope_head_dim*2)
        position_embeddings = (emb.cos(), emb.sin())  # Tuple of (cos, sin)

        # 3. Windowing (or Full Attention based on implementation)
        # Assume get_window_index provides correct indices and boundaries
        # window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        # cu_window_seqlens = torch.tensor(cu_window_seqlens, device=hidden_states.device, dtype=torch.int32)
        # cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        # Reorder hidden_states and position_embeddings based on window_index
        # This step is skipped if not using windowing logic from HF
        # hidden_states = hidden_states[window_index, :]
        # cos_w = position_embeddings[0][window_index, :]
        # sin_w = position_embeddings[1][window_index, :]
        # position_embeddings_windowed = (cos_w, sin_w)

        # Prepare cu_seqlens for full attention blocks (used if layer is in fullatt_block_indexes)
        # cu_seqlens_full = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
        # cu_seqlens_full = F.pad(cu_seqlens_full, (1, 0), value=0) # Add 0 at the start

        # --- Simplified cu_seqlens for full attention only ---
        # Calculate total patches per item in batch
        patches_per_item = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
        cu_seqlens_full = torch.cat(
            [
                torch.tensor([0], device=hidden_states.device, dtype=torch.int32),
                patches_per_item.cumsum(0).to(torch.int32),
            ]
        )
        # --- End Simplified ---

        # 4. Transformer Blocks
        for layer_num, blk in enumerate(self.blocks):
            # Determine if using full attention or windowed attention for this layer
            # Use cu_seqlens_full for all blocks in simplified case
            current_cu_seqlens = cu_seqlens_full
            # Pass windowed pos embeddings if windowing, else full
            current_pos_embed = position_embeddings

            # Apply block
            # Add gradient checkpointing control if needed
            hidden_states = blk(
                hidden_states,
                cu_seqlens=current_cu_seqlens,
                position_embeddings=current_pos_embed,
            )

        # 5. Reverse Windowing (if applied)
        # If windowing was used, reverse the permutation
        # reverse_indices = torch.argsort(window_index)
        # hidden_states = hidden_states[reverse_indices, :]

        # 6. Patch Merging
        # Input to merger: (TotalPatches, hidden_size)
        # Output: (TotalPatches / merge_unit, out_hidden_size)
        # Note: HF PatchMerger takes context_dim which is hidden_size here
        # and dim which is out_hidden_size
        hidden_states = self.merger(hidden_states)
        # print(f"ViT after merger shape: {hidden_states.shape}") # Debug

        # Final output shape should be (Total Merged Patches, LLM Hidden Size)
        return hidden_states


# --- LLM Components (Adapted from Qwen2 / Qwen2_5_VL) ---
class Qwen2_5_VLRotaryEmbedding(nn.Module):
    # ... (Copy/Adapt implementation from modeling_qwen2_5_vl.py) ...
    # Handles the 3D M-RoPE for combined text/vision sequences in the LLM part
    def __init__(self, config: Qwen2_5_VLConfig, device=None):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.max_seq_len_cached = config.max_position_embeddings
        self.base = config.rope_theta

        # Calculate inverse frequencies
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
                / self.head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Scaling factor (usually 1.0 unless using specific scaling like NTK)
        self.attention_scaling = 1.0  # Default, adjust if rope_scaling is used

    def forward(self, x, position_ids):
        # x: The input tensor (only used for dtype and device)
        # position_ids: (3, BatchSize, SeqLen) tensor containing T, H, W indices

        # inv_freq shape: (head_dim / 2)
        # Expand inv_freq for matmul: (1, 1, head_dim / 2, 1) -> (3, 1, head_dim/2, 1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].expand(3, 1, -1, 1)

        # position_ids shape: (3, BatchSize, SeqLen)
        # Expand for matmul: (3, BatchSize, 1, SeqLen)
        position_ids_expanded = position_ids.unsqueeze(2).float()

        # Calculate frequencies: (inv_freq @ pos_ids)
        # (3, 1, D/2, 1) @ (3, B, 1, S) -> (3, B, D/2, S)
        freqs = torch.matmul(inv_freq_expanded.float(), position_ids_expanded.float())

        # Transpose to (3, B, S, D/2)
        freqs = freqs.transpose(2, 3)

        # Concatenate for full head dim: (3, B, S, D)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Calculate cos and sin: shape (3, B, S, D)
        cos = (emb.cos() * self.attention_scaling).to(x.dtype)
        sin = (emb.sin() * self.attention_scaling).to(x.dtype)

        # Return tuple (cos, sin), each of shape (3, B, S, D)
        return cos, sin


class Qwen2MLP(nn.Module):
    # ... (Copy implementation from modeling_qwen2_5_vl.py) ...
    def __init__(self, config):  # Assumes LLM config
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# --- LLM Attention Classes (Select one) ---
# Option 1: Eager Attention
class Qwen2_5_VLAttention(nn.Module):
    # ... (Copy/Adapt Qwen2_5_VLAttention implementation) ...
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx  # Needed for KV cache management

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True  # Standard for decoders
        self.attention_dropout = config.attention_dropout
        # RoPE scaling info needed by apply_multimodal_rotary_pos_emb
        self.rope_scaling_mrope_section = config.rope_scaling.get(
            "mrope_section", self.head_dim // 3
        )  # Default/fallback needed

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # RoPE embedding module is instantiated in the main model and passed down/used via position_embeddings
        # self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config) # Not needed here if pos embeds are precomputed

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[
            torch.Tensor
        ] = None,  # Expected shape (B, 1, Q_len, K_len)
        position_ids: Optional[
            torch.LongTensor
        ] = None,  # Not directly used if position_embeddings provided
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # KV Cache
        output_attentions: bool = False,
        use_cache: bool = False,
        # No cache_position needed here, handled by KV cache update logic
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # Tuple of (cos, sin), shape (3, B, S, D)
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape Q, K, V: (B, SeqLen, NumHeads/NumKVHeads, HeadDim) -> (B, NumHeads/NumKVHeads, SeqLen, HeadDim)
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # Apply RoPE
        if position_embeddings is None:
            raise ValueError(
                "position_embeddings (cos, sin) are required for LLM attention."
            )
        cos, sin = position_embeddings  # Shape (3, B, S, D)

        # apply_multimodal_rotary_pos_emb expects cos/sin of shape (B, S, D) ? No, (3, B, S, D) is correct.
        # It handles the splitting internally based on mrope_section.
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            self.rope_scaling_mrope_section,
            unsqueeze_dim=1,  # dim 1 for heads
        )

        # KV Cache update
        if past_key_value is not None:
            # Append current K, V to cached K, V
            # Shape of cache: (B, NumKVHeads, SeqLen_cached, HeadDim)
            # Shape of current K, V: (B, NumKVHeads, q_len, HeadDim)
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        # Store updated KV cache if requested
        present_key_value = (key_states, value_states) if use_cache else None

        # Grouped Query Attention: Repeat K/V heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # Now K/V shape: (B, NumHeads, SeqLen_full, HeadDim)

        # Attention calculation
        # Q: (B, NumHeads, q_len, HeadDim)
        # K: (B, NumHeads, SeqLen_full, HeadDim)
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.size() != (
                bsz,
                1,
                q_len,
                key_states.shape[2],
            ):  # K_len = key_states.shape[2]
                raise ValueError(
                    f"Attention mask size mismatch: {attention_mask.size()} vs expected {(bsz, 1, q_len, key_states.shape[2])}"
                )
            attn_weights = (
                attn_weights + attention_mask
            )  # Mask has large negative values where attention is prohibited

        # Softmax
        # Handle potential overflows/NaNs in float16
        if attn_weights.dtype == torch.float16:
            attn_weights = torch.nan_to_num(
                attn_weights
            )  # More robust than HF's isinf check?
            # attn_weights = torch.where(torch.isinf(attn_weights), torch.full_like(attn_weights, -1e4), attn_weights) # Alternative
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        # Apply attention weights to Value
        # V: (B, NumHeads, SeqLen_full, HeadDim)
        attn_output = torch.matmul(
            attn_weights, value_states
        )  # (B, NumHeads, q_len, HeadDim)

        # Reshape output: (B, q_len, NumHeads, HeadDim) -> (B, q_len, HiddenSize)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return (
            attn_output,
            attn_weights if output_attentions else None,
            present_key_value,
        )


# Option 2: SDPA Attention (requires torch >= 2.0)
class Qwen2_5_VLSdpaAttention(nn.Module):
    # ... (Copy/Adapt Qwen2_5_VLSdpaAttention implementation) ...
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: Optional[int] = None):
        super().__init__()
        # Reuse init from Eager version
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True  # Standard for decoders
        self.attention_dropout = config.attention_dropout
        self.rope_scaling_mrope_section = config.rope_scaling.get(
            "mrope_section", self.head_dim // 3
        )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[
            torch.Tensor
        ] = None,  # Boolean mask for SDPA (B, Q_len, K_len) or (B, H, Q_len, K_len)
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,  # Not supported by SDPA
        use_cache: bool = False,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # (cos, sin)
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        if output_attentions:
            print(
                "Warning: output_attentions=True is not supported with SDPA. Returning None for attentions."
            )
            # Optionally fallback to Eager attention here if needed

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape Q, K, V: (B, SeqLen, NumHeads/NumKVHeads, HeadDim) -> (B, NumHeads/NumKVHeads, SeqLen, HeadDim)
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # Apply RoPE
        if position_embeddings is None:
            raise ValueError(
                "position_embeddings (cos, sin) are required for LLM attention."
            )
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            self.rope_scaling_mrope_section,
            unsqueeze_dim=1,
        )

        # KV Cache update
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None

        # GQA: Repeat K/V heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # Now K/V shape: (B, NumHeads, SeqLen_full, HeadDim)

        # Prepare for SDPA
        # Q, K, V need to be contiguous for some backends if using custom mask
        # HF code suggests this for CUDA + custom mask.
        if attention_mask is not None and query_states.device.type == "cuda":
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # SDPA mask: Boolean, True means MASK OUT.
        # Input attention_mask: Float, large negative means MASK OUT. Convert.
        sdpa_mask = None
        is_causal_sdpa = False
        if attention_mask is not None:
            # Convert float mask to boolean mask if needed
            # Assuming attention_mask shape is (B, 1, Q, K) or similar
            # SDPA prefers (B, Q, K) or (B, H, Q, K)
            # Let's assume input mask is correctly shaped float mask from _prepare_4d_causal_attention_mask
            sdpa_mask = attention_mask < -1.0  # True where masked
            if sdpa_mask.shape[1] == 1:  # (B, 1, Q, K) -> (B, Q, K) if possible
                sdpa_mask = sdpa_mask.squeeze(1)
        else:
            # If no mask provided, assume causal unless q_len is 1
            if q_len > 1:
                is_causal_sdpa = True

        # Call SDPA
        attn_output = F.scaled_dot_product_attention(
            query_states,  # (B, H, Q_len, D)
            key_states,  # (B, H, K_len, D)
            value_states,  # (B, H, K_len, D)
            attn_mask=sdpa_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal_sdpa,
        )  # Output: (B, H, Q_len, D)

        # Reshape output: (B, Q_len, H, D) -> (B, Q_len, HiddenSize)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, None, present_key_value  # No attention weights from SDPA


# Option 3: Flash Attention 2
if IS_FLASH_ATTN_AVAILABLE:

    class Qwen2_5_VLFlashAttention2(nn.Module):
        # ... (Copy/Adapt Qwen2_5_VLFlashAttention2 implementation) ...
        # Very similar to the Vision FA2 version, but uses LLM RoPE and causal mask option
        def __init__(self, config: Qwen2_5_VLConfig, layer_idx: Optional[int] = None):
            super().__init__()
            self.config = config
            self.layer_idx = layer_idx
            self.hidden_size = config.hidden_size
            self.num_heads = config.num_attention_heads
            self.head_dim = self.hidden_size // self.num_heads
            self.num_key_value_heads = config.num_key_value_heads
            self.num_key_value_groups = self.num_heads // self.num_key_value_heads
            self.is_causal = True
            self.attention_dropout = config.attention_dropout
            self.rope_scaling_mrope_section = config.rope_scaling.get(
                "mrope_section", self.head_dim // 3
            )

            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.head_dim, bias=True
            )
            self.k_proj = nn.Linear(
                self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
            )
            self.v_proj = nn.Linear(
                self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
            )
            self.o_proj = nn.Linear(
                self.num_heads * self.head_dim, self.hidden_size, bias=False
            )

            # Sliding window setup (if used)
            self.use_sliding_window = config.use_sliding_window
            self.sliding_window = (
                config.sliding_window if self.use_sliding_window else None
            )
            self.max_window_layers = config.max_window_layers

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[
                torch.Tensor
            ] = None,  # FA2 generally ignores mask if causal=True and no padding
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[
                Tuple[torch.Tensor, torch.Tensor]
            ] = None,  # FA2 needs KV cache handling
            output_attentions: bool = False,  # Not supported by FA2
            use_cache: bool = False,
            position_embeddings: Optional[
                Tuple[torch.Tensor, torch.Tensor]
            ] = None,  # (cos, sin)
        ):
            if output_attentions:
                print(
                    "Warning: output_attentions=True is not supported with FlashAttention. Returning None."
                )

            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            # Reshape for FA2: (B, SeqLen, NumHeads/NumKVHeads, HeadDim)
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
            key_states = key_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            )
            value_states = value_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            )

            # Apply RoPE
            if position_embeddings is None:
                raise ValueError("position_embeddings (cos, sin) are required.")
            cos, sin = position_embeddings  # Shape (3, B, S, D)

            # RoPE needs Q/K in (..., S, D) or similar for apply_rotary_emb
            # We have (B, S, H, D). Apply RoPE per head.
            # This needs the multimodal RoPE helper!
            # apply_multimodal_rotary_pos_emb needs Q(B,H,S,D), K(B,KVH,S,D) and cos/sin(3,B,S,D)
            # Let's transpose Q/K for RoPE helper
            query_states_t = query_states.transpose(1, 2)  # (B, H, S, D)
            key_states_t = key_states.transpose(1, 2)  # (B, KVH, S, D)
            query_states_rot, key_states_rot = apply_multimodal_rotary_pos_emb(
                query_states_t,
                key_states_t,
                cos,
                sin,
                self.rope_scaling_mrope_section,
                unsqueeze_dim=1,  # RoPE adds head dim
            )
            # Transpose back: (B, S, H, D) and (B, S, KVH, D)
            query_states = query_states_rot.transpose(1, 2)
            key_states = key_states_rot.transpose(1, 2)

            # KV Cache update for FA2 (handled differently)
            # FA2 often uses a cache object or expects concatenated K/V
            # Let's assume simple concatenation here for compatibility
            if past_key_value is not None:
                past_key, past_value = past_key_value
                # Concatenate along sequence dim (dim 1)
                key_states = torch.cat([past_key, key_states], dim=1)
                value_states = torch.cat([past_value, value_states], dim=1)

            present_key_value = (key_states, value_states) if use_cache else None
            k_len = key_states.shape[1]  # Full key length including cache

            # GQA repeat K/V (needed *after* caching and RoPE for FA2 input)
            # FA2 varlen function expects (TotalSeqLen, NumHeads, HeadDim) input usually.
            # The standard flash_attn_func might work with (B, S, H, D)
            # Let's try flash_attn_func with explicit cache handling.

            # Repeat KV heads
            key_states_gqa = repeat_kv(
                key_states.transpose(1, 2), self.num_key_value_groups
            ).transpose(1, 2)
            value_states_gqa = repeat_kv(
                value_states.transpose(1, 2), self.num_key_value_groups
            ).transpose(1, 2)
            # Shape: (B, K_len, NumHeads, HeadDim)

            # Determine sliding window setting for this layer
            current_sliding_window = None
            if self.use_sliding_window and self.layer_idx < self.max_window_layers:
                current_sliding_window = self.sliding_window

            # Call flash_attn_func
            # Inputs Q, K, V: (B, SeqLen, NumHeads, HeadDim)
            attn_output = flash_attn_func(
                query_states,
                key_states_gqa,
                value_states_gqa,
                dropout_p=self.attention_dropout if self.training else 0.0,
                softmax_scale=None,  # Defaults to 1/sqrt(head_dim)
                causal=True,
                window_size=(current_sliding_window, current_sliding_window)
                if current_sliding_window is not None
                else (-1, -1),
                return_attn_probs=False,  # Cannot get attention weights
            )  # Output: (B, q_len, NumHeads, HeadDim)

            # Reshape output: (B, q_len, HiddenSize)
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

            # Output projection
            attn_output = self.o_proj(attn_output)

            return attn_output, None, present_key_value


# --- LLM Decoder Layer ---
class Qwen2_5_VLDecoderLayer(nn.Module):
    # ... (Copy/Adapt implementation from modeling_qwen2_5_vl.py) ...
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Select Attention Implementation
        attn_implementation = config._attn_implementation
        if attn_implementation == "flash_attention_2" and IS_FLASH_ATTN_AVAILABLE:
            ATTN_CLASS = Qwen2_5_VLFlashAttention2
        elif attn_implementation == "sdpa":
            ATTN_CLASS = Qwen2_5_VLSdpaAttention
        else:  # eager
            ATTN_CLASS = Qwen2_5_VLAttention
            if config.use_sliding_window:
                print(
                    "Warning: Sliding window attention not fully supported in eager mode."
                )

        self.self_attn = ATTN_CLASS(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,  # Used by RoPE indirectly
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # (cos, sin)
    ) -> Tuple[
        torch.FloatTensor,
        Optional[Tuple[torch.FloatTensor, torch.FloatTensor]],
        Optional[Tuple[torch.Tensor]],
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,  # Pass along if needed by attn impl
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
        hidden_states = attn_outputs[0]
        attn_weights = attn_outputs[1]  # Can be None
        present_key_value = attn_outputs[2]  # Can be None

        hidden_states = residual + hidden_states

        # Fully Connected Layer (MLP)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs  # type: ignore


# --- LLM Model ---
class Qwen2_5_VLModel(nn.Module):
    # ... (Copy/Adapt implementation from modeling_qwen2_5_vl.py) ...
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self._attn_implementation = (
            config._attn_implementation
        )  # Store for mask creation

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen2_5_VLDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Instantiate RoPE module here
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)

        # self.gradient_checkpointing = False # Control externally

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Helper to create causal mask
    def _prepare_4d_causal_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_shape: Tuple[int, int],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
    ):
        """Creates a causal 4D mask based on input mask and past length."""
        bsz, tgt_len = input_shape
        # 4d mask is used for the layers
        if tgt_len > 1:
            mask = torch.full(
                (tgt_len, tgt_len),
                torch.finfo(inputs_embeds.dtype).min,
                device=inputs_embeds.device,
            )
            mask_cond = torch.arange(mask.size(-1), device=inputs_embeds.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(inputs_embeds.dtype)
        else:
            mask = None  # No mask needed for single token generation

        if past_key_values_length > 0:
            # If we have cached keys, the mask needs to accommodate them
            # Mask shape should be (bsz, 1, q_len, k_len) where k_len = past_len + q_len
            if mask is not None:  # q_len > 1
                mask = torch.cat(
                    [
                        torch.zeros(
                            tgt_len,
                            past_key_values_length,
                            dtype=mask.dtype,
                            device=mask.device,
                        ),
                        mask,
                    ],
                    dim=-1,
                )
            else:  # q_len = 1
                # Causal mask for single query token against all keys
                mask = torch.zeros(
                    1,
                    1,
                    1,
                    past_key_values_length + 1,
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device,
                )

        # Add batch dimension and expand mask if necessary
        if mask is not None:
            causal_mask = mask[None, None, :, :].expand(
                bsz, 1, tgt_len, -1
            )  # Use K length from mask itself
        else:
            causal_mask = None

        # Apply input attention mask (for padding)
        if attention_mask is not None:
            # Ensure attention_mask covers the full key length
            if attention_mask.dim() == 2:  # (bsz, full_seq_len)
                # Expand to (bsz, 1, q_len, k_len)
                expanded_mask = attention_mask[:, None, None, :].expand(
                    bsz, 1, tgt_len, -1
                )  # Use K length from attention_mask
            elif attention_mask.dim() == 4:  # Already (bsz, 1, q_len, k_len)
                expanded_mask = attention_mask
            else:
                raise ValueError("Invalid attention_mask shape")

            # Combine causal mask and padding mask
            if causal_mask is None:
                # If only padding mask (e.g., q_len=1, past=0, but padding exists)
                # SDPA/FA2 might handle padding differently. Eager needs float mask.
                # Create a float mask where padding is large negative.
                padding_mask_float = (
                    expanded_mask.eq(0.0).to(inputs_embeds.dtype)
                    * torch.finfo(inputs_embeds.dtype).min
                )
                causal_mask = padding_mask_float
            else:
                # Merge: Where expanded_mask is 0 (padding), set causal_mask to large negative
                pad_indices = expanded_mask == 0
                # Ensure shapes match before masking. Causal mask K dim might be larger if no input mask was provided initially.
                k_len_causal = causal_mask.shape[-1]
                k_len_padding = pad_indices.shape[-1]
                if k_len_causal != k_len_padding:
                    # This case needs careful handling - likely means inconsistent input mask length
                    # For now, assume they match or the padding mask dictates the effective K length
                    if k_len_padding < k_len_causal:
                        # Pad the padding mask? Or trim the causal mask? Trim is safer.
                        causal_mask = causal_mask[..., :k_len_padding]
                    else:
                        # Should not happen if causal mask generated correctly based on past_len
                        raise ValueError(
                            "Padding mask longer than causal mask K length"
                        )

                causal_mask = causal_mask.masked_fill(
                    pad_indices[:, :, :tgt_len, :], torch.finfo(inputs_embeds.dtype).min
                )

        # For SDPA/Flash Attention, mask should be boolean (True means mask out) or None
        if self._attn_implementation in ["sdpa", "flash_attention_2"]:
            if causal_mask is not None:
                # Convert float mask to boolean
                causal_mask = causal_mask < -1.0  # True where masked
            # Further optimization: if mask is purely causal and no padding, return None and set is_causal=True in SDPA/FA2 call
            # This requires checking if attention_mask was None and tgt_len > 1

        return causal_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # Input mask (B, S)
        position_ids: Optional[torch.LongTensor] = None,  # M-RoPE pos ids (3, B, S)
        past_key_values: Optional[
            List[Tuple[torch.Tensor, torch.Tensor]]
        ] = None,  # List per layer
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # No cache_position, derive from past_key_values
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions if output_attentions is not None else False
        )  # Default
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )  # Default
        use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )  # Default from config
        return_dict = return_dict if return_dict is not None else False  # Default

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both.")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be specified.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Retrieve kv cache length
        past_key_values_length = 0
        if past_key_values is not None and len(past_key_values) > 0:
            # Shape: B, H, S, D -> index 2 is sequence length
            past_key_values_length = past_key_values[0][0].shape[2]

        # Create attention mask
        # Input shape required for mask creation
        input_shape = inputs_embeds.size()[:-1]  # (bsz, q_len)
        combined_attention_mask = self._prepare_4d_causal_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # Calculate RoPE embeddings for the current sequence part
        if position_ids is None:
            # Create standard sequential position ids if not provided (e.g., text-only)
            # This needs adjustment for M-RoPE if vision parts are present and position_ids were not pre-calculated
            seq_length = inputs_embeds.shape[1]
            device = inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            # Expand for M-RoPE (assuming text-only here, needs fixing for multimodal)
            position_ids = position_ids.unsqueeze(0).expand(
                3, inputs_embeds.shape[0], -1
            )
            print(
                "Warning: Creating default sequential position_ids. M-RoPE might be incorrect if vision tokens present."
            )

        # Get cos/sin embeddings using the rotary embedding module
        # Need to slice position_ids if q_len < total sequence length (relevant during generation)
        current_position_ids = position_ids[
            ...,
            past_key_values_length : past_key_values_length + inputs_embeds.shape[1],
        ]
        position_embeddings = self.rotary_emb(
            inputs_embeds, current_position_ids
        )  # (cos, sin) tuple

        # Decoder layers
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = [] if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            # Add gradient checkpointing control if needed

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=combined_attention_mask,
                position_ids=None,  # Not directly needed by layer if position_embeddings is passed
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache.append(
                    layer_outputs[-1]
                )  # Last element is kv cache tuple

            if output_attentions:
                all_self_attns += (layer_outputs[1],)  # Second element is attn weights

        hidden_states = self.norm(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Convert cache list to tuple
        next_cache = tuple(next_decoder_cache) if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# --- Full Model with LM Head ---
class Qwen2_5_VLForConditionalGeneration(nn.Module):
    # ... (Adapt implementation from modeling_qwen2_5_vl.py) ...
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__()
        self.config = config  # Store combined config
        # Instantiate Vision Transformer using vision_config
        self.visual = Qwen2_5_VisionTransformer(config.vision_config)
        # Instantiate LLM using llm_config (accessible via top-level config)
        self.model = Qwen2_5_VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # M-RoPE related attributes from HF model (might be needed)
        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id
        self.vision_start_token_id = config.vision_start_token_id
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.tokens_per_second = config.vision_config.tokens_per_second
        # self.rope_deltas = None # Cache for rope calculation delta during generation

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # --- RoPE Index Calculation (Crucial for Multimodal Input) ---
    def get_rope_index(
        self,
        input_ids: torch.LongTensor,  # (B, S)
        image_grid_thw: Optional[
            torch.LongTensor
        ] = None,  # (NumImages, 3) - Patches T, H, W
        video_grid_thw: Optional[
            torch.LongTensor
        ] = None,  # (NumVideos, 3) - Patches T, H, W
        second_per_grid_ts: Optional[torch.Tensor] = None,  # (NumVideos,)
        attention_mask: Optional[torch.Tensor] = None,  # (B, S)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ... (Copy/Adapt implementation from Qwen2_5_VLForConditionalGeneration.get_rope_index) ...
        # This function is complex and calculates the 3D position IDs based on
        # the sequence structure (text vs image/video tokens) and the grid dimensions.
        # Returns: position_ids (3, B, S), mrope_position_deltas (B, 1)

        # --- Simplified version for text-only or if multimodal part is complex ---
        if image_grid_thw is None and video_grid_thw is None:
            # Text-only case: standard sequential position IDs
            bsz, seq_len = input_ids.shape
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            # Create position IDs based on attention mask (handle padding)
            position_ids_1d = attention_mask.long().cumsum(-1) - 1
            position_ids_1d.masked_fill_(
                attention_mask == 0, 0
            )  # Use 0 for padding? Or a specific index? HF uses 1, check. Let's use 0.

            # Expand to 3D for M-RoPE structure (all dimensions are the same for text)
            position_ids = (
                position_ids_1d.unsqueeze(0).expand(3, bsz, seq_len).clone()
            )  # Need clone?

            # Calculate delta (difference between max pos ID and seq length)
            # Max position ID is max(cumsum-1) where mask==1
            max_pos_ids = position_ids_1d.max(dim=-1, keepdim=True)[0]
            mrope_position_deltas = (
                max_pos_ids + 1 - seq_len
            )  # Should be 0 if no padding left/middle
            mrope_position_deltas = mrope_position_deltas.unsqueeze(-1)  # (B, 1)

            return position_ids.to(input_ids.device), mrope_position_deltas.to(
                input_ids.device
            )
        else:
            # --- Full Multimodal RoPE Index Calculation ---
            # This requires careful porting of the HF logic.
            # It iterates through the batch, finds vision tokens, calculates
            # vision pos IDs based on grid_thw/second_per_grid_ts, and
            # calculates text pos IDs sequentially between/after vision parts.
            # print("Warning: Full multimodal get_rope_index logic not implemented in this skeleton.")
            # --- Placeholder: Copy the exact HF code here ---
            spatial_merge_size = self.spatial_merge_size
            image_token_id = self.image_token_id
            video_token_id = self.video_token_id
            vision_start_token_id = self.vision_start_token_id
            mrope_position_deltas_list = []  # Changed name to avoid conflict
            total_input_ids = input_ids
            bsz, seq_len_total = total_input_ids.shape

            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)

            # Initialize position_ids tensor
            final_position_ids = torch.zeros(
                3, bsz, seq_len_total, dtype=torch.long, device=input_ids.device
            )

            image_idx_global, video_idx_global = 0, 0

            for i in range(bsz):  # Process each sequence in the batch
                current_input_ids = total_input_ids[i]
                current_attn_mask = attention_mask[i]
                valid_input_ids = current_input_ids[current_attn_mask == 1]
                seq_len_valid = valid_input_ids.shape[0]

                # Find vision markers in the valid sequence part
                vision_start_indices = torch.where(
                    valid_input_ids == vision_start_token_id
                )[0]
                # Identify token type *after* vision start token
                vision_tokens = (
                    valid_input_ids[vision_start_indices + 1]
                    if len(vision_start_indices) > 0
                    else torch.tensor([], dtype=torch.long)
                )

                num_images_sample = (vision_tokens == image_token_id).sum().item()
                num_videos_sample = (vision_tokens == video_token_id).sum().item()

                input_tokens_list = valid_input_ids.tolist()
                llm_pos_ids_list_sample: list = []  # Stores pos ID segments for this sample
                current_pos = 0  # Tracks the current maximum position ID used
                text_start_idx = (
                    0  # Start index of current text segment in input_tokens_list
                )
                image_idx_sample, video_idx_sample = 0, 0

                # Find indices of image/video placeholders in the original list
                placeholder_indices = []
                try:
                    start_search = 0
                    while True:
                        idx = input_tokens_list.index(image_token_id, start_search)
                        placeholder_indices.append({"type": "image", "index": idx})
                        start_search = idx + 1
                except ValueError:
                    pass  # No more image tokens
                try:
                    start_search = 0
                    while True:
                        idx = input_tokens_list.index(video_token_id, start_search)
                        placeholder_indices.append({"type": "video", "index": idx})
                        start_search = idx + 1
                except ValueError:
                    pass  # No more video tokens

                # Sort placeholders by their index
                placeholder_indices.sort(key=lambda x: x["index"])

                for placeholder in placeholder_indices:
                    placeholder_idx = placeholder["index"]
                    placeholder_type = placeholder["type"]

                    # 1. Add text segment before this placeholder
                    text_len = placeholder_idx - text_start_idx
                    if text_len > 0:
                        text_pos_ids = torch.arange(
                            current_pos, current_pos + text_len, device=input_ids.device
                        )
                        llm_pos_ids_list_sample.append(
                            text_pos_ids.unsqueeze(0).expand(3, -1)
                        )
                        current_pos += text_len

                    # 2. Add vision segment
                    if placeholder_type == "image":
                        if (
                            image_idx_sample >= num_images_sample
                            or image_grid_thw is None
                        ):
                            raise ValueError(
                                f"Mismatch in image tokens/grid info for sample {i}"
                            )
                        t, h, w = image_grid_thw[image_idx_global + image_idx_sample]
                        current_second_per_grid_t = 0  # Images have T=1 conceptually
                        image_idx_sample += 1
                    else:  # video
                        if (
                            video_idx_sample >= num_videos_sample
                            or video_grid_thw is None
                        ):
                            raise ValueError(
                                f"Mismatch in video tokens/grid info for sample {i}"
                            )
                        t, h, w = video_grid_thw[video_idx_global + video_idx_sample]
                        current_second_per_grid_t = (
                            second_per_grid_ts[video_idx_global + video_idx_sample]
                            if second_per_grid_ts is not None
                            else 1.0
                        )
                        video_idx_sample += 1

                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    num_vision_tokens = llm_grid_t * llm_grid_h * llm_grid_w

                    # Calculate vision position IDs
                    # Temporal IDs
                    time_indices = torch.arange(
                        llm_grid_t, device=input_ids.device
                    ).view(-1, 1, 1)
                    time_pos_ids = (
                        time_indices
                        * current_second_per_grid_t
                        * self.tokens_per_second
                    ).long()
                    t_pos = time_pos_ids.expand(-1, llm_grid_h, llm_grid_w).flatten()

                    # Height IDs
                    h_indices = torch.arange(llm_grid_h, device=input_ids.device).view(
                        1, -1, 1
                    )
                    h_pos = h_indices.expand(llm_grid_t, -1, llm_grid_w).flatten()

                    # Width IDs
                    w_indices = torch.arange(llm_grid_w, device=input_ids.device).view(
                        1, 1, -1
                    )
                    w_pos = w_indices.expand(llm_grid_t, llm_grid_h, -1).flatten()

                    # Combine and shift by current_pos
                    vision_pos_ids = (
                        torch.stack([t_pos, h_pos, w_pos], dim=0) + current_pos
                    )
                    llm_pos_ids_list_sample.append(vision_pos_ids)

                    # Update current_pos to the maximum ID used by the vision block
                    current_pos = vision_pos_ids.max() + 1
                    # Update text_start_idx to point after the placeholder token
                    text_start_idx = (
                        placeholder_idx + 1
                    )  # Move past the single placeholder token

                # 3. Add remaining text segment after the last placeholder
                text_len = len(input_tokens_list) - text_start_idx
                if text_len > 0:
                    text_pos_ids = torch.arange(
                        current_pos, current_pos + text_len, device=input_ids.device
                    )
                    llm_pos_ids_list_sample.append(
                        text_pos_ids.unsqueeze(0).expand(3, -1)
                    )
                    current_pos += text_len

                # Concatenate all segments for the current sample
                if not llm_pos_ids_list_sample:
                    # Handle case with no vision tokens (should not happen if called correctly)
                    sample_positions = (
                        torch.arange(seq_len_valid, device=input_ids.device)
                        .unsqueeze(0)
                        .expand(3, -1)
                    )
                    max_pos_sample = seq_len_valid - 1 if seq_len_valid > 0 else -1
                else:
                    sample_positions = torch.cat(
                        llm_pos_ids_list_sample, dim=1
                    )  # (3, seq_len_valid)
                    max_pos_sample = (
                        sample_positions.max().item()
                        if sample_positions.numel() > 0
                        else -1
                    )

                # Assign to the final tensor using the attention mask
                final_position_ids[:, i, current_attn_mask == 1] = sample_positions.to(
                    final_position_ids.device
                )

                # Calculate delta for this sample
                delta = (
                    max_pos_sample + 1 - seq_len_valid
                )  # Should be seq_len_total? No, valid len.
                mrope_position_deltas_list.append(delta)

                # Update global indices for grid_thw access
                image_idx_global += num_images_sample
                video_idx_global += num_videos_sample

            mrope_position_deltas = torch.tensor(
                mrope_position_deltas_list, device=input_ids.device
            ).view(bsz, 1)
            return final_position_ids, mrope_position_deltas
            # --- End Full Multimodal ---

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # (B, S)
        position_ids: Optional[
            torch.LongTensor
        ] = None,  # If provided, overrides get_rope_index
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,  # (NumImages, C, T, H, W)
        pixel_values_videos: Optional[
            torch.FloatTensor
        ] = None,  # (NumVideos, C, T, H, W)
        image_grid_thw: Optional[
            torch.LongTensor
        ] = None,  # (NumImages, 3) - Patches T, H, W
        video_grid_thw: Optional[
            torch.LongTensor
        ] = None,  # (NumVideos, 3) - Patches T, H, W
        second_per_grid_ts: Optional[torch.Tensor] = None,  # (NumVideos,)
        # No rope_deltas input, calculated internally if needed
        # No cache_position, handled via past_key_values
    ) -> Union[
        Tuple, Qwen2_5_VLCausalLMOutputWithPast
    ]:  # Use HF output class for structure
        output_attentions = (
            output_attentions if output_attentions is not None else False
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else False

        # 1. Handle Inputs: Get input_embeds from input_ids if needed
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)  # (B, S, D)

        # 2. Process Visual Inputs and Embed Them
        image_embeds_list = []
        video_embeds_list = []
        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("image_grid_thw must be provided with pixel_values")
            # Assume pixel_values is (TotalImagePatches, C, T_patch, H_patch, W_patch) ???
            # Or (NumImages, C, T, H, W) -> Needs patching first.
            # Let's assume Qwen2_5_VisionTransformer handles (NumImages, C, T, H, W)
            # print(f"Forward: pixel_values shape: {pixel_values.shape}")
            # print(f"Forward: image_grid_thw: {image_grid_thw}")
            image_embeds = self.visual(
                pixel_values.to(self.visual.patch_embed.proj.weight.dtype),
                grid_thw=image_grid_thw,
            )  # (TotalImageMergedPatches, D)
            image_embeds_list.append(image_embeds)

        if pixel_values_videos is not None:
            if video_grid_thw is None:
                raise ValueError(
                    "video_grid_thw must be provided with pixel_values_videos"
                )
            video_embeds = self.visual(
                pixel_values_videos.to(self.visual.patch_embed.proj.weight.dtype),
                grid_thw=video_grid_thw,
            )  # (TotalVideoMergedPatches, D)
            video_embeds_list.append(video_embeds)

        # Concatenate visual embeddings if both present (ensure order matches input_ids)
        # This assumes a single batch item for simplicity in merging. Batching needs careful index handling.
        # The merging logic below assumes batch size 1 or requires splitting embeds by batch item.
        vision_embeds = (
            torch.cat(image_embeds_list + video_embeds_list, dim=0)
            if image_embeds_list or video_embeds_list
            else None
        )

        # 3. Insert Visual Embeddings into Text Embeddings
        if vision_embeds is not None:
            # Find placeholder token indices in input_ids
            # This is tricky with batching. Let's assume B=1 for now.
            if input_ids.shape[0] != 1:
                raise NotImplementedError(
                    "Batch size > 1 requires more complex vision embedding insertion."
                )

            image_mask = input_ids == self.image_token_id
            video_mask = input_ids == self.video_token_id
            vision_mask = image_mask | video_mask  # Locations of placeholders

            num_vision_placeholders = vision_mask.sum().item()
            num_vision_embeds = vision_embeds.shape[0]

            if num_vision_placeholders != num_vision_embeds:
                # This can happen if placeholder expansion was done beforehand by processor
                # We need to handle the case where one placeholder maps to multiple embeds
                # Let's assume the input_ids ALREADY have expanded placeholders.
                # Example: ... T T T <img_tok> <img_tok> <img_tok> T T ...
                if num_vision_embeds % num_vision_placeholders != 0:
                    raise ValueError(
                        f"Number of vision embeddings ({num_vision_embeds}) "
                        f"is not compatible with placeholder count ({num_vision_placeholders})."
                    )
                # If compatible, scatter the embeddings
                inputs_embeds = inputs_embeds.masked_scatter(
                    vision_mask.unsqueeze(-1).expand_as(inputs_embeds),
                    vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype),
                )

            else:
                # Simple 1-to-1 replacement (should not happen with Qwen-VL?)
                inputs_embeds[vision_mask.squeeze(0)] = vision_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )

        # 4. Calculate M-RoPE Position IDs if not provided
        current_rope_deltas = None  # Store for potential use in generation loop
        if position_ids is None:
            # We need the original input_ids structure for get_rope_index
            if input_ids is None:
                raise ValueError(
                    "input_ids are required to calculate M-RoPE position IDs when position_ids is None."
                )
            # Calculate position IDs based on the potentially modified input_ids structure
            position_ids, mrope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            current_rope_deltas = mrope_deltas  # Cache for generation

        # 5. Pass combined embeddings and position IDs to the LLM
        outputs = self.model(
            input_ids=None,  # Pass embeds instead
            attention_mask=attention_mask,  # Pass original B, S mask
            position_ids=position_ids,  # Pass calculated M-RoPE IDs
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,  # Use internal return_dict
        )

        # 6. Get hidden states from LLM output
        hidden_states = (
            outputs[0]
            if not self.model.config.use_return_dict
            else outputs.last_hidden_state
        )

        # 7. Apply LM Head
        logits = self.lm_head(hidden_states)

        # 8. Calculate Loss if labels provided
        loss = None
        if labels is not None:
            logits = logits.float()  # Upcast for loss calculation
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Move labels to correct device
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # 9. Format Output
        if not return_dict:
            output = (logits,) + outputs[1:]  # Add logits to model output tuple
            return (loss,) + output if loss is not None else output

        # Use the HF dataclass structure for consistency
        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=current_rope_deltas,  # Include calculated deltas
        )

    # Add generate method manually here
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        eos_token_id: Optional[int] = None,
        # Add other generation params: temp, top_k, top_p, do_sample etc.
        **kwargs,  # Absorb other potential args
    ):
        # Simplified greedy search implementation

        # Get EOS token ID from config if not provided
        if eos_token_id is None:
            eos_token_id = self.config.pad_token_id  # Use pad/eos token

        # Prepare initial inputs (handle embeddings and vision)
        # This part mirrors the start of the forward pass

        # Get text embeddings
        inputs_embeds = self.model.embed_tokens(input_ids)  # (B, S_init, D)
        bsz, seq_len_init = input_ids.shape

        # Process and insert vision embeddings
        image_embeds_list = []
        video_embeds_list = []
        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("image_grid_thw needed")
            image_embeds = self.visual(
                pixel_values.to(self.visual.patch_embed.proj.weight.dtype),
                grid_thw=image_grid_thw,
            )
            image_embeds_list.append(image_embeds)
        if pixel_values_videos is not None:
            if video_grid_thw is None:
                raise ValueError("video_grid_thw needed")
            video_embeds = self.visual(
                pixel_values_videos.to(self.visual.patch_embed.proj.weight.dtype),
                grid_thw=video_grid_thw,
            )
            video_embeds_list.append(video_embeds)

        vision_embeds = (
            torch.cat(image_embeds_list + video_embeds_list, dim=0)
            if image_embeds_list or video_embeds_list
            else None
        )

        if vision_embeds is not None:
            if bsz != 1:
                raise NotImplementedError(
                    "Batch size > 1 in generation not fully supported here."
                )
            image_mask = input_ids == self.image_token_id
            video_mask = input_ids == self.video_token_id
            vision_mask = image_mask | video_mask
            inputs_embeds = inputs_embeds.masked_scatter(
                vision_mask.unsqueeze(-1).expand_as(inputs_embeds),
                vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype),
            )

        # Calculate initial M-RoPE position IDs
        position_ids, rope_deltas = self.get_rope_index(
            input_ids,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts,
            attention_mask,
        )
        # self.rope_deltas = rope_deltas # Store if needed for subsequent calls?

        past_key_values = None
        generated_ids = input_ids.clone()  # Start with input_ids

        for _ in range(max_new_tokens):
            # Prepare inputs for this step
            current_input_ids = generated_ids[:, -1:]  # Only the last token ID
            current_seq_len = generated_ids.shape[1]

            # Get embeddings for the new token(s)
            # If first step, use precomputed inputs_embeds, else embed last token
            if past_key_values is None:  # First step
                model_inputs_embeds = inputs_embeds
                current_position_ids = position_ids  # Full pos ids
            else:  # Subsequent steps
                model_inputs_embeds = self.model.embed_tokens(
                    current_input_ids
                )  # (B, 1, D)
                # Calculate position ID for the new token
                # Need to use rope_deltas and past length
                past_len = past_key_values[0][0].shape[2]
                new_pos = (
                    torch.tensor([[past_len]], device=input_ids.device) + rope_deltas
                ).long()  # (B, 1)
                # Expand to 3D
                current_position_ids = new_pos.unsqueeze(0).expand(
                    3, bsz, 1
                )  # (3, B, 1)

            # Forward pass through the LLM part
            outputs = self.model(
                input_ids=None,
                attention_mask=attention_mask,  # Needs to be updated/extended!
                position_ids=current_position_ids,  # Pass current step's pos IDs
                past_key_values=past_key_values,
                inputs_embeds=model_inputs_embeds,
                use_cache=True,
                return_dict=True,
            )

            # Get logits for the last token
            next_token_logits = self.lm_head(
                outputs.last_hidden_state[:, -1, :]
            )  # (B, V)

            # Greedy decoding
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(
                -1
            )  # (B, 1)

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

            # Update KV cache
            past_key_values = outputs.past_key_values

            # Update attention mask (append a 1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token_id)], dim=1
            )

            # Check for EOS token
            if eos_token_id is not None and (next_token_id == eos_token_id).all():
                break

        return generated_ids


# ============================================================================
# 3. Weight Loading (Manual)
# ============================================================================
def load_weights(model, weight_path):
    # weight_path: Path to .safetensors file(s) or .bin file(s)
    # Use safetensors library or torch.load for .bin
    try:
        from safetensors import safe_open

        tensors = {}
        # Handle multi-file checkpoints if necessary
        if os.path.isdir(weight_path):
            filenames = [
                os.path.join(weight_path, fn)
                for fn in os.listdir(weight_path)
                if fn.endswith(".safetensors")
            ]
        elif weight_path.endswith(".safetensors"):
            filenames = [weight_path]
        else:
            raise ValueError(
                "Provide path to .safetensors file or directory containing them."
            )

        for filename in filenames:
            with safe_open(
                filename, framework="pt", device="cpu"
            ) as f:  # Load to CPU first
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
        model.load_state_dict(tensors, strict=True)  # Use strict=True initially

    except ImportError:
        print(
            "sagetensors not found. Trying torch.load (might be slow/memory intensive)"
        )
        if os.path.isdir(weight_path):
            # Find pytorch_model.bin etc.
            bin_files = [
                os.path.join(weight_path, fn)
                for fn in os.listdir(weight_path)
                if fn.endswith(".bin")
            ]
            if not bin_files:
                raise ValueError(".bin files not found")
            state_dict = {}
            for bf in bin_files:
                state_dict.update(torch.load(bf, map_location="cpu"))
        elif weight_path.endswith(".bin"):
            state_dict = torch.load(weight_path, map_location="cpu")
        else:
            raise ValueError("Provide path to .bin file or directory containing them.")

        model.load_state_dict(state_dict, strict=True)

    print("Weights loaded successfully.")


# ============================================================================
# 4. Tokenizer and Image Processor (Manual Implementation or HF)
# ============================================================================

# --- Tokenizer ---
# Using HF Tokenizer for convenience
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
# Special tokens needed later
pad_token_id = (
    tokenizer.pad_token_id
    if tokenizer.pad_token_id is not None
    else tokenizer.eos_token_id
)
image_token = "<|image_pad|>"  # As defined in HF processor
image_token_id = tokenizer.convert_tokens_to_ids(image_token)
# video_token = "<|video_pad|>"
# video_token_id = tokenizer.convert_tokens_to_ids(video_token)


# --- Image Processing ---
# This needs to replicate the *exact* steps of Qwen2_5_VLImageProcessor
# Including resizing, normalization, padding, patching, and grid calculation.
# This is highly model-specific and complex.
# Simplified version for demonstration:
from torchvision import transforms


def preprocess_image(
    image_input, target_size=448, patch_size=14, temporal_patch_size=2
):
    # Load image if URL/path
    if isinstance(image_input, str):
        if image_input.startswith("http"):
            response = requests.get(image_input, stream=True)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise TypeError("Unsupported image input type")

    # Basic transforms (Likely need Qwen specific resizing/cropping)
    # Common practice: Resize shorter edge, center crop. Check Qwen's approach.
    # Assuming simple resize and center crop for now
    transform = transforms.Compose(
        [
            transforms.Resize(
                target_size
            ),  # Resize shortest edge? Check HF processor config
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.48145466,
                    0.4578275,
                    0.40821073,
                ],  # CLIP defaults? Check Qwen's values
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    processed_image = transform(image)  # Shape (C, H, W)

    # Add Temporal dimension (T=1 for images, T_video for videos)
    # Qwen ViT uses Conv3D, expects T dimension.
    # How is T handled for single images? Often T=temporal_patch_size ?
    # Let's assume T=temporal_patch_size for images based on Conv3D kernel
    processed_image = processed_image.unsqueeze(1).repeat(
        1, temporal_patch_size, 1, 1
    )  # (C, T_patch, H, W)

    # Calculate grid_thw (in patch units)
    # This depends on the *final* size fed to the patch embedder.
    c, t, h, w = processed_image.shape
    grid_t = t // temporal_patch_size  # Should be 1 for our repeated image
    grid_h = h // patch_size
    grid_w = w // patch_size
    grid_thw = torch.tensor(
        [[grid_t, grid_h, grid_w]], dtype=torch.long
    )  # (NumImages=1, 3)

    # Reshape/Prepare for patch_embed?
    # Qwen2_5_VisionTransformer expects (B, C, T, H, W) input
    # Add batch dimension
    pixel_values = processed_image.unsqueeze(0)  # (1, C, T_patch, H, W)

    return pixel_values, grid_thw


# --- process_vision_info (Manual version) ---
def process_vision_info(messages):
    # Extracts image/video data from the message structure
    image_inputs_data = []
    # video_inputs_data = [] # Add if handling video
    for message in messages:
        if message["role"] == "user":
            for item in message["content"]:
                if item["type"] == "image":
                    image_inputs_data.append(item["image"])  # URL or PIL image
                # Add video handling if needed
    return image_inputs_data, None  # Return image_data, video_data


# --- apply_chat_template (Manual version) ---
def apply_chat_template(messages, tokenizer, add_generation_prompt=True):
    # Replicate the specific chat template logic for Qwen2-VL
    # Example for a simple template (check Qwen's actual template)
    prompt = ""
    for message in messages:
        role = message["role"]
        content = ""
        for item in message["content"]:
            if item["type"] == "text":
                content += item["text"]
            elif item["type"] == "image":
                content += f"\n{image_token}\n"  # Use the placeholder token
            # Add video handling
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    if add_generation_prompt:
        prompt += "<|im_start|>assistant\n"
    return prompt


# ============================================================================
# 5. Main Execution Logic
# ============================================================================

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)  # or float32

# --- Instantiate Model ---
print("Instantiating model...")
model = Qwen2_5_VLForConditionalGeneration(config)
model.eval()  # Set to evaluation mode

# --- Load Weights ---
# Set path to your downloaded weights (directory or .safetensors/.bin file)
WEIGHTS_PATH = "path/to/your/Qwen2.5-VL-3B-Instruct/weights"  # CHANGE THIS
print(f"Loading weights from {WEIGHTS_PATH}...")
# load_weights(model, WEIGHTS_PATH) # Uncomment when path is set
print("Moving model to device...")
model.to(device=device, dtype=dtype)
print("Model ready.")

# --- Prepare Inputs ---
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Process vision info
image_data, _ = process_vision_info(messages)
if not image_data:
    raise ValueError("No image found in messages")
image_url = image_data[0]  # Assuming one image

# Preprocess image
# Use vision config details for target size etc.
pixel_values, image_grid_thw = preprocess_image(
    image_url,
    target_size=VISION_CONFIG["image_size"],
    patch_size=VISION_CONFIG["patch_size"],
    temporal_patch_size=VISION_CONFIG["temporal_patch_size"],
)
pixel_values = pixel_values.to(device=device, dtype=dtype)
image_grid_thw = image_grid_thw.to(device=device)

# Apply chat template
text = apply_chat_template(messages, tokenizer, add_generation_prompt=True)

# Tokenize text
# Important: Replace placeholder *after* initial tokenization if needed,
# or ensure tokenizer handles the image token correctly.
# The HF processor replaces *after* templating but *before* full tokenization call.
# Let's replicate that: Replace in the string first.

# Calculate number of image tokens needed based on grid
merge_size = config.vision_config.spatial_merge_size
num_image_tokens = image_grid_thw[0].prod().item() // (merge_size**2)
expanded_image_token_str = image_token * num_image_tokens

# Replace single placeholder with multiple
text_with_expanded_placeholders = text.replace(
    image_token, expanded_image_token_str, 1
)  # Replace only first instance

# Tokenize the final string
# Add padding? Generation usually doesn't need left padding if KV cache is used correctly.
# HF code uses padding=True. Let's try without first for manual gen.
# But the processor call uses padding=True, return_tensors="pt". Let's match that.
inputs = tokenizer(
    text_with_expanded_placeholders,
    return_tensors="pt",
    padding=True,  # Pad to max length in batch (here, batch size 1)
    truncation=True,
    max_length=2048,  # Set a reasonable max length
)

input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# --- Manual Generation ---
print("Generating...")
# Use the manually implemented generate method
with torch.no_grad():
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        # video inputs would go here if used
        max_new_tokens=128,
        eos_token_id=tokenizer.eos_token_id,
    )

# --- Decode Output ---
# Trim the input tokens from the generated sequence
generated_ids_trimmed = generated_ids[:, input_ids.shape[1] :]

# Decode
output_text = tokenizer.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,  # Match HF example
)

print("\nGenerated Text:")
print(output_text[0])  # Assuming batch size 1
