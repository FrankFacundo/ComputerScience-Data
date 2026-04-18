"""Basic building blocks: RMSNorm variants and SwiGLU MLP."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


ACT2FN = {
    "silu": F.silu,
    "gelu": F.gelu,
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "relu": F.relu,
}


class RMSNorm(nn.Module):
    """RMSNorm used in the Qwen3.5 text stack.

    Stored weight is zero-initialized; the effective scale is `1 + weight`
    (matches the reference implementation).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        out = x32 * (1.0 + self.weight.float())
        return out.to(in_dtype)


class RMSNormGated(nn.Module):
    """RMSNorm with a multiplicative SiLU-gate (used inside GatedDeltaNet).

    Weight is 1-initialized here (unlike `RMSNorm`) — this matches the
    `Qwen3_5RMSNormGated` in transformers.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.float()
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.variance_epsilon)
        x_scaled = (self.weight * x32.to(in_dtype))
        out = x_scaled.float() * F.silu(gate.float())
        return out.to(in_dtype)


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
