"""Building blocks shared across Qwen3.6: RMSNorm variants and SwiGLU MLP.

The MoE block (see :mod:`moe`) reuses :class:`SwiGLUMLP` for its **shared
expert**, and the routed experts pack the same gate / up / down trio per
expert into stacked tensors.

See qwen3_5_torch/layers.py for math + references; the formulas are
identical.
"""

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
    r"""Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    .. math::
        \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \varepsilon}}
        \;\odot\; (1 + w)

    The Qwen3.x convention stores the weight as a *delta* from the identity
    scale and zero-initialises it. Variance is accumulated in fp32 to avoid
    fp16/bf16 overflow.
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
    r"""RMSNorm with a multiplicative SiLU gate (used inside GatedDeltaNet).

    .. math::
        y = (w \odot \text{RMSNorm}(x)) \;\odot\; \text{SiLU}(z)

    Weight is **1-initialised** (vs zero-init for plain RMSNorm) — the
    convention used by ``Qwen3_5RMSNormGated``.
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
    r"""SwiGLU feed-forward block.

    .. math::
        \text{SwiGLU}(x) = W_{\text{down}}\big(\text{SiLU}(W_{\text{gate}}x) \odot W_{\text{up}}x\big)
    """

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
