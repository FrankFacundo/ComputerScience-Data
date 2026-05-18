"""Basic building blocks: RMSNorm variants and SwiGLU MLP.

References
----------
RMSNorm
    Zhang & Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019.
    https://arxiv.org/abs/1910.07467

SwiGLU
    Shazeer, "GLU Variants Improve Transformer", 2020.
    https://arxiv.org/abs/2002.05202

SiLU / Swish activation
    Ramachandran, Zoph & Le, "Searching for Activation Functions", 2017.
    https://arxiv.org/abs/1710.05941

GELU
    Hendrycks & Gimpel, "Gaussian Error Linear Units", 2016.
    https://arxiv.org/abs/1606.08415
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Activation function registry.
#
#   silu(x)             = x * sigmoid(x)                              (Ramachandran+ 2017)
#   gelu(x)             = x * Phi(x)  ≈  0.5 x (1 + erf(x / sqrt(2))) (Hendrycks+ 2016)
#   gelu_pytorch_tanh(x)= 0.5 x (1 + tanh( sqrt(2/pi) (x + 0.044715 x^3) ))
#   relu(x)             = max(0, x)
#
# Qwen3.5 text side uses SiLU inside the MLP (SwiGLU). The vision tower uses
# the tanh-approximation of GELU (matches SigLIP / Qwen-VL vision).
# --------------------------------------------------------------------------

ACT2FN = {
    "silu": F.silu,
    "gelu": F.gelu,
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "relu": F.relu,
}


class RMSNorm(nn.Module):
    r"""Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Math
    ----
    Given an input :math:`x \in \mathbb{R}^{d}`:

    .. math::
        \text{RMS}(x) = \sqrt{\tfrac{1}{d}\sum_{i=1}^{d} x_i^2 \;+\; \varepsilon}

    .. math::
        \text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot (1 + w)

    where :math:`w \in \mathbb{R}^d` is a learnable per-channel weight and
    :math:`\odot` is elementwise multiplication. Unlike LayerNorm, no mean
    subtraction and no learnable bias are used — this saves a first moment and
    empirically matches LayerNorm quality at lower cost.

    Qwen3.5 convention
    ------------------
    The stored weight is zero-initialized and the effective scale is
    ``1 + weight`` (the reference implementation writes the *delta* from the
    identity scale). This matches Hugging Face transformers' ``Qwen3_5RMSNorm``.

    Why fp32 inside?
        Squaring fp16/bf16 activations can overflow; the variance is therefore
        accumulated in fp32 and cast back to the original dtype at the end.

    Reference: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.float()
        # rsqrt(mean(x^2) + eps)  ==  1 / RMS(x)
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        out = x32 * (1.0 + self.weight.float())
        return out.to(in_dtype)


class RMSNormGated(nn.Module):
    r"""RMSNorm with a multiplicative SiLU gate (used inside GatedDeltaNet).

    Math
    ----
    For input :math:`x` and gate :math:`z`:

    .. math::
        y = \big(w \odot \text{RMSNorm}(x)\big) \;\odot\; \text{SiLU}(z)

    where ``SiLU(z) = z * sigmoid(z)``.

    This is the same "normalize then gate" pattern used in Mamba / Mamba-2
    (Gu & Dao 2023, 2024) for the output of the SSM block. The gate is
    computed from the same input stream and broadcasts elementwise, giving
    per-feature multiplicative control over the normalized output.

    Unlike :class:`RMSNorm`, the weight is **1-initialized** here — this is
    the convention used in the reference ``Qwen3_5RMSNormGated`` class.

    References
    ----------
    * Mamba-2: Dao & Gu, "Transformers are SSMs". https://arxiv.org/abs/2405.21060
    * SiLU:   Ramachandran et al. 2017. https://arxiv.org/abs/1710.05941
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
        # gate * SiLU(gate) applied elementwise
        out = x_scaled.float() * F.silu(gate.float())
        return out.to(in_dtype)


class SwiGLUMLP(nn.Module):
    r"""SwiGLU feed-forward block (Shazeer 2020, "GLU Variants Improve Transformer").

    Math
    ----
    Three linear projections :math:`W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}`
    define

    .. math::
        \text{SwiGLU}(x) \;=\; W_{\text{down}}\!\Big(
             \underbrace{\text{SiLU}(W_{\text{gate}}\, x)}_{\text{gate branch}}
             \;\odot\;
             \underbrace{W_{\text{up}}\, x}_{\text{value branch}}
        \Big)

    i.e. a gated-linear-unit with SiLU as the gating nonlinearity. Compared to
    a plain ``FFN(x) = W_2 \sigma(W_1 x)``, SwiGLU empirically improves loss at
    matched parameter count; it is used in PaLM, LLaMA, Qwen, etc.

    Hidden size
        ``gate_proj`` and ``up_proj`` are Linear(hidden → intermediate),
        ``down_proj`` is Linear(intermediate → hidden). Qwen3.5 has no bias on
        any of the three.

    Reference: https://arxiv.org/abs/2002.05202
    """

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # W_down ( SiLU(W_gate x) ⊙ (W_up x) )
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
