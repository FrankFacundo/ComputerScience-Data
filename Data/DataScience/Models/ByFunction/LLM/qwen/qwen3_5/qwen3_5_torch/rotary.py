"""Rotary position embeddings for Qwen3.5.

Qwen3.5 uses **interleaved multi-axis RoPE (M-RoPE)** with partial rotary. The
text rotary embedding accepts position ids of shape ``(3, bs, seq)`` where the
three rows correspond to ``(temporal, height, width)``, and projects them
through a shared ``inv_freq`` — then interleaves the per-axis frequencies into
a single per-token rotary vector. The vision tower uses a simpler single-axis
RoPE over ``(row, col)``.

Math overview
-------------
Standard RoPE (Su et al. 2021) encodes an absolute position :math:`m` into a
rotation of each 2-D sub-space of the query/key vector. For head dim ``d`` the
inverse frequencies are

.. math::
    \theta_i = \text{base}^{-2i/d}, \qquad i = 0, 1, \dots, d/2 - 1.

For position :math:`m` and a 2-D pair :math:`(x_{2i}, x_{2i+1})`:

.. math::
    R_{\theta_i, m}\!\begin{bmatrix}x_{2i}\\x_{2i+1}\end{bmatrix}
    \;=\;\begin{bmatrix}\cos m\theta_i & -\sin m\theta_i\\
                        \sin m\theta_i & \cos m\theta_i\end{bmatrix}
         \begin{bmatrix}x_{2i}\\x_{2i+1}\end{bmatrix}.

The dot-product :math:`\langle R\,q_m, R\,k_n\rangle` depends only on the
relative offset :math:`m-n`, giving relative-position-aware attention with
absolute-position code. Equivalently (using the half-interleaved layout that
transformers adopts):

.. math::
    \text{RoPE}(x, m) = x \odot \cos(m\theta) + \text{rotate\_half}(x) \odot \sin(m\theta),

where ``rotate_half([a, b]) = [-b, a]`` splits the head into two halves.

Partial rotary
--------------
Qwen3.5 sets ``partial_rotary_factor = 0.25``: only the first ``rotary_dim =
0.25 * head_dim`` channels of q/k are rotated, the remainder pass through
unchanged. This reduces the compute of RoPE and was empirically found to work
as well as full rotary for some architectures.

Interleaved M-RoPE (Qwen2-VL / Qwen3.5)
---------------------------------------
For multimodal inputs, each *token* has three position indices
:math:`(t, h, w)`. Rather than averaging axis embeddings, M-RoPE splits the
inv_freq channels into three disjoint sections with widths ``mrope_section``
(Qwen3.5 uses ``[11, 11, 10]``), and **interleaves** them across channel
index modulo 3: channel positions ``≡ 0 (mod 3)`` carry the *temporal*
frequency, ``≡ 1`` the *height*, ``≡ 2`` the *width*. Applied to a single
channel :math:`i`:

.. math::
    \theta_i \;=\; \begin{cases}
        m_t \cdot \theta^{(t)}_i & \text{if } i \bmod 3 = 0 \\
        m_h \cdot \theta^{(h)}_i & \text{if } i \bmod 3 = 1 \\
        m_w \cdot \theta^{(w)}_i & \text{if } i \bmod 3 = 2 \\
    \end{cases}

References
----------
* RoFormer / RoPE: Su et al., 2021 — https://arxiv.org/abs/2104.09864
* Qwen2-VL (M-RoPE): Wang et al., 2024 — https://arxiv.org/abs/2409.12191
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .config import Qwen3_5TextConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    r"""Rotate the last dim by 90° in its (first-half, second-half) 2-D pairing.

    Math
    ----
    For :math:`x = [x_0 \;\|\; x_1]` split into two equal halves along the last
    axis, returns

    .. math::
        \text{rotate\_half}(x) \;=\; [-x_1 \;\|\; x_0].

    Combined with ``x * cos + rotate_half(x) * sin`` this realises the 2-D
    rotation used by RoPE. The "half-then-half" layout (as opposed to strict
    2-D interleave ``(x_0,x_1,x_2,x_3,\ldots)``) is the convention used by
    LLaMA / Qwen / HuggingFace transformers.
    """
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Apply partial rotary embeddings to q/k.

    Math
    ----
    Given precomputed ``cos`` and ``sin`` tables of shape
    ``(bs, seq, rotary_dim)`` (broadcastable to head-axis by ``unsqueeze_dim``):

    .. math::
        q' = q \odot \cos + \text{rotate\_half}(q) \odot \sin

    .. math::
        k' = k \odot \cos + \text{rotate\_half}(k) \odot \sin

    When ``cos.shape[-1] < head_dim`` (partial rotary), the rotation is applied
    only to the leading ``rotary_dim`` channels of ``q`` and ``k``; the trailing
    channels are concatenated back unchanged.

    Why this formulation
        Writing the rotation in ``(cos, -sin; sin, cos)`` form with the
        "half-split" permutation gives the same geometry as the classical
        2-D rotation :math:`(x_{2i}, x_{2i+1}) \mapsto (x_{2i}\cos\theta -
        x_{2i+1}\sin\theta,\; x_{2i}\sin\theta + x_{2i+1}\cos\theta)` but maps
        cleanly to contiguous GPU memory.

    Reference: https://arxiv.org/abs/2104.09864
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def _compute_default_inv_freq(config: Qwen3_5TextConfig, device=None) -> torch.Tensor:
    r"""Compute the base RoPE inverse frequencies.

    Math
    ----
    With base :math:`\theta_{\text{base}}` (``rope_theta``) and rotary width
    :math:`d = \text{head\_dim} \times \text{partial\_rotary\_factor}`,

    .. math::
        \text{inv\_freq}_i = \theta_{\text{base}}^{-2i / d}, \qquad
        i = 0, 1, \dots, d/2 - 1.

    Qwen3.5 uses a **very large base** (``rope_theta ≈ 1e7``), which spreads the
    frequencies out so that the phase ``m * theta_i`` wraps less often over the
    262k-token context window (the "NTK/long-context" trick popularised after
    the original RoFormer paper).
    """
    rope_params = config.rope_parameters
    base = rope_params["rope_theta"]
    partial = rope_params.get("partial_rotary_factor", 1.0)
    head_dim = config.head_dim
    dim = int(head_dim * partial)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
            / dim
        )
    )
    return inv_freq


def _apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
    r"""Interleave per-axis frequencies for (T, H, W) into a single vector.

    Math
    ----
    Input ``freqs`` has shape ``(3, bs, seq, dim_half)`` where axis 0 indexes
    ``(t, h, w)`` and the last axis carries the channel index
    :math:`i = 0, \dots, d/2 - 1`. The output has shape ``(bs, seq, dim_half)``
    where channel :math:`i` is taken from axis:

    .. math::
        \text{axis}(i) \;=\; \begin{cases}
            0 & i \bmod 3 = 0 \quad\text{(time)} \\
            1 & i \bmod 3 = 1 \quad\text{(height)} \\
            2 & i \bmod 3 = 2 \quad\text{(width)} \\
        \end{cases}

    Effectively every 3rd channel belongs to the temporal axis, every 3rd
    (offset 1) to height, every 3rd (offset 2) to width — producing the
    interleaved M-RoPE layout from Qwen2-VL. ``mrope_section`` lengths
    (default ``[11, 11, 10]``) describe how many channels each axis occupies
    after the mod-3 selection.
    """
    freqs_t = freqs[0].clone()
    for axis, offset in enumerate((1, 2), start=1):  # H, W
        length = mrope_section[axis] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[axis, ..., idx]
    return freqs_t


class TextRotaryEmbedding(nn.Module):
    r"""Qwen3.5 text-side rotary embedding with interleaved M-RoPE.

    Given position ids of shape ``(3, bs, seq)`` — the (T, H, W) axes — this
    module returns ``(cos, sin)`` tables of shape ``(bs, seq, rotary_dim)``
    that are applied to the queries/keys by :func:`apply_rotary_pos_emb`.

    Pipeline (per forward call)
    ---------------------------
    1. **Outer product** per axis: ``freqs[a] = pos[a] * inv_freq`` →
       ``(3, bs, seq, dim_half)``.
    2. **Interleave** across channel index mod 3 via
       :func:`_apply_interleaved_mrope` → ``(bs, seq, dim_half)``.
    3. **Double** the channel dim with ``cat([freqs, freqs], -1)`` so that
       ``cos/sin`` match the two halves consumed by ``rotate_half``.
    4. Compute ``cos``, ``sin`` and scale by ``attention_scaling`` (1.0 in this
       implementation — slot for long-context NTK-aware scaling schemes).

    Reference: https://arxiv.org/abs/2409.12191 (Qwen2-VL M-RoPE)
    """

    def __init__(self, config: Qwen3_5TextConfig, device=None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])
        self.attention_scaling = 1.0

        inv_freq = _compute_default_inv_freq(config, device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        position_ids:
            - ``(bs, seq)`` — text-only; expanded to ``(3, bs, seq)`` with T=H=W.
            - ``(3, bs, seq)`` — multimodal ``(T, H, W)`` positions.

        Returns
        -------
        ``(cos, sin)`` each of shape ``(bs, seq, rotary_dim)`` and dtype == x.dtype.
        """
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # inv_freq: (dim_half,) -> (3, bs, dim_half, 1)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()  # (3, bs, 1, seq)

        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # outer product: (3, bs, dim_half, 1) @ (3, bs, 1, seq) -> (3, bs, dim_half, seq)
            # → transpose to (3, bs, seq, dim_half)
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = _apply_interleaved_mrope(freqs, self.mrope_section)  # (bs, seq, dim_half)
            # duplicate so cos/sin align with rotate_half split
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(x.dtype), sin.to(x.dtype)


class VisionRotaryEmbedding(nn.Module):
    r"""Single-axis rotary embedding used inside the vision tower.

    Identical to classical RoPE but precomputes a frequency *table* indexed by
    integer position — the vision side looks up ``freq[row]`` and ``freq[col]``
    separately, then concatenates them along the last dim to get a 2-D
    (row, col) rotary encoding.

    .. math::
        \text{inv\_freq}_i \;=\; \theta^{-2i/d}, \quad
        \text{freqs}[m, i] \;=\; m \cdot \text{inv\_freq}_i.

    Reference: RoFormer — https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        """Return a ``(seqlen, dim/2)`` table where ``out[m, i] = m * inv_freq[i]``."""
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""RoPE for the vision tower (full rotary over all head channels).

    Same formula as :func:`apply_rotary_pos_emb` but:
      * no ``partial_rotary`` — all channels are rotated;
      * computed in fp32 for numerical stability and cast back at the end.

    .. math::
        q' = q\cos + \text{rotate\_half}(q)\sin, \qquad
        k' = k\cos + \text{rotate\_half}(k)\sin.
    """
    orig_q, orig_k = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q), k_embed.to(orig_k)
