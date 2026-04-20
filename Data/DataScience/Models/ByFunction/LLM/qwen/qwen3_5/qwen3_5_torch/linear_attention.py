r"""Gated DeltaNet — the linear-attention branch used in Qwen3.5 layers.

This is a pure-PyTorch port of the reference ``torch_chunk_gated_delta_rule`` /
``torch_recurrent_gated_delta_rule`` paths in transformers. No triton, no fla,
no causal_conv1d kernels — just ``F.conv1d`` and a gated-delta-rule loop.

Mathematical background
=======================

Gated DeltaNet generalises two earlier linear-attention families:

1. **Linear attention** (Katharopoulos et al. 2020,
   https://arxiv.org/abs/2006.16236). With feature map :math:`\phi(\cdot)` the
   attention at step :math:`t` becomes the recurrence

   .. math::
       S_t = S_{t-1} + \phi(k_t) v_t^\top, \qquad
       o_t = \phi(q_t)^\top S_t \,/\, z_t.

2. **DeltaNet** (Schlag et al. 2021; see also Yang et al. 2024
   https://arxiv.org/abs/2406.06484) replaces the additive rule by an
   **associative delta rule**:

   .. math::
       S_t = S_{t-1} + \beta_t (v_t - S_{t-1}\,k_t)\,k_t^\top,

   which is an online least-squares / associative-memory update — it writes
   the *residual* ``v_t - S_{t-1} k_t`` rather than ``v_t`` itself. This
   dramatically improves in-context recall vs linear attention.

3. **Gated DeltaNet** (Yang et al. 2024, https://arxiv.org/abs/2412.06464)
   combines Mamba-2's data-dependent **forget gate** ``g_t`` with the delta
   rule:

   .. math::
       S_t = \alpha_t S_{t-1} + \beta_t \big(v_t - (\alpha_t S_{t-1})\,k_t\big)\,k_t^\top,
       \qquad
       o_t = q_t^\top S_t,

   where :math:`\alpha_t = \exp(g_t)` with :math:`g_t \le 0` (so :math:`\alpha_t
   \in (0, 1]`) is the multiplicative decay, and :math:`\beta_t = \sigma(b_t)`
   controls the size of the delta update. This gives the model both a
   **forget** mechanism (like Mamba/SSMs) and a **precise overwrite**
   mechanism (like DeltaNet). In practice the forget is derived as

   .. math::
       g_t = -A \cdot \text{softplus}(a_t + \text{dt\_bias}), \quad A = \exp(A_{\text{log}}),

   matching the ``A_log`` / ``dt`` parameterisation of Mamba-2.

Structure of this file
----------------------
* :func:`l2norm`, :func:`apply_mask_to_padding_states`,
  :func:`torch_causal_conv1d_update` — helpers.
* :func:`torch_chunk_gated_delta_rule`     — fast chunked evaluation (prefill).
* :func:`torch_recurrent_gated_delta_rule` — step-by-step recurrence (decoding).
* :class:`Qwen3_5GatedDeltaNet` — the nn.Module that wraps everything with
  the qkv depthwise conv, gating projections, RMSNormGated, and output proj.

References
----------
* Linear attention:  Katharopoulos+ 2020  — https://arxiv.org/abs/2006.16236
* DeltaNet:          Yang+ 2024 (chunking) — https://arxiv.org/abs/2406.06484
* Gated DeltaNet:    Yang+ 2024            — https://arxiv.org/abs/2412.06464
* Mamba-2 (for the A/dt gate parameterisation): Dao & Gu 2024 —
  https://arxiv.org/abs/2405.21060
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import HybridCache
from .config import Qwen3_5TextConfig
from .layers import RMSNormGated, ACT2FN


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    r"""L2 normalise along ``dim`` with epsilon stabilisation.

    Math
    ----
    .. math::
        \text{l2norm}(x)_i = \frac{x_i}{\sqrt{\sum_j x_j^2 + \varepsilon}}.

    Used on q and k inside the kernel so their dot-product behaves like a
    cosine similarity — a known stabiliser for long-horizon linear attention.
    """
    inv = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv


def apply_mask_to_padding_states(hidden_states: torch.Tensor, attention_mask) -> torch.Tensor:
    r"""Zero-out hidden states for padding tokens (Mamba-style).

    Math
    ----
    For a 2-D attention mask :math:`m \in \{0, 1\}^{B \times S}` the output is
    ``hidden_states * m[:, :, None]``. Since Gated DeltaNet is a causal
    recurrence, padding contributes nothing once its contribution is zeroed
    before entering the recurrence.
    """
    if attention_mask is not None and attention_mask.ndim == 2 and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


def torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: str | None = None,
) -> torch.Tensor:
    r"""Single-step causal conv1d that rolls the conv_state in place.

    Math
    ----
    Conceptually this is a depthwise 1-D convolution with kernel width :math:`K`:

    .. math::
        y_t = \sum_{\tau=0}^{K-1} W[\tau] \cdot x_{t-\tau} \;+\; b

    At each decode step we maintain ``conv_state = [x_{t-K+1}, \dots, x_t]``,
    roll the new tokens in, apply the conv, take the last ``seq_len`` outputs,
    and optionally apply a SiLU. The rolled state is copied **in place** so
    the cache updates without extra allocation.
    """
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    merged = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(merged[:, :, -state_len:])
    out = F.conv1d(merged, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    if activation == "silu":
        out = F.silu(out[:, :, -seq_len:])
    else:
        out = out[:, :, -seq_len:]
    return out.to(hidden_states.dtype)


# --------------------------------------------------------------------------
# gated delta rule (pure-torch)
# --------------------------------------------------------------------------

def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    r"""Chunked gated-delta-rule attention (prefill path).

    Inputs
    ------
    * ``query, key, value``: ``(B, T, H, D)``.
    * ``g`` (= :math:`\log \alpha`): cumulative log-decay scores, ``(B, T, H)``.
    * ``beta``: delta-rule step sizes :math:`\beta_t \in (0, 1)`, ``(B, T, H)``.
    * ``chunk_size``: decomposition length ``C`` for the chunked algorithm.

    Output
    ------
    * ``core_attn_out``: ``(B, T, H, Dv)``.
    * ``last_recurrent_state``: optional ``(B, H, Dk, Dv)``.

    Math
    ----
    The gated delta rule

    .. math::
        S_t = \alpha_t S_{t-1} + \beta_t\big(v_t - (\alpha_t S_{t-1}) k_t\big)k_t^\top

    is cast into an **exactly-equivalent chunked form** following Yang et al.
    (2024, Sec. 3). Let :math:`G_{i,j} = \exp(\sum_{\ell=j}^{i} g_\ell)` be the
    cumulative decay from step :math:`j` to step :math:`i` inside the chunk.
    The intra-chunk triangular solve

    .. math::
        (I + L[k_\beta k^\top \odot G]) \;\tilde v = v_\beta

    (computed by the explicit for-loop over ``chunk_size``) linearises the
    delta recursion inside one chunk; the inter-chunk state update is the
    scalar-gated outer product

    .. math::
        S_{\text{next}} = e^{g_{[-1]}} S \;+\; k^\top \big(e^{g_{[-1]} - g}\odot \tilde v\big).

    The output of chunk :math:`c` is split into an **intra** term
    :math:`\text{inter\_attn} \tilde v` (triangular mask over the chunk) and an
    **inter** term :math:`q\,e^{g}\, S_{prev}` (contribution of previous state).
    This is the same structure as Mamba-2's chunked SSM (Dao & Gu 2024),
    specialised to the delta-rule update.

    ``use_qk_l2norm_in_kernel=True`` applies :func:`l2norm` to q and k before
    the kernel — the configuration used by Qwen3.5.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # (B, H, T, D) / (B, H, T) in fp32 — the kernel is numerically sensitive.
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch, heads, seq_len, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad))
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    total = seq_len + pad

    scale = 1.0 / (k_head_dim ** 0.5)
    query = query * scale

    # beta-weighted keys and values (from the delta update formula).
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # (B, H, chunks, chunk, D)
    def _chunk(x):
        return x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])

    query, key, value, k_beta, v_beta = [_chunk(x) for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    mask_incl = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )
    # Cumulative log-decay inside each chunk.
    g = g.cumsum(dim=-1)
    # decay_mask[i, j] = exp(g_i - g_j) for i >= j, else 0 — this is the
    # lower-triangular G from the math block above.
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    # (I + L)^{-1}-like intra-chunk propagation of deltas.
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask_incl, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    # Propagate v_beta and (k_beta ⊙ exp(g)) through the intra-chunk mixing.
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_recurrent_state = (
        torch.zeros(batch, heads, k_head_dim, v_head_dim, device=query.device, dtype=query.dtype)
        if initial_state is None
        else initial_state.to(query.dtype).to(query.device)
    )
    core_attn_out = torch.zeros_like(value)
    mask_excl = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    num_chunks = total // chunk_size
    for i in range(num_chunks):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        # Intra-chunk: strict-lower (q k^T * decay) — excludes same position.
        inter = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask_excl, 0)
        # Remove the "echo" of previous state from the current chunk's values
        # before writing it: v' = v - k_cumdecay @ S_prev.
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        # Inter-chunk contribution: q · exp(g) · S_prev.
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + inter @ v_new
        # Update the recurrent state with the end-of-chunk decay.
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )[:, :, :seq_len]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = False,
):
    r"""Step-by-step recurrent gated-delta-rule (single-token decoding path).

    Math
    ----
    Explicit loop of the closed-form recurrence

    .. math::
        \begin{aligned}
            S_t^{\prime} &= \alpha_t S_{t-1} \\
            \tilde v_t   &= v_t - S_t^{\prime\top} k_t \\
            S_t          &= S_t^{\prime} + \beta_t\; k_t\, \tilde v_t^\top \\
            o_t          &= S_t^\top q_t
        \end{aligned}

    with :math:`\alpha_t = \exp(g_t)` and :math:`\beta_t` already in :math:`(0, 1)`.
    This matches Eq. (1)–(2) of Yang et al. 2024 but written in our ``(Dk, Dv)``
    outer-product layout. Cheaper than the chunked path when decoding one
    token at a time.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch, heads, seq_len, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1.0 / (k_head_dim ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch, heads, seq_len, v_head_dim, device=query.device, dtype=query.dtype)
    last_recurrent_state = (
        torch.zeros(batch, heads, k_head_dim, v_head_dim, device=query.device, dtype=query.dtype)
        if initial_state is None
        else initial_state.to(query.dtype).to(query.device)
    )

    for i in range(seq_len):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)   # α_t
        beta_t = beta[:, :, i].unsqueeze(-1)

        # Decay: S' = α S_{t-1}
        last_recurrent_state = last_recurrent_state * g_t
        # Predicted value given current key: S'^T k_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        # Residual δ = β (v_t - kv_mem)
        delta = (v_t - kv_mem) * beta_t
        # State update: S = S' + k ⊗ δ
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # Readout: o_t = S^T q_t
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# --------------------------------------------------------------------------
# GatedDeltaNet module
# --------------------------------------------------------------------------

class Qwen3_5GatedDeltaNet(nn.Module):
    r"""Gated DeltaNet block used in place of attention in ~3/4 of Qwen3.5 layers.

    Pipeline (matches `Qwen3_5GatedDeltaNet` in transformers)
    ---------------------------------------------------------
    Inputs: ``x ∈ R^{B×S×d}``, cache, padding mask.

    1. Zero-out padding rows (Mamba-style).
    2. Project x to three streams:
         * ``mixed_qkv = [Q | K | V] = W_{qkv} x``, shape ``(B, conv_dim, S)``
         * ``z = W_z x`` → gate for the output RMSNormGated.
         * ``b = W_b x`` → raw β (before sigmoid).
         * ``a = W_a x`` → raw a (before the ``A exp · softplus`` map → g).
    3. Depthwise causal conv1d over the concatenated qkv channel axis
       (width ``conv_dim = 2·d_k + d_v``). During decoding the conv_state is
       a ring buffer of width ``conv_kernel_size``.
    4. Split into ``Q, K, V`` and reshape to per-head. If ``H_v > H_k``,
       queries and keys are repeated to match the value-head count (similar
       to GQA).
    5. Derive gates:

       .. math::
           \beta_t = \sigma(b_t), \qquad g_t = -A \cdot \text{softplus}(a_t + \text{dt\_bias})

       so that :math:`\alpha_t = e^{g_t} \in (0, 1]`.

    6. Delta-rule kernel — chunked for prefill, recurrent for single-step
       decoding — producing ``core_out`` of shape ``(B, S, H_v, d_v)``.
    7. :class:`RMSNormGated` with the gate ``z``: ``y = (W · RMSNorm(o)) ⊙ SiLU(z)``.
    8. Output projection ``y ← W_out y``.

    Parameters
    ----------
    ``A_log`` : learnable; ``A = exp(A_log)`` is the per-head decay magnitude
                (matches Mamba-2's ``A`` parameter).
    ``dt_bias`` : learnable additive bias in the softplus, giving an initial
                  "time-step" analogous to ``Δ`` in Mamba-2.

    References: see module docstring.
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        # Depthwise conv over (q | k | v) along the time axis — groups = channels.
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # Mamba-2 style gate parameterisation:  g_t = -A · softplus(a + dt_bias),
        # where A = exp(A_log) so A > 0.
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # Input projections for the qkv stream, z-gate, β, a.
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[HybridCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        # Decode-step fast path: a single query, cache already warm.
        use_precomputed = (
            cache_params is not None
            and cache_params.has_linear_state(self.layer_idx)
            and seq_len == 1
        )

        # Projections: qkv stream, z-gate, β pre-activation, a pre-activation.
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # (B, conv_dim, T)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed:
            # single-step causal conv update — reads/writes the ring buffer.
            conv_state = cache_params.get_linear(self.layer_idx).conv_states
            mixed_qkv = torch_causal_conv1d_update(
                mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation
            )
        else:
            if cache_params is not None:
                # F.pad with a negative left pad crops, so this yields the
                # last `conv_kernel_size` tokens regardless of seq_len.
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.update_linear_conv_state(self.layer_idx, conv_state)
            # depthwise causal conv1d followed by SiLU, truncated to seq_len.
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)  # (B, T, conv_dim)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        # Gate parameters:
        #   β_t = σ(b_t)                                ∈ (0, 1)
        #   g_t = -A · softplus(a + dt_bias),  A > 0    ≤ 0
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # "GQA"-style repeat if there are more value heads than key heads.
        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        if use_precomputed:
            # Decode step: continue the recurrence from the cached state.
            recurrent = cache_params.get_linear(self.layer_idx).recurrent_states
            core_out, last_state = torch_recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=recurrent,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            # Prefill: chunked algorithm from zero initial state.
            core_out, last_state = torch_chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None and last_state is not None:
            cache_params.update_linear_recurrent_state(self.layer_idx, last_state)

        # Output: RMSNormGated with the z-gate, then project back to d_model.
        core_out = core_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        core_out = self.norm(core_out, z_flat)
        core_out = core_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_out)
