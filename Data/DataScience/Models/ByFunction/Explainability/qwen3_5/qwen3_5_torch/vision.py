r"""Vision transformer for Qwen3.5 multimodal.

Processes pre-patchified pixel inputs of shape
``(N, C · t_patch · patch · patch)`` and returns per-token features projected
into the text ``out_hidden_size``. This is the Qwen2-VL / Qwen3-VL "native
resolution" ViT: patch size is a small constant (16), but the input grid
``(T, H, W)`` adapts to the image aspect ratio via ``smart_resize`` in
:mod:`image_processor`.

Pipeline overview
-----------------
1. **Patch embed** (``Qwen3_5VisionPatchEmbed``) — Conv3d with kernel = stride
   = (t_patch, patch, patch) over the input volume, producing one embedding
   per patch.
2. **Absolute learned positions** — a 2D learned ``pos_embed`` table of side
   ``sqrt(num_position_embeddings)`` is **bilinearly interpolated** to each
   image's grid via :meth:`Qwen3_5VisionModel.fast_pos_embed_interpolate`.
   This lets the same learned grid generalise to arbitrary aspect ratios.
3. **Rotary position embedding** — per-token (row, col) rotary via
   :meth:`Qwen3_5VisionModel.rot_pos_emb` (single-axis RoPE, see
   :mod:`rotary`).
4. **Transformer blocks** — L layers of pre-norm attention + MLP with
   *packed-sequence* attention: multiple images live in the same batch entry
   and ``cu_seqlens`` delimits per-image boundaries so images do not attend
   to each other.
5. **Patch merger** — 2x2 spatial merger collapses 4 adjacent patches into 1,
   projects into the text model's ``d_model`` via a 2-layer MLP. This
   reduces the number of visual tokens by 4x before they are spliced into
   the text stream.

References
----------
* ViT: Dosovitskiy et al., "An Image is Worth 16x16 Words", 2020 —
  https://arxiv.org/abs/2010.11929
* Qwen2-VL (native resolution, 2D RoPE, packed seq attention):
  Wang et al., 2024 — https://arxiv.org/abs/2409.12191
* SigLIP vision backbone conventions (tanh-GELU, GELU MLPs):
  Zhai et al., 2023 — https://arxiv.org/abs/2303.15343
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3_5VisionConfig
from .layers import ACT2FN
from .rotary import VisionRotaryEmbedding, apply_rotary_pos_emb_vision


class Qwen3_5VisionMLP(nn.Module):
    r"""Two-layer MLP used in each vision block.

    Math
    ----
    .. math::
        \text{MLP}(x) \;=\; W_2\,\phi(W_1 x + b_1) + b_2

    where :math:`\phi = \text{gelu\_tanh}` is the tanh approximation of GELU
    (used by SigLIP / Qwen-VL vision). Unlike the text stack this is a plain
    FFN, *not* a gated (SwiGLU) MLP.
    """

    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Qwen3_5VisionPatchEmbed(nn.Module):
    r"""Patch embedding via a Conv3d over ``(t_patch, patch, patch)`` volumes.

    Math
    ----
    For an input patch :math:`x \in \mathbb{R}^{C \times t_p \times p \times p}`:

    .. math::
        \text{embed}(x) = W \;\ast\; x \;+\; b  \;\;\in\; \mathbb{R}^{d}

    where ``W`` has shape ``(d, C, t_p, p, p)`` and stride == kernel so patches
    do not overlap. The linear Conv3d with kernel == stride is the standard
    ViT patch-embed trick (Dosovitskiy 2020) extended to video with a temporal
    kernel ``t_p`` (here 2 — one frame pair).
    """

    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen3_5VisionPatchMerger(nn.Module):
    r"""2x2 spatial merger + projection into the text model's ``d_model``.

    Math
    ----
    Let ``M = spatial_merge_size = 2``. The merger reinterprets every ``M · M``
    adjacent tokens as a single super-token by concatenating their channels:

    .. math::
        \tilde x \in \mathbb{R}^{N/M^2 \times (M^2 \cdot d_v)}.

    It then applies LayerNorm → Linear → GELU → Linear to project into
    ``out_hidden_size`` (text ``d_model``). This is the 4x token compression
    used by Qwen2-VL: vision tokens are dense but expensive, so merging is a
    cheap way to shrink the sequence before it enters the text decoder.

    Conditional LayerNorm
        ``use_postshuffle_norm=True`` applies norm *after* the channel
        concatenation (different per-feature statistics). Default False here
        matches the Qwen3.5 recipe.
    """

    def __init__(self, config: Qwen3_5VisionConfig, use_postshuffle_norm: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.hidden_size if use_postshuffle_norm else config.hidden_size
        self.norm = nn.LayerNorm(norm_dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class Qwen3_5VisionAttention(nn.Module):
    r"""Vision self-attention on a **packed sequence** of multiple images.

    Math
    ----
    Per image (delimited by ``cu_seqlens``), the usual scaled dot-product:

    .. math::
        A = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right), \qquad
        \text{out} = A V.

    ``cu_seqlens`` carries cumulative per-image token counts; attention is
    computed independently within each chunk so images do not attend to each
    other. Unlike the text stack there is **no causal mask** — images use
    full bi-directional attention.

    The ``qkv`` projection produces all three streams at once for efficiency,
    then rotary is applied to ``q`` and ``k`` via :func:`apply_rotary_pos_emb_vision`.
    """

    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (seq_len, hidden_size)
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3)
        q, k, v = qkv.unbind(0)  # each (seq, heads, head_dim)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # reshape to (1, H, S, D) and split by images
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        outputs = []
        # One softmax per image — images don't attend across boundaries.
        for q_i, k_i, v_i in zip(
            torch.split(q, lengths, dim=2),
            torch.split(k, lengths, dim=2),
            torch.split(v, lengths, dim=2),
        ):
            scores = torch.matmul(q_i, k_i.transpose(-1, -2)) * self.scaling
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_i.dtype)
            out = torch.matmul(probs, v_i)  # (1, H, s, D)
            outputs.append(out.transpose(1, 2))  # (1, s, H, D)
        attn_output = torch.cat(outputs, dim=1).reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class Qwen3_5VisionBlock(nn.Module):
    r"""Pre-norm ViT block: ``x += attn(norm(x)); x += mlp(norm(x))``.

    Math
    ----
    .. math::
        \begin{aligned}
            h_1 &= x + \text{Attn}(\text{LN}_1(x)) \\
            h_2 &= h_1 + \text{MLP}(\text{LN}_2(h_1))
        \end{aligned}

    with **LayerNorm** (not RMSNorm — unlike the text stack).
    """

    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3_5VisionAttention(config)
        self.mlp = Qwen3_5VisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3_5VisionModel(nn.Module):
    r"""Qwen3.5 vision tower.

    Expects ``hidden_states`` of shape ``(N, C · t_patch · patch · patch)``
    where ``N`` is the total number of patches across every image/frame, and
    ``grid_thw`` rows of ``(num_frames, h_grid, w_grid)`` describing the
    patch grid for each image.

    End-to-end math
    ---------------
    .. math::
        \begin{aligned}
            e^{(0)} &= \text{PatchEmbed}(x) \\
            e^{(0)} &\leftarrow e^{(0)} + \text{Interp}(\text{PosEmbed},\, \text{grid\_thw}) \\
            (\cos, \sin) &= \text{RoPE}(\text{rot\_pos\_emb}(\text{grid\_thw})) \\
            e^{(\ell+1)} &= \text{Block}_\ell(e^{(\ell)};\, \text{cu\_seqlens},\, (\cos,\sin)) \\
            z           &= \text{Merger}(e^{(L)})   \quad\text{(4x token compression)}
        \end{aligned}

    ``z`` is the vector that gets **spliced into the text embeddings** at each
    ``<|image_pad|>`` position by :class:`Qwen3_5Model.forward`.
    """

    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size ** 2

        self.patch_embed = Qwen3_5VisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)

        # Single-axis rotary over half the head dim — the other half mirrors it
        # so the cat((rot, rot), -1) trick (see forward) gives full head RoPE.
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen3_5VisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3_5VisionPatchMerger(config, use_postshuffle_norm=False)

    # -- rotary / positional helpers -----------------------------------

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        r"""Per-token (row, col) rotary frequencies concatenated along ``dim=-1``.

        Math
        ----
        For each token at position :math:`(r, c)` inside an image grid the
        rotary "angle table" is built by looking up a shared 1-D frequency
        table ``freq_table[·] ∈ R^{d/2}``:

        .. math::
            \text{out}[n] = \big[\,\text{freq\_table}[r_n] \;\|\;
                                    \text{freq\_table}[c_n]\,\big] \in \mathbb{R}^{d}.

        The 2x2 spatial-merge layout is baked in: within a 2x2 super-block the
        4 sub-positions are generated by nested offsets before being mapped
        through ``freq_table``. For videos (``num_frames > 1``) the same 2D
        coordinates are repeated across frames (temporal RoPE is handled
        separately via the text-side M-RoPE).
        """
        merge_size = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()

        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw_list:
            merged_h, merged_w = height // merge_size, width // merge_size

            # (row, col) indices that respect the 2x2 spatial merger layout.
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # (N, 2, dim//2)
        embeddings = embeddings.flatten(1)  # (N, dim)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        r"""Bilinearly interpolate learned ``pos_embed`` to each image's grid.

        Math
        ----
        The learned ``pos_embed`` is a ``num_grid_per_side × num_grid_per_side``
        table. For a target grid of size ``(h, w)`` we place ``h`` equally-spaced
        samples along each axis and read the table at the 4 surrounding integer
        corners with bilinear weights:

        .. math::
            \tilde e[y, x] = \sum_{(i,j) \in \{0,1\}^2}
                 w_{ij}(dy, dx)\; \text{pos\_embed}[y_0 + i,\, x_0 + j],

        with ``dy, dx`` the fractional parts of the sample positions and

        .. math::
            w_{00} = (1-dy)(1-dx),\quad w_{01} = (1-dy)\,dx, \\
            w_{10} = dy\,(1-dx),  \quad   w_{11} = dy\,dx.

        This allows the ViT to support arbitrary grids without retraining a new
        positional table — essentially inverting the ViT-style "interpolate
        positional embedding to higher resolution" trick (Touvron et al. 2022,
        "Deit III"; originally from ViT's appendix).
        """
        grid_thw_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]
        device = self.pos_embed.weight.device

        idx_list: list[list[int]] = [[] for _ in range(4)]
        weight_list: list[list[float]] = [[] for _ in range(4)]

        for t, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            # 4 corner indices and their bilinear weights
            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),        # (y0, x0)
                (base_h[None].T + w_idxs_ceil[None]).flatten(),         # (y0, x1)
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),   # (y1, x0)
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),    # (y1, x1)
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        # Repeat temporally and reorder to the (merged_h, merged_w, merge_h, merge_w) layout.
        permuted: list[torch.Tensor] = []
        merge_size = self.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            permuted.append(pos_embed)
        return torch.cat(permuted)

    # -- forward -------------------------------------------------------

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> dict:
        r"""Run the vision tower.

        Steps
        -----
        1. Patch embed: ``e = Conv3d(x)`` per patch.
        2. Add interpolated absolute position embeddings.
        3. Compute per-token rotary (row, col) frequencies; duplicate to full
           head width with ``cat((r, r), -1)`` and take cos/sin.
        4. Build ``cu_seqlens`` from ``grid_thw`` — per-image token counts.
        5. L transformer blocks (packed-sequence attention).
        6. ``merger(·)`` — 2x2 spatial merge + projection to text ``d_model``.
        """
        hidden_states = self.patch_embed(hidden_states)

        # Learned abs. pos (interpolated) -> added to token embeddings.
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        # 2D RoPE (row, col) — shared by all blocks.
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len = hidden_states.shape[0]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # Per-image boundaries in the packed sequence.
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings
            )

        # 4x compression + projection into text d_model.
        merged_hidden_states = self.merger(hidden_states)
        return {
            "last_hidden_state": hidden_states,
            "pooler_output": merged_hidden_states,
        }
