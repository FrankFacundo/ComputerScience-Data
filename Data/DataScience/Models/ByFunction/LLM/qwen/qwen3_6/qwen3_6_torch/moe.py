r"""Sparse Mixture-of-Experts MLP block for Qwen3.6.

The Qwen3.6 (a.k.a. ``Qwen3_5MoE``) checkpoints replace the dense SwiGLU MLP
of Qwen3.5 with a **shared expert + top-K routed experts** block. For an
input token :math:`x \in \mathbb{R}^{d}`:

1. **Router** — a linear layer produces logits over experts:

   .. math::
       r(x) = W_{\text{gate}}\, x \;\in\; \mathbb{R}^{E}.

   Top-:math:`K` experts are picked, their softmaxed scores are
   re-normalised so they sum to 1
   (:math:`p_e = \tfrac{\exp r_e}{\sum_{e' \in \text{top-}K} \exp r_{e'}}`).

2. **Routed experts** — each expert :math:`e` is a SwiGLU MLP packed into
   two fused tensors shared with the others:

   .. math::
       y_e(x) = W^{(e)}_{\text{down}}\big(\text{SiLU}(W^{(e)}_{\text{gate}} x) \odot W^{(e)}_{\text{up}} x\big).

   The contribution of expert :math:`e` to the output is :math:`p_e\, y_e(x)`,
   summed over the chosen experts.

3. **Shared expert** — a single ordinary SwiGLU MLP that runs for every
   token, multiplied by a sigmoid scalar derived from
   :math:`\sigma(W_{\text{shared\_gate}}\, x) \in (0, 1)`. This gives the
   block a "always-on" path even when the router is unsure.

The total per-token MLP output is

.. math::
    \text{MoE}(x) \;=\; \sigma(W_{\text{sh\_gate}}\, x) \cdot W_{\text{sh}}(x)
    \;+\; \sum_{e \in \text{top-}K(x)} p_e(x)\, y_e(x).

References
----------
* Sparse MoE / soft routing: Shazeer et al. 2017 — https://arxiv.org/abs/1701.06538
* Switch / GShard top-K: Fedus et al. 2021 — https://arxiv.org/abs/2101.03961
* DeepSeek-MoE shared expert: DeepSeek 2024 — https://arxiv.org/abs/2401.06066
* Qwen2-MoE / Qwen3-MoE checkpoints — Qwen team, 2024 — https://qwenlm.github.io/

Implementation notes
--------------------
The Hugging Face checkpoint stores the routed experts as **packed** tensors:
``experts.gate_up_proj`` has shape ``(E, 2·d_moe, d)`` (gate stacked above up
along the output axis) and ``experts.down_proj`` has shape ``(E, d, d_moe)``.
We keep that layout here so the safetensors loader can copy weights byte-
for-byte without reshaping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3_6TextConfig
from .layers import ACT2FN, SwiGLUMLP


class Qwen3_6MoeRoutedExperts(nn.Module):
    r"""Container for the :math:`E` routed SwiGLU experts (packed weights).

    Parameters
    ----------
    ``gate_up_proj`` : ``(E, 2·d_moe, d)`` — fused gate / up projection per expert
        (gate is the first ``d_moe`` rows, up is the next ``d_moe``).
    ``down_proj``    : ``(E, d, d_moe)`` — output projection per expert.
    """

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # NOTE: out_features = 2 * intermediate_size for the fused gate+up.
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))

    def expert_forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        r"""Apply expert ``expert_idx`` to a batch of tokens ``x``.

        Math:  :math:`y = W^{(e)}_{\text{down}}(\text{SiLU}(W^{(e)}_{\text{gate}} x) \odot W^{(e)}_{\text{up}} x)`.
        """
        gu_w = self.gate_up_proj[expert_idx]                 # (2·d_moe, d)
        down_w = self.down_proj[expert_idx]                  # (d, d_moe)
        gu = F.linear(x, gu_w)                                # (..., 2·d_moe)
        gate, up = gu.chunk(2, dim=-1)
        h = F.silu(gate) * up
        return F.linear(h, down_w)                            # (..., d)


class Qwen3_6SparseMoeBlock(nn.Module):
    r"""Sparse MoE block (router + routed experts + shared expert).

    Pipeline (per token)
    --------------------
    1. ``router_logits = gate(x)``                                — :math:`(E,)`.
    2. Pick top-:math:`K` experts; softmax over the selected logits.
       (When ``norm_topk_prob=True`` we re-normalise so weights sum to 1.)
    3. Dispatch each token to its chosen experts; sum the weighted outputs.
    4. Compute the shared expert output and a sigmoid scalar gate; add to
       the routed sum.

    The routed loop iterates over experts (not tokens) and uses ``torch.where``
    to gather the rows that selected each expert. This is the vanilla
    "for each expert, find its tokens" implementation — readable and exact,
    with no triton / fused-MoE kernels.
    """

    def __init__(self, config: Qwen3_6TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = Qwen3_6MoeRoutedExperts(
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
        )

        # Shared expert: an "always-on" SwiGLU MLP, scaled by a sigmoid.
        self.shared_expert = SwiGLUMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""Run the MoE block over ``hidden_states`` of shape ``(B, S, d)``.

        Returns a tensor of the same shape; no router-loss / aux-loss terms
        are returned (this is inference-only).
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.reshape(-1, hidden_dim)                   # (N, d)

        # Router: logits → softmax → top-K selection.
        router_logits = self.gate(x)                                # (N, E)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights.to(x.dtype)

        # Shared expert is independent of the routed loop; launching it first
        # lets its kernels overlap with the per-expert dispatch below.
        shared = self.shared_expert(x)
        shared_gate = torch.sigmoid(self.shared_expert_gate(x))         # (N, 1)
        out = shared_gate * shared

        # Determine which experts received at least one token. Building the
        # full hit list once (with a single device→CPU sync) is dramatically
        # faster than checking `token_mask.any()` inside a 256-iteration loop,
        # which forces a sync per iteration on accelerators (notably MPS).
        # Shape: (E, top_k, N) so `expert_mask[e]` indexes (top_k_pos, token).
        with torch.no_grad():
            expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = expert_mask.sum(dim=(-1, -2)).gt(0).nonzero(as_tuple=False).flatten().tolist()

        gate_up_w = self.experts.gate_up_proj
        down_w = self.experts.down_proj
        for expert_idx in expert_hit:
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            x_e = x[token_idx]                                            # (n_e, d)
            gu = F.linear(x_e, gate_up_w[expert_idx])                     # (n_e, 2·d_moe)
            gate, up = gu.chunk(2, dim=-1)
            h_e = F.silu(gate) * up
            y_e = F.linear(h_e, down_w[expert_idx])                       # (n_e, d)
            w_e = topk_weights[token_idx, top_k_pos, None]                # (n_e, 1)
            out.index_add_(0, token_idx, (y_e * w_e).to(out.dtype))

        return out.view(batch_size, seq_len, hidden_dim)
