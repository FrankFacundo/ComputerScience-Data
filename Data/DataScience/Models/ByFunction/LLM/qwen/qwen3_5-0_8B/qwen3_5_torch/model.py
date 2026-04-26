r"""Multimodal Qwen3.5 (vision + text) and conditional-generation wrapper.

This file glues the vision tower (:class:`Qwen3_5VisionModel`) and the text
decoder (:class:`Qwen3_5TextModel`) together into a single model. The recipe
is the **Qwen2-VL / Qwen3-VL "late-fusion" scheme**:

1. The text tokenizer produces a sequence that contains special image
   placeholder tokens ``<|image_pad|>`` (one per merged-vision token).
2. The vision tower turns each image into a sequence of merged vision
   embeddings of the same width as the text model.
3. The multimodal model **splices** vision embeddings into the text embedding
   stream by ``masked_scatter`` — so at each placeholder position we replace
   the learned ``<|image_pad|>`` embedding by the corresponding vision
   embedding.
4. A 3-axis **M-RoPE position id** is built per token, with text tokens using
   a single running index across all three axes and vision tokens receiving
   ``(t, h, w)`` coordinates derived from their grid.
5. The spliced embeddings are run through the text decoder.
6. The final hidden states are projected to vocab logits.

Paper: Wang et al., "Qwen2-VL", 2024 — https://arxiv.org/abs/2409.12191
"""

from __future__ import annotations

import itertools
from typing import Optional

import torch
import torch.nn as nn

from .cache import HybridCache
from .config import Qwen3_5Config
from .decoder import Qwen3_5TextModel
from .vision import Qwen3_5VisionModel


class Qwen3_5Model(nn.Module):
    r"""Top-level multimodal model: vision tower + hybrid text decoder.

    Accepts batched ``input_ids`` with image/video placeholder tokens. For
    each placeholder, the corresponding vision features (from ``pixel_values``
    or ``pixel_values_videos``) are spliced into the text embedding stream via
    ``masked_scatter``. 3D M-RoPE position ids are computed from
    ``mm_token_type_ids`` + per-image ``grid_thw`` tensors.

    Math at a glance
    ----------------
    .. math::
        e^{(0)} = E[\,\text{input\_ids}\,]
        \qquad
        v_i = \text{Vision}(\text{pixels}_i, \text{grid}_i)

    .. math::
        \tilde e^{(0)} = \text{masked\_scatter}\big(e^{(0)},\; \text{img\_mask},\; v\big)

    .. math::
        (t_n, h_n, w_n) = \text{RoPEIndex}_n \quad \text{(per token)}

    .. math::
        y = \text{TextDecoder}(\tilde e^{(0)};\, (t, h, w))
    """

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.visual = Qwen3_5VisionModel(config.vision_config)
        self.language_model = Qwen3_5TextModel(config.text_config)
        self.rope_deltas: Optional[torch.Tensor] = None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.language_model.set_input_embeddings(value)

    # -- vision feature extraction -------------------------------------

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> list[torch.Tensor]:
        r"""Run the vision tower and split its output by image.

        Shapes
        ------
        * Input ``pixel_values`` is the packed tensor of patches across all
          images in the batch.
        * Output is a list of per-image embeddings
          :math:`v_i \in \mathbb{R}^{(T_i H_i W_i / M^2) \times d_{\text{text}}}`
          where ``M = spatial_merge_size`` and each image contributes
          ``T·H·W / M²`` tokens (after the 2x2 merger).
        """
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds = vision_output["pooler_output"]
        # After merger each image contributes T*H*W / M^2 tokens.
        split_sizes = (
            image_grid_thw.prod(-1) // (self.visual.spatial_merge_size ** 2)
        ).tolist()
        return list(torch.split(image_embeds, split_sizes))

    def get_video_features(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Identical machinery as :meth:`get_image_features` — videos are just
        images with ``T_i > 1`` frames."""
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    # -- multimodal placeholder masking --------------------------------

    def get_placeholder_mask(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Return boolean masks marking image / video placeholder positions.

        When ``input_ids`` is available we simply compare ids. When only
        ``inputs_embeds`` is available (e.g. for embedding-level decoding), we
        identify placeholders by testing whether the embedding row equals the
        learned placeholder embedding :math:`E[\text{image\_token\_id}]`.
        """
        if input_ids is None:
            embed = self.get_input_embeddings()
            img_token_emb = embed(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            vid_token_emb = embed(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = (inputs_embeds == img_token_emb).all(-1)
            video_mask = (inputs_embeds == vid_token_emb).all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id
            video_mask = input_ids == self.config.video_token_id
        image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds)
        return image_mask, video_mask

    # -- 3D M-RoPE position id construction ---------------------------

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        r"""Build the ``(t, h, w)`` positions for one image's vision tokens.

        Math
        ----
        For a grid of size ``(T, H, W)`` (already divided by the respective
        merge sizes), we enumerate all ``T·H·W`` spatial positions as:

        .. math::
            (t_n, h_n, w_n) \text{ with }
            w_n = s + (n \bmod W),\quad
            h_n = s + \lfloor n / W \rfloor \bmod H, \quad
            t_n = s \cdot \tau,

        where ``s = start_position`` and :math:`\tau = \text{time\_interval}`.
        The temporal component is constant per frame because Qwen3.5 inserts
        explicit *timestamp tokens* in the text between video frames — the
        per-frame temporal offset lives there, not in the vision ids.
        """
        t = grid_thw[0].item() // temp_merge_size
        h = grid_thw[1].item() // spatial_merge_size
        w = grid_thw[2].item() // spatial_merge_size

        seq_len = t * h * w
        pos_w = torch.arange(start_position, start_position + w, device=device).repeat(h * t)
        pos_h = torch.arange(start_position, start_position + h, device=device).repeat_interleave(w * t)
        pos_t = torch.full((seq_len,), start_position, device=device, dtype=torch.long) * time_interval
        return torch.stack([pos_t, pos_h, pos_w], dim=0)

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Compute the 3-axis M-RoPE position ids for an entire sequence.

        Math / algorithm
        ----------------
        A single running counter :math:`p` walks left-to-right through the
        token stream. Each maximal contiguous group of tokens of a single
        modality (text = 0, image = 1, video = 2) contributes positions:

        * **Text group of length L**:
          :math:`(p, p+1, \dots, p+L-1)` on all 3 axes; advance ``p += L``.
        * **Vision group (image/video)**:
          use ``get_vision_position_ids(p, grid)`` for per-token ``(t, h, w)``;
          advance ``p += max(H, W) / M`` (the longer spatial side of the
          grid, divided by the merge size). Using the longer side makes
          subsequent text positions larger than the maximum vision coordinate,
          so the text still sees strictly increasing positions.

        Returns
        -------
        * ``position_ids`` of shape ``(3, B, S)`` — the M-RoPE ids.
        * ``mrope_deltas`` of shape ``(B, 1)`` — the difference between the
          final position and the raw token count. Used during KV-cache
          decoding to extend the counter past the prefill correctly.
        """
        # Videos with multiple frames are expanded since Qwen3.5 inserts
        # timestamp tokens between frames (each frame is treated separately).
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw = video_grid_thw.clone()
            video_grid_thw[:, 0] = 1
        spatial_merge_size = self.config.vision_config.spatial_merge_size

        mrope_deltas: list[torch.Tensor] = []
        position_ids = torch.zeros(
            3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
        )
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }

        for batch_idx, current_input_ids in enumerate(input_ids):
            token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
                token_type = token_type[attention_mask[batch_idx].bool()]

            # Group consecutive tokens by modality: [(modality, start, end), ...]
            groups = []
            for key, g in itertools.groupby(enumerate(token_type.tolist()), lambda x: x[1]):
                g = list(g)
                groups.append((key, g[0][0], g[-1][0] + 1))

            current_pos = 0
            chunks: list[torch.Tensor] = []
            for modality, start, end in groups:
                if modality == 0:
                    # text: simple arithmetic progression on all 3 axes.
                    text_len = end - start
                    chunks.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len
                else:
                    # vision: (t, h, w) coordinates starting at current_pos.
                    grid_thw = next(grid_iters[modality])
                    vision_pos = self.get_vision_position_ids(
                        current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device
                    )
                    chunks.append(vision_pos)
                    # advance by max(H, W) // merge_size so subsequent text
                    # positions are strictly larger than any vision coord.
                    current_pos += int(max(grid_thw[1], grid_thw[2])) // spatial_merge_size
            llm_pos = torch.cat(chunks, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_pos.to(position_ids.device)
            else:
                position_ids[:, batch_idx] = llm_pos.to(position_ids.device)
            mrope_deltas.append(llm_pos.max() + 1 - len(current_input_ids))

        mrope_deltas_t = torch.tensor(mrope_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_deltas_t

    def compute_3d_position_ids(
        self,
        input_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[HybridCache] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        r"""Return 3-axis position ids for the current forward.

        Prefill (``past_len == 0``)
            Build positions from scratch via :meth:`get_rope_index`. Store
            ``rope_deltas = max_pos + 1 - num_tokens`` on self so we can
            **extend** them during decoding without re-walking the prefix.

        Decode (``past_len > 0``)
            Next-token position = ``past_len + offset + rope_deltas``. Since
            text positions are monotonic and we advanced by ``max(H, W)/M``
            for vision groups, this stays consistent with the prefill.
        """
        past_len = 0 if past_key_values is None else past_key_values.get_seq_length()
        has_mm = image_grid_thw is not None or video_grid_thw is not None
        if has_mm and mm_token_type_ids is None and input_ids is not None:
            raise ValueError("mm_token_type_ids required when image_grid_thw/video_grid_thw is provided")

        can_compute = input_ids is not None and mm_token_type_ids is not None and has_mm
        if can_compute and (self.rope_deltas is None or past_len == 0):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
            return position_ids
        if self.rope_deltas is not None and (past_len > 0 or input_ids is None):
            batch_size, seq_length, _ = inputs_embeds.shape
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
                position_ids = position_ids.view(1, batch_size, -1).repeat(3, 1, 1).to(inputs_embeds.device)
            else:
                position_ids = torch.arange(past_len, past_len + seq_length)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1).to(inputs_embeds.device)
            delta = self.rope_deltas.repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)
            return position_ids + delta.to(device=inputs_embeds.device)
        return None

    # -- forward -------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        use_cache: bool = False,
    ) -> dict:
        r"""End-to-end multimodal forward pass.

        Stages
        ------
        1. Embed ``input_ids`` → ``inputs_embeds``.
        2. If ``pixel_values`` is given: run the vision tower, splice image
           features into ``inputs_embeds`` at placeholder positions.
        3. If ``pixel_values_videos``: same, for videos.
        4. Build 3-axis M-RoPE position ids (new for prefill, shifted for decode).
        5. Run the text decoder over the spliced embeddings.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            # Vision → list of per-image embeddings → flat concat in order.
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_embeds)
            # Replace placeholder embeddings in-place at masked positions.
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        out = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        out["rope_deltas"] = self.rope_deltas
        return out


class Qwen3_5ForConditionalGeneration(nn.Module):
    r"""Multimodal causal-LM head.

    Adds the language-model softmax projection on top of :class:`Qwen3_5Model`:

    .. math::
        \text{logits} = W_{\text{lm}}\; y,

    with ``y`` the final hidden state from the decoder. Optionally ties
    :math:`W_{\text{lm}}` to the input embedding (see :class:`Qwen3_5ForCausalLM`).
    """

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        use_cache: bool = False,
    ) -> dict:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=use_cache,
        )
        hidden_states = out["last_hidden_state"]
        logits = self.lm_head(hidden_states)
        return {
            "logits": logits,
            "last_hidden_state": hidden_states,
            "past_key_values": out["past_key_values"],
            "rope_deltas": out.get("rope_deltas"),
        }
