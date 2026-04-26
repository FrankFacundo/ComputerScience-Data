r"""Multimodal Qwen3.6 (vision + text) and conditional-generation wrapper.

Glues the vision tower (:class:`Qwen3_6VisionModel`) and the text decoder
(:class:`Qwen3_6TextModel`) together. The late-fusion recipe is identical
to Qwen3.5: vision embeddings are *masked-scattered* into the text
embedding stream at every ``<|image_pad|>`` placeholder, then 3-axis
M-RoPE positions are built so vision tokens get ``(t, h, w)`` coordinates
and text tokens get a running scalar.

The MoE routing happens inside each text decoder layer (see
:class:`Qwen3_6SparseMoeBlock`); from this file's perspective nothing
changes vs. the dense Qwen3.5 model.

References — see qwen3_5_torch/model.py and the Qwen2-VL paper.
"""

from __future__ import annotations

import itertools
from typing import Optional

import torch
import torch.nn as nn

from .cache import HybridCache
from .config import Qwen3_6Config
from .decoder import Qwen3_6TextModel
from .vision import Qwen3_6VisionModel


class Qwen3_6Model(nn.Module):
    r"""Top-level multimodal model: vision tower + hybrid text decoder."""

    def __init__(self, config: Qwen3_6Config):
        super().__init__()
        self.config = config
        self.visual = Qwen3_6VisionModel(config.vision_config)
        self.language_model = Qwen3_6TextModel(config.text_config)
        self.rope_deltas: Optional[torch.Tensor] = None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.language_model.set_input_embeddings(value)

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> list[torch.Tensor]:
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds = vision_output["pooler_output"]
        split_sizes = (
            image_grid_thw.prod(-1) // (self.visual.spatial_merge_size ** 2)
        ).tolist()
        return list(torch.split(image_embeds, split_sizes))

    def get_video_features(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
    ) -> list[torch.Tensor]:
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def get_placeholder_mask(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
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

            groups = []
            for key, g in itertools.groupby(enumerate(token_type.tolist()), lambda x: x[1]):
                g = list(g)
                groups.append((key, g[0][0], g[-1][0] + 1))

            current_pos = 0
            chunks: list[torch.Tensor] = []
            for modality, start, end in groups:
                if modality == 0:
                    text_len = end - start
                    chunks.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len
                else:
                    grid_thw = next(grid_iters[modality])
                    vision_pos = self.get_vision_position_ids(
                        current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device
                    )
                    chunks.append(vision_pos)
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
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_embeds)
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


class Qwen3_6ForConditionalGeneration(nn.Module):
    r"""Multimodal causal-LM head on top of :class:`Qwen3_6Model`."""

    def __init__(self, config: Qwen3_6Config):
        super().__init__()
        self.config = config
        self.model = Qwen3_6Model(config)
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
