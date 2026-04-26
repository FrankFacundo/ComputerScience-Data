r"""Qwen2VL-style image preprocessor, pure torch + PIL + torchvision.

Produces the same ``(pixel_values, image_grid_thw)`` that the transformers
``Qwen2VLImageProcessorFast`` produces, with values matching bit-for-bit when
the underlying torchvision/PIL is identical.

Pipeline (per image)
--------------------
1. **PIL → tensor** (``C, H, W``, uint8) via ``torchvision.functional``.
2. **Smart resize**: round ``(H, W)`` to multiples of ``factor = patch · merge``
   while clipping total pixel count into ``[min_pixels, max_pixels]`` and
   preserving the aspect ratio up to a rounding step. See :func:`smart_resize`.
3. **Rescale**: ``x ← x / 255`` so values lie in ``[0, 1]``.
4. **Normalize**: ``x ← (x - mean) / std`` (channel-wise).
5. **Patchify**: reshape into an ordered sequence of
   ``(C · t_patch · patch · patch)`` flat patches using a permutation that
   groups 2×2 "merger blocks" together so the vision model's spatial merger
   can simply reshape.

Output
------
* ``pixel_values`` of shape ``(N_total, C · t_patch · patch · patch)``
* ``image_grid_thw`` of shape ``(num_images, 3)`` — rows
  ``(grid_t, grid_h, grid_w)``.

Reference
---------
Qwen2-VL: native-resolution image preprocessing,
https://arxiv.org/abs/2409.12191
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torchvision.transforms.v2 import functional as tvF


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    r"""Round ``(h, w)`` to multiples of ``factor`` while staying within a pixel budget.

    Math
    ----
    Let :math:`f = \text{factor}` (``patch_size · merge_size``) and initial
    target :math:`\bar h = f \cdot \text{round}(h/f),\; \bar w = f \cdot \text{round}(w/f)`.

    * If :math:`\bar h \bar w > \text{max\_pixels}`: shrink both sides by
      :math:`\beta = \sqrt{h\,w / \text{max\_pixels}}` then floor to multiples of ``f``.
    * If :math:`\bar h \bar w < \text{min\_pixels}`: enlarge by
      :math:`\beta = \sqrt{\text{min\_pixels} / (h\,w)}` then ceil to multiples of ``f``.

    The choice of ``floor`` for max-pixel and ``ceil`` for min-pixel is
    deliberate: both hit the target budget from the **inside**, preserving
    the budget invariants after rounding.

    Aspect ratio guard
        If the source ratio exceeds 200:1, raise — beyond this the bilinear
        pos-embed interpolation becomes numerically unstable.

    Reference: mirrors ``transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize``.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class Qwen2VLImageProcessor:
    r"""Pure-python equivalent of ``Qwen2VLImageProcessorFast`` for Qwen3.5.

    The output packs patches into a single long sequence along the first axis,
    and returns ``image_grid_thw`` of shape ``(num_images, 3)`` with each row
    ``(grid_t, grid_h, grid_w)``.

    The per-patch layout is chosen so the vision model's ``PatchEmbed`` (a
    Conv3d with kernel=stride) and the 2x2 spatial merger can both operate
    with plain reshapes — see :meth:`_process_one`.
    """

    def __init__(
        self,
        *,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        min_pixels: int = 56 * 56,
        max_pixels: int = 28 * 28 * 1280,
        rescale_factor: float = 1.0 / 255.0,
        do_resize: bool = True,
        do_rescale: bool = True,
        do_normalize: bool = True,
        do_convert_rgb: bool = True,
    ):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.image_mean = tuple(image_mean)
        self.image_std = tuple(image_std)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.rescale_factor = rescale_factor
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.do_convert_rgb = do_convert_rgb

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "Qwen2VLImageProcessor":
        with open(Path(model_dir) / "preprocessor_config.json") as f:
            cfg = json.load(f)
        size = cfg.get("size", {})
        kwargs = dict(
            patch_size=int(cfg.get("patch_size", 14)),
            temporal_patch_size=int(cfg.get("temporal_patch_size", 2)),
            merge_size=int(cfg.get("merge_size", 2)),
            image_mean=tuple(cfg.get("image_mean", (0.5, 0.5, 0.5))),
            image_std=tuple(cfg.get("image_std", (0.5, 0.5, 0.5))),
            min_pixels=int(size.get("shortest_edge", cfg.get("min_pixels", 56 * 56))),
            max_pixels=int(size.get("longest_edge", cfg.get("max_pixels", 28 * 28 * 1280))),
        )
        return cls(**kwargs)

    # -- public ---------------------------------------------------------

    def __call__(
        self, images: Image.Image | Iterable[Image.Image], *, return_tensors: str = "pt"
    ) -> dict:
        return self.preprocess(images, return_tensors=return_tensors)

    def preprocess(
        self, images: Image.Image | Iterable[Image.Image], *, return_tensors: str = "pt"
    ) -> dict:
        """Preprocess one or more images into the packed vision input format.

        Produces a single ``pixel_values`` tensor concatenating every image's
        patches along the first axis, and an ``image_grid_thw`` tensor listing
        each image's ``(grid_t, grid_h, grid_w)``.
        """
        if isinstance(images, Image.Image):
            images = [images]

        flat_patch_list: list[torch.Tensor] = []
        grids: list[tuple[int, int, int]] = []

        for image in images:
            patches, grid = self._process_one(image)
            flat_patch_list.append(patches)
            grids.append(grid)

        pixel_values = torch.cat(flat_patch_list, dim=0)
        image_grid_thw = torch.tensor(grids, dtype=torch.long)
        return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}

    @property
    def size(self) -> dict[str, int]:
        return {"shortest_edge": self.min_pixels, "longest_edge": self.max_pixels}

    # -- implementation -------------------------------------------------

    def _process_one(self, image: Image.Image) -> tuple[torch.Tensor, tuple[int, int, int]]:
        r"""Preprocess one image end-to-end.

        Math / pipeline
        ---------------
        1. **RGB convert** (optional) and uint8 CHW tensor.
        2. **Smart resize** to :math:`(\bar h, \bar w)` with
           :math:`\bar h, \bar w \equiv 0 \pmod{p \cdot m}` (``p = patch_size``,
           ``m = merge_size``).
        3. **Rescale** :math:`x \leftarrow x / 255`.
        4. **Normalize** :math:`x \leftarrow (x - \mu) / \sigma` per channel.
        5. **Temporal pad** to a multiple of ``temporal_patch_size`` by
           repeating the last frame (needed for the Conv3d kernel).
        6. **Reshape into patch/merge structure**:

           .. math::
               (B,\, \text{grid\_t} \cdot t_p,\, C,\, H,\, W)
               \longrightarrow
               (B,\, \text{grid\_t},\, t_p,\, C,
                \tfrac{H}{p m},\, m,\, p,\,
                \tfrac{W}{p m},\, m,\, p)

           then permute to
           ``(B, grid_t, H/pm, W/pm, m, m, C, t_p, p, p)`` and finally flatten
           to ``(grid_t · grid_h · grid_w,  C · t_p · p · p)`` so the vision
           tower sees one flat patch per row, grouped by 2×2 merger blocks.

        Returns
        -------
        ``(patches, (grid_t, grid_h, grid_w))`` for this image.
        """
        if self.do_convert_rgb and image.mode != "RGB":
            image = image.convert("RGB")

        # PIL -> (C, H, W) uint8 tensor
        tensor = tvF.pil_to_tensor(image)

        height, width = tensor.shape[-2:]
        factor = self.patch_size * self.merge_size
        if self.do_resize:
            h_bar, w_bar = smart_resize(
                height, width, factor, self.min_pixels, self.max_pixels
            )
            tensor = tvF.resize(
                tensor,
                size=[h_bar, w_bar],
                interpolation=tvF.InterpolationMode.BICUBIC,
                antialias=True,
            )
            height, width = h_bar, w_bar

        # dtype to float, rescale, normalize
        x = tensor.to(torch.float32)
        if self.do_rescale:
            x = x * self.rescale_factor
        if self.do_normalize:
            mean = torch.tensor(self.image_mean, dtype=x.dtype).view(-1, 1, 1)
            std = torch.tensor(self.image_std, dtype=x.dtype).view(-1, 1, 1)
            x = (x - mean) / std

        # (C, H, W) -> (1, 1, C, H, W), pad temporal to `temporal_patch_size`
        x = x.unsqueeze(0).unsqueeze(0)  # batch=1, time=1
        if x.shape[1] % self.temporal_patch_size != 0:
            repeats = x[:, -1:].repeat(1, self.temporal_patch_size - 1, 1, 1, 1)
            x = torch.cat([x, repeats], dim=1)
        batch_size, grid_t, channel = x.shape[:3]
        grid_t = grid_t // self.temporal_patch_size
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size

        x = x.view(
            batch_size,
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        flat = x.reshape(
            batch_size,
            grid_t * grid_h * grid_w,
            channel * self.temporal_patch_size * self.patch_size * self.patch_size,
        )
        # drop the singleton batch dim so the output concatenates across images
        return flat.squeeze(0), (grid_t, grid_h, grid_w)
