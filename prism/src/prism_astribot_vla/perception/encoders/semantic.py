"""Qwen-VL semantic visual encoder."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class QwenVLVisualEncoder(nn.Module):
    """Frozen Qwen-VL visual tower returning global and spatial tokens."""

    def __init__(
        self,
        model_name: str,
        *,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        trust_remote_code: bool = True,
    ):
        super().__init__()
        if not model_name:
            raise ValueError(
                "semantic_model_name must be set when the semantic visual encoder is enabled."
            )

        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        vision_cfg = getattr(self.model.config, "vision_config", None)
        text_cfg = getattr(self.model.config, "text_config", None)
        self.embed_dim = int(
            getattr(vision_cfg, "out_hidden_size", 0)
            or getattr(text_cfg, "hidden_size", 0)
            or getattr(self.model.config, "hidden_size", 0)
        )
        if self.embed_dim <= 0:
            raise ValueError("Could not infer Qwen-VL visual output dimension.")

    def _to_pil_list(
        self,
        images: list | np.ndarray | Image.Image | torch.Tensor,
    ) -> list[Image.Image]:
        if isinstance(images, Image.Image):
            return [images.convert("RGB")]
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                images = images[None]
            if images.dtype != np.uint8:
                images = (
                    images * 255 if images.max() <= 1.5 else images
                ).clip(0, 255).astype(np.uint8)
            return [Image.fromarray(img).convert("RGB") for img in images]
        if isinstance(images, torch.Tensor):
            if images.ndim == 3:
                images = images.unsqueeze(0)
            if images.ndim != 4:
                raise ValueError(
                    f"Expected image tensor rank 3/4, got shape={tuple(images.shape)}"
                )
            images_cpu = images.detach().cpu()
            if images_cpu.shape[1] == 3 and images_cpu.shape[-1] != 3:
                images_cpu = images_cpu.permute(0, 2, 3, 1)
            if images_cpu.dtype != torch.uint8:
                if float(images_cpu.max()) <= 1.5:
                    images_cpu = images_cpu * 255.0
                images_cpu = images_cpu.clamp(0, 255).to(torch.uint8)
            return [Image.fromarray(img.numpy()).convert("RGB") for img in images_cpu]
        return [
            img.convert("RGB")
            if isinstance(img, Image.Image)
            else Image.fromarray(img).convert("RGB")
            for img in images
        ]

    @torch.no_grad()
    def forward(
        self,
        images: list | np.ndarray | Image.Image | torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = next(self.model.parameters()).device

        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            raise RuntimeError("Qwen-VL processor does not expose an image_processor.")
        encoded = image_processor(
            images=self._to_pil_list(images),
            return_tensors="pt",
        )
        pixel_values = encoded["pixel_values"].to(device)
        image_grid_thw = encoded["image_grid_thw"].to(device)
        outputs = self.model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        token_pieces = outputs.pooler_output
        if isinstance(token_pieces, torch.Tensor):
            split_sizes = (
                image_grid_thw.prod(-1) // self.model.visual.spatial_merge_size**2
            ).tolist()
            token_pieces = torch.split(token_pieces, split_sizes)

        token_list = list(token_pieces)
        token_lens = {int(tokens.shape[0]) for tokens in token_list}
        if len(token_lens) != 1:
            raise ValueError(
                "Qwen-VL visual encoder produced variable token counts in one batch. "
                "Use fixed-size images or add padding/masks before enabling this path."
            )
        visual_tokens = torch.stack(token_list, dim=0)
        return visual_tokens.mean(dim=1), visual_tokens

    def extract_features(
        self,
        images: list | np.ndarray | Image.Image | torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(images, device=device)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype
