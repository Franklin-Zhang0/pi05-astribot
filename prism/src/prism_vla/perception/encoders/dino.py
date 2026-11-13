"""DINOv3 vision encoder wrapper for PRISM-VLA.

Uses timm to load a DINOv3-ViT-B/16 model and extract CLS + patch features.
DINOv3 ViT-B/16 with 224x224 input produces:
  - 1 CLS token (768-d)
  - 4 register tokens (768-d), available through the auxiliary-token API
  - 196 patch tokens (14x14 grid, each 768-d)
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def load_aligned_dino_backbone(
    backbone: nn.Module,
    checkpoint_path: str | os.PathLike[str],
) -> int:
    """Load a qwen-dino-align backbone checkpoint with exact key/shape checks."""
    path = os.fspath(checkpoint_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"DINO initialization checkpoint does not exist: {path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model_trainable"), dict
    ):
        raise RuntimeError(
            "Aligned DINO checkpoint must contain a `model_trainable` state dictionary: "
            f"{path}"
        )

    prefix = "vision.backbone."
    source_state = {
        key[len(prefix) :]: value
        for key, value in checkpoint["model_trainable"].items()
        if isinstance(key, str) and key.startswith(prefix)
    }
    target_state = backbone.state_dict()
    missing = sorted(set(target_state) - set(source_state))
    unexpected = sorted(set(source_state) - set(target_state))
    shape_mismatches = sorted(
        key
        for key in set(target_state) & set(source_state)
        if tuple(target_state[key].shape) != tuple(source_state[key].shape)
    )
    if missing or unexpected or shape_mismatches:
        def summarize(keys: list[str]) -> list[str]:
            return keys[:8] + ([f"... ({len(keys)} total)"] if len(keys) > 8 else [])

        raise RuntimeError(
            "Aligned DINO checkpoint is not an exact backbone match: "
            f"missing={summarize(missing)}, "
            f"unexpected={summarize(unexpected)}, "
            f"shape_mismatches={summarize(shape_mismatches)}, path={path}"
        )

    backbone.load_state_dict(source_state, strict=True)
    return len(source_state)


class DINOv3Encoder(nn.Module):
    """DINOv3 encoder using timm.

    Extracts CLS and patch tokens from images using a pretrained DINOv3-ViT-B/16.
    Parameters are frozen by default; pass ``trainable=True`` for finetuning.

    Attributes:
        model: The timm ViT model.
        image_size: Expected input image resolution (default 224).
        embed_dim: Output feature dimension (768 for ViT-B).
        num_patches: Number of spatial patch tokens (196 for 224x224 with patch_size=16).
        num_registers: Number of register tokens in DINOv3 (typically 4).
    """

    # Known DINOv3 model names in timm and their register token counts.
    # DINOv3 (DINOv2 with registers) models use 4 register tokens.
    REGISTER_COUNTS = {
        "vit_base_patch16_dinov3": 4,
        "vit_base_patch16_dinov3.lvd1689m": 4,
        "vit_large_patch14_dinov3": 4,
        "vit_large_patch14_dinov3.lvd1689m": 4,
        "vit_giant_patch14_dinov3": 4,
        "vit_giant_patch14_dinov3.lvd1689m": 4,
    }

    def __init__(
        self,
        model_name: str = "vit_base_patch16_dinov3",
        image_size: int = 224,
        pretrained: bool = True,
        trainable: bool = False,
        tuning_mode: str = "full",
        gradient_checkpointing: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[list[str] | tuple[str, ...]] = None,
        dynamic_resolution: bool = False,
    ):
        super().__init__()
        import timm

        # Respect HF_ENDPOINT env var for mirror support
        hf_endpoint = os.environ.get("HF_ENDPOINT")
        if hf_endpoint:
            os.environ.setdefault("TIMM_HUB_URL", hf_endpoint)

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=image_size,
            num_classes=0,  # Remove classification head
            dynamic_img_size=bool(dynamic_resolution),
        )
        tuning_mode = str(tuning_mode or "full").lower()
        if tuning_mode not in {"full", "lora"}:
            raise ValueError(f"Unsupported DINO tuning_mode={tuning_mode!r}; expected 'full' or 'lora'.")
        if trainable and gradient_checkpointing and hasattr(self.model, "set_grad_checkpointing"):
            self.model.set_grad_checkpointing(True)

        self.lora_enabled = bool(trainable and tuning_mode == "lora")
        if self.lora_enabled:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as exc:
                raise ImportError(
                    "DINO LoRA tuning requires the `peft` package. "
                    "Install project dependencies with `uv sync` after adding peft."
                ) from exc

            target_modules = tuple(lora_target_modules or ("qkv", "attn.proj"))
            lora_config = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                target_modules=list(target_modules),
                lora_dropout=float(lora_dropout),
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.model = get_peft_model(self.model, lora_config)
        else:
            for param in self.model.parameters():
                param.requires_grad = trainable

        self.model.train(mode=trainable)

        self.image_size = image_size
        self.dynamic_resolution = bool(dynamic_resolution)
        self.embed_dim = self.model.embed_dim  # 768 for ViT-B
        self.num_registers = self.REGISTER_COUNTS.get(model_name, 4)

        # Compute number of patches from model config
        patch_size = self.model.patch_embed.patch_size
        if isinstance(patch_size, (tuple, list)):
            if len(patch_size) != 2:
                raise ValueError(f"Expected 2D DINO patch size, got {patch_size!r}.")
            patch_height, patch_width = (int(patch_size[0]), int(patch_size[1]))
        else:
            patch_height = patch_width = int(patch_size)
        self.patch_size = (patch_height, patch_width)
        # Keep the nominal square count stable for checkpoint/config contracts.
        self.num_patches = (image_size // patch_height) * (image_size // patch_width)

        # Get the data transform config from timm for preprocessing
        data_config = timm.data.resolve_model_data_config(self.model)
        # Override input_size to match our image_size (timm default may differ)
        data_config["input_size"] = (3, image_size, image_size)
        self._transform = timm.data.create_transform(**data_config, is_training=False)
        self.register_buffer(
            "_preprocess_mean",
            torch.tensor(data_config["mean"], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_preprocess_std",
            torch.tensor(data_config["std"], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward_with_aux_tokens(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract CLS, register, and patch features from preprocessed tensors.

        Args:
            images: Preprocessed image tensor of shape [B, 3, H, W], typically
                    normalized with ImageNet stats and resized to image_size.

        Returns:
            cls_tokens: [B, 768] CLS token features.
            aux_global_tokens: [B, 4, 768] DINO register token features.
            patch_tokens: [B, P, 768] Spatial patch features (registers removed).
        """
        expected_patches = self.num_patches_for_resolution(images.shape[-2:])

        # timm ViT forward_features returns [B, 1 + num_registers + num_patches, D]
        # where the first token is CLS, followed by register tokens, then patch tokens.
        features = self.model.forward_features(images)

        # Extract CLS token (always index 0)
        cls_tokens = features[:, 0]  # [B, 768]

        # Skip register tokens and extract patch tokens
        # Layout: [CLS, reg_1, ..., reg_R, patch_1, ..., patch_N]
        patch_start = 1 + self.num_registers
        aux_global_tokens = features[:, 1:patch_start]
        patch_tokens = features[:, patch_start:]  # [B, num_patches, 768]

        assert aux_global_tokens.shape[1] == self.num_registers, (
            f"Expected {self.num_registers} registers, got "
            f"{aux_global_tokens.shape[1]}."
        )
        expected_dim = int(getattr(self, "embed_dim", patch_tokens.shape[2]))
        if patch_tokens.shape[1] != expected_patches or patch_tokens.shape[2] != expected_dim:
            raise ValueError(
                "DINO output shape disagrees with the runtime encoder contract: "
                f"got {tuple(patch_tokens.shape)}, expected [B, {expected_patches}, "
                f"{expected_dim}]; num_registers={self.num_registers}."
            )

        return cls_tokens, aux_global_tokens, patch_tokens

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract CLS and patch features from preprocessed image tensors.

        The auxiliary register-token API is opt-in so existing callers retain
        the historical two-tensor return contract.
        """
        cls_tokens, _, patch_tokens = self.forward_with_aux_tokens(images)
        return cls_tokens, patch_tokens

    @staticmethod
    def _array_to_pil_list(array: np.ndarray) -> list[Image.Image]:
        if array.ndim == 3:
            array = array[None]
        if array.ndim != 4:
            raise ValueError(f"Expected image array rank-3/4, got shape={array.shape}")

        if array.shape[-1] == 3:
            hwc = array
        elif array.shape[1] == 3:
            hwc = np.transpose(array, (0, 2, 3, 1))
        else:
            raise ValueError(f"Expected RGB images, got shape={array.shape}")

        if hwc.dtype != np.uint8:
            min_value = float(np.nanmin(hwc))
            max_value = float(np.nanmax(hwc))
            if min_value < 0.0:
                raise ValueError("DINO preprocessing expects raw unnormalized images.")
            if max_value <= 1.5:
                hwc = hwc * 255.0
            hwc = np.clip(hwc, 0, 255).astype(np.uint8)

        return [Image.fromarray(np.ascontiguousarray(img)).convert("RGB") for img in hwc]

    @classmethod
    def _tensor_to_pil_list(cls, tensor: torch.Tensor) -> list[Image.Image]:
        tensor = tensor.detach().cpu()
        if tensor.ndim == 5:
            if tensor.shape[-1] == 3:
                tensor = tensor.reshape(-1, tensor.shape[-3], tensor.shape[-2], tensor.shape[-1])
            elif tensor.shape[2] == 3:
                tensor = tensor.reshape(-1, tensor.shape[2], tensor.shape[3], tensor.shape[4])
            else:
                raise ValueError(f"Expected RGB images, got shape={tuple(tensor.shape)}")
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(f"Expected image tensor rank-3/4/5, got shape={tuple(tensor.shape)}")

        if tensor.shape[-1] == 3:
            array = tensor.numpy()
        elif tensor.shape[1] == 3:
            array = tensor.permute(0, 2, 3, 1).numpy()
        else:
            raise ValueError(f"Expected RGB images, got shape={tuple(tensor.shape)}")
        return cls._array_to_pil_list(array)

    def _to_pil_list(self, images: list | np.ndarray | Image.Image | torch.Tensor) -> list[Image.Image]:
        if isinstance(images, Image.Image):
            return [images.convert("RGB")]
        if isinstance(images, np.ndarray):
            return self._array_to_pil_list(images)
        if isinstance(images, torch.Tensor):
            return self._tensor_to_pil_list(images)
        if isinstance(images, (list, tuple)):
            pil_images: list[Image.Image] = []
            for img in images:
                pil_images.extend(self._to_pil_list(img))
            return pil_images
        raise TypeError(f"Unsupported image input type: {type(images)!r}")

    def preprocess_images(
        self,
        images: list | np.ndarray | Image.Image | torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Apply the timm eval transform to raw RGB images.

        Accepts uint8 HWC / CHW images or float raw images in [0, 1] or
        [0, 255]. The returned tensor is normalized and ready for forward().
        """
        if device is None:
            device = next(self.model.parameters()).device

        pil_images = self._to_pil_list(images)
        if self.dynamic_resolution:
            arrays = [np.asarray(image.convert("RGB")) for image in pil_images]
            shapes = {tuple(array.shape) for array in arrays}
            if len(shapes) != 1:
                raise ValueError(
                    "Dynamic DINO preprocessing requires one H/W bucket per batch, "
                    f"got image shapes={sorted(shapes)!r}."
                )
            batch = torch.from_numpy(np.ascontiguousarray(np.stack(arrays, axis=0)))
            return self.preprocess_image_tensor(batch, device=device)

        processed = [self._transform(img) for img in pil_images]
        return torch.stack(processed).to(device=device)

    def preprocess_image_tensor(
        self,
        images: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Tensor-native version of the timm eval transform.

        This follows the DINO timm eval preprocessing config for batched raw RGB
        tensors: resize the shorter side to ``image_size`` with bicubic
        antialiasing, center crop, convert to [0, 1], then normalize.
        """
        if device is None:
            device = next(self.model.parameters()).device
        if images.ndim == 5:
            if images.shape[-1] == 3:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            elif images.shape[2] == 3:
                images = images.reshape(-1, images.shape[2], images.shape[3], images.shape[4])
            else:
                raise ValueError(f"Expected RGB images, got shape={tuple(images.shape)}")
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError(f"Expected image tensor rank-3/4/5, got shape={tuple(images.shape)}")

        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        elif images.shape[1] != 3:
            raise ValueError(f"Expected RGB images, got shape={tuple(images.shape)}")

        images = images.contiguous()
        if images.dtype == torch.uint8:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
        else:
            if images.dtype.is_floating_point:
                images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            else:
                images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            if torch.amin(images) < 0.0:
                raise ValueError("DINO preprocessing expects raw unnormalized images.")
            if torch.amax(images) > 1.5:
                images = images.div(255.0)

        _, _, height, width = images.shape
        if self.dynamic_resolution:
            self.num_patches_for_resolution((height, width))
        elif height != self.image_size or width != self.image_size:
            if height <= 0 or width <= 0:
                raise ValueError(f"Invalid image size: {(height, width)}")
            if height < width:
                resized_h = self.image_size
                resized_w = int(round(width * self.image_size / height))
            else:
                resized_w = self.image_size
                resized_h = int(round(height * self.image_size / width))
            images = F.interpolate(
                images,
                size=(resized_h, resized_w),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            top = max((resized_h - self.image_size) // 2, 0)
            left = max((resized_w - self.image_size) // 2, 0)
            images = images[:, :, top : top + self.image_size, left : left + self.image_size]

        mean = self._preprocess_mean.to(device=images.device, dtype=images.dtype)
        std = self._preprocess_std.to(device=images.device, dtype=images.dtype)
        return (images - mean) / std

    def num_patches_for_resolution(self, resolution: tuple[int, int]) -> int:
        """Validate a runtime H/W bucket and return its spatial token count."""
        if len(resolution) != 2:
            raise ValueError(f"Expected image resolution (H, W), got {resolution!r}.")
        height, width = (int(resolution[0]), int(resolution[1]))
        patch_size = getattr(self, "patch_size", None)
        if patch_size is None:
            # Preserve manually constructed legacy wrappers used by downstream
            # callers and tests. Normal construction always records patch_size.
            return int(self.num_patches)
        patch_height, patch_width = patch_size
        if height <= 0 or width <= 0:
            raise ValueError(f"DINO image dimensions must be positive, got {(height, width)}.")
        if height % patch_height != 0 or width % patch_width != 0:
            raise ValueError(
                "DINO dynamic resolution must be patch-aligned: "
                f"patch_size={patch_height}x{patch_width}, got HxW={height}x{width}."
            )
        return (height // patch_height) * (width // patch_width)

    def extract_features(
        self,
        images: list | np.ndarray | Image.Image | torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract features from PIL images, numpy arrays, or tensors.

        This is a convenience method that handles preprocessing. For batched
        training with already-preprocessed tensors, use forward() directly.

        Args:
            images: One of:
                - Single PIL Image
                - Single numpy array [H, W, 3] (uint8 or float)
                - List of PIL Images or numpy arrays
                - Preprocessed tensor [B, 3, H, W]
            device: Target device. If None, uses the model's device.

        Returns:
            cls_tokens: [B, 768] CLS token features.
            patch_tokens: [B, P, 768] Spatial patch features.
        """
        cls_tokens, _, patch_tokens = self.extract_features_with_aux_tokens(
            images,
            device=device,
        )
        return cls_tokens, patch_tokens

    def extract_features_with_aux_tokens(
        self,
        images: list | np.ndarray | Image.Image | torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract CLS, register, and patch features with preprocessing."""
        if device is None:
            device = next(self.model.parameters()).device

        if isinstance(images, torch.Tensor) and images.ndim == 4 and images.shape[1] == 3:
            # Already a batch of preprocessed CHW tensors.
            batch = images.to(device)
        else:
            batch = self.preprocess_images(images, device=device)
        return self.forward_with_aux_tokens(batch)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype
