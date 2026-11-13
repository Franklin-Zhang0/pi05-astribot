"""Dense visual encoder selection for PRISM-VLA."""

from __future__ import annotations

from typing import Any

from torch import nn

from prism_vla.perception.encoders.cradio import (
    CRADIO_SUMMARY_TOKEN_PARAMETER,
    CRADIO_SO400M_CLS_TOKENS,
    CRADIO_SO400M_REGISTER_TOKENS,
    CRADIO_SO400M_SUMMARY_TOKEN_SHAPE,
    CRADIO_TUNING_MODES,
    CRADIOEncoder,
    configure_cradio_positional_chunking,
    cradio_stochastic_module_names,
)
from prism_vla.perception.encoders.dino import (
    DINOv3Encoder,
    load_aligned_dino_backbone,
)
from prism_vla.perception.encoders.semantic import QwenVLVisualEncoder


CRADIO_SO400M_MODEL_NAME = "nvidia/C-RADIOv4-SO400M"
CRADIO_SO400M_REVISION = "c0457f5dc26ca145f954cd4fc5bb6114e5705ad8"


def normalize_visual_encoder_type(value: str | None) -> str:
    encoder_type = str(value or "dino").strip().lower().replace("-", "_")
    aliases = {
        "dinov3": "dino",
        "dino_v3": "dino",
        "c_radio": "cradio",
        "c_radio_v4": "cradio",
        "cradio_v4": "cradio",
    }
    encoder_type = aliases.get(encoder_type, encoder_type)
    if encoder_type not in {"dino", "cradio"}:
        raise ValueError(
            f"Unsupported visual_encoder_type={value!r}; expected 'dino' or 'cradio'."
        )
    return encoder_type


def default_visual_encoder_model_name(encoder_type: str) -> str:
    encoder_type = normalize_visual_encoder_type(encoder_type)
    if encoder_type == "cradio":
        return CRADIO_SO400M_MODEL_NAME
    return "vit_base_patch16_dinov3"


def default_visual_encoder_image_size(encoder_type: str) -> int:
    return 256 if normalize_visual_encoder_type(encoder_type) == "cradio" else 224


def default_visual_encoder_revision(encoder_type: str) -> str:
    return CRADIO_SO400M_REVISION if normalize_visual_encoder_type(encoder_type) == "cradio" else ""


def normalize_visual_encoder_dtype(value: str | None) -> str:
    dtype = str(value or "bf16").strip().lower()
    aliases = {
        "bfloat16": "bf16",
        "float32": "fp32",
        "float": "fp32",
    }
    dtype = aliases.get(dtype, dtype)
    if dtype not in {"bf16", "fp32"}:
        raise ValueError(
            f"Unsupported visual_encoder_dtype={value!r}; expected 'bf16' or 'fp32'."
        )
    return dtype


def expected_visual_feature_dim(encoder_type: str, model_name: str) -> int | None:
    encoder_type = normalize_visual_encoder_type(encoder_type)
    if encoder_type == "cradio":
        if str(model_name).rstrip("/") != CRADIO_SO400M_MODEL_NAME:
            raise ValueError(
                "This branch currently supports only "
                f"{CRADIO_SO400M_MODEL_NAME!r} for visual_encoder_type='cradio'; "
                f"got {model_name!r}."
            )
        return 1152
    return None


def infer_visual_patch_size(encoder_type: str, model_name: str) -> int:
    encoder_type = normalize_visual_encoder_type(encoder_type)
    if encoder_type == "cradio":
        return 16
    lowered = str(model_name).lower()
    if "patch14" in lowered:
        return 14
    if "patch16" in lowered:
        return 16
    raise ValueError(
        f"Cannot infer DINO patch size from model name {model_name!r}; "
        "expected a name containing patch14 or patch16."
    )


def infer_visual_num_patches(
    encoder_type: str,
    model_name: str,
    image_size: int,
    num_views: int,
) -> int:
    patch_size = infer_visual_patch_size(encoder_type, model_name)
    if image_size % patch_size != 0:
        raise ValueError(
            f"visual_encoder_image_size={image_size} is not divisible by "
            f"patch_size={patch_size}."
        )
    if normalize_visual_encoder_type(encoder_type) == "cradio" and image_size < 256:
        raise ValueError(
            "C-RADIOv4 requires visual_encoder_image_size >= 256; "
            f"got {image_size}."
        )
    return (image_size // patch_size) ** 2 * max(1, int(num_views))


def build_visual_encoder(
    encoder_type: str,
    *,
    model_name: str,
    image_size: int,
    pretrained: bool = True,
    trainable: bool = False,
    revision: str = "",
    summary_pooling: str = "mean",
    **kwargs: Any,
) -> nn.Module:
    """Build a dense visual encoder with the common CLS/spatial API."""
    encoder_type = normalize_visual_encoder_type(encoder_type)
    pairs = kwargs.pop("cradio_reentrant_checkpoint_pairs", False)
    if type(pairs) is not bool or (pairs and encoder_type != "cradio"):
        raise ValueError("cradio_reentrant_checkpoint_pairs requires a bool and a C-RADIO encoder")
    if encoder_type == "cradio":
        tuning_mode = (
            str(kwargs.get("tuning_mode", "full") or "full")
            .strip()
            .lower()
            .replace("-", "_")
        )
        if trainable and tuning_mode not in CRADIO_TUNING_MODES:
            raise ValueError(
                "Trainable C-RADIO supports only full finetuning or summary-token tuning; "
                f"got tuning_mode={tuning_mode!r}."
            )
        return CRADIOEncoder(
            model_name=model_name,
            image_size=image_size,
            pretrained=pretrained,
            trainable=trainable,
            tuning_mode=tuning_mode,
            gradient_checkpointing=bool(kwargs.get("gradient_checkpointing", False)),
            cradio_reentrant_checkpoint_pairs=pairs,
            revision=revision,
            summary_pooling=summary_pooling,
            dynamic_resolution=bool(kwargs.get("dynamic_resolution", False)),
        )
    if revision:
        raise ValueError(
            "visual_encoder_revision is not supported for timm DINO encoders; "
            "leave it empty."
        )
    return DINOv3Encoder(
        model_name=model_name,
        image_size=image_size,
        pretrained=pretrained,
        trainable=trainable,
        **kwargs,
    )


__all__ = [
    "CRADIO_SO400M_MODEL_NAME",
    "CRADIO_SO400M_REVISION",
    "CRADIO_SUMMARY_TOKEN_PARAMETER",
    "CRADIO_SO400M_CLS_TOKENS",
    "CRADIO_SO400M_REGISTER_TOKENS",
    "CRADIO_SO400M_SUMMARY_TOKEN_SHAPE",
    "CRADIO_TUNING_MODES",
    "CRADIOEncoder",
    "configure_cradio_positional_chunking",
    "cradio_stochastic_module_names",
    "DINOv3Encoder",
    "QwenVLVisualEncoder",
    "build_visual_encoder",
    "default_visual_encoder_image_size",
    "default_visual_encoder_model_name",
    "default_visual_encoder_revision",
    "expected_visual_feature_dim",
    "infer_visual_num_patches",
    "infer_visual_patch_size",
    "load_aligned_dino_backbone",
    "normalize_visual_encoder_type",
    "normalize_visual_encoder_dtype",
]
