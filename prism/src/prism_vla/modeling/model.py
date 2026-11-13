"""PRISM-VLA model wrapper.

The heavy training, inference, checkpoint, and runtime helper methods live in
neighbor modules so this file stays focused on model construction and the public
`PrismVLA` import path.
"""

from __future__ import annotations

import json
import torch
import torch.nn as nn

from prism_vla.data.datasets.ego_schema import (
    ego_action_token_dim,
    ego_action_tokens_per_step,
    ego_proprio_dim,
    normalize_ego_pose_representation,
    normalize_rotation_representation,
)
from prism_vla.modeling.backbone_layers import (
    apply_qwen_layer_skip,
    normalize_qwen_keep_layers,
)
from prism_vla.modeling.checkpointing import PrismVLACheckpointMixin
from prism_vla.modeling.config import (
    GLOBAL_QUERY_GENERATION_MODES,
    MULTI_QUERY_GENERATION_MODES,
    TASK_QUERY_GENERATION_MODES,
    PrismVLAConfig,
    validate_query_generation_config,
)
from prism_vla.modeling.forward import PrismVLAForwardMixin
from prism_vla.modeling.idm_backends import build_idm_backend
from prism_vla.modeling.inference import PrismVLAInferenceMixin
from prism_vla.modeling.instruction_encoder import (
    FrozenQwenInstructionEncoder,
    load_qwen_text_causal_lm,
)
from prism_vla.modeling.losses import normalize_regression_loss_type
from prism_vla.modeling.proprio import (
    build_proprio_encoder,
    initialize_proprio_encoder,
    normalize_proprio_encoder_type,
)
from prism_vla.modeling.qwen35_deltanet_patch import (
    convert_qwen35_deltanet_to_kda,
    normalize_linear_attention_variant,
    patch_qwen35_deltanet_chunk_cache,
)
from prism_vla.modeling.runtime import PrismVLARuntimeMixin
from prism_vla.modeling.direct_query_stream import PrismVLADirectQueryMixin
from prism_vla.modeling.attention_policy import attention_window_policy_description
from prism_vla.modeling.visual_fusion import (
    ResidualMLPFusion,
    normalize_visual_encoder_mode,
    uses_dino,
    uses_semantic,
)
from prism_vla.perception.encoders import (
    CRADIO_TUNING_MODES,
    build_visual_encoder,
    default_visual_encoder_image_size,
    default_visual_encoder_model_name,
    default_visual_encoder_revision,
    normalize_visual_encoder_dtype,
    normalize_visual_encoder_type,
)


def _normalize_dino_tuning_mode(
    value: str | None,
    *,
    visual_encoder_type: str,
    train_dino_backbone: bool,
    train_dino_cls_encoder: bool,
    use_frozen_dino_for_cls: bool,
) -> tuple[str, bool]:
    """Normalize tuning mode and resolve shared C-RADIO ownership."""
    encoder_type = normalize_visual_encoder_type(visual_encoder_type)
    tuning_mode = str(value or "full").strip().lower().replace("-", "_")
    allowed_modes = (
        set(CRADIO_TUNING_MODES)
        if encoder_type == "cradio"
        else {"full", "lora"}
    )
    if tuning_mode not in allowed_modes:
        raise ValueError(
            f"visual_encoder_type={encoder_type!r} does not support "
            f"dino_tuning_mode={tuning_mode!r}; expected one of "
            f"{sorted(allowed_modes)!r}."
        )

    shared = tuning_mode == "summary_tokens"
    if shared:
        invalid_reasons: list[str] = []
        if encoder_type != "cradio":
            invalid_reasons.append("visual_encoder_type must be 'cradio'")
        if not train_dino_cls_encoder:
            invalid_reasons.append("train_dino_cls_encoder must be true")
        if train_dino_backbone:
            invalid_reasons.append("train_dino_backbone must be false")
        if use_frozen_dino_for_cls:
            invalid_reasons.append("use_frozen_dino_for_cls must be false")
        if invalid_reasons:
            raise ValueError(
                "dino_tuning_mode='summary_tokens' requires one checkpoint-owned "
                "C-RADIO encoder for CLS/aux and patch features: "
                + "; ".join(invalid_reasons)
                + "."
            )
    return tuning_mode, shared


def _make_query_source_parameters(
    mode: str,
    *,
    num_global_query_tokens: int,
    num_task_query_tokens: int,
    qwen_dim: int,
) -> tuple[nn.Parameter | None, nn.Parameter | None]:
    global_tokens = (
        nn.Parameter(torch.empty(num_global_query_tokens, qwen_dim))
        if mode in GLOBAL_QUERY_GENERATION_MODES
        else None
    )
    task_tokens = (
        nn.Parameter(torch.empty(num_task_query_tokens, qwen_dim))
        if mode in TASK_QUERY_GENERATION_MODES
        else None
    )
    return global_tokens, task_tokens


class PrismVLA(
    PrismVLACheckpointMixin,
    PrismVLAForwardMixin,
    PrismVLAInferenceMixin,
    PrismVLADirectQueryMixin,
    PrismVLARuntimeMixin,
    nn.Module,
):
    """PRISM-VLA v2 with interleaved AR layout.

    Training sequence (standard causal mask):
      [INST(I)] [PATCH_0(P)] [CS_0 | CLS_0 | QS_0 | CS_1 | CLS_1 | QS_1 | ... | CS_{T-1} | CLS_{T-1} | QS_{T-1}]

    Each step t contributes the current compact token, current CLS token, and
    current query-state token. Prediction heads on the current block supervise
    the next compact / CLS targets.

    Total tokens: I + P + 3T.
    """

    # Hugging Face Trainer checks this during checkpoint resume. Define it on
    # the raw module so torch.compile wrappers can forward the lookup safely.
    _keys_to_ignore_on_save = None

    def __init__(self, config: PrismVLAConfig):
        super().__init__()
        self.config = config
        self.attention_window_policy = attention_window_policy_description(config)
        print('[attention-window] ' + json.dumps(self.attention_window_policy, sort_keys=True), flush=True)
        self.config.qwen_keep_layers = normalize_qwen_keep_layers(
            getattr(config, "qwen_keep_layers", ())
        )
        self.config.loss_type = normalize_regression_loss_type(config.loss_type)
        self.config.huber_delta = float(config.huber_delta)
        self.config.linear_attention_variant = normalize_linear_attention_variant(
            getattr(config, "linear_attention_variant", "gated_deltanet")
        )
        if self.config.huber_delta <= 0.0:
            raise ValueError(f"config.huber_delta must be positive, got {self.config.huber_delta}.")
        if bool(getattr(config, "blockwise_full_attention", False)):
            backend = str(
                getattr(config, "blockwise_full_attention_backend", "dense_sdpa")
            ).strip().lower()
            if backend != "dense_sdpa":
                raise ValueError(
                    "The SDPA branch requires "
                    "config.blockwise_full_attention_backend='dense_sdpa'."
                )
            if not bool(getattr(config, "sliding_window_attention", False)):
                raise ValueError(
                    "config.blockwise_full_attention=True requires sliding_window_attention=True."
                )
            if not bool(getattr(config, "pass_deltanet_state_between_chunks", True)):
                raise ValueError(
                    "config.blockwise_full_attention=True requires global DeltaNet state passing."
                )
            if int(getattr(config, "attention_window_timesteps", 0)) <= 0:
                raise ValueError(
                    "config.attention_window_timesteps must be positive for blockwise full attention."
                )
            if str(getattr(config, "attn_implementation", "sdpa")) != "sdpa":
                raise ValueError(
                    "blockwise_full_attention_backend='dense_sdpa' requires "
                    "config.attn_implementation='sdpa'."
                )
        self.config.w_sigreg = float(getattr(config, "w_sigreg", 0.0))
        self.config.sigreg_num_slices = int(getattr(config, "sigreg_num_slices", 256))
        self.config.sigreg_num_knots = int(getattr(config, "sigreg_num_knots", 17))
        self.config.sigreg_t_max = float(getattr(config, "sigreg_t_max", 3.0))
        self.config.sigreg_seed = int(getattr(config, "sigreg_seed", 0))
        if self.config.w_sigreg < 0.0:
            raise ValueError(f"config.w_sigreg must be non-negative, got {self.config.w_sigreg}.")
        if self.config.sigreg_num_slices <= 0:
            raise ValueError(
                "config.sigreg_num_slices must be positive, "
                f"got {self.config.sigreg_num_slices}."
            )
        if self.config.sigreg_num_knots < 2:
            raise ValueError(
                "config.sigreg_num_knots must be at least 2, "
                f"got {self.config.sigreg_num_knots}."
            )
        if self.config.sigreg_t_max <= 0.0:
            raise ValueError(
                f"config.sigreg_t_max must be positive, got {self.config.sigreg_t_max}."
            )
        if self.config.sigreg_seed < 0:
            raise ValueError(
                f"config.sigreg_seed must be non-negative, got {self.config.sigreg_seed}."
            )
        self.dino_dim = config.dino_dim
        self.max_inst_len = config.max_inst_len
        self.num_patches = config.num_patches
        self.max_fast_len = config.max_fast_len
        self.fast_vocab_size = config.fast_vocab_size
        self.action_horizon = config.action_horizon
        action_tokens_per_step = int(
            getattr(config, "action_tokens_per_step", 1)
        )
        if config.flow_output_projection in {"per_action_type", "per_effector_type"} and action_tokens_per_step != 5:
            raise ValueError("per_action_type requires the separated five-token action layout.")
        if action_tokens_per_step <= 0 or self.action_horizon % action_tokens_per_step != 0:
            raise ValueError(
                "config.action_horizon must be divisible by positive "
                "config.action_tokens_per_step: "
                f"{self.action_horizon} % {action_tokens_per_step}."
            )
        action_step_horizon = int(
            getattr(
                config,
                "action_step_horizon",
                self.action_horizon // action_tokens_per_step,
            )
        )
        if action_step_horizon * action_tokens_per_step != self.action_horizon:
            raise ValueError(
                "Action horizon contract mismatch: "
                f"{action_step_horizon} physical steps * {action_tokens_per_step} "
                f"tokens != {self.action_horizon}."
            )
        pose_representation = normalize_ego_pose_representation(
            getattr(config, "pose_representation", "standard")
        )
        rotation_representation = normalize_rotation_representation(
            getattr(config, "rotation_representation", "euler")
        )
        if bool(getattr(config, "ego_layout_id", "")):
            ego_tokens_per_step = ego_action_tokens_per_step(action_tokens_per_step)
            expected_action_dim = ego_action_token_dim(
                rotation_representation,
                ego_tokens_per_step,
            )
            expected_proprio_dim = (
                15
                if getattr(config, "robot_proprio_mode", "ego_tokens")
                == "native_absolute_state_joint_v1"
                else 28 if getattr(config, "robot_proprio_mode", "ego_tokens")
                == "robodojo_absolute_state_joint_v1"
                else ego_proprio_dim(rotation_representation, ego_tokens_per_step)
            )
            if int(config.action_dim) != expected_action_dim or (
                bool(config.use_proprio)
                and int(config.proprio_dim) != expected_proprio_dim
            ):
                raise ValueError(
                    "Resolved ego model shape mismatch: "
                    f"pose={pose_representation}, rotation={rotation_representation}, "
                    f"tokens_per_step={ego_tokens_per_step}, expected action_dim="
                    f"{expected_action_dim}, expected proprio_dim={expected_proprio_dim}, "
                    f"got action_dim={config.action_dim}, proprio_dim={config.proprio_dim}."
                )
        self.visual_encoder_mode = normalize_visual_encoder_mode(config.visual_encoder_mode)
        self.use_dino_visual = uses_dino(self.visual_encoder_mode)
        self.use_semantic_visual = uses_semantic(self.visual_encoder_mode)
        self.visual_encoder_type = normalize_visual_encoder_type(
            getattr(config, "visual_encoder_type", "dino")
        )
        config.visual_encoder_type = self.visual_encoder_type
        config.visual_encoder_model_name = str(
            getattr(config, "visual_encoder_model_name", "")
            or (
                config.dino_model_name
                if self.visual_encoder_type == "dino"
                else default_visual_encoder_model_name(self.visual_encoder_type)
            )
        )
        config.visual_encoder_image_size = int(
            getattr(config, "visual_encoder_image_size", 0)
            or (
                config.dino_image_size
                if self.visual_encoder_type == "dino"
                else default_visual_encoder_image_size(self.visual_encoder_type)
            )
        )
        config.visual_encoder_revision = str(
            getattr(config, "visual_encoder_revision", "")
            or default_visual_encoder_revision(self.visual_encoder_type)
        )
        config.visual_encoder_summary_pooling = str(
            getattr(config, "visual_encoder_summary_pooling", "mean") or "mean"
        ).strip().lower()
        if config.visual_encoder_summary_pooling not in {"mean", "first"}:
            raise ValueError(
                "visual_encoder_summary_pooling must be 'mean' or 'first', "
                f"got {config.visual_encoder_summary_pooling!r}."
            )
        config.visual_encoder_dtype = normalize_visual_encoder_dtype(
            getattr(config, "visual_encoder_dtype", "bf16")
        )
        config.visual_encoder_preprocess_backend = str(
            getattr(config, "visual_encoder_preprocess_backend", "timm_tensor")
            or "timm_tensor"
        ).strip().lower()
        if config.visual_encoder_preprocess_backend not in {"timm_tensor", "timm_pil"}:
            raise ValueError(
                "visual_encoder_preprocess_backend must be 'timm_tensor' or "
                f"'timm_pil', got {config.visual_encoder_preprocess_backend!r}."
            )
        raw_eval_image_hw = getattr(config, "visual_encoder_eval_image_hw", ())
        if raw_eval_image_hw:
            if len(raw_eval_image_hw) != 2:
                raise ValueError(
                    "visual_encoder_eval_image_hw must contain exactly H/W, got "
                    f"{raw_eval_image_hw!r}."
                )
            eval_height, eval_width = (
                int(raw_eval_image_hw[0]),
                int(raw_eval_image_hw[1]),
            )
            if eval_height <= 0 or eval_width <= 0:
                raise ValueError(
                    "visual_encoder_eval_image_hw must be positive, got "
                    f"{(eval_height, eval_width)}."
                )
            config.visual_encoder_eval_image_hw = (eval_height, eval_width)
        else:
            config.visual_encoder_eval_image_hw = ()
        self.train_dino_backbone = bool(getattr(config, "train_dino_backbone", False))
        self.train_dino_cls_encoder = bool(
            getattr(config, "train_dino_cls_encoder", False)
        )
        self.use_frozen_dino_for_cls = bool(getattr(config, "use_frozen_dino_for_cls", False))
        (
            config.dino_tuning_mode,
            self.share_dino_cls_encoder_for_patches,
        ) = _normalize_dino_tuning_mode(
            getattr(config, "dino_tuning_mode", "full"),
            visual_encoder_type=self.visual_encoder_type,
            train_dino_backbone=self.train_dino_backbone,
            train_dino_cls_encoder=self.train_dino_cls_encoder,
            use_frozen_dino_for_cls=self.use_frozen_dino_for_cls,
        )
        if self.train_dino_backbone and not self.use_dino_visual:
            raise ValueError("train_dino_backbone=True requires a DINO visual stream.")
        if self.train_dino_cls_encoder and not self.use_dino_visual:
            raise ValueError("train_dino_cls_encoder=True requires a dense visual stream.")
        if self.train_dino_backbone and self.visual_encoder_type != "dino":
            raise ValueError(
                "train_dino_backbone=True is only supported with "
                "visual_encoder_type='dino'."
            )
        if self.use_frozen_dino_for_cls and not self.train_dino_backbone:
            raise ValueError("use_frozen_dino_for_cls=True requires train_dino_backbone=True.")
        if self.use_frozen_dino_for_cls and self.train_dino_cls_encoder:
            raise ValueError(
                "use_frozen_dino_for_cls and train_dino_cls_encoder are mutually exclusive."
            )
        self.dino_encoder = (
            build_visual_encoder(
                self.visual_encoder_type,
                model_name=config.visual_encoder_model_name,
                image_size=config.visual_encoder_image_size,
                pretrained=True,
                trainable=True,
                revision=config.visual_encoder_revision,
                summary_pooling=config.visual_encoder_summary_pooling,
                tuning_mode=str(config.dino_tuning_mode),
                gradient_checkpointing=bool(config.dino_gradient_checkpointing),
                cradio_reentrant_checkpoint_pairs=config.cradio_reentrant_checkpoint_pairs,
                dynamic_resolution=bool(config.visual_encoder_dynamic_resolution),
                lora_r=int(config.dino_lora_r),
                lora_alpha=int(config.dino_lora_alpha),
                lora_dropout=float(config.dino_lora_dropout),
                lora_target_modules=tuple(config.dino_lora_target_modules),
            )
            if self.train_dino_backbone
            else None
        )
        self.dino_cls_encoder = (
            build_visual_encoder(
                self.visual_encoder_type,
                model_name=config.visual_encoder_model_name,
                image_size=config.visual_encoder_image_size,
                pretrained=True,
                trainable=self.train_dino_cls_encoder,
                revision=config.visual_encoder_revision,
                summary_pooling=config.visual_encoder_summary_pooling,
                tuning_mode=str(config.dino_tuning_mode),
                gradient_checkpointing=bool(config.dino_gradient_checkpointing),
                cradio_reentrant_checkpoint_pairs=config.cradio_reentrant_checkpoint_pairs,
                dynamic_resolution=bool(config.visual_encoder_dynamic_resolution),
                lora_r=int(config.dino_lora_r),
                lora_alpha=int(config.dino_lora_alpha),
                lora_dropout=float(config.dino_lora_dropout),
                lora_target_modules=tuple(config.dino_lora_target_modules),
            )
            if self.use_frozen_dino_for_cls or self.train_dino_cls_encoder
            else None
        )
        if self.dino_cls_encoder is not None and not self.train_dino_cls_encoder:
            self.dino_cls_encoder.eval()
            for param in self.dino_cls_encoder.parameters():
                param.requires_grad = False
        if self.share_dino_cls_encoder_for_patches and self.dino_cls_encoder is None:
            raise RuntimeError(
                "summary_tokens tuning did not construct the checkpoint-owned "
                "C-RADIO CLS/patch encoder."
            )
        self.compact_query_mode = str(config.compact_query_mode).lower()
        self.compact_token_layout = config.compact_token_layout
        direct_query = getattr(config, "query_design", "legacy") == "direct_query"
        self.direct_query_layout = None
        if direct_query:
            from prism_vla.modeling.direct_query import DirectQueryLayout
            self.direct_query_layout = DirectQueryLayout(
                config.direct_query_global_tokens_per_camera,
                config.direct_query_task_tokens_per_camera,
                config.direct_query_generated_tokens_per_camera,
            )
        if self.compact_query_mode not in {"qwen", "native_dino"}:
            raise ValueError(
                "config.compact_query_mode must be 'qwen' or 'native_dino', "
                f"got {config.compact_query_mode!r}."
            )
        self.num_query_tokens = int(config.num_query_tokens)
        if self.num_query_tokens <= 0:
            raise ValueError(f"config.num_query_tokens must be positive, got {self.num_query_tokens}.")
        self.compact_output_tokens = int(getattr(config, "compact_output_tokens", 1))
        if self.compact_output_tokens <= 0:
            raise ValueError(
                f"config.compact_output_tokens must be positive, got {self.compact_output_tokens}."
            )
        if self.compact_output_tokens > self.num_query_tokens:
            raise ValueError(
                "config.compact_output_tokens cannot exceed num_query_tokens: "
                f"{self.compact_output_tokens} > {self.num_query_tokens}."
            )
        self.query_generation_mode = str(
            getattr(config, "query_generation_mode", "cls")
        ).strip().lower()
        self.num_global_query_tokens = int(
            getattr(config, "num_global_query_tokens", 0)
        )
        self.num_task_query_tokens = int(
            getattr(config, "num_task_query_tokens", 0)
        )
        self.num_generated_query_tokens = int(
            getattr(config, "num_generated_query_tokens", 1)
        )
        self.use_cls_tokens = bool(getattr(config, "use_cls_tokens", True))
        if direct_query:
            self.num_global_query_tokens = sum(self.direct_query_layout.global_counts)
            self.num_task_query_tokens = sum(self.direct_query_layout.task_counts)
            self.num_generated_query_tokens = self.direct_query_layout.num_generated
            # These modes describe seed-bank availability only. Direct mode has
            # its own layout and bypasses the legacy source-expansion contract.
            self.query_generation_mode = ("global_task_cls" if self.num_task_query_tokens
                else "global_cls" if self.num_global_query_tokens else "cls")
            source_tokens = self.direct_query_layout.num_compacts
        else:
            self.query_generation_mode, source_tokens = validate_query_generation_config(
                self.query_generation_mode,
                num_global_query_tokens=self.num_global_query_tokens,
                num_task_query_tokens=self.num_task_query_tokens,
                compact_output_tokens=self.compact_output_tokens,
                use_cls_tokens=self.use_cls_tokens,
                num_generated_query_tokens=self.num_generated_query_tokens,
            )
        # Query sources (S), cameras (C), and per-camera compact slots (M) are independent.
        self.query_source_count = source_tokens
        from prism_vla.modeling.camera_token_budget import camera_token_counts
        self.camera_compact_tokens_per_camera = getattr(config, "camera_compact_tokens_per_camera", 1)
        self.camera_compact_token_counts = (tuple(g + t + r for g, t, r in zip(
            self.direct_query_layout.global_counts, self.direct_query_layout.task_counts,
            self.direct_query_layout.generated_counts)) if direct_query else camera_token_counts(
            self.camera_compact_tokens_per_camera, int(config.num_dino_cameras)))
        if direct_query:
            self.camera_compact_tokens_per_camera = self.camera_compact_token_counts
        self.compact_tokens_per_step = (
            sum(self.camera_compact_token_counts)
            if self.compact_token_layout == "camera" else source_tokens
        )
        self.config.query_generation_mode = self.query_generation_mode
        self.config.num_generated_query_tokens = self.num_generated_query_tokens
        self.cross_attn_depth = int(config.cross_attn_depth)
        if self.cross_attn_depth <= 0:
            raise ValueError(f"config.cross_attn_depth must be positive, got {self.cross_attn_depth}.")
        if self.compact_query_mode == "native_dino" and not self.use_dino_visual:
            raise ValueError("compact_query_mode='native_dino' requires DINO visual tokens.")
        if self.config.w_sigreg > 0.0 and self.compact_query_mode != "native_dino":
            raise ValueError("w_sigreg > 0 requires compact_query_mode='native_dino'.")
        sigreg_query_tokens = (
            (self.compact_tokens_per_step if direct_query else config.camera_compact_queries_per_source * self.query_source_count)
            if self.compact_token_layout == "camera" else self.num_query_tokens
        )
        if self.config.w_sigreg > 0.0 and sigreg_query_tokens < 2:
            raise ValueError("w_sigreg > 0 requires at least two query tokens.")
        if (self.compact_query_mode == "native_dino" and self.compact_token_layout == "source"
                and config.dino_dim % int(config.native_dino_nhead) != 0):
            raise ValueError(
                f"dino_dim={config.dino_dim} must be divisible by native_dino_nhead={config.native_dino_nhead}."
            )
        legacy_cls_condition = (
            bool(getattr(config, "use_cls_condition_for_idm", True))
            and self.use_cls_tokens
        )
        self.use_cls_condition_for_idm = legacy_cls_condition
        self.use_current_cls_condition_for_idm = (
            bool(getattr(config, "use_current_cls_condition_for_idm", legacy_cls_condition))
            and self.use_cls_tokens
        )
        self.use_future_cls_condition_for_idm = (
            bool(getattr(config, "use_future_cls_condition_for_idm", legacy_cls_condition))
            and self.use_cls_tokens
        )
        self.use_current_compact_condition_for_idm = bool(
            getattr(config, "use_current_compact_condition_for_idm", True)
        )
        self.use_future_compact_condition_for_idm = bool(
            getattr(config, "use_future_compact_condition_for_idm", True)
        )
        compact_condition_tokens = int(self.compact_tokens_per_step)
        self.use_start_patch_visual_tokens = bool(
            getattr(config, "use_start_patch_visual_tokens", False)
        )
        self.dino_camera_fusion = str(getattr(config, "dino_camera_fusion", "concat")).lower()
        self.dino_camera_fusion_temperature = float(
            getattr(config, "dino_camera_fusion_temperature", 0.1)
        )
        if self.dino_camera_fusion_temperature <= 0.0:
            raise ValueError(
                "config.dino_camera_fusion_temperature must be positive, "
                f"got {self.dino_camera_fusion_temperature}."
            )
        if self.dino_camera_fusion not in {"concat", "weighted_average"}:
            raise ValueError(
                "config.dino_camera_fusion must be 'concat' or 'weighted_average', "
                f"got {config.dino_camera_fusion!r}."
            )
        self.num_dino_cameras = max(1, int(getattr(config, "num_dino_cameras", 1)))
        self.camera_compact_camera_ids = (
            tuple(config.camera_compact_camera_ids) or tuple(range(self.num_dino_cameras))
        ) if self.compact_token_layout == "camera" else ()
        if self.compact_token_layout == "camera" and (
            self.compact_query_mode != "native_dino" or self.visual_encoder_mode != "dino"
            or self.dino_camera_fusion != "concat"
        ):
            raise ValueError("Camera compact layout requires native_dino, DINO-only, unmerged camera patches")
        if self.dino_camera_fusion == "weighted_average":
            if self.compact_query_mode != "native_dino":
                raise ValueError("dino_camera_fusion='weighted_average' requires compact_query_mode='native_dino'.")
            if int(config.num_patches) % self.num_dino_cameras != 0:
                raise ValueError(
                    f"num_patches={config.num_patches} is incompatible with "
                    f"num_dino_cameras={self.num_dino_cameras}."
                )
            if self.num_query_tokens % self.num_dino_cameras != 0:
                raise ValueError(
                    f"num_query_tokens={self.num_query_tokens} must be divisible by "
                    f"num_dino_cameras={self.num_dino_cameras} for camera-weighted DINO fusion."
                )
        self.use_proprio = bool(config.use_proprio)
        if self.use_proprio and int(config.proprio_dim) <= 0:
            raise ValueError("config.proprio_dim must be positive when use_proprio=True.")
        self.proprio_token_placement = str(
            getattr(config, "proprio_token_placement", "condition")
        ).strip().lower().replace("-", "_")
        if self.proprio_token_placement not in {"condition", "action"}:
            raise ValueError(
                "config.proprio_token_placement must be 'condition' or 'action', "
                f"got {config.proprio_token_placement!r}."
            )
        if self.proprio_token_placement == "action":
            if not self.use_proprio:
                raise ValueError(
                    "config.proprio_token_placement='action' requires use_proprio=True."
                )
            flow_arch = str(config.flow_idm_arch).strip().lower().replace("-", "_")
            if config.idm_type != "flow_matching" or flow_arch not in {
                "temporal_transformer",
                "transformer",
            }:
                raise ValueError(
                    "config.proprio_token_placement='action' requires the temporal "
                    "flow-matching IDM."
                )
        self.config.proprio_token_placement = self.proprio_token_placement
        self.proprio_encoder_type = normalize_proprio_encoder_type(
            getattr(config, "proprio_encoder_type", "mlp")
        )
        self.config.proprio_encoder_type = self.proprio_encoder_type
        if self.use_semantic_visual and int(config.semantic_dim) <= 0:
            raise ValueError(
                "config.semantic_dim must be set when visual_encoder_mode is 'semantic' or 'both'. "
                "Stage 1 can infer it while loading Qwen-VL; Stage 2 needs it before model construction."
            )
        self._configure_training_attention_backends()

        # ------------------------------------------------------------------
        # Load Qwen3.5-4B backbone
        # ------------------------------------------------------------------
        use_kda = self.config.linear_attention_variant == "kda"
        if bool(getattr(config, "sliding_window_attention", False)) or use_kda:
            deltanet_patch_ready = patch_qwen35_deltanet_chunk_cache()
            if not deltanet_patch_ready:
                raise RuntimeError(
                    "The selected Qwen3.5 linear-attention runtime requires the PRISM "
                    "cached-chunk patch, but the installed Transformers version is incompatible."
                )

        self.qwen = load_qwen_text_causal_lm(
            config.qwen_model_name,
            attn_implementation=config.attn_implementation,
        )
        if config.main_stream_action_token:
            from prism_vla.modeling.act_observer import install_qwen_action_observer_cache_filters
            install_qwen_action_observer_cache_filters()
        self.qwen_layer_skip_info = apply_qwen_layer_skip(
            self.qwen,
            self.config.qwen_keep_layers,
        )
        self.config.qwen_original_num_hidden_layers = (
            self.qwen_layer_skip_info.original_num_layers
            if self.config.qwen_keep_layers
            else 0
        )
        if use_kda:
            converted_layers = convert_qwen35_deltanet_to_kda(
                self.qwen,
                gate_rank=int(getattr(config, "kda_gate_rank", 128)),
                chunk_size=int(getattr(config, "kda_chunk_size", 64)),
                disable_recompute=bool(
                    getattr(config, "kda_disable_recompute", False)
                ),
                safe_gate=bool(getattr(config, "kda_safe_gate", False)),
                lower_bound=float(getattr(config, "kda_lower_bound", -5.0)),
            )
            if converted_layers <= 0:
                raise RuntimeError(
                    "linear_attention_variant='kda' did not find any Qwen3.5 "
                    "Gated DeltaNet layers to convert."
                )
            print(
                "[KDA] converted "
                f"{converted_layers} Qwen3.5 linear-attention layers "
                f"with gate_rank={config.kda_gate_rank} "
                f"chunk_size={config.kda_chunk_size} "
                f"safe_gate={config.kda_safe_gate}",
                flush=True,
            )
        if config.qwen_saved_activation_cpu_offload_min_bytes:
            from prism_vla.modeling.qwen_activation_offload import validate_qwen_activation_offload_backbone

            validate_qwen_activation_offload_backbone(
                self.qwen.model, config.qwen_saved_activation_cpu_offload_min_bytes)
            print('[memory] Qwen saved-activation pinned-CPU storage: '
                  f'minimum_bytes={config.qwen_saved_activation_cpu_offload_min_bytes}; '
                  'rank>=3; blocking_D2H/nonblocking_H2D', flush=True)
        if config.qwen_mlp_gradient_checkpointing:
            from prism_vla.modeling.qwen_mlp_checkpoint import (
                install_qwen_mlp_checkpointing, validate_qwen_mlp_packing_backbone,
            )

            if config.qwen_mlp_pack_valid_tokens:
                if any(float(getattr(config, name, 0.0)) != 0.0 for name in (
                        'history_noise_std', 'history_token_noise_std', 'input_history_noise_std')):
                    raise ValueError('Qwen MLP packing does not support history-token input noise')
                if (getattr(self._assemble_main_stream_tokens, '__func__', None)
                        is not PrismVLARuntimeMixin._assemble_main_stream_tokens):
                    raise ValueError('Qwen MLP packing requires the native zero-preserving main-stream assembler')
                validate_qwen_mlp_packing_backbone(self.qwen.model)
            checkpointed_mlps = install_qwen_mlp_checkpointing(tuple(self.qwen.model.layers),
                pack_valid_tokens=config.qwen_mlp_pack_valid_tokens)
            print(f"[memory] Qwen stateless MLP checkpointing: {checkpointed_mlps}", flush=True)
        if self._uses_blockwise_full_attention() or self._uses_rolling_full_attention():
            text_config = getattr(self.qwen.config, "text_config", self.qwen.config)
            layer_types = tuple(getattr(text_config, "layer_types", ()))
            if "full_attention" not in layer_types or "linear_attention" not in layer_types:
                raise RuntimeError(
                    ("Rolling" if self._uses_rolling_full_attention() else "Blockwise")
                    + " full attention requires a hybrid Qwen3.5 backbone with both "
                    "full-attention and DeltaNet layers."
                )

        hf_cfg = self.qwen.config
        if hasattr(hf_cfg, "text_config") and hasattr(hf_cfg.text_config, "hidden_size"):
            self.qwen_dim = hf_cfg.text_config.hidden_size
        elif hasattr(hf_cfg, "hidden_size"):
            self.qwen_dim = hf_cfg.hidden_size
        else:
            self.qwen_dim = self.qwen.model.embed_tokens.weight.shape[1]
        if config.idm_type in {"act_patch", "patchwam"}:
            if config.action_patch_token_dim not in (0, self.qwen_dim):
                raise ValueError("action_patch_token_dim differs from the actual backbone width")
            config.action_patch_token_dim = int(self.qwen_dim)

        # A vector parameter stays in the ordinary AdamW/FP32 shard path.
        # Registering None preserves legacy state dictionaries and RNG draws.
        self.register_parameter("main_action_token", nn.Parameter(torch.zeros(self.qwen_dim))
            if config.main_stream_action_token and config.main_action_token_input == "learned" else None)
        self.register_parameter("main_action_step_embeddings",
            nn.Parameter(torch.empty(config.action_horizon // action_tokens_per_step, self.qwen_dim))
            if config.main_stream_action_token and config.idm_type in {"act_patch", "patchwam"} else None)
        if self.main_action_step_embeddings is not None:
            nn.init.normal_(self.main_action_step_embeddings, std=self.qwen_dim ** -0.5)
        self.action_cvae_posterior = None
        self.action_cvae_projection = None
        if config.main_action_token_input == "cvae":
            from prism_vla.modeling.action_latent import ActionChunkPosterior
            self.action_cvae_posterior = ActionChunkPosterior(
                action_dim=config.action_dim, horizon=config.action_horizon,
                proprio_dim=config.proprio_dim if config.use_proprio else 0,
                latent_dim=config.action_cvae_latent_dim,
                encoder_dim=config.action_cvae_encoder_dim,
                layers=config.action_cvae_encoder_layers, heads=config.action_cvae_encoder_heads)
            self.action_cvae_projection = nn.Linear(config.action_cvae_latent_dim, self.qwen_dim)
            # z=0 must still give a well-scaled seed in BF16 recurrent layers.
            nn.init.normal_(self.action_cvae_projection.bias, std=1.0)
            with torch.no_grad():
                self.action_cvae_projection.bias.div_(
                    self.action_cvae_projection.bias.float().square().mean().sqrt().clamp_min(1e-12))

        self.instruction_encoder = None
        if getattr(config, "instruction_encoder_model_name", None):
            self.instruction_encoder = FrozenQwenInstructionEncoder(
                config.instruction_encoder_model_name,
                self.qwen_dim,
                revision=config.instruction_encoder_revision,
                nhead=config.instruction_encoder_nhead,
                dim_feedforward=config.instruction_encoder_dim_feedforward,
                dropout=config.instruction_encoder_dropout,
                attn_implementation=config.attn_implementation,
                text_input_deepstack=config.text_input_deepstack,
                deepstack_layer_indices=tuple(dict.fromkeys(
                    source for _, source in config.text_input_deepstack_layer_map
                )),
            )
            if config.text_input_deepstack:
                from prism_vla.modeling.text_deepstack import install_text_deepstack_attention
                install_text_deepstack_attention(
                    self.qwen, self.instruction_encoder.encoder,
                    layer_mapping=config.text_input_deepstack_layer_map,
                )
                layers = self.qwen.model.layers
                geometry = tuple(
                    (*layers[target].self_attn.text_deepstack_adapter.source_geometry,
                     *layers[target].self_attn.text_deepstack_adapter.target_geometry)
                    for target, _ in config.text_input_deepstack_layer_map
                )
                if config.text_input_deepstack_geometry and config.text_input_deepstack_geometry != geometry:
                    raise ValueError("Text input deepstack checkpoint geometry differs from loaded models")
                config.text_input_deepstack_geometry = geometry
            # Instruction IDs belong to the encoder's tokenizer. The backbone
            # receives continuous embeddings, so its token table is inactive.
            self.qwen.get_input_embeddings().requires_grad_(False)

        self.use_cls_register_fusion = bool(
            getattr(config, "use_cls_register_fusion", False)
        )
        self.cls_register_nhead = int(getattr(config, "cls_register_nhead", 8))
        self.cls_register_gate_init = float(
            getattr(config, "cls_register_gate_init", 0.001)
        )
        if self.use_cls_register_fusion:
            if not self.use_dino_visual or not self.use_cls_tokens:
                raise ValueError(
                    "use_cls_register_fusion requires the DINO-style visual stream "
                    "and use_cls_tokens=true."
                )
            if self.cls_register_nhead <= 0 or self.qwen_dim % self.cls_register_nhead != 0:
                raise ValueError(
                    "qwen_dim must be divisible by cls_register_nhead; "
                    f"got qwen_dim={self.qwen_dim}, nhead={self.cls_register_nhead}."
                )

        self.qwen.config.use_cache = False

        if config.gradient_checkpointing:
            self.qwen.gradient_checkpointing_enable()

        # ------------------------------------------------------------------
        # Visual projector: dino_dim -> qwen_dim (for PATCH, CLS, CS embeddings)
        # ------------------------------------------------------------------
        self.visual_proj = nn.Linear(config.dino_dim, self.qwen_dim) if config.dino_dim != self.qwen_dim else nn.Identity()
        self.cls_register_query = (
            nn.Parameter(torch.empty(1, self.qwen_dim))
            if self.use_cls_register_fusion
            else None
        )
        self.cls_register_attn = (
            nn.MultiheadAttention(
                embed_dim=self.qwen_dim,
                num_heads=self.cls_register_nhead,
                batch_first=True,
                dropout=0.0,
            )
            if self.use_cls_register_fusion
            else None
        )
        self.cls_register_gate = (
            nn.Parameter(torch.empty(()))
            if self.use_cls_register_fusion
            else None
        )
        self.semantic_visual_proj = (
            nn.Linear(config.semantic_dim, self.qwen_dim)
            if self.use_semantic_visual and config.semantic_dim != self.qwen_dim
            else (nn.Identity() if self.use_semantic_visual else None)
        )
        self.proprio_encoder = (
            build_proprio_encoder(
                encoder_type=self.proprio_encoder_type,
                input_dim=int(config.proprio_dim),
                output_dim=self.qwen_dim,
                hidden_mult=int(config.proprio_hidden_mult),
                structured_hidden_dim=int(config.proprio_structured_hidden_dim),
                structured_num_heads=int(config.proprio_structured_nhead),
                structured_num_layers=int(config.proprio_structured_layers),
                structured_ff_mult=int(config.proprio_structured_ff_mult),
                structured_dropout=float(config.proprio_structured_dropout),
                action_tokens_per_step=action_tokens_per_step,
                rotation_representation=rotation_representation,
                proprio_delta_policy=getattr(
                    config,
                    "proprio_delta_policy",
                    None,
                ),
            )
            if self.use_proprio
            else None
        )

        if self.proprio_encoder is not None and config.proprio_encoder_by_frame:
            from prism_vla.modeling.proprio import FrameSpecificProprioEncoder
            self.proprio_encoder = FrameSpecificProprioEncoder(
                self.proprio_encoder, tuple(config.proprio_frame_names))

        # ------------------------------------------------------------------
        # Cross-attention in backbone dim for compact computation
        # ------------------------------------------------------------------
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.qwen_dim,
            num_heads=8,
            batch_first=True,
        )
        self.cross_attn_extra = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.qwen_dim,
                    num_heads=8,
                    batch_first=True,
                )
                for _ in range(max(0, self.cross_attn_depth - 1))
            ]
        )
        self.cross_attn_extra_norms = nn.ModuleList(
            [nn.LayerNorm(self.qwen_dim) for _ in range(max(0, self.cross_attn_depth - 1))]
        )
        self.semantic_cross_attn = (
            nn.MultiheadAttention(embed_dim=self.qwen_dim, num_heads=8, batch_first=True)
            if self.use_semantic_visual
            else None
        )
        self.semantic_cross_attn_extra = (
            nn.ModuleList(
                [
                    nn.MultiheadAttention(embed_dim=self.qwen_dim, num_heads=8, batch_first=True)
                    for _ in range(max(0, self.cross_attn_depth - 1))
                ]
            )
            if self.use_semantic_visual
            else None
        )
        self.semantic_cross_attn_extra_norms = (
            nn.ModuleList(
                [nn.LayerNorm(self.qwen_dim) for _ in range(max(0, self.cross_attn_depth - 1))]
            )
            if self.use_semantic_visual
            else None
        )
        self.dino_query_proj_head = (
            nn.Linear(self.qwen_dim, self.num_query_tokens * config.dino_dim)
            if self.compact_query_mode == "native_dino" and self.compact_token_layout == "source"
            else None
        )
        self.dino_native_cross_attn = (
            nn.MultiheadAttention(
                embed_dim=config.dino_dim,
                num_heads=int(config.native_dino_nhead),
                batch_first=True,
            )
            if self.compact_query_mode == "native_dino" and self.compact_token_layout == "source"
            else None
        )
        self.dino_native_cross_attn_extra = (
            nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        embed_dim=config.dino_dim,
                        num_heads=int(config.native_dino_nhead),
                        batch_first=True,
                    )
                    for _ in range(max(0, self.cross_attn_depth - 1))
                ]
            )
            if self.compact_query_mode == "native_dino" and self.compact_token_layout == "source"
            else None
        )
        self.dino_native_cross_attn_extra_norms = (
            nn.ModuleList(
                [nn.LayerNorm(config.dino_dim) for _ in range(max(0, self.cross_attn_depth - 1))]
            )
            if self.compact_query_mode == "native_dino" and self.compact_token_layout == "source"
            else None
        )
        self.dino_compact_out_proj = (
            nn.Linear(self.num_query_tokens * config.dino_dim, self.qwen_dim)
            if self.compact_query_mode == "native_dino" and self.compact_token_layout == "source"
            else None
        )
        self.dino_token_compact_out_proj = (
            nn.Linear(config.dino_dim, self.qwen_dim)
            if self.compact_query_mode == "native_dino" and self.compact_token_layout == "source"
            else None
        )
        dino_queries_per_camera = max(1, self.num_query_tokens // self.num_dino_cameras)
        if self.dino_camera_fusion == "weighted_average" and self.compact_output_tokens > dino_queries_per_camera:
            raise ValueError(
                "compact_output_tokens cannot exceed queries per camera when "
                "dino_camera_fusion='weighted_average': "
                f"{self.compact_output_tokens} > {dino_queries_per_camera}."
            )
        self.dino_camera_compact_out_proj = (
            nn.Linear(dino_queries_per_camera * config.dino_dim, self.qwen_dim)
            if self.dino_camera_fusion == "weighted_average"
            else None
        )
        self.dino_camera_weight_head = (
            nn.Linear(self.qwen_dim, self.num_dino_cameras)
            if self.dino_camera_fusion == "weighted_average"
            else None
        )
        self.camera_compact_resampler = None
        self.direct_query_resampler = None
        if direct_query:
            from prism_vla.modeling.camera_compact import resolve_camera_pool_num_heads
            from prism_vla.modeling.direct_query import DirectQueryResampler
            self.direct_query_resampler = DirectQueryResampler(
                self.direct_query_layout, qwen_dim=self.qwen_dim, vision_dim=config.dino_dim,
                num_heads=resolve_camera_pool_num_heads(
                    self.qwen_dim, config.camera_compact_pool_num_heads),
                depth=config.direct_query_depth,
                include_registers=config.direct_query_include_registers,
            )
        elif self.compact_token_layout == "camera":
            from prism_vla.modeling.camera_compact import CameraCompactResampler

            self.camera_compact_resampler = CameraCompactResampler(
                self.qwen_dim, config.dino_dim,
                queries_per_source=config.camera_compact_queries_per_source,
                depth=config.camera_compact_depth,
                num_heads=config.camera_compact_num_heads,
                pool_num_heads=config.camera_compact_pool_num_heads,
                ff_mult=config.camera_compact_ff_mult,
                camera_identity_count=config.camera_compact_identity_count,
                tokens_per_camera=self.camera_compact_tokens_per_camera,
            )
        self.cls_fusion = (
            ResidualMLPFusion(self.qwen_dim, hidden_mult=config.fusion_hidden_mult)
            if self.visual_encoder_mode == "both"
            else None
        )
        compact_fusion_inputs = (
            int(self.use_dino_visual)
            + int(self.use_semantic_visual)
        )
        self.compact_fusion = (
            ResidualMLPFusion(
                self.qwen_dim,
                hidden_mult=config.fusion_hidden_mult,
                num_inputs=compact_fusion_inputs,
                residual_pooling=config.compact_fusion_residual,
            )
            if compact_fusion_inputs > 1
            else None
        )

        # ------------------------------------------------------------------
        # Prediction heads (analogous to lm_head in LLMs)
        # Maps deep hidden states -> input embedding space for loss computation
        # ------------------------------------------------------------------
        self.cls_head = nn.Linear(self.qwen_dim, self.qwen_dim)
        self.compact_head = (
            nn.Linear(self.qwen_dim, self.compact_tokens_per_step * self.qwen_dim)
            if self.compact_token_layout == "source" else None
        )
        self.camera_compact_head = None
        if self.compact_token_layout == "camera":
            from prism_vla.modeling.camera_compact import CameraSpecificLinearPredictor

            self.camera_compact_head = CameraSpecificLinearPredictor(
                self.qwen_dim, self.camera_compact_camera_ids,
                tokens_per_camera=self.camera_compact_tokens_per_camera,
                **({"allow_zero_tokens": True} if direct_query else {}))
        self.compact_action_adapter = nn.Linear(self.qwen_dim, self.qwen_dim)
        self.query_proj_head = nn.Linear(self.qwen_dim, self.qwen_dim)
        if direct_query:
            self.global_query_tokens = (nn.Parameter(torch.empty(self.num_global_query_tokens, self.qwen_dim))
                if self.num_global_query_tokens else None)
            self.task_query_seed_tokens = (nn.Parameter(torch.empty(self.num_task_query_tokens, self.qwen_dim))
                if self.num_task_query_tokens else None)
            self.direct_generated_query_seed_embeddings = nn.Parameter(
                torch.empty(self.num_generated_query_tokens, self.qwen_dim))
        else:
            self.global_query_tokens, self.task_query_seed_tokens = _make_query_source_parameters(
                self.query_generation_mode,
                num_global_query_tokens=self.num_global_query_tokens,
                num_task_query_tokens=self.num_task_query_tokens,
                qwen_dim=self.qwen_dim,
            )
        self.no_cls_query_token = (
            nn.Parameter(torch.empty(self.qwen_dim)) if not self.use_cls_tokens else None
        )
        # Keep the legacy parameter layout; "embed" routes learned seeds to AdamW.
        if not direct_query and self.num_generated_query_tokens > 1:
            self.generated_query_seed_embeddings = nn.Parameter(
                torch.empty(self.num_generated_query_tokens - 1, self.qwen_dim)
            )

        # ------------------------------------------------------------------
        # Switchable IDM backend
        # ------------------------------------------------------------------
        backend_options = dict(
            cond_dim=self.qwen_dim,
            d_model=config.idm_d_model,
            nhead=config.idm_nhead,
            n_layers=config.idm_n_layers,
            ff_mult=config.idm_ff_mult,
            idm_attention_scale_limit=config.idm_attention_scale_limit,
            idm_attention_shift_limit=config.idm_attention_shift_limit,
            idm_qk_control_mode=config.idm_qk_control_mode,
            idm_qk_logit_limit=config.idm_qk_logit_limit,
            idm_qk_norm_temperature=config.idm_qk_norm_temperature,
            idm_condition_rms_limit=config.idm_condition_rms_limit,
            idm_attention_gate_limit=config.idm_attention_gate_limit,
            idm_mlp_scale_limit=config.idm_mlp_scale_limit,
            idm_mlp_shift_limit=config.idm_mlp_shift_limit,
            idm_mlp_gate_limit=config.idm_mlp_gate_limit,
            idm_final_scale_limit=config.idm_final_scale_limit,
            idm_final_shift_limit=config.idm_final_shift_limit,
            fast_vocab_size=config.fast_vocab_size,
            max_fast_len=config.max_fast_len,
            fast_ffn_gradient_checkpointing=config.fast_ffn_gradient_checkpointing,
            fast_layer_gradient_checkpointing=config.fast_layer_gradient_checkpointing,
            fast_broadcast_attention=config.fast_broadcast_attention,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            action_clip_abs_max=config.action_clip_abs_max,
            sampled_action_clip_abs_max=config.sampled_action_clip_abs_max,
            fast_pretrained_path=config.fast_pretrained_path,
            flow_idm_arch=config.flow_idm_arch,
            flow_conditioning=config.flow_conditioning,
            flow_output_projection=config.flow_output_projection,
            action_token_types=config.action_token_types,
            robot_action_layout=config.robot_action_layout,
            flow_cross_attn_gate_init=config.flow_cross_attn_gate_init,
            flow_cross_block_gradient_checkpointing=config.flow_cross_block_gradient_checkpointing,
            flow_qk_norm=config.flow_qk_norm,
            flow_time_embed_dim=config.flow_time_embed_dim,
            flow_sampling_steps=config.flow_sampling_steps,
            flow_sampling_cache_static_kv=config.flow_sampling_cache_static_kv,
            loss_type=config.loss_type,
            huber_delta=config.huber_delta,
            n_cond_tokens=1 if config.main_stream_action_token else (
                int(self.use_current_cls_condition_for_idm)
                + int(self.use_current_compact_condition_for_idm) * compact_condition_tokens
                + int(
                    self.use_proprio
                    and self.proprio_token_placement == "condition"
                    and config.action_conditioning != "predicted_compact"
                )
                + int(self.use_future_cls_condition_for_idm)
                + int(self.use_future_compact_condition_for_idm) * compact_condition_tokens
            ),
            proprio_as_action_token=(self.proprio_token_placement == "action"),
        )
        if config.idm_type == "act_patch":
            backend_options["action_tokens_per_step"] = action_tokens_per_step

        if config.action_heads:
            from prism_vla.modeling.multi_action_heads import MultiActionHeadIDM
            native_kv_config = None
            if any(head["type"] == "pi05" for head in config.action_heads):
                text_config = getattr(self.qwen.config, "text_config", self.qwen.config)
                layer_types = getattr(text_config, "layer_types", ())
                full_layers = sum(kind == "full_attention" for kind in layer_types)
                if not full_layers:
                    raise ValueError("pi05 requires native full-attention layers")
                head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)
                rope = getattr(text_config, "rope_parameters", {}) or {}
                if rope.get("rope_type", "default") != "default":
                    raise ValueError("pi05 currently requires the native default text RoPE")
                native_kv_config = dict(native_kv_layers=full_layers, native_head_dim=head_dim,
                    native_rotary_dim=int(head_dim * rope.get("partial_rotary_factor", getattr(text_config, "partial_rotary_factor", 1.0))),
                    native_rope_theta=rope.get("rope_theta", getattr(text_config, "rope_theta", 10000.0)))
            self.idm = MultiActionHeadIDM(config.action_heads,
                primary_head=config.action_primary_head, native_kv_config=native_kv_config,
                pi05_block_gradient_checkpointing=config.pi05_block_gradient_checkpointing,
                auxiliary_sampled_metrics=config.action_auxiliary_sampled_metrics, **backend_options)
            tokenizer_contracts = self.idm.tokenizer_contracts()
            if config.action_head_tokenizer_contracts is not None and config.action_head_tokenizer_contracts != tokenizer_contracts:
                raise ValueError("Action-head tokenizer differs from the checkpoint vocabulary/physical layout")
            config.action_head_tokenizer_contracts = tokenizer_contracts
        else:
            self.idm = build_idm_backend(config.idm_type, **backend_options)

        self.idm.weak_flow_sampling_steps = int(config.weak_flow_sampling_steps)
        if self.idm.weak_flow_sampling_steps < 1:
            raise ValueError("weak_flow_sampling_steps must be positive")

        self.camera_patch_reconstruction = None
        if (float(getattr(config, "w_patch_recon", 0.0)) > 0.0
                and tuple(getattr(config, "global_compact_indices", ()))):
            from prism_vla.modeling.camera_reconstruction import CameraPatchReconstruction

            self.camera_patch_reconstruction = CameraPatchReconstruction(
                self.qwen_dim, config.dino_dim, tuple(config.global_compact_indices),
                group_size=config.patch_recon_group_size,
                decoder_dim=config.patch_recon_decoder_dim,
                num_layers=config.patch_recon_decoder_layers,
                num_heads=config.patch_recon_decoder_num_heads,
                ff_mult=config.patch_recon_decoder_ff_mult,
            )

        self._init_new_modules()

        # Optional reference query from Stage 1 checkpoint (for compatibility).
        self.register_buffer("frozen_learnable_query", None)

        if config.stage1_checkpoint:
            self.load_stage1_weights(config.stage1_checkpoint)

        self._freeze_inactive_compact_modules()

        # Construct native additions after all historical initialization calls:
        # enabling this interface cannot consume RNG before an inherited tensor.
        if config.engram_native is not None:
            from prism_vla.modeling.engram_native import EngramJointDiT, EngramJointProprioEncoder
            extension = config.engram_native
            if (config.proprio_dim != 75 or not config.use_proprio
                    or config.action_conditioning != "predicted_compact"
                    or config.main_stream_action_token or not config.use_cls_tokens):
                raise ValueError("Native Engram requires the pretrained 75D predicted-compact proprio interface")
            if (config.action_dim != 15 or config.action_tokens_per_step != 5
                    or extension["physical_horizon"] != config.action_step_horizon):
                raise ValueError("Native Engram must preserve the source five15D-token physical horizon")
            heads = {head["name"]: head for head in config.action_heads}
            selected = heads.get(extension["source_head"])
            if selected is None or selected["type"] != "flow_matching":
                raise ValueError("Native Engram DiT requires an explicitly selected pretrained flow head")
            if config.flow_idm_arch != "temporal_transformer" or config.flow_conditioning != "cross_attn":
                raise ValueError("Native Engram DiT requires the pretrained temporal cross-attention trunk")
            source_head = self.idm.heads[extension["source_head"]]
            self.engram_action_head = EngramJointDiT(
                physical_horizon=extension["physical_horizon"], cond_dim=self.qwen_dim,
                d_model=source_head.hidden_dim, nhead=source_head.blocks[0].self_attn.num_heads,
                n_layers=len(source_head.blocks),
                ff_mult=source_head.blocks[0].mlp[0].out_features // source_head.hidden_dim,
                n_cond_tokens=source_head.n_cond_tokens,
                flow_time_embed_dim=source_head.flow_time_embed_dim,
                flow_sampling_steps=source_head.flow_sampling_steps,
                flow_cross_attn_gate_init=source_head.flow_cross_attn_gate_init,
                flow_cross_block_gradient_checkpointing=source_head.flow_cross_block_gradient_checkpointing,
                idm_qk_control_mode=source_head.idm_qk_control_mode,
                idm_qk_logit_limit=source_head.idm_qk_logit_limit,
                flow_qk_norm=source_head.flow_qk_norm,
                loss_type=config.loss_type, huber_delta=config.huber_delta,
            )
            self.engram_proprio_encoder = EngramJointProprioEncoder(
                self.qwen_dim, hidden_dim=self.qwen_dim * config.proprio_hidden_mult)
            self.idm.requires_grad_(False)
            self.proprio_encoder.requires_grad_(False)
            if extension.get("mode", "joint") == "joint_and_robot_eef":
                eef_spec = heads.get(extension["robot_eef_head"])
                expected_rep = dict(frame="robot", frame_anchor="chunk_start", target="chunk_offset")
                if (eef_spec is None or eef_spec["type"] != "flow_matching"
                        or any(eef_spec["action_representation"][key] != value for key, value in expected_rep.items())
                        or config.rotation_representation != "rotation_6d"):
                    raise ValueError("Hybrid Engram requires the pretrained robot/chunk-start offset head with rotation6D")
                robot_head = self.idm.heads[extension["robot_eef_head"]]
                robot_head.requires_grad_(True)
                # Raw Linker joint targets live in the native head. Only the
                # source EEF regressors receive observed pose supervision.
                for name, projection in robot_head.final_layer.linears.items():
                    if name != "eef":
                        projection.requires_grad_(False)

    def _clear_text_deepstack_memory(self) -> None:
        for name in ("_text_deepstack_memory", "_text_deepstack_input_ids", "_text_deepstack_attention_mask"):
            if hasattr(self, name):
                setattr(self, name, None)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._clear_text_deepstack_memory()
        return result

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self._clear_text_deepstack_memory()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _init_new_modules(self) -> None:
        if isinstance(self.visual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.visual_proj.weight)
            if self.visual_proj.bias is not None:
                nn.init.zeros_(self.visual_proj.bias)
        if isinstance(self.semantic_visual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.semantic_visual_proj.weight)
            if self.semantic_visual_proj.bias is not None:
                nn.init.zeros_(self.semantic_visual_proj.bias)
        if self.cls_register_query is not None:
            nn.init.normal_(
                self.cls_register_query,
                mean=0.0,
                std=self.qwen_dim ** -0.5,
            )
        if self.cls_register_gate is not None:
            nn.init.constant_(self.cls_register_gate, self.cls_register_gate_init)
        for module in (
            self.dino_query_proj_head,
            self.dino_compact_out_proj,
            self.dino_token_compact_out_proj,
            self.dino_camera_compact_out_proj,
            self.dino_camera_weight_head,
        ):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.proprio_encoder is not None:
            initialize_proprio_encoder(self.proprio_encoder)
        # Init prediction heads close to identity so they start as pass-through
        nn.init.eye_(self.cls_head.weight)
        nn.init.zeros_(self.cls_head.bias)
        if self.compact_head is not None:
            nn.init.zeros_(self.compact_head.weight)
            with torch.no_grad():
                eye = torch.eye(self.qwen_dim, device=self.compact_head.weight.device, dtype=self.compact_head.weight.dtype)
                if self.query_generation_mode in MULTI_QUERY_GENERATION_MODES:
                    for token_idx in range(self.compact_tokens_per_step):
                        start = token_idx * self.qwen_dim
                        self.compact_head.weight[start : start + self.qwen_dim].copy_(eye)
                else:
                    self.compact_head.weight[: self.qwen_dim].copy_(eye)
            nn.init.zeros_(self.compact_head.bias)
        if self.camera_compact_head is not None:
            for camera_id, count in zip(self.camera_compact_camera_ids, self.camera_compact_token_counts):
                if not count:
                    continue
                head = self.camera_compact_head.heads[str(camera_id)]
                if count == 1:
                    nn.init.eye_(head.weight)
                else:
                    with torch.no_grad():
                        head.weight.copy_(torch.eye(self.qwen_dim,
                            device=head.weight.device, dtype=head.weight.dtype).repeat(
                                count, 1))
                nn.init.zeros_(head.bias)
        nn.init.eye_(self.compact_action_adapter.weight)
        nn.init.zeros_(self.compact_action_adapter.bias)
        nn.init.eye_(self.query_proj_head.weight)
        nn.init.zeros_(self.query_proj_head.bias)
        if self.global_query_tokens is not None:
            nn.init.normal_(
                self.global_query_tokens,
                mean=0.0,
                std=self.qwen_dim ** -0.5,
            )
        if self.task_query_seed_tokens is not None:
            nn.init.normal_(
                self.task_query_seed_tokens,
                mean=0.0,
                std=self.qwen_dim ** -0.5,
            )
        if self.no_cls_query_token is not None:
            nn.init.normal_(self.no_cls_query_token, mean=0.0, std=self.qwen_dim ** -0.5)
        if getattr(self.config, "query_design", "legacy") == "direct_query":
            nn.init.normal_(self.direct_generated_query_seed_embeddings,
                mean=0.0, std=self.qwen_dim ** -0.5)
        elif self.num_generated_query_tokens > 1:
            nn.init.normal_(
                self.generated_query_seed_embeddings, mean=0.0, std=self.qwen_dim ** -0.5
            )
        if self.main_action_token is not None:
            nn.init.normal_(self.main_action_token, mean=0.0, std=self.qwen_dim ** -0.5)
            # Visual/CLS main-stream inputs have unit RMS. A tiny ACT seed
            # amplifies BF16 chunk/step differences in the pretrained decoder;
            # align only its initial scale, retaining a fully learned vector.
            with torch.no_grad():
                initial_rms = float(self.main_action_token.float().square().mean().sqrt())
                self.main_action_token.mul_(1.0 / max(initial_rms, 1e-12))

    @staticmethod
    def _freeze_module_parameters(module: nn.Module | None) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad_(False)

    def _freeze_inactive_compact_modules(self) -> None:
        if self.config.main_stream_action_token:
            # ACT is consumed directly; this old compact-only projection has
            # no loss path and must not enter the distributed optimizer.
            self._freeze_module_parameters(self.compact_action_adapter)
        if self.compact_token_layout == "camera":
            for module in (
                self.cross_attn, self.cross_attn_extra, self.cross_attn_extra_norms,
                self.dino_query_proj_head, self.dino_native_cross_attn,
                self.dino_native_cross_attn_extra, self.dino_native_cross_attn_extra_norms,
                self.dino_compact_out_proj, self.dino_token_compact_out_proj,
                self.dino_camera_compact_out_proj, self.dino_camera_weight_head,
            ):
                self._freeze_module_parameters(module)
            return
        if self.compact_query_mode == "native_dino":
            self._freeze_module_parameters(self.cross_attn)
            self._freeze_module_parameters(self.cross_attn_extra)
            self._freeze_module_parameters(self.cross_attn_extra_norms)
            if self.compact_output_tokens > 1:
                self._freeze_module_parameters(self.dino_compact_out_proj)
                self._freeze_module_parameters(self.dino_camera_compact_out_proj)
            elif self.dino_camera_fusion == "weighted_average":
                self._freeze_module_parameters(self.dino_compact_out_proj)
                self._freeze_module_parameters(self.dino_token_compact_out_proj)
            else:
                self._freeze_module_parameters(self.dino_token_compact_out_proj)
                self._freeze_module_parameters(self.dino_camera_compact_out_proj)
                self._freeze_module_parameters(self.dino_camera_weight_head)
        else:
            self._freeze_module_parameters(self.dino_query_proj_head)
            self._freeze_module_parameters(self.dino_native_cross_attn)
            self._freeze_module_parameters(self.dino_native_cross_attn_extra)
            self._freeze_module_parameters(self.dino_native_cross_attn_extra_norms)
            self._freeze_module_parameters(self.dino_compact_out_proj)
            self._freeze_module_parameters(self.dino_token_compact_out_proj)
            self._freeze_module_parameters(self.dino_camera_compact_out_proj)
            self._freeze_module_parameters(self.dino_camera_weight_head)

    def gradient_checkpointing_enable(self, **kwargs):
        self.qwen.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.qwen.gradient_checkpointing_disable()

    # ------------------------------------------------------------------
