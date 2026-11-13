"""Configuration for the PRISM-VLA model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math

from prism_vla.data.instruction_conditioning import (
    normalize_instruction_conditioning_config,
    normalize_instruction_format,
)

def instruction_tokenizer_pretrained_kwargs(config: object) -> dict:
    """Resolve the tokenizer belonging to the instruction consumer's vocabulary."""
    get = config.get if isinstance(config, dict) else lambda name, default=None: getattr(config, name, default)
    encoder_name = get("instruction_encoder_model_name")
    result = {"pretrained_model_name_or_path": encoder_name or get("qwen_model_name")}
    if encoder_name and get("instruction_encoder_revision") is not None:
        result["revision"] = get("instruction_encoder_revision")
    return result

def validate_robot_action_normalization(value: dict | None) -> dict | None:
    """Use the same serialized-statistics contract as the dataset and decoder."""
    if value is None:
        return None
    from prism_vla.data.datasets.robot_action_normalization import validate_action_normalization_metadata
    return validate_action_normalization_metadata(value)


QUERY_GENERATION_MODES = frozenset(
    {"cls", "global_cls", "global_task_cls", "task_cls"}
)
MULTI_QUERY_GENERATION_MODES = frozenset(
    {"global_cls", "global_task_cls", "task_cls"}
)
GLOBAL_QUERY_GENERATION_MODES = frozenset({"global_cls", "global_task_cls"})
TASK_QUERY_GENERATION_MODES = frozenset({"global_task_cls", "task_cls"})


IDM_RECOVERY_LIMIT_FIELDS = (
    "idm_condition_rms_limit",
    "idm_attention_gate_limit",
    "idm_mlp_scale_limit",
    "idm_mlp_shift_limit",
    "idm_mlp_gate_limit",
    "idm_final_scale_limit",
    "idm_final_shift_limit",
)

ACTION_CVAE_DEFAULTS = {
    "main_action_token_input": "learned",
    "action_cvae_latent_dim": 32,
    "action_cvae_encoder_dim": 256,
    "action_cvae_encoder_layers": 2,
    "action_cvae_encoder_heads": 4,
    "action_cvae_kl_weight": 1e-3,
    "action_cvae_inference_mode": "zero",
}


def action_cvae_config_contract(config):
    """Omit learned mode so pre-CVAE optimizer manifests stay identical."""
    get = config.get if isinstance(config, dict) else lambda key, default: getattr(config, key, default)
    if get("main_action_token_input", "learned") == "learned":
        return None
    return {key: get(key, default) for key, default in ACTION_CVAE_DEFAULTS.items()}


def action_patch_config_contract(config):
    """Bind per-step raw ACT geometry without extending legacy manifests."""
    get = config.get if isinstance(config, dict) else lambda key, default: getattr(config, key, default)
    if get("idm_type", "fast_ar") not in {"act_patch", "patchwam"}:
        return None
    return {
        "schema": "prism_act_patch_per_action_step_v1",
        "layout_version": get("action_patch_token_layout_version", 1),
        "action_horizon": get("action_horizon", 10),
        "action_dim": get("action_dim", 7),
        "action_tokens_per_step": get("action_tokens_per_step", 1),
        "token_dim": get("action_patch_token_dim", 0),
    }


def validate_idm_recovery_controls(
    *,
    idm_qk_control_mode: str = "none",
    idm_qk_logit_limit: float = 100.0,
    idm_qk_norm_temperature: float = 8.0,
    idm_condition_rms_limit: float | None = None,
    idm_attention_gate_limit: float | None = None,
    idm_mlp_scale_limit: float | None = None,
    idm_mlp_shift_limit: float | None = None,
    idm_mlp_gate_limit: float | None = None,
    idm_final_scale_limit: float | None = None,
    idm_final_shift_limit: float | None = None,
    idm_type: str = "flow_matching",
    flow_idm_arch: str = "temporal_transformer",
    flow_conditioning: str = "adaln_zero",
) -> dict[str, str | float | None]:
    """Validate legacy recovery metadata and the post-update MuonClip mode.

    Legacy limits remain readable for inference with old checkpoints. New
    training separately rejects those retired controls.
    """
    mode = str(idm_qk_control_mode).strip().lower()
    if mode not in {"none", "qk_norm", "logit_rescale", "muonclip"}:
        raise ValueError("idm_qk_control_mode must be none, qk_norm, logit_rescale or muonclip")
    result: dict[str, str | float | None] = {"idm_qk_control_mode": mode}
    values = locals()
    for name in ("idm_qk_logit_limit", "idm_qk_norm_temperature", *IDM_RECOVERY_LIMIT_FIELDS):
        value = values[name]
        if value is None and name in IDM_RECOVERY_LIMIT_FIELDS:
            result[name] = None
            continue
        try:
            bound = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be finite and positive, got {value!r}") from None
        if isinstance(value, bool) or not math.isfinite(bound) or bound <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")
        result[name] = bound
    has_limits = any(result[name] is not None for name in IDM_RECOVERY_LIMIT_FIELDS)
    enabled = mode != "none" or has_limits
    conditioning = str(flow_conditioning).strip().lower().replace("-", "_")
    if enabled and (
        idm_type != "flow_matching"
        or str(flow_idm_arch).strip().lower() not in {"temporal_transformer", "transformer"}
    ):
        raise ValueError("IDM Q/K and conditioning/modulation controls require a flow_matching temporal transformer.")
    if mode == "muonclip":
        if conditioning not in {"cross_attn", "cross_attention", "crossattn"} or has_limits:
            raise ValueError("Post-update MuonClip requires cross_attn conditioning without legacy modulation limits.")
    elif enabled and conditioning not in {"adaln_zero", "adaln0"}:
        raise ValueError(
            "IDM Q/K and conditioning/modulation controls require flow_matching "
            "with a temporal transformer and adaln_zero conditioning."
        )
    return result


def validate_idm_attention_limits(
    scale_limit: float | None,
    shift_limit: float | None,
    *,
    idm_type: str,
    flow_idm_arch: str,
    flow_conditioning: str,
) -> tuple[float | None, float | None]:
    """Validate optional limits on temporal AdaLN attention modulation only."""
    limits: list[float | None] = []
    for name, value in (
        ("idm_attention_scale_limit", scale_limit),
        ("idm_attention_shift_limit", shift_limit),
    ):
        if value is None:
            limits.append(None)
            continue
        try:
            bound = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be finite and positive or null, got {value!r}") from None
        if isinstance(value, bool) or not math.isfinite(bound) or bound <= 0:
            raise ValueError(f"{name} must be finite and positive or null, got {value!r}")
        limits.append(bound)
    if any(bound is not None for bound in limits) and (
        idm_type != "flow_matching"
        or str(flow_idm_arch).strip().lower() not in {"temporal_transformer", "transformer"}
        or str(flow_conditioning).strip().lower().replace("-", "_") not in {"adaln_zero", "adaln0"}
    ):
        raise ValueError(
            "IDM attention scale/shift limits require flow_matching with a temporal "
            "transformer and adaln_zero conditioning."
        )
    return limits[0], limits[1]


def validate_flow_output_projection(
    projection: str,
    *,
    action_token_types: tuple[str, ...],
    action_horizon: int | None = None,
    action_tokens_per_step: int | None = None,
    idm_type: str = "flow_matching",
    flow_idm_arch: str = "temporal_transformer",
) -> tuple[str, tuple[str, ...]]:
    """Validate terminal flow routing; token metadata describes one physical step."""
    projection = str(projection).strip().lower()
    if projection not in {"shared", "per_action_type", "per_effector_type"}:
        raise ValueError(
            "flow_output_projection must be 'shared', 'per_action_type' or 'per_effector_type', "
            f"got {projection!r}."
        )
    if not isinstance(action_token_types, (tuple, list)):
        raise ValueError("action_token_types must be a tuple or list of token types.")
    token_types = tuple(action_token_types)
    if any(token_type not in ("head", "eef", "hand") for token_type in token_types):
        raise ValueError("action_token_types supports only 'head', 'eef', and 'hand'.")
    if token_types:
        if action_tokens_per_step is not None and len(token_types) != action_tokens_per_step:
            raise ValueError("action_token_types must match action_tokens_per_step.")
        if action_horizon is not None and (
            action_horizon <= 0 or action_horizon % len(token_types) != 0
        ):
            raise ValueError("action_horizon must be divisible by len(action_token_types).")
    if projection in {"per_action_type", "per_effector_type"}:
        if idm_type != "flow_matching" or str(flow_idm_arch).lower() not in {
            "temporal_transformer", "transformer"
        }:
            raise ValueError(
                "flow_output_projection='per_action_type' requires a temporal "
                "flow-matching IDM; flat MLP and FAST backends are unsupported."
            )
        if not token_types:
            raise ValueError("per_action_type requires explicit action_token_types metadata.")
        if len(token_types) != 5 or (
            action_tokens_per_step is not None and action_tokens_per_step != 5
        ):
            raise ValueError(
                "per_action_type requires the separated five-token action layout; "
                "the packed three-token layout is unsupported."
            )
    if projection == "per_effector_type" and token_types != ("head", "eef", "hand", "eef", "hand"):
        raise ValueError("per_effector_type requires canonical head/left-eef/left-hand/right-eef/right-hand order")
    return projection, token_types


def validate_query_generation_config(
    mode: str,
    *,
    num_global_query_tokens: int,
    num_task_query_tokens: int,
    compact_output_tokens: int,
    use_cls_tokens: bool,
    num_generated_query_tokens: int = 1,
) -> tuple[str, int]:
    """Normalize query generation settings and return compact tokens per step."""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in QUERY_GENERATION_MODES:
        raise ValueError(
            "query_generation_mode must be 'cls', 'global_cls', "
            "'global_task_cls', or 'task_cls', "
            f"got {mode!r}."
        )
    if num_global_query_tokens < 0 or num_task_query_tokens < 0:
        raise ValueError(
            "num_global_query_tokens and num_task_query_tokens must be non-negative, "
            f"got {num_global_query_tokens} and {num_task_query_tokens}."
        )

    if num_generated_query_tokens < 1:
        raise ValueError("num_generated_query_tokens must be positive.")
    if normalized_mode == "cls":
        if num_generated_query_tokens != 1:
            raise ValueError(
                "num_generated_query_tokens > 1 requires a multi-source "
                "query_generation_mode, such as 'global_task_cls'."
            )
        if num_global_query_tokens != 0 or num_task_query_tokens != 0:
            raise ValueError(
                "num_global_query_tokens and num_task_query_tokens must both be zero "
                "when query_generation_mode='cls'."
            )
        return normalized_mode, compact_output_tokens

    if compact_output_tokens != 1:
        raise ValueError(
            f"query_generation_mode={normalized_mode!r} requires compact_output_tokens=1 "
            "because each source query produces one compact token."
        )
    if not use_cls_tokens:
        raise ValueError(
            f"query_generation_mode={normalized_mode!r} requires use_cls_tokens=true."
        )

    if normalized_mode == "global_cls":
        if num_global_query_tokens <= 0 or num_task_query_tokens != 0:
            raise ValueError(
                "query_generation_mode='global_cls' requires positive "
                "num_global_query_tokens and num_task_query_tokens=0."
            )
        return normalized_mode, num_global_query_tokens + num_generated_query_tokens

    if normalized_mode == "global_task_cls":
        if num_global_query_tokens <= 0 or num_task_query_tokens <= 0:
            raise ValueError(
                "query_generation_mode='global_task_cls' requires positive "
                "num_global_query_tokens and num_task_query_tokens."
            )
        return normalized_mode, (
            num_global_query_tokens + num_task_query_tokens + num_generated_query_tokens
        )

    if num_global_query_tokens != 0:
        raise ValueError(
            "query_generation_mode='task_cls' requires num_global_query_tokens=0."
        )
    if num_task_query_tokens <= 0:
        raise ValueError(
            "query_generation_mode='task_cls' requires positive num_task_query_tokens."
        )
    return normalized_mode, num_task_query_tokens + num_generated_query_tokens


@dataclass
class PrismVLAConfig:
    """Configuration for PRISM-VLA v2."""
    qwen_model_name: str = "Qwen/Qwen3.5-4B"
    # The instruction encoder has its own vocabulary and hidden width. None
    # retains direct backbone token embeddings for existing configurations.
    instruction_encoder_model_name: str | None = None
    instruction_encoder_revision: str | None = None
    instruction_encoder_nhead: int = 8
    instruction_encoder_dim_feedforward: int | None = None
    instruction_encoder_dropout: float = 0.0
    # A separate architecture: external frozen text K/V, no hidden-state prefix.
    text_input_deepstack: bool = False
    # (active PRISM decoder index, frozen instruction decoder index).
    text_input_deepstack_layer_map: tuple[tuple[int, int], ...] = ()
    text_input_deepstack_position_policy: str = "reserved_text_prefix_v1"
    # Resolved at construction, ordered like layer_map: source heads/dim, target heads/dim.
    text_input_deepstack_geometry: tuple[tuple[int, int, int, int], ...] = ()
    qwen_keep_layers: tuple[int, ...] = ()
    qwen_original_num_hidden_layers: int = 0
    # Execution only: replay stateless MLPs without replaying attention/caches.
    qwen_mlp_gradient_checkpointing: bool = False
    # Execution only: reuse scoped KDA indices for native zero-preserving MLPs.
    qwen_mlp_pack_valid_tokens: bool = False
    # Execution only: exact pinned-CPU storage for large rank>=3 autograd saves.
    qwen_saved_activation_cpu_offload_min_bytes: int = 0
    visual_encoder_type: str = "dino"
    visual_encoder_model_name: str = ""
    visual_encoder_image_size: int = 0
    visual_encoder_revision: str = ""
    visual_encoder_summary_pooling: str = "mean"
    visual_encoder_dtype: str = "bf16"
    visual_encoder_preprocess_backend: str = "timm_tensor"
    visual_encoder_dynamic_resolution: bool = False
    visual_encoder_eval_image_hw: tuple[int, int] = ()
    dino_dim: int = 768
    dino_model_name: str = "vit_base_patch16_dinov3"
    dino_image_size: int = 224
    train_dino_backbone: bool = False
    train_dino_cls_encoder: bool = False
    use_frozen_dino_for_cls: bool = False
    dino_tuning_mode: str = "full"
    dino_gradient_checkpointing: bool = False
    cradio_reentrant_checkpoint_pairs: bool = False
    cpu_transition_row_indices: bool = False
    dino_lora_r: int = 16
    dino_lora_alpha: int = 32
    dino_lora_dropout: float = 0.05
    dino_lora_target_modules: tuple[str, ...] = ("qkv", "attn.proj")
    num_patches: int = 196
    max_inst_len: int = 64
    instruction_format: str = "plain"
    instruction_conditioning: dict | None = None
    max_fast_len: int = 20
    fast_vocab_size: int = 2049  # 2048 BPE + 1 EOS
    # Execution only: keep this outside action_heads/normalization identities.
    fast_ffn_gradient_checkpointing: bool = False
    fast_layer_gradient_checkpointing: bool = False
    fast_broadcast_attention: bool = False
    pi05_block_gradient_checkpointing: bool = False
    action_dim: int = 7
    action_horizon: int = 10
    action_tokens_per_step: int = 1
    action_token_types: tuple[str, ...] = ()
    action_step_horizon: int = 10
    ego_layout_id: str = ""
    pose_representation: str = "standard"
    rotation_representation: str = "euler"
    action_delta_policy: dict[str, bool] | None = None
    action_target_frame: str = "transition_local"
    ego_action_representation: dict[str, object] | None = None
    ego_action_normalization: dict[str, object] | None = None
    cotrain_contract: dict | None = None
    # Robot targets keep the pretrained five-token ego layout and distinguish
    # EEF motion from the commanded gripper coordinate.
    robot_action_layout: str = ""
    robot_proprio_mode: str = "ego_tokens"
    allow_robot_action_frame_transfer: bool = False
    robot_action_normalization: dict | None = None
    robot_proprio_normalization: dict | None = None
    chunk_frame_position_scale: float = 1.0
    proprio_delta_policy: dict[str, bool] | None = None
    ego_frame_policy: dict[str, object] | None = None
    action_clip_abs_max: float | None = 1.0
    sampled_action_clip_abs_max: float | None = None
    replan_steps: int = 5
    replan_physical_steps: int = 5
    replan_token_horizon: int = 5
    frame_skip: int = 1
    fast_pretrained_path: str = "physical-intelligence/fast"
    idm_type: str = "fast_ar"
    action_heads: tuple[dict, ...] = ()
    # Opt-in bottleneck: predicted compact is the sole FM/FAST condition.
    action_conditioning: str = "legacy"
    # Evaluation execution only; always retain every auxiliary training loss.
    action_auxiliary_sampled_metrics: bool = True
    action_primary_head: str = ""
    # Legacy ego moments or independent source-keyed cotrain contracts by head.
    action_head_normalization: dict | None = None
    action_head_tokenizer_contracts: dict | None = None
    # FAST label rounding before DCT/BPE; distinct from optimizer precision.
    action_head_target_dtypes: dict[str, str] | None = None
    # IDM config
    idm_d_model: int = 512
    idm_nhead: int = 8
    idm_n_layers: int = 4
    idm_ff_mult: int = 4
    # Limits apply to attention scale deltas and shifts, not residual gates or MLP/final modulation.
    idm_attention_scale_limit: float | None = None
    idm_attention_shift_limit: float | None = None
    # These independent controls add no model parameters or checkpoint tensors.
    idm_qk_control_mode: str = "none"
    idm_qk_logit_limit: float = 100.0
    idm_qk_norm_temperature: float = 8.0
    idm_condition_rms_limit: float | None = None
    idm_attention_gate_limit: float | None = None
    idm_mlp_scale_limit: float | None = None
    idm_mlp_shift_limit: float | None = None
    idm_mlp_gate_limit: float | None = None
    idm_final_scale_limit: float | None = None
    idm_final_shift_limit: float | None = None
    flow_idm_arch: str = "mlp"
    flow_conditioning: str = "adaln_zero"
    flow_output_projection: str = "shared"
    flow_cross_attn_gate_init: float = 0.0
    # Execution only: replay the stateless action block, never cached KDA state.
    flow_cross_block_gradient_checkpointing: bool = False
    # Parameter-free per-head FP32 RMS normalization after flow Q/K projection.
    flow_qk_norm: bool = False
    flow_time_embed_dim: int = 128
    flow_sampling_steps: int = 16
    flow_sampling_cache_static_kv: bool = True
    weak_flow_sampling_steps: int = 4
    # Loss weights
    w_cls: float = 1.0
    # None retains the legacy next-observation target. Otherwise use a physical
    # horizon in raw source frames (nearest frame, ties rounded forward).
    future_cls_horizon_seconds: Optional[float] = None
    future_cls_source_fps: Optional[float] = None
    w_compact: float = 1.0
    w_patch_recon: float = 0.0
    w_action: float = 1.0
    w_sigreg: float = 0.0
    w_summary_token_similarity: float = 0.0
    sigreg_num_slices: int = 256
    sigreg_num_knots: int = 17
    sigreg_t_max: float = 3.0
    sigreg_seed: int = 0
    loss_type: str = "mse"
    huber_delta: float = 0.1
    action_pred_compact_gamma: float = 0.5
    # Zero retains legacy GT endpoint execution on every update.
    action_gt_diagnostic_interval: int = 0
    # Stage2 resolves this from the global dataset recipe, never rank-local rows.
    action_gt_weak_supervision_possible: bool = True
    detach_pred_compact_for_action: bool = True
    stage1_checkpoint: str = ""
    # Provenance only; the Stage-2 launcher loads this after Stage-1 initialization.
    backbone_checkpoint: str = ""
    stage1_allow_action_head_transfer: bool = False
    gradient_checkpointing: bool = True
    attn_implementation: str = "sdpa"
    cached_sdpa_backend: str = "math"
    disable_cudnn_sdpa: bool = True
    visual_encoder_mode: str = "dino"
    semantic_model_name: str = ""
    semantic_dim: int = 2560
    semantic_attn_implementation: str = "sdpa"
    fusion_hidden_mult: int = 2
    compact_query_mode: str = "qwen"
    compact_token_layout: str = "source"
    # Opt-in direct per-camera queries; legacy checkpoints retain their layout.
    query_design: str = "legacy"
    direct_query_global_tokens_per_camera: tuple[int, ...] = ()
    direct_query_task_tokens_per_camera: tuple[int, ...] = ()
    direct_query_generated_tokens_per_camera: tuple[int, ...] = ()
    direct_query_context: str = "global_task_cls"
    direct_query_include_registers: bool = True
    direct_query_depth: int = 1
    camera_compact_architecture_version: int = 2
    camera_compact_depth: int = 1
    camera_compact_queries_per_source: int = 4
    camera_compact_num_heads: int = 8
    camera_compact_pool_num_heads: Optional[int] = None
    camera_compact_ff_mult: int = 4
    camera_compact_identity_count: int = 32
    camera_compact_camera_ids: tuple[int, ...] = ()
    camera_compact_tokens_per_camera: int | tuple[int, ...] = 1
    # Camera-local slot indices. These slots use patch reconstruction instead
    # of next-compact regression, while remaining action/backbone conditions.
    global_compact_indices: tuple[int, ...] = ()
    patch_recon_group_size: int = 1
    patch_recon_decoder_dim: int = 512
    patch_recon_decoder_layers: int = 2
    patch_recon_decoder_num_heads: int = 8
    patch_recon_decoder_ff_mult: int = 4
    num_query_tokens: int = 1
    compact_output_tokens: int = 1
    query_generation_mode: str = "cls"
    num_global_query_tokens: int = 0
    num_task_query_tokens: int = 0
    num_generated_query_tokens: int = 1
    native_dino_nhead: int = 8
    num_dino_cameras: int = 1
    dino_camera_fusion: str = "concat"
    dino_camera_fusion_temperature: float = 0.1
    dino_cls_camera_fusion: str = "mean"
    use_cls_register_fusion: bool = False
    cls_register_nhead: int = 8
    cls_register_gate_init: float = 0.001
    cross_attn_depth: int = 1
    use_cls_tokens: bool = True
    use_cls_condition_for_idm: bool = True
    use_current_cls_condition_for_idm: bool = True
    use_current_compact_condition_for_idm: bool = True
    use_future_cls_condition_for_idm: bool = True
    use_future_compact_condition_for_idm: bool = True
    use_start_patch_visual_tokens: bool = False
    use_proprio: bool = False
    proprio_dim: int = 15
    proprio_hidden_mult: int = 2
    proprio_token_placement: str = "condition"
    proprio_encoder_type: str = "mlp"
    proprio_encoder_by_frame: bool = False
    proprio_frame_names: tuple[str, ...] = ("head", "camera", "robot", "eef", "world", "head_eef", "camera_eef", "robot_eef", "world_eef")
    proprio_token_dropout: float = 0.0
    # Append-only native joint interfaces; geometric heads/banks stay intact.
    engram_native: dict | None = None
    proprio_structured_hidden_dim: int = 256
    proprio_structured_nhead: int = 8
    proprio_structured_layers: int = 2
    proprio_structured_ff_mult: int = 4
    proprio_structured_dropout: float = 0.0
    compact_fusion_residual: str = "mean"
    sliding_window_attention: bool = False
    attention_window_timesteps: int = 16
    blockwise_full_attention: bool = False
    # Historical checkpoints omit this field and retain their boolean policy.
    # Rolling uses W timesteps INCLUDING the current one, in both streams.
    full_attention_window_policy: str = "legacy"
    # Only the main stream changes; the query stream retains its causal masks.
    main_stream_timestep_causal: bool = False
    # Add a learned ACT slot before Q and condition the expert on its output.
    # Visual/CLS prediction heads keep their corresponding output positions.
    main_stream_action_token: bool = False
    # Version 2 makes ACT a transient observer, absent from persistent memory.
    main_stream_action_token_version: int = 2
    main_action_token_input: str = "learned"
    action_cvae_latent_dim: int = 32
    action_cvae_encoder_dim: int = 256
    action_cvae_encoder_layers: int = 2
    action_cvae_encoder_heads: int = 4
    action_cvae_kl_weight: float = 1e-3
    action_cvae_inference_mode: str = "zero"
    action_patch_token_layout_version: int = 1
    action_patch_token_dim: int = 0
    blockwise_full_attention_backend: str = "dense_sdpa"
    linear_attention_variant: str = "gated_deltanet"
    kda_gate_rank: int = 128
    kda_chunk_size: int = 64
    kda_disable_recompute: bool = False
    kda_safe_gate: bool = False
    kda_lower_bound: float = -5.0
    pass_deltanet_state_between_chunks: bool = True
    deltanet_state_noise: bool = False
    deltanet_state_noise_std: float = 0.0

    def __post_init__(self) -> None:
        if self.engram_native is not None:
            from prism_vla.modeling.engram_finetune import validate_engram_native_config
            self.engram_native = validate_engram_native_config(self.engram_native)
        if type(self.qwen_mlp_gradient_checkpointing) is not bool:
            raise ValueError("qwen_mlp_gradient_checkpointing must be a bool")
        if type(self.qwen_mlp_pack_valid_tokens) is not bool:
            raise ValueError("qwen_mlp_pack_valid_tokens must be a bool")
        if (type(self.qwen_saved_activation_cpu_offload_min_bytes) is not int
                or self.qwen_saved_activation_cpu_offload_min_bytes < 0):
            raise ValueError("qwen_saved_activation_cpu_offload_min_bytes must be a nonnegative integer")
        if self.qwen_saved_activation_cpu_offload_min_bytes and self.compact_token_layout != "camera":
            raise ValueError("Qwen saved-activation offload currently requires camera compact layout")
        if type(self.max_inst_len) is not int or self.max_inst_len <= 0:
            raise ValueError("max_inst_len must be a positive integer")
        self.instruction_format = normalize_instruction_format(self.instruction_format)
        self.instruction_conditioning = normalize_instruction_conditioning_config(
            self.instruction_conditioning
            if self.instruction_conditioning is not None else {"format": self.instruction_format}
        )
        if self.instruction_conditioning["format"] != self.instruction_format:
            raise ValueError("instruction_format must match instruction_conditioning.format")
        from prism_vla.modeling.attention_policy import resolved_full_attention_window_policy
        self.full_attention_window_policy = str(self.full_attention_window_policy).strip().lower()
        resolved_full_attention_window_policy(self)
        from prism_vla.data.future_cls import future_cls_frame_offset
        future_cls_frame_offset(self.future_cls_horizon_seconds, self.future_cls_source_fps)
        from prism_vla.modeling.action_head_config import normalize_action_heads
        self.compact_token_layout = str(self.compact_token_layout).strip().lower()
        if self.compact_token_layout not in {"source", "camera"}:
            raise ValueError("compact_token_layout must be 'source' or 'camera'")
        self.query_design = str(self.query_design).strip().lower()
        if self.query_design not in {"legacy", "direct_query"}:
            raise ValueError("query_design must be 'legacy' or 'direct_query'")
        self.direct_query_context = str(self.direct_query_context).strip().lower()
        if self.direct_query_context not in {"cls", "global_cls", "global_task_cls"}:
            raise ValueError("direct_query_context must be cls, global_cls, or global_task_cls")
        if type(self.direct_query_include_registers) is not bool:
            raise ValueError("direct_query_include_registers must be a bool")
        if type(self.direct_query_depth) is not int or self.direct_query_depth < 1:
            raise ValueError("direct_query_depth must be a positive integer")
        if self.query_design == "direct_query":
            if self.compact_token_layout != "camera":
                raise ValueError("direct_query requires compact_token_layout='camera'")
            if self.use_cls_tokens is not True:
                raise ValueError("direct_query requires use_cls_tokens=True")
            from prism_vla.modeling.direct_query import DirectQueryLayout
            layout = DirectQueryLayout.from_config(self)
            if type(self.num_dino_cameras) is not int or layout.num_cameras != self.num_dino_cameras:
                raise ValueError("direct_query count tuples must have one entry per camera")
            if layout.num_generated < 1:
                raise ValueError("direct_query requires at least one generated query in total")
            self.direct_query_global_tokens_per_camera = layout.global_counts
            self.direct_query_task_tokens_per_camera = layout.task_counts
            self.direct_query_generated_tokens_per_camera = layout.generated_counts
            self.num_global_query_tokens = sum(layout.global_counts)
            self.num_task_query_tokens = sum(layout.task_counts)
            self.num_generated_query_tokens = layout.num_generated
        if type(self.main_stream_timestep_causal) is not bool:
            raise ValueError("main_stream_timestep_causal must be a bool")
        if self.main_stream_timestep_causal and (
            self.compact_token_layout != "camera" or self.use_cls_tokens is not True
        ):
            raise ValueError("main_stream_timestep_causal requires camera layout and use_cls_tokens=True")
        if self.main_stream_timestep_causal and self.attn_implementation != "sdpa":
            raise ValueError("main_stream_timestep_causal requires attn_implementation='sdpa'")
        if self.qwen_mlp_pack_valid_tokens:
            if not self.qwen_mlp_gradient_checkpointing:
                raise ValueError("qwen_mlp_pack_valid_tokens requires qwen_mlp_gradient_checkpointing=True")
            if (self.compact_token_layout != "camera" or self.query_design != "direct_query"
                    or not self.main_stream_timestep_causal
                    or resolved_full_attention_window_policy(self) != "rolling"
                    or self.gradient_checkpointing):
                raise ValueError("qwen_mlp_pack_valid_tokens requires camera/direct_query/timestep/rolling "
                                 "with full-backbone gradient_checkpointing=False")
            # These extensions do not exist in this implementation. Keep a
            # config carrying a later nonzero history-noise policy from silently
            # violating the exact-zero missing MAIN-camera invariant.
            for name in ("history_noise_std", "history_token_noise_std", "input_history_noise_std"):
                if float(getattr(self, name, 0.0)) != 0.0:
                    raise ValueError("qwen_mlp_pack_valid_tokens does not support history-token input noise")
        if type(self.main_stream_action_token) is not bool:
            raise ValueError("main_stream_action_token must be a bool")
        if self.main_stream_action_token and not self.main_stream_timestep_causal:
            raise ValueError("main_stream_action_token requires main_stream_timestep_causal=True")
        if str(self.idm_type).strip().lower() in {"act_patch", "patchwam"}:
            self.idm_type = "act_patch"
            if not self.main_stream_action_token:
                raise ValueError("act_patch requires main_stream_action_token=True")
            if self.use_proprio and self.proprio_token_placement != "condition":
                raise ValueError("act_patch requires condition-placed proprio fused into ACT")
            if (type(self.action_tokens_per_step) is not int or self.action_tokens_per_step < 1
                    or type(self.action_horizon) is not int or self.action_horizon < 1
                    or self.action_horizon % self.action_tokens_per_step):
                raise ValueError("act_patch action_horizon must contain complete physical action steps")
        if (type(self.main_stream_action_token_version) is not int
                or self.main_stream_action_token_version != 2):
            raise ValueError("main_stream_action_token_version must be 2")
        self.main_action_token_input = str(self.main_action_token_input).strip().lower()
        if self.main_action_token_input not in {"learned", "cvae"}:
            raise ValueError("main_action_token_input must be 'learned' or 'cvae'")
        if self.main_action_token_input == "cvae" and not self.main_stream_action_token:
            raise ValueError("main_action_token_input='cvae' requires main_stream_action_token=True")
        for name in ("action_cvae_latent_dim", "action_cvae_encoder_dim",
                     "action_cvae_encoder_layers", "action_cvae_encoder_heads"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.action_cvae_encoder_dim % self.action_cvae_encoder_heads:
            raise ValueError("action_cvae_encoder_dim must be divisible by action_cvae_encoder_heads")
        try:
            kl_weight = float(self.action_cvae_kl_weight)
        except (TypeError, ValueError):
            raise ValueError("action_cvae_kl_weight must be finite and nonnegative") from None
        if (isinstance(self.action_cvae_kl_weight, bool)
                or not math.isfinite(kl_weight) or kl_weight < 0):
            raise ValueError("action_cvae_kl_weight must be finite and nonnegative")
        self.action_cvae_kl_weight = kl_weight
        self.action_cvae_inference_mode = str(self.action_cvae_inference_mode).strip().lower()
        if self.action_cvae_inference_mode not in {"zero", "sample"}:
            raise ValueError("action_cvae_inference_mode must be 'zero' or 'sample'")
        if type(self.action_patch_token_layout_version) is not int or self.action_patch_token_layout_version != 1:
            raise ValueError("action_patch_token_layout_version must be 1")
        if type(self.action_patch_token_dim) is not int or self.action_patch_token_dim < 0:
            raise ValueError("action_patch_token_dim must be a nonnegative integer (0 infers backbone width)")
        if type(self.action_gt_diagnostic_interval) is not int or self.action_gt_diagnostic_interval < 0:
            raise ValueError("action_gt_diagnostic_interval must be a non-negative integer")
        if type(self.action_gt_weak_supervision_possible) is not bool:
            raise ValueError("action_gt_weak_supervision_possible must be a bool")
        if type(self.flow_sampling_cache_static_kv) is not bool:
            raise ValueError("flow_sampling_cache_static_kv must be a bool")
        for name in (
            "camera_compact_depth", "camera_compact_queries_per_source",
            "camera_compact_num_heads", "camera_compact_ff_mult",
            "camera_compact_identity_count",
            "patch_recon_group_size",
            "patch_recon_decoder_dim", "patch_recon_decoder_layers",
            "patch_recon_decoder_num_heads", "patch_recon_decoder_ff_mult",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (self.camera_compact_pool_num_heads is not None
                and (type(self.camera_compact_pool_num_heads) is not int
                    or self.camera_compact_pool_num_heads <= 0)):
            raise ValueError("camera_compact_pool_num_heads must be None or a positive integer")
        if not isinstance(self.camera_compact_camera_ids, (tuple, list)):
            raise ValueError("camera_compact_camera_ids must be an ordered tuple/list of unique integer IDs")
        ids = tuple(self.camera_compact_camera_ids)
        if (any(type(value) is not int or not 0 <= value < self.camera_compact_identity_count for value in ids)
                or len(set(ids)) != len(ids)
                or ids and len(ids) != self.num_dino_cameras):
            raise ValueError("camera_compact_camera_ids must contain one distinct in-vocabulary ID per camera slot")
        self.camera_compact_camera_ids = ids
        from prism_vla.modeling.camera_token_budget import camera_token_counts
        if self.query_design == "direct_query":
            camera_counts = tuple(
                global_count + task_count + generated_count
                for global_count, task_count, generated_count in zip(
                    self.direct_query_global_tokens_per_camera,
                    self.direct_query_task_tokens_per_camera,
                    self.direct_query_generated_tokens_per_camera,
                )
            )
            self.camera_compact_tokens_per_camera = camera_counts
        else:
            camera_counts = camera_token_counts(self.camera_compact_tokens_per_camera, self.num_dino_cameras)
            if isinstance(self.camera_compact_tokens_per_camera, (tuple, list)):
                self.camera_compact_tokens_per_camera = camera_counts
        if not isinstance(self.global_compact_indices, (tuple, list)):
            raise ValueError("global_compact_indices must be an ordered tuple/list of unique slot indices")
        slots = tuple(self.global_compact_indices)
        if (any(type(value) is not int or not 0 <= value < min(camera_counts)
                for value in slots) or len(set(slots)) != len(slots)):
            raise ValueError("global_compact_indices must contain distinct in-range camera-local slot indices")
        self.global_compact_indices = slots
        if self.patch_recon_decoder_dim % self.patch_recon_decoder_num_heads:
            raise ValueError("patch_recon_decoder_dim must be divisible by patch_recon_decoder_num_heads")
        if (isinstance(self.w_patch_recon, bool) or not isinstance(self.w_patch_recon, (int, float))
                or not math.isfinite(self.w_patch_recon) or self.w_patch_recon < 0):
            raise ValueError("w_patch_recon must be finite and non-negative")
        if self.query_design == "direct_query" and self.w_patch_recon > 0:
            raise ValueError("direct_query currently requires w_patch_recon=0")
        if self.compact_token_layout != "camera" and (
                self.camera_compact_tokens_per_camera != 1 or slots or self.w_patch_recon > 0):
            raise ValueError("Multiple/global camera compacts require compact_token_layout='camera'")
        if self.w_patch_recon > 0 and not slots:
            raise ValueError("w_patch_recon > 0 requires at least one global_compact_indices slot")
        if self.compact_token_layout == "camera":
            if (type(self.camera_compact_architecture_version) is not int
                    or self.camera_compact_architecture_version != 2):
                raise ValueError("camera_compact_architecture_version must be 2: backbone-width pooling and camera-specific predictors require a new camera architecture checkpoint")
            if (str(self.compact_query_mode).strip().lower() != "native_dino"
                    or str(self.visual_encoder_mode).strip().lower() not in {"dino", "dinov3"}):
                raise ValueError("compact_token_layout='camera' requires native_dino and the DINO-only visual stream")
            if str(self.dino_camera_fusion).strip().lower() != "concat":
                raise ValueError("Camera compact tokens cannot use camera-weighted fusion")
            if self.use_cls_tokens is not True:
                raise ValueError("Camera compact layout requires use_cls_tokens=True")
            if self.gradient_checkpointing is not False:
                raise ValueError("Camera compact layout requires gradient_checkpointing=False until recompute contexts are supported")
            if type(self.num_dino_cameras) is not int or self.num_dino_cameras <= 0:
                raise ValueError("Camera compact layout requires a positive integer num_dino_cameras")
            if self.num_dino_cameras > self.camera_compact_identity_count:
                raise ValueError("num_dino_cameras cannot exceed camera_compact_identity_count")
            if self.compact_output_tokens != 1:
                raise ValueError("Camera compact layout requires compact_output_tokens=1; output slots are cameras")
            if self.dino_dim % self.camera_compact_num_heads:
                raise ValueError("dino_dim must be divisible by camera_compact_num_heads")
        if type(self.fast_ffn_gradient_checkpointing) is not bool:
            raise ValueError("fast_ffn_gradient_checkpointing must be a bool")
        if type(self.fast_layer_gradient_checkpointing) is not bool:
            raise ValueError("fast_layer_gradient_checkpointing must be a bool")
        if type(self.fast_broadcast_attention) is not bool:
            raise ValueError("fast_broadcast_attention must be a bool")
        if type(self.pi05_block_gradient_checkpointing) is not bool:
            raise ValueError("pi05_block_gradient_checkpointing must be a bool")
        if type(self.flow_cross_block_gradient_checkpointing) is not bool:
            raise ValueError("flow_cross_block_gradient_checkpointing must be a bool")
        if type(self.flow_qk_norm) is not bool:
            raise ValueError("flow_qk_norm must be a bool")
        if type(self.cpu_transition_row_indices) is not bool:
            raise ValueError("cpu_transition_row_indices must be a bool")
        if type(self.cradio_reentrant_checkpoint_pairs) is not bool:
            raise ValueError("cradio_reentrant_checkpoint_pairs must be a bool")
        if self.fast_ffn_gradient_checkpointing and self.fast_layer_gradient_checkpointing:
            raise ValueError("FAST FFN and whole-layer checkpointing are mutually exclusive")
        if self.sampled_action_clip_abs_max is not None:
            bound = float(self.sampled_action_clip_abs_max)
            if not math.isfinite(bound) or bound <= 0:
                raise ValueError("sampled_action_clip_abs_max must be finite and positive or null")
            self.sampled_action_clip_abs_max = bound
        self.action_heads = normalize_action_heads(self.action_heads)
        if self.action_conditioning not in {"legacy", "predicted_compact"}:
            raise ValueError("action_conditioning must be legacy or predicted_compact")
        if type(self.action_auxiliary_sampled_metrics) is not bool:
            raise ValueError("action_auxiliary_sampled_metrics must be a bool")
        if self.action_conditioning == "predicted_compact":
            if (self.use_current_cls_condition_for_idm or self.use_current_compact_condition_for_idm
                    or self.use_future_cls_condition_for_idm or not self.use_future_compact_condition_for_idm
                    or self.action_pred_compact_gamma != 1.0 or self.detach_pred_compact_for_action
                    or self.w_compact != 0 or self.w_patch_recon != 0 or self.w_sigreg != 0
                    or self.main_stream_action_token or self.proprio_token_placement != "condition"):
                raise ValueError("predicted_compact requires only future compact conditions, action-only compact supervision, gamma=1 and no direct proprio/action-token bypass")
        if any(head["type"] == "pi05" for head in self.action_heads) and self.action_conditioning != "predicted_compact":
            raise ValueError("pi05 heads require explicit predicted_compact action conditioning")
        if self.action_head_target_dtypes is not None:
            fast_heads = {head["name"] for head in self.action_heads if head["type"] == "fast_ar"}
            if (not isinstance(self.action_head_target_dtypes, dict)
                    or set(self.action_head_target_dtypes) != fast_heads
                    or any(not isinstance(value, str) or value not in {"float32", "bfloat16"}
                           for value in self.action_head_target_dtypes.values())):
                raise ValueError("action_head_target_dtypes must record float32/bfloat16 for every FAST head")
        if self.action_heads:
            primary = self.action_heads[0]
            if self.action_tokens_per_step == 5:
                token_types = ("head", "eef", "hand", "eef", "hand")
                if self.action_token_types and tuple(self.action_token_types) != token_types:
                    raise ValueError("Multiple ego action heads require the canonical five-token schema")
                self.action_token_types = token_types
            self.action_primary_head = self.action_primary_head or primary["name"]
            if self.action_primary_head != primary["name"]:
                raise ValueError("The first configured head is the primary metric/inference head")
            if self.idm_type != primary["type"]:
                raise ValueError("idm_type must match the primary action head")
            if self.proprio_token_placement != "condition":
                raise ValueError("Multiple heads require shared proprio condition tokens")
            if self.stage1_checkpoint:
                raise ValueError("Multi-head Stage-1 initialization requires explicit per-head mapping")
            if self.ego_action_representation is not None and self.ego_action_representation != primary["action_representation"]:
                raise ValueError("Primary head representation differs from the shared dataset contract")
            if self.action_head_normalization is not None:
                from prism_vla.data.datasets.ego_action_normalization import validate_ego_action_normalization
                if set(self.action_head_normalization) != {head["name"] for head in self.action_heads}:
                    raise ValueError("Action-head normalization requires every named head")
                for head in self.action_heads:
                    metadata = self.action_head_normalization[head["name"]]
                    if metadata is not None:
                        cotrain_metadata = isinstance(metadata, dict) and metadata.get("schema") == "cotrain_action_contract_v1"
                        if cotrain_metadata:
                            if self.cotrain_contract is None:
                                raise ValueError("Source-keyed action-head normalization requires cotrain_contract")
                            from prism_vla.data.datasets.cotrain_contract import validate_cotrain_contract
                            metadata = validate_cotrain_contract(metadata)
                        else:
                            metadata = validate_ego_action_normalization(metadata)
                        if metadata["representation"] != head["action_representation"] or metadata["horizon"] != self.action_step_horizon:
                            raise ValueError(f"Head {head['name']}: normalization representation/horizon mismatch")
                        expected_policy = "clip" if head["type"] == "fast_ar" else "mask"
                        if not cotrain_metadata and metadata.get("action_normalization_max_abs_policy", "mask") != expected_policy:
                            raise ValueError(
                                f"Head {head['name']}: normalization range policy must be {expected_policy}; "
                                "FAST codec clipping must preserve observed target coordinates"
                            )
                        self.action_head_normalization[head["name"]] = metadata
        if self.instruction_encoder_model_name is not None and (
            not isinstance(self.instruction_encoder_model_name, str)
            or not self.instruction_encoder_model_name.strip()
        ):
            raise ValueError("instruction_encoder_model_name must be a nonempty model name or null")
        if type(self.instruction_encoder_nhead) is not int or self.instruction_encoder_nhead <= 0:
            raise ValueError("instruction_encoder_nhead must be a positive integer")
        if self.instruction_encoder_dim_feedforward is not None and (
            type(self.instruction_encoder_dim_feedforward) is not int
            or self.instruction_encoder_dim_feedforward <= 0
        ):
            raise ValueError("instruction_encoder_dim_feedforward must be a positive integer or null")
        if not math.isfinite(self.instruction_encoder_dropout) or not 0 <= self.instruction_encoder_dropout < 1:
            raise ValueError("instruction_encoder_dropout must be finite and in [0, 1)")
        if type(self.text_input_deepstack) is not bool:
            raise ValueError("text_input_deepstack must be boolean")
        if self.text_input_deepstack_position_policy != "reserved_text_prefix_v1":
            raise ValueError("Unsupported text_input_deepstack_position_policy")
        mapping = self.text_input_deepstack_layer_map
        if not isinstance(mapping, (list, tuple)) or any(
            not isinstance(pair, (list, tuple)) or len(pair) != 2
            or any(type(index) is not int or index < 0 for index in pair)
            for pair in mapping
        ):
            raise ValueError("text_input_deepstack_layer_map must contain nonnegative (PRISM, encoder) layer pairs")
        self.text_input_deepstack_layer_map = tuple(tuple(pair) for pair in mapping)
        if len({pair[0] for pair in mapping}) != len(mapping):
            raise ValueError("text_input_deepstack_layer_map contains duplicate PRISM layers")
        geometry = self.text_input_deepstack_geometry
        if not isinstance(geometry, (list, tuple)) or any(
            not isinstance(item, (list, tuple)) or len(item) != 4
            or any(type(value) is not int or value <= 0 for value in item)
            for item in geometry
        ):
            raise ValueError("Invalid text_input_deepstack_geometry")
        self.text_input_deepstack_geometry = tuple(tuple(item) for item in geometry)
        if self.text_input_deepstack:
            if not self.instruction_encoder_model_name or not mapping:
                raise ValueError("text_input_deepstack requires an instruction encoder and explicit layer map")
            if geometry and len(geometry) != len(mapping):
                raise ValueError("text_input_deepstack_geometry must match layer map length")
        elif mapping or geometry:
            raise ValueError("text_input_deepstack layer map/geometry require enabling the architecture")
        self.idm_attention_scale_limit, self.idm_attention_shift_limit = validate_idm_attention_limits(
            self.idm_attention_scale_limit,
            self.idm_attention_shift_limit,
            idm_type=self.idm_type,
            flow_idm_arch=self.flow_idm_arch,
            flow_conditioning=self.flow_conditioning,
        )
        controls = validate_idm_recovery_controls(
            **{name: getattr(self, name) for name in (
                "idm_qk_control_mode", "idm_qk_logit_limit", "idm_qk_norm_temperature",
                *IDM_RECOVERY_LIMIT_FIELDS,
            )},
            idm_type=self.idm_type,
            flow_idm_arch=self.flow_idm_arch,
            flow_conditioning=self.flow_conditioning,
        )
        for name, value in controls.items():
            setattr(self, name, value)
        if self.ego_action_normalization is not None:
            from prism_vla.data.datasets.ego_action_normalization import validate_ego_action_normalization
            self.ego_action_normalization = validate_ego_action_normalization(self.ego_action_normalization)
            if self.action_target_frame != "factorized":
                raise ValueError("Chunk action normalization requires factorized targets")
            if (self.ego_action_normalization["representation"] != self.ego_action_representation
                    or self.ego_action_normalization["horizon"] != self.action_step_horizon
                    or self.ego_action_normalization["position_scale"] != self.chunk_frame_position_scale):
                raise ValueError("Action normalization geometry/horizon differs from model contract")
        self.robot_action_normalization = validate_robot_action_normalization(self.robot_action_normalization)
        from prism_vla.data.datasets.robot_proprio_normalization import validate_proprio_normalization_metadata
        self.robot_proprio_normalization = validate_proprio_normalization_metadata(self.robot_proprio_normalization)
        if self.robot_proprio_normalization is not None and self.robot_proprio_mode != "robodojo_absolute_state_joint_v1":
            raise ValueError("robot_proprio_normalization requires the native 28-D RoboDojo proprio contract")
        if self.robot_action_normalization is not None and not self.robot_action_layout:
            raise ValueError("robot_action_normalization requires robot_action_layout")
        if self.robot_action_layout not in {"", "ego_robot_controls_v1:libero", "ego_robot_controls_v1:robocasa365",
                                            "ego_robot_command_trajectory_v2:libero",
                                            "ego_robot_command_trajectory_v3_camera_masked:libero",
                                            "ego_robot_command_trajectory_v4_world:libero",
                                            "ego_robot_command_trajectory_v5_camera_hand_gripper:libero",
                                            "ego_robot_absolute_trajectory_v1_camera_hand_gripper:robodojo"}:
            raise ValueError("Unsupported robot_action_layout")
        if self.robot_action_layout and self.action_target_frame != "factorized":
            raise ValueError("robot_action_layout requires factorized geometric targets")
        if self.robot_proprio_mode not in {"ego_tokens", "native_absolute_state_joint_v1", "robodojo_absolute_state_joint_v1"}:
            raise ValueError("Unsupported robot_proprio_mode")
        if self.robot_proprio_mode == "robodojo_absolute_state_joint_v1":
            if self.proprio_dim != 28 or self.robot_action_layout != "ego_robot_absolute_trajectory_v1_camera_hand_gripper:robodojo":
                raise ValueError("RoboDojo absolute state and joints require its explicit bimanual layout and proprio_dim=28")
        if self.robot_proprio_mode == "native_absolute_state_joint_v1":
            native_direct = (
                not self.robot_action_layout
                and self.action_dim == 7
                and self.action_tokens_per_step == 1
                and self.action_target_frame == "transition_local"
            )
            if self.proprio_dim != 15 or not (
                self.robot_action_layout.endswith(":libero") or native_direct
            ):
                raise ValueError(
                    "Native absolute robot proprio requires LIBERO robot tokens "
                    "or direct 7-D commands, with proprio_dim=15"
                )
        if self.action_target_frame not in {"transition_local", "chunk_start_camera", "chunk_start_eef", "factorized"}:
            raise ValueError("Unsupported action_target_frame")
        if self.action_target_frame == "factorized":
            from prism_vla.data.datasets.ego_action_representation import normalize_action_representation
            self.ego_action_representation = normalize_action_representation(self.ego_action_representation)
        elif self.ego_action_representation is not None:
            raise ValueError("ego_action_representation requires action_target_frame=factorized")
        if not math.isfinite(self.chunk_frame_position_scale) or self.chunk_frame_position_scale <= 0:
            raise ValueError("chunk_frame_position_scale must be finite and positive")
        if self.cotrain_contract is not None:
            from prism_vla.data.datasets.cotrain_contract import validate_cotrain_contract
            self.cotrain_contract = validate_cotrain_contract(self.cotrain_contract)
            contract = self.cotrain_contract
            if self.action_heads:
                heads = contract.get("heads")
                if (not isinstance(heads, dict) or set(heads) != {head["name"] for head in self.action_heads}
                        or contract.get("primary_head") != self.action_primary_head):
                    raise ValueError("cotrain requires a source contract for every configured action head")
                for head in self.action_heads:
                    metadata = heads[head["name"]]
                    if (metadata["representation"] != head["action_representation"]
                            or metadata["horizon"] != self.action_step_horizon
                            or metadata["position_scale"] != self.chunk_frame_position_scale):
                        raise ValueError(f"Head {head['name']}: cotrain representation/horizon/scale mismatch")
                if self.action_head_normalization is None:
                    self.action_head_normalization = heads
                elif self.action_head_normalization != heads:
                    raise ValueError("cotrain action-head normalization differs from its source contracts")
            elif contract.get("heads") is not None:
                raise ValueError("Multihead cotrain contracts require configured action_heads")
            if (self.robot_action_layout or self.robot_action_normalization is not None
                    or self.robot_proprio_normalization is not None or self.ego_action_normalization is not None):
                raise ValueError("cotrain requires source-keyed layouts and normalization, not global robot/ego statistics")
            if (self.action_target_frame != "factorized"
                    or self.ego_action_representation != contract["representation"]
                    or self.action_dim != contract["action_dim"]
                    or self.action_tokens_per_step != contract["action_tokens_per_step"]
                    or self.action_step_horizon != contract["horizon"]
                    or self.action_horizon != contract["horizon"] * contract["action_tokens_per_step"]
                    or self.pose_representation != contract["pose_representation"]
                    or self.rotation_representation != contract["rotation_representation"]
                    or self.chunk_frame_position_scale != contract["position_scale"]):
                raise ValueError("cotrain common action geometry differs from the model contract")
            if self.robot_proprio_mode != "ego_tokens" or (self.use_proprio and self.proprio_dim != contract["proprio_dim"]):
                raise ValueError("cotrain requires the common ego_tokens proprio contract")
        if self.action_target_frame in {"chunk_start_camera", "chunk_start_eef", "factorized"} and (
            self.action_tokens_per_step != 5 or self.action_dim != 15
            or self.rotation_representation != "rotation_6d"
        ):
            raise ValueError(f"{self.action_target_frame} requires five 15-D rotation_6d action tokens")
        if self.action_target_frame == "chunk_start_eef" and self.pose_representation != "umi_local":
            raise ValueError("chunk_start_eef requires pose_representation=umi_local")
        if type(self.proprio_encoder_by_frame) is not bool:
            raise ValueError("proprio_encoder_by_frame must be boolean")
        self.proprio_frame_names = tuple(self.proprio_frame_names)
        if (not self.proprio_frame_names or len(set(self.proprio_frame_names)) != len(self.proprio_frame_names)
                or any(not isinstance(name, str) or not name.isidentifier() for name in self.proprio_frame_names)):
            raise ValueError("proprio_frame_names must contain unique identifier names")
        if not 0.0 <= float(self.proprio_token_dropout) <= 1.0:
            raise ValueError("proprio_token_dropout must be between 0 and 1")
        self.flow_output_projection, self.action_token_types = validate_flow_output_projection(
            self.flow_output_projection,
            action_token_types=self.action_token_types,
            action_horizon=self.action_horizon,
            action_tokens_per_step=self.action_tokens_per_step,
            idm_type=self.idm_type,
            flow_idm_arch=self.flow_idm_arch,
        )

    @classmethod
    def from_dict(cls, values: dict) -> "PrismVLAConfig":
        """Build a config with explicit, fail-closed legacy compatibility."""
        normalized = dict(values)
        # Older two-camera checkpoints predate the explicit camera-count field.
        # Infer the only compatible module shape from the saved DINO patch count
        # instead of silently reconstructing a one-camera model.
        if "num_dino_cameras" not in normalized:
            num_patches = int(normalized.get("num_patches", 196))
            if num_patches <= 0 or num_patches % 196 != 0:
                raise ValueError(
                    "Cannot infer legacy num_dino_cameras from "
                    f"num_patches={num_patches}; expected a positive multiple of 196."
                )
            normalized["num_dino_cameras"] = num_patches // 196
        if "lambda_action" in normalized:
            if (
                "w_action" in normalized
                and normalized["w_action"] != normalized["lambda_action"]
            ):
                raise ValueError(
                    "Conflicting PRISM-VLA config values for legacy `lambda_action` "
                    f"({normalized['lambda_action']!r}) and `w_action` "
                    f"({normalized['w_action']!r})."
                )
            normalized.setdefault("w_action", normalized["lambda_action"])
        for key in ("lambda_action", "lambda_query_align", "query_align_decay_steps"):
            normalized.pop(key, None)
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(normalized) - allowed)
        if unknown:
            raise ValueError(
                "Unknown PRISM-VLA config keys: "
                f"{unknown}. This checkpoint may require a different code branch/worktree."
            )
        return cls(**normalized)

    def to_dict(self) -> dict:
        result = asdict(self)
        if self.engram_native is None:
            result.pop("engram_native")
        return result

    def instruction_tokenizer_pretrained_kwargs(self) -> dict:
        """Return independent encoder tokenizer identity, including its revision."""
        return instruction_tokenizer_pretrained_kwargs(self)
