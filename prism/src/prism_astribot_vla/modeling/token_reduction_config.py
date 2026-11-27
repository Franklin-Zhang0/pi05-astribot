"""Closed, default-off architecture contract for fixed visual token selection."""
from __future__ import annotations


def resolved_token_reduction_windows(config):
    """Cap each rolling stage window; include the current observation."""
    get = config.get if isinstance(config, dict) else lambda k, d=None: getattr(config, k, d)
    windows = tuple(get("token_reduction_windows", (1, 2, 4, 8, 16, 32)))
    cap = get("token_reduction_max_attention_window_timesteps", None)
    if cap is not None and (type(cap) is not int or cap < 1):
        raise ValueError("token_reduction_max_attention_window_timesteps must be None or a positive integer")
    return windows if cap is None else tuple(min(window, cap) for window in windows)


def token_reduction_contract(config):
    get = config.get if isinstance(config, dict) else lambda k, d=None: getattr(config, k, d)
    if get("query_design", "legacy") != "token_reduction":
        return None
    if isinstance(config, dict):
        required = {"token_reduction_version", "token_reduction_stage_boundaries",
                    "token_reduction_windows", "token_reduction_main_camera_id",
                    "token_reduction_condition_capacity", "camera_compact_camera_ids", "num_dino_cameras", "w_compact"}
        if required - set(config):
            raise ValueError(f"Incomplete token-reduction checkpoint contract: missing {sorted(required - set(config))}")
    contract = {
        "schema": "prism_fixed_token_reduction_v1",
        "version": get("token_reduction_version", 1),
        "selection": "checkerboard_original_positions_v1",
        "stage_boundaries": list(get("token_reduction_stage_boundaries", (4,8,12,16,20))),
        "windows": list(get("token_reduction_windows", (1,2,4,8,16,32))),
        "main_camera_id": get("token_reduction_main_camera_id", 0),
        "camera_ids": list(get("camera_compact_camera_ids", ())) or list(range(get("num_dino_cameras", 1))),
        "condition_capacity": get("token_reduction_condition_capacity", 64),
        "frozen_checkpoint_vision": get("token_reduction_frozen_vision_encoder", False),
        "wrist_front_merge": "valid_edge_mean2x2_fp32",
        "position_policy": "original_time_xy_camera_v1",
        "action_conditioning": "final_visual_tokens",
        "future_predictors_enabled": bool(get("w_compact", 0.0)),
        "pi05_conditioning": "final_compact_per_block_kv_v1",
        "future_compact_horizon_seconds": get("future_cls_horizon_seconds", None),
        "future_source_fps": get("future_cls_source_fps", None),
    }
    # Omit default extensions to retain exact legacy optimizer manifests.
    cap = get("token_reduction_max_attention_window_timesteps", None)
    if cap is not None:
        contract["max_attention_window_timesteps"] = cap
        contract["effective_windows"] = list(resolved_token_reduction_windows(config))
    granularity = get("kda_update_granularity", "token")
    if granularity != "token":
        contract["kda_update_granularity"] = granularity
        contract["timestep_kda_policy"] = "masked_mean_simultaneous_delta_v1"
    return contract


def validate_token_reduction_config(config):
    resolved_token_reduction_windows(config)
    if config.kda_update_granularity not in {"token", "timestep"}:
        raise ValueError("kda_update_granularity must be token or timestep")
    if config.kda_update_granularity == "timestep" and config.linear_attention_variant != "kda":
        raise ValueError("Timestep KDA requires linear_attention_variant=kda")
    if type(config.token_reduction_frozen_vision_encoder) is not bool:
        raise ValueError("token_reduction_frozen_vision_encoder must be a bool")
    if type(config.token_reduction_cpu_kda_reference) is not bool:
        raise ValueError("token_reduction_cpu_kda_reference must be a bool")
    if type(config.token_reduction_version) is not int or config.token_reduction_version != 1:
        raise ValueError("token_reduction_version must be 1")
    if config.qwen_keep_layers:
        raise ValueError("token_reduction does not support qwen_keep_layers remapping")
    boundaries = tuple(config.token_reduction_stage_boundaries)
    windows = tuple(config.token_reduction_windows)
    if (any(type(x) is not int or x <= 0 for x in boundaries)
            or tuple(sorted(set(boundaries))) != boundaries
            or len(windows) != len(boundaries) + 1
            or any(type(x) is not int or x <= 0 for x in windows)
            or tuple(sorted(windows)) != windows):
        raise ValueError("Token reduction requires increasing layer boundaries and positive stage windows")
    config.token_reduction_stage_boundaries, config.token_reduction_windows = boundaries, windows
    if type(config.token_reduction_main_camera_id) is not int or config.token_reduction_main_camera_id < 0:
        raise ValueError("token_reduction_main_camera_id must be a nonnegative camera identity")
    if type(config.token_reduction_condition_capacity) is not int or config.token_reduction_condition_capacity < 1:
        raise ValueError("token_reduction_condition_capacity must be positive")
    if config.compact_token_layout != "camera" or config.compact_query_mode != "native_dino":
        raise ValueError("token_reduction requires camera layout and native_dino patch features")
    if config.use_cls_tokens or config.main_stream_action_token or config.main_action_token_input != "learned":
        raise ValueError("token_reduction has no CLS/ACT/CVAE backbone tokens")
    if config.use_start_patch_visual_tokens or config.proprio_token_placement != "condition":
        raise ValueError("token_reduction requires no start-patch prefix and condition-placed proprio")
    if config.gradient_checkpointing:
        raise ValueError("Use stateless MLP checkpointing instead of whole-backbone checkpointing in token_reduction")
    if (config.train_dino_backbone or config.train_dino_cls_encoder or config.use_frozen_dino_for_cls
            or config.dino_tuning_mode != "full"):
        raise ValueError("token_reduction requires a fully frozen vision encoder with dino_tuning_mode=full")
    import math
    for name in ("w_action", "w_compact"):
        weight = getattr(config, name)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if config.use_cls_register_fusion:
        raise ValueError("token_reduction has no CLS/register fusion path")
    if config.w_cls or config.w_sigreg or config.w_patch_recon or config.global_compact_indices:
        raise ValueError("token_reduction requires w_cls=w_sigreg=w_patch_recon=0 and no global compact slots")
    if config.visual_encoder_mode not in {"dino", "dinov3"} or config.dino_camera_fusion != "concat":
        raise ValueError("token_reduction requires unmixed camera-local visual patch grids")
    if not config.main_stream_timestep_causal or not config.pass_deltanet_state_between_chunks:
        raise ValueError("token_reduction requires timestep causality and global recurrent state")
    if config.action_pred_compact_gamma != 1.0 or config.detach_pred_compact_for_action:
        raise ValueError("token_reduction conditions directly on differentiable final tokens: gamma=1, detach=false")
    if config.attn_implementation != "sdpa":
        raise ValueError("token_reduction requires SDPA attention")


def assert_token_reduction_compatible(source, target, context="Checkpoint"):
    original, requested = token_reduction_contract(source), token_reduction_contract(target)
    if original != requested:
        raise ValueError(f"{context} token-reduction architecture mismatch: {original!r} != {requested!r}")
