"""Dense visual input, fixed spatial gathers, and direct final-token actions.

This mode has one main stream. Future feature tensors are labels only; they
never enter the Qwen runner or its persistent state.
"""
from __future__ import annotations

import torch
from torch import nn

from prism_astribot_vla.modeling.condition_mask import ConditionList, repeat_conditions
from prism_astribot_vla.modeling.token_reduction_layout import FixedTokenLayout, camera_patch_mean2x2


class PrismVLATokenReductionMixin:
    def _initialize_token_reduction(self):
        from prism_astribot_vla.modeling.token_reduction_backbone import install_token_reduction_attention
        from prism_astribot_vla.modeling.token_reduction_config import token_reduction_contract
        layers = self.qwen.model.layers
        if self.config.token_reduction_stage_boundaries and self.config.token_reduction_stage_boundaries[-1] >= len(layers):
            raise ValueError("Token-reduction boundaries must lie inside the actual backbone")
        for boundary in self.config.token_reduction_stage_boundaries:
            if getattr(layers[boundary - 1], "layer_type", None) != "full_attention":
                raise ValueError("Fixed selection must follow a full-attention layer")
        # No unused resampler/query/readout parameters enter the new optimizer.
        for name in ("cls_head", "compact_head", "camera_compact_head", "compact_action_adapter",
                     "query_proj_head", "camera_compact_resampler", "direct_query_resampler",
                     "global_query_tokens", "task_query_seed_tokens", "no_cls_query_token",
                     "generated_query_seed_embeddings", "cross_attn", "cross_attn_extra",
                     "cross_attn_extra_norms", "dino_query_proj_head", "dino_native_cross_attn",
                     "dino_native_cross_attn_extra", "dino_native_cross_attn_extra_norms",
                     "dino_compact_out_proj", "dino_token_compact_out_proj", "dino_camera_compact_out_proj",
                     "dino_camera_weight_head", "cls_proj", "cls_register_attn", "cls_register_norm",
                     "cls_register_query", "cls_register_gate"):
            if hasattr(self, name):
                setattr(self, name, None)
        self.token_reduction_camera_embed = nn.Embedding(self.config.camera_compact_identity_count, self.qwen_dim)
        nn.init.normal_(self.token_reduction_camera_embed.weight, std=self.qwen_dim ** -.5)
        self.token_reduction_predictors = nn.ModuleDict()
        if self.config.w_compact:
            for camera_id in self.camera_compact_camera_ids:
                self.token_reduction_predictors[str(camera_id)] = nn.Linear(self.qwen_dim, self.dino_dim)
        for layer in layers:
            attention = getattr(layer, "linear_attn", None)
            if attention is not None:
                attention._prism_token_reduction_cpu_kda_reference = self.config.token_reduction_cpu_kda_reference
        install_token_reduction_attention(self.qwen)
        active = ("qwen.model.layers.", "qwen.model.norm.", "visual_proj.", "proprio_encoder.",
                  "idm.", "instruction_encoder.", "vlm_image_age_encoder.", "token_reduction_")
        for name, parameter in self.named_parameters():
            if not name.startswith(active):
                parameter.requires_grad_(False)
        self.token_reduction_layout_contract = token_reduction_contract(self.config)
        import json
        print('[token-reduction] ' + json.dumps(self.token_reduction_layout_contract, sort_keys=True), flush=True)

    def _token_reduction_features(self, patches, grid, camera_mask=None):
        if patches is None or patches.ndim != 4 or patches.shape[-1] != self.dino_dim:
            raise ValueError("Token reduction requires frozen patches [B,T,C*H*W,Dvision]")
        if grid is None or len(grid) != 2:
            raise ValueError("Token reduction requires explicit dino_patch_grid_hw")
        grid = tuple(int(x) for x in grid)
        batch, steps, count, _ = patches.shape
        if count != self.num_dino_cameras * grid[0] * grid[1]:
            raise ValueError("Camera patch count differs from the declared grid")
        if camera_mask is None:
            camera_mask = torch.ones(batch, steps, self.num_dino_cameras, device=patches.device, dtype=torch.bool)
        if tuple(camera_mask.shape) != (batch, steps, self.num_dino_cameras):
            raise ValueError("Camera validity must cover all supplied patch frames")
        present = camera_mask.to(device=patches.device, dtype=torch.bool)
        raw = patches.detach().reshape(batch, steps, self.num_dino_cameras, grid[0] * grid[1], self.dino_dim)
        parts, grids, masks = [], [], []
        for slot, camera_id in enumerate(self.camera_compact_camera_ids):
            value = torch.where(present[:, :, slot, None, None], raw[:, :, slot], 0)
            camera_grid = grid
            if camera_id != self.config.token_reduction_main_camera_id:
                value, camera_grid = camera_patch_mean2x2(value, grid)
            parts.append(value)
            grids.append(camera_grid)
            masks.append(present[:, :, slot, None].expand(batch, steps, value.shape[-2]))
        layout = FixedTokenLayout(tuple(grids), self.camera_compact_camera_ids,
                                  len(self.config.token_reduction_windows))
        return torch.cat(parts, -2), torch.cat(masks, -1), layout

    def _token_reduction_inference_reset(self, instruction_ids, instruction_mask):
        device = self._model_device()
        instruction_ids, instruction_mask = instruction_ids.to(device), instruction_mask.to(device)
        with_images = getattr(self.config, "vlm_image_input", False)
        if with_images:
            # Encode image and instruction together when the first observation arrives.
            prefix = torch.empty(instruction_ids.shape[0], 0, self.qwen_dim, device=device, dtype=self._qwen_dtype())
        else:
            prefix = self._encode_instruction(instruction_ids, instruction_mask).to(self._qwen_dtype())
        return {"token_reduction": True, "main_cache": None, "kv_cache": None,
                "prefix_embeds": prefix, "prefix_mask": self._instruction_prefix_mask(instruction_mask).bool(),
                "input_ids": instruction_ids.detach().clone(), "instruction_mask": instruction_mask.detach().clone(),
                "next_timestep": torch.zeros(instruction_ids.shape[0], device=device), "step": 0,
                **({} if with_images else self._text_deepstack_state())}

    def _token_reduction_refresh_instruction(self, state, ids, mask, rows):
        rows = torch.as_tensor(rows, device=ids.device, dtype=torch.bool)
        if getattr(self.config, "vlm_image_input", False):
            state = dict(state)
            pending = state.get("vlm_text_refresh_rows")
            state["vlm_text_refresh_rows"] = rows if pending is None else (pending.to(rows) | rows)
            state["input_ids"] = torch.where(rows[:, None], ids, state["input_ids"])
            state["instruction_mask"] = torch.where(rows[:, None], mask, state["instruction_mask"])
            return state
        encode = bool(rows.any())
        if not self.config.text_input_deepstack:
            # Hidden-state instruction encoders and prefix-prefill Qwen layers
            # can be sharded. Rank-local subtask changes must keep their forward
            # collectives aligned, even on peers with no changed local rows.
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                requested = torch.tensor(int(encode), device=ids.device)
                dist.all_reduce(requested, op=dist.ReduceOp.MAX)
                encode = bool(requested.item())
        if not encode:
            self._activate_text_deepstack_state(state)
            return state
        prefix = self._encode_instruction(ids, mask).to(self._qwen_dtype())
        replacement = self._text_deepstack_state()
        if self.config.text_input_deepstack:
            self._activate_text_deepstack_state(state, replacement, rows)
        else:
            from prism_astribot_vla.modeling.token_reduction_backbone import refresh_token_reduction_prefix
            if state.get("main_cache") is not None:
                refresh_token_reduction_prefix(self.qwen, state["main_cache"], prefix,
                    self._instruction_prefix_mask(mask).bool(), rows)
            if state["prefix_embeds"].shape != prefix.shape:
                raise ValueError("Cached instruction refresh requires the same padded prefix width")
            state["prefix_embeds"] = torch.where(rows[:, None, None], prefix, state["prefix_embeds"])
            state["prefix_mask"] = torch.where(rows[:, None], self._instruction_prefix_mask(mask).bool(), state["prefix_mask"])
        state["input_ids"] = torch.where(rows[:, None], ids, state["input_ids"])
        state["instruction_mask"] = torch.where(rows[:, None], mask, state["instruction_mask"])
        return state

    def _token_reduction_run(self, patches, grid, camera_mask, ids, instruction_mask,
                             *, state=None, reset_mask=None, refresh_mask=None,
                             positions=None, valid_timesteps=None, vlm_images=None, observation_timestamps_s=None, vlm_capture_timestamps_s=None, vlm_prefix_future=None):
        from prism_astribot_vla.modeling.token_reduction_backbone import run_token_reduction_backbone
        raw, valid, layout = self._token_reduction_features(patches, grid, camera_mask)
        batch, steps = raw.shape[:2]
        if state is None:
            state = self._token_reduction_inference_reset(ids, instruction_mask)
        else:
            state = dict(state)
            if not getattr(self.config, "vlm_image_input", False):
                self._activate_text_deepstack_state(state)
            changed = ((state["input_ids"].to(ids) != ids).any(-1)
                       | (state["instruction_mask"].to(instruction_mask) != instruction_mask).any(-1))
            if refresh_mask is not None:
                changed |= refresh_mask.to(changed.device).bool()
            state = self._token_reduction_refresh_instruction(state, ids, instruction_mask, changed)
        if reset_mask is not None and bool(reset_mask.any()):
            if state.get("main_cache") is not None:
                state["main_cache"].reset_rows(reset_mask)
            state["next_timestep"] = torch.where(reset_mask.to(raw.device), 0, state["next_timestep"].to(raw.device))
        if positions is None:
            positions = state["next_timestep"].to(raw.device)[:, None] + torch.arange(steps, device=raw.device)[None]
        if tuple(positions.shape) != (batch, steps):
            raise ValueError("Observation positions must have shape [B,T] in reference-cadence units")
        if valid_timesteps is not None:
            valid = valid & valid_timesteps.to(valid.device).unsqueeze(-1)
        raw = torch.where(valid.unsqueeze(-1), raw, 0)
        metadata = layout.metadata(0, device=raw.device)
        features = self._project_dino(raw)
        features = features + self.token_reduction_camera_embed(metadata.camera_ids).to(features)[None, None]
        memory = state.get("text_deepstack_memory")
        if getattr(self.config, "vlm_image_input", False):
            from prism_astribot_vla.modeling.vlm_snapshot_schedule import prepare_snapshot_memory
            primary = self.camera_compact_camera_ids.index(self.config.token_reduction_main_camera_id)
            observed = (valid.any(-1) if valid_timesteps is None else valid_timesteps.bool())
            main_present = observed if camera_mask is None else camera_mask[:, :, primary].bool()
            changed = state.pop("vlm_text_refresh_rows", None)
            if vlm_prefix_future is not None:
                memory, age_features, snapshot = vlm_prefix_future.take()
            else:
                memory, age_features, snapshot = prepare_snapshot_memory(
                    self.instruction_encoder, hz=self.config.vlm_image_update_hz, images=vlm_images,
                    seconds=observation_timestamps_s, capture_seconds=vlm_capture_timestamps_s, valid=observed, present=main_present,
                    ids=ids, mask=instruction_mask, previous=state.get("vlm_snapshot"), reset=reset_mask,
                    changed=changed, compute_age=self.config.vlm_image_age_embedding)
            state["vlm_snapshot"] = snapshot
            if self.config.vlm_image_age_embedding:
                features = features + self.vlm_image_age_encoder(
                    self._to_module_dtype(age_features, self.vlm_image_age_encoder)).to(features).unsqueeze(-2)
        features = torch.where(valid.unsqueeze(-1), features, 0)
        result = run_token_reduction_backbone(self.qwen, features, valid, positions, layout,
            cache=state.get("main_cache"), stage_boundaries=self.config.token_reduction_stage_boundaries,
            windows=self.config.token_reduction_windows, prefix_embeds=state["prefix_embeds"],
            max_attention_window_timesteps=self.config.token_reduction_max_attention_window_timesteps,
            kda_update_granularity=self.config.kda_update_granularity,
            prefix_mask=state["prefix_mask"], text_deepstack_memory=memory,
            detach_cache=True)
        state.update(main_cache=result.cache, kv_cache=result.cache)
        frame_valid = valid.any(-1)
        next_time = torch.where(frame_valid, positions.to(raw.device) + 1, -torch.inf).amax(1)
        fallback_time = (state["next_timestep"].to(next_time) if getattr(self.config, "vlm_image_input", False)
                         else positions[:, 0])
        state["next_timestep"] = torch.where(frame_valid.any(1), next_time, fallback_time)
        state["prefix_embeds"] = state["prefix_embeds"].detach()
        state["step"] = state.get("step", 0) + steps
        return result, state, raw, layout

    def _token_reduction_conditions(self, hidden, valid, *, proprio=None, proprio_frame_ids=None, end_effector_type_ids=None,
                                    engram_route_ids=None, engram_state_mask=None):
        values, masks = list(hidden.unbind(-2)), list(valid.unbind(-1))
        if self.use_proprio:
            if proprio is None:
                raise ValueError("Configured proprio is required for token-reduction action conditions")
            token = self._project_proprio(proprio.to(hidden), proprio_frame_ids,
                **({"engram_route_ids": engram_route_ids, "engram_state_mask": engram_state_mask}
                   if engram_route_ids is not None or engram_state_mask is not None else {}))
            values.append(token)
            masks.append(torch.ones(hidden.shape[0], device=hidden.device, dtype=torch.bool))
        if len(values) > self.config.token_reduction_condition_capacity:
            raise ValueError("Final visual/proprio condition count exceeds token_reduction_condition_capacity")
        return ConditionList(values, valid_mask=torch.stack(masks, -1), end_effector_type_ids=end_effector_type_ids)

    def _token_reduction_inference_step(self, *, state, dino_patches_new, dino_patch_grid_hw_new,
            dino_camera_mask_new=None, proprio_new=None, proprio_frame_ids=None, end_effector_type_ids=None,
            decode_actions=True, flow_initial_noise=None, return_all_action_heads=False,
            instruction_ids=None, instruction_mask=None, prefix_refresh_mask=None,
            action_eval_callback=None, observation_position=None, flow_guidance_scale=1.0,
            vlm_image_new=None, observation_timestamp_s=None, vlm_image_timestamp_s=None, observation_valid=None,
            flow_sample_batch_size=None, action_latent=None, engram_route_ids=None,
            engram_state_mask=None, engram_action_mask=None, engram_robot_eef_mask=None, **unused):
        if type(decode_actions) is not bool:
            raise TypeError("decode_actions must be a boolean")
        if flow_sample_batch_size is not None and (type(flow_sample_batch_size) is not int or flow_sample_batch_size < 1):
            raise ValueError("flow_sample_batch_size must be a positive integer or None")
        if action_latent is not None:
            raise ValueError("token_reduction has no CVAE action_latent input")
        if not decode_actions and (return_all_action_heads or action_eval_callback is not None):
            raise ValueError("Action diagnostics require decode_actions=True")
        native = getattr(self.config, "engram_native", None) is not None
        if native and (return_all_action_heads or flow_guidance_scale != 1.0):
            raise ValueError("Native Engram inference requires its routed heads and guidance_scale=1")
        if return_all_action_heads:
            if not hasattr(self.idm, "generate_all_actions"):
                raise ValueError("return_all_action_heads requires a multi-head checkpoint")
            if flow_initial_noise is not None or flow_guidance_scale != 1.0:
                raise ValueError("All-head inference uses default guidance and individual head noise")
        batch = dino_patches_new.shape[0]
        if flow_initial_noise is not None and not native:
            if flow_initial_noise.ndim not in (3, 4):
                raise ValueError("flow_initial_noise must be [B,H,A] or [S,B,H,A]")
            if flow_initial_noise.shape[-3] != batch or (flow_initial_noise.ndim == 4 and flow_initial_noise.shape[0] < 1):
                raise ValueError("flow_initial_noise must contain samples with the observation batch size")
        ids = state["input_ids"] if instruction_ids is None else instruction_ids
        inst_mask = state["instruction_mask"] if instruction_mask is None else instruction_mask
        snapshot_args = {}
        if getattr(self.config, "vlm_image_input", False):
            if observation_timestamp_s is None or tuple(observation_timestamp_s.shape) != (batch,):
                raise ValueError("VLM image inference requires observation_timestamp_s [B] in seconds")
            if vlm_image_new is not None and (vlm_image_new.ndim != 4 or vlm_image_new.shape[:2] != (batch, 3)):
                raise ValueError("VLM image inference requires main-camera RGB vlm_image_new [B,3,H,W]")
            if observation_valid is not None and tuple(observation_valid.shape) != (batch,):
                raise ValueError("observation_valid must be [B]")
            if vlm_image_timestamp_s is not None and tuple(vlm_image_timestamp_s.shape) != (batch,):
                raise ValueError("vlm_image_timestamp_s must be [B]")
            snapshot_args = dict(vlm_images=None if vlm_image_new is None else vlm_image_new[:, None],
                                 vlm_capture_timestamps_s=None if vlm_image_timestamp_s is None else vlm_image_timestamp_s[:, None],
                                 observation_timestamps_s=observation_timestamp_s[:, None],
                                 valid_timesteps=None if observation_valid is None else observation_valid[:, None])
        output, state, _, _ = self._token_reduction_run(dino_patches_new[:, None], dino_patch_grid_hw_new,
            None if dino_camera_mask_new is None else dino_camera_mask_new[:, None], ids, inst_mask,
            state=state, refresh_mask=prefix_refresh_mask, **snapshot_args,
            positions=None if observation_position is None else observation_position.reshape(-1, 1))
        state["token_reduction_hidden"] = output.hidden_states[:, 0]
        state["token_reduction_mask"] = output.valid[:, 0]
        if not decode_actions:
            return None, state
        conditions = self._token_reduction_conditions(output.hidden_states[:, 0], output.valid[:, 0],
            proprio=proprio_new, proprio_frame_ids=proprio_frame_ids, end_effector_type_ids=end_effector_type_ids,
            engram_route_ids=engram_route_ids, engram_state_mask=engram_state_mask)
        base_conditions = conditions
        if native:
            native_mask = engram_state_mask if engram_action_mask is None else engram_action_mask
            if self.config.engram_native.get("mode", "joint") == "joint_and_robot_eef":
                from prism_astribot_vla.modeling.engram_native import hybrid_native_inference_mask, robot_eef_action_mask, sample_robot_eef_actions
                head = self.idm.heads[self.config.engram_native["robot_eef_head"]]
                eef_mask = robot_eef_action_mask(engram_route_ids, horizon=head.action_horizon, action_mask=engram_robot_eef_mask)
                native_mask = hybrid_native_inference_mask(self.engram_action_head, conditions,
                    embodiment_ids=engram_route_ids, state_mask=native_mask, eef_mask=eef_mask)
                if isinstance(flow_initial_noise, dict):
                    if set(flow_initial_noise) != {"native_joint", "robot_eef"}:
                        raise ValueError("Hybrid flow noise requires native_joint and robot_eef tensors")
                    native_noise, eef_noise = flow_initial_noise["native_joint"], flow_initial_noise["robot_eef"]
                else:
                    native_noise = eef_noise = flow_initial_noise
                actions = dict(native_joint=self.engram_action_head.generate_actions(conditions,
                    embodiment_ids=engram_route_ids, action_mask=native_mask, initial_noise=native_noise),
                    robot_eef=sample_robot_eef_actions(head, conditions, embodiment_ids=engram_route_ids,
                        action_mask=eef_mask, initial_noise=eef_noise))
            else:
                actions = self.engram_action_head.generate_actions(conditions, embodiment_ids=engram_route_ids,
                    action_mask=native_mask, initial_noise=flow_initial_noise)
        elif return_all_action_heads:
            actions = self.idm.generate_all_actions(conditions)
        elif flow_initial_noise is not None and flow_initial_noise.ndim == 4:
            samples = flow_initial_noise.shape[0]
            group_size = samples if flow_sample_batch_size is None else flow_sample_batch_size
            chunks = []
            for start in range(0, samples, group_size):
                noise = flow_initial_noise[start:start + group_size]
                count = noise.shape[0]
                generated = self.idm.generate_actions(repeat_conditions(conditions, count),
                    initial_noise=noise.reshape(-1, *noise.shape[2:]), guidance_scale=flow_guidance_scale)
                chunks.append(generated.reshape(count, batch, *generated.shape[1:]))
            actions = torch.cat(chunks, 0)
        else:
            actions = self.idm.generate_actions(conditions, initial_noise=flow_initial_noise,
                guidance_scale=flow_guidance_scale)
        if action_eval_callback is not None:
            action_eval_callback(base_conditions)
        return actions, state
