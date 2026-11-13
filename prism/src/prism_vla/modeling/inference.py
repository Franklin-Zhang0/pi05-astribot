"""Inference and KV-cache rollout path for PRISM-VLA."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from prism_vla.modeling.camera_layout import (
    camera_layout_enabled, camera_validity, camera_compact_loss,
    mask_camera_tokens, run_qwen,
)


def should_crop_attention_cache(*, completed_steps: int, interval: int) -> bool:
    """Return whether an observation reaches the explicit cache boundary."""
    if completed_steps <= 0:
        raise ValueError(f"completed_steps must be positive, got {completed_steps}.")
    if interval <= 0:
        raise ValueError(f"cache crop interval must be positive, got {interval}.")
    return completed_steps % interval == 0


def resolve_attention_cache_interval(
    requested: Optional[int], *, blockwise_full_attention: bool, window_steps: int,
    rolling_full_attention: bool = False,
) -> int:
    """Bind retention cadence to the checkpoint's explicit attention policy."""
    if rolling_full_attention and blockwise_full_attention:
        raise ValueError("Rolling and blockwise full attention are mutually exclusive.")
    interval = (window_steps if blockwise_full_attention else 1) if requested is None else int(requested)
    if interval <= 0:
        raise ValueError(f"cache_crop_interval must be positive, got {interval}.")
    if blockwise_full_attention and interval != window_steps:
        raise ValueError(
            "Blockwise inference cache cadence must match attention_window_timesteps: "
            f"cache_crop_interval={interval}, window_steps={window_steps}."
        )
    if rolling_full_attention and interval != 1:
        raise ValueError(
            "Rolling full attention requires cache_crop_interval=1 to retain "
            "exactly attention_window_timesteps-1 preceding timesteps."
        )
    return interval


class PrismVLAInferenceMixin:
    @torch.no_grad()
    def inference_reset(
        self,
        instruction_ids: torch.Tensor,
        instruction_mask: torch.Tensor,
        dino_cls_0: Optional[torch.Tensor],
        dino_patches_0: Optional[torch.Tensor],
        use_query_cache: bool = True,
        semantic_cls_0: Optional[torch.Tensor] = None,
        semantic_tokens_0: Optional[torch.Tensor] = None,
        dino_aux_tokens_0: Optional[torch.Tensor] = None,
        dino_camera_mask_0: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Initialize inference state with the instruction and frame-0 patches.

        The current frame is consumed by the first ``inference_step`` call. We
        therefore only cache the static prefix here and let step 0 append the
        current [CS_0, CLS_0, QS_0] block.
        """
        if getattr(self.config, "query_design", "legacy") == "direct_query" and not use_query_cache:
            raise ValueError("Direct queries require the separate cached causal query stream")
        # ``dino_cls_0`` stays in the API for caller compatibility even though
        # reset only caches the static prefix and step 0 consumes frame 0.
        B = instruction_ids.shape[0]
        if dino_camera_mask_0 is not None and self.use_start_patch_visual_tokens:
            raise ValueError("Optional dense camera masks require use_start_patch_visual_tokens=False")
        mdtype = self._model_dtype()
        qdtype = self._qwen_dtype()
        device = self._model_device()

        if int(getattr(self, "num_generated_query_tokens", 1)) > 1 and not use_query_cache:
            raise ValueError("num_generated_query_tokens > 1 requires use_query_cache=True.")
        if (
            self._uses_blockwise_full_attention()
            and self.use_cls_tokens
            and not use_query_cache
        ):
            raise ValueError(
                "Blockwise full attention requires use_query_cache=True so query-token "
                "inference obeys the same block boundaries as training."
            )
        if self._uses_rolling_full_attention() and self.use_cls_tokens and not use_query_cache:
            raise ValueError(
                "Rolling full attention requires use_query_cache=True so the "
                "query stream preserves global KDA state and the rolling full-attention window."
            )

        self.qwen.config.use_cache = True

        # Ensure inputs are on the correct device
        instruction_ids = instruction_ids.to(device)
        instruction_mask = instruction_mask.to(device)
        if dino_patches_0 is not None:
            dino_patches_0 = dino_patches_0.to(device)
        if semantic_tokens_0 is not None:
            semantic_tokens_0 = semantic_tokens_0.to(device)

        inst_embeds = self._encode_instruction(instruction_ids, instruction_mask).to(qdtype)
        task_queries = (
            self._compute_task_queries(inst_embeds, instruction_mask)
            if self._uses_task_query_sources()
            else None
        )
        visual_prefix = self._build_visual_prefix(
            dino_patches_0,
            semantic_tokens_0,
            mdtype=mdtype,
            qdtype=qdtype,
        )

        # Prefix used by CLS-query pass: [INST | PATCH_0]
        query_prefix_embeds = torch.cat([inst_embeds, visual_prefix], dim=1)
        query_prefix_mask = torch.cat(
            [
                self._instruction_prefix_mask(instruction_mask).long(),
                torch.ones(B, visual_prefix.shape[1], device=device, dtype=torch.long),
                ],
            dim=1,
        )
        if use_query_cache and self.use_cls_tokens:
            query_pos_prefix = torch.arange(query_prefix_embeds.shape[1], device=device).unsqueeze(0).expand(B, -1)
            with self._cached_inference_attention_context():
                query_outputs_prefix = run_qwen(self,
                    inputs_embeds=query_prefix_embeds,
                    attention_mask=query_prefix_mask,
                    position_ids=query_pos_prefix,
                    use_cache=True,
                )
            query_kv_cache = query_outputs_prefix.past_key_values
            query_cache_seq_len = query_prefix_embeds.shape[1]
            query_attn_mask = query_prefix_mask
        else:
            query_kv_cache = None
            query_cache_seq_len = 0
            query_attn_mask = None
        prefix = query_prefix_embeds
        prefix_len = query_prefix_embeds.shape[1]

        mask_2d = query_prefix_mask
        position_ids = torch.arange(prefix_len, device=device).unsqueeze(0).expand(B, -1)

        with self._cached_inference_attention_context():
            outputs = run_qwen(self,
                inputs_embeds=prefix,
                attention_mask=mask_2d,
                position_ids=position_ids,
                use_cache=True,
            )

        return {
            "kv_cache": outputs.past_key_values,
            "cache_seq_len": prefix_len,
            "attn_mask": mask_2d,
            "prefix_len": prefix_len,
            "prefix_mask": mask_2d,
            "cls_history": [],
            "inst_mask": instruction_mask,
            **self._text_deepstack_state(),
            "query_use_cache": bool(use_query_cache and self.use_cls_tokens),
            "query_kv_cache": query_kv_cache,
            "query_cache_seq_len": query_cache_seq_len,
            "query_attn_mask": query_attn_mask,
            "query_prefix_len": query_prefix_embeds.shape[1],
            "query_last": None,
            "query_prefix_embeds": query_prefix_embeds,
            "query_prefix_mask": query_prefix_mask,
            "task_queries": task_queries,
            "step": 0,
        }

    @torch.no_grad()
    def inference_refresh_instruction(
        self, state: dict[str, Any], instruction_ids: torch.Tensor,
        instruction_mask: torch.Tensor, *, prefix_refresh_mask: torch.Tensor,
    ) -> dict[str, Any]:
        """Re-encode changed text without resetting the observation history.

        Instructions must retain their padded width so cached visual RoPE
        positions remain valid. Only instruction full-attention K/V, text masks
        and selected task queries change. Prior visual K/V, KDA recurrent/conv
        state, CLS history, the previous generated query and step offsets stay.
        Consequently the old instruction's influence remains embedded in KDA
        memory; refreshing text does not replay or erase episode history.

        Sharded callers must invoke this API on the same schedule on all ranks,
        using an all-false row mask where no local instruction changed.
        """
        if self.use_start_patch_visual_tokens:
            raise ValueError("Instruction refresh requires use_start_patch_visual_tokens=False")
        old_instruction_mask = state["inst_mask"]
        if (instruction_ids.ndim != 2 or instruction_mask.ndim != 2
                or instruction_ids.shape != instruction_mask.shape
                or instruction_mask.shape != old_instruction_mask.shape):
            raise ValueError("Instruction refresh requires the same padded [B, instruction_width] inputs")
        batch, instruction_tokens = instruction_ids.shape
        device = self._model_device()
        refresh = self._instruction_refresh_mask(prefix_refresh_mask, batch_size=batch, device=device)
        instruction_ids, instruction_mask = instruction_ids.to(device), instruction_mask.to(device)
        if bool(getattr(self.config, "text_input_deepstack", False)):
            inst_embeds = self._encode_instruction(instruction_ids, instruction_mask)
            replacement = {"text_deepstack_memory": self._text_deepstack_memory,
                "text_deepstack_input_ids": self._text_deepstack_input_ids,
                "text_deepstack_attention_mask": self._text_deepstack_attention_mask}
            self._activate_text_deepstack_state(state, replacement, refresh)
            state["inst_mask"] = torch.where(refresh[:, None], instruction_mask,
                old_instruction_mask.to(instruction_mask))
            if self._uses_task_query_sources():
                task_queries = self._compute_task_queries(inst_embeds, instruction_mask)
                old_queries = state["task_queries"]
                rows = refresh.view(batch, *([1] * (task_queries.ndim - 1)))
                state["task_queries"] = torch.where(rows, task_queries, old_queries)
            return state
        old_prefix = state["query_prefix_embeds"]
        prefix_len = int(state["prefix_len"])
        if instruction_tokens != prefix_len:
            raise ValueError("Instruction refresh currently requires an instruction-only prefix")
        if (old_prefix.shape[:2] != (batch, prefix_len)
                or instruction_tokens > prefix_len
                or int(state["query_prefix_len"]) != prefix_len):
            raise ValueError("Instruction refresh requires compatible fixed main/query prefixes")
        inst_embeds = self._encode_instruction(instruction_ids, instruction_mask).to(old_prefix)
        new_prefix = torch.cat((inst_embeds, old_prefix[:, instruction_tokens:]), dim=1)
        new_prefix_mask = torch.cat((instruction_mask.to(dtype=state["prefix_mask"].dtype),
            state["prefix_mask"][:, instruction_tokens:].to(device)), dim=1)
        positions = torch.arange(prefix_len, device=device).unsqueeze(0).expand(batch, -1)
        with self._cached_inference_attention_context():
            main = run_qwen(self, inputs_embeds=new_prefix, attention_mask=new_prefix_mask,
                position_ids=positions, use_cache=True)
            query = (run_qwen(self, inputs_embeds=new_prefix, attention_mask=new_prefix_mask,
                position_ids=positions, use_cache=True)
                if state.get("query_kv_cache") is not None else None)
        state["kv_cache"] = self._refresh_qwen_instruction_rows(state["kv_cache"], main.past_key_values,
            refresh, instruction_tokens=instruction_tokens)
        if query is not None:
            state["query_kv_cache"] = self._refresh_qwen_instruction_rows(
                state["query_kv_cache"], query.past_key_values, refresh,
                instruction_tokens=instruction_tokens)
        for name in ("attn_mask", "prefix_mask", "query_attn_mask", "query_prefix_mask", "inst_mask"):
            if state.get(name) is not None:
                state[name] = self._refresh_instruction_mask_rows(state[name], new_prefix_mask,
                    refresh, instruction_tokens=instruction_tokens)
        state["query_prefix_embeds"] = torch.cat((torch.where(refresh[:, None, None],
            inst_embeds, old_prefix[:, :instruction_tokens]), old_prefix[:, instruction_tokens:]), dim=1)
        if self._uses_task_query_sources():
            task_queries = self._compute_task_queries(inst_embeds, instruction_mask)
            old_queries = state.get("task_queries")
            if old_queries is None or old_queries.shape != task_queries.shape:
                raise ValueError("Instruction refresh requires the initialized task-query inventory")
            rows = refresh.view(batch, *([1] * (task_queries.ndim - 1)))
            state["task_queries"] = torch.where(rows, task_queries, old_queries)
        return state

    @torch.no_grad()
    def inference_step(
        self,
        state: dict[str, Any],
        dino_cls_new: Optional[torch.Tensor],
        dino_patches_new: Optional[torch.Tensor],
        use_query_cache: Optional[bool] = None,
        semantic_cls_new: Optional[torch.Tensor] = None,
        semantic_tokens_new: Optional[torch.Tensor] = None,
        proprio_new: Optional[torch.Tensor] = None,
        proprio_frame_ids: Optional[torch.Tensor] = None,
        end_effector_type_ids: Optional[torch.Tensor] = None,
        engram_route_ids: Optional[torch.Tensor] = None,
        engram_state_mask: Optional[torch.Tensor] = None,
        engram_action_mask: Optional[torch.Tensor] = None,
        engram_robot_eef_mask: Optional[torch.Tensor] = None,
        flow_initial_noise: Optional[torch.Tensor] = None,
        action_latent: Optional[torch.Tensor] = None,
        flow_guidance_scale: float = 1.0,
        return_all_action_heads: bool = False,
        dino_aux_tokens_new: Optional[torch.Tensor] = None,
        dino_camera_mask_new: Optional[torch.Tensor] = None,
        cache_crop_interval: Optional[int] = None,
        decode_actions: bool = True,
        dino_registers_new: Optional[torch.Tensor] = None,
        dino_patch_grid_hw_new: Optional[tuple[int, int]] = None,
        flow_sample_batch_size: Optional[int] = None,
        instruction_ids: Optional[torch.Tensor] = None,
        instruction_mask: Optional[torch.Tensor] = None,
        prefix_refresh_mask: Optional[torch.Tensor] = None,
        action_eval_callback: Optional[Callable[..., None]] = None,
    ) -> tuple[Optional[torch.Tensor], dict[str, Any]]:
        """Single inference step: consume the current frame and decode action.

        Args:
            state: State dict from reset or previous step.
            dino_cls_new: [B, 768] DINOv3 CLS for the current frame t.
            dino_patches_new: [B, 196, 768] DINOv3 patches for the current frame t.
            dino_aux_tokens_new: [B, R, dino_dim] register or summary tokens.
            decode_actions: If False, skip only action generation. All observation,
                latent, recurrent and KV-cache updates still run normally.
                Stateful evaluation can keep one episode state and set this
                false between fixed scoring observations without consuming
                flow-sampling noise. Sharded callers must keep the decoder
                call schedule synchronized across ranks.
            flow_sample_batch_size: Optional positive maximum number of flow
                samples decoded together from [S, B, H, A] initial noise.
                Splits only the sample dimension and advances the backbone
                once. All ranks must use the same sample count and batch size.
            action_eval_callback: Internal held-out objective observer. Called
                once with the same causal decoder conditions after sampling;
                it must not mutate conditions, caches, or global RNG state.

        Returns:
            action output tensor:
                - `fast_ar`: FAST token IDs, [B, max_fast_len]
                - `flow_matching`: [B, action_horizon, action_dim], or
                  [S, B, action_horizon, action_dim] with multiple noise samples
                - None when decode_actions=False
            updated state
        """
        if type(decode_actions) is not bool:
            raise TypeError("decode_actions must be a boolean")
        if flow_sample_batch_size is not None and (
            type(flow_sample_batch_size) is not int or flow_sample_batch_size < 1
        ):
            raise ValueError("flow_sample_batch_size must be a positive integer or None")
        if not decode_actions and return_all_action_heads:
            raise ValueError("return_all_action_heads requires decode_actions=True")
        if action_eval_callback is not None and not decode_actions:
            raise ValueError("action_eval_callback requires decode_actions=True")
        if return_all_action_heads:
            if not hasattr(self.idm, "generate_all_actions"):
                raise ValueError("return_all_action_heads requires a multi-head checkpoint")
            if flow_initial_noise is not None or flow_guidance_scale != 1.0:
                raise ValueError("All-head inference uses default guidance and individual head noise")
        if self.use_cls_tokens and self.use_dino_visual:
            if dino_cls_new is None:
                raise ValueError("dino_cls_new is required by this visual_encoder_mode.")
            B = dino_cls_new.shape[0]
        elif self.use_cls_tokens:
            if semantic_cls_new is None:
                raise ValueError("semantic_cls_new is required by this visual_encoder_mode.")
            B = semantic_cls_new.shape[0]
        elif self.use_dino_visual:
            if dino_patches_new is None:
                raise ValueError("dino_patches_new is required by this visual_encoder_mode.")
            B = dino_patches_new.shape[0]
        elif self.use_semantic_visual:
            if semantic_tokens_new is None:
                raise ValueError("semantic_tokens_new is required by this visual_encoder_mode.")
            B = semantic_tokens_new.shape[0]
        elif proprio_new is not None:
            B = proprio_new.shape[0]
        else:
            raise ValueError("Unable to infer batch size for no-CLS inference step.")
        mdtype = self._model_dtype()
        qdtype = self._qwen_dtype()
        device = self._model_device()
        self._activate_text_deepstack_state(state)
        if prefix_refresh_mask is not None:
            if instruction_ids is None or instruction_mask is None:
                raise ValueError("Explicit prefix refresh requires current instruction_ids and instruction_mask")
            state = self.inference_refresh_instruction(state, instruction_ids, instruction_mask,
                prefix_refresh_mask=prefix_refresh_mask)
        elif instruction_ids is not None or instruction_mask is not None:
            raise ValueError("Changed inference instructions require an explicit prefix_refresh_mask")
        kv_cache = state["kv_cache"]
        cache_len = state["cache_seq_len"]
        step_idx = int(state.get("step", 0))
        query_use_cache = state.get("query_use_cache", True) if use_query_cache is None else bool(use_query_cache)
        empty_query_prefix = (bool(getattr(self.config, "text_input_deepstack", False))
            and int(state["query_prefix_len"]) == 0)
        generated_count = int(getattr(self, "num_generated_query_tokens", 1))
        if generated_count > 1 and (
            not query_use_cache or (state.get("query_kv_cache") is None and not empty_query_prefix)
        ):
            raise ValueError("Multiple generated queries require initialized cached query inference.")
        sliding_window_attention = bool(getattr(self.config, "sliding_window_attention", False))
        window_steps = max(1, int(getattr(self.config, "attention_window_timesteps", 16)))
        history_steps = self._attention_history_steps()
        blockwise_full_attention = self._uses_blockwise_full_attention()
        rolling_full_attention = self._uses_rolling_full_attention()
        cache_crop_interval = resolve_attention_cache_interval(
            cache_crop_interval,
            blockwise_full_attention=blockwise_full_attention,
            window_steps=window_steps,
            rolling_full_attention=rolling_full_attention,
        )
        crop_attention_cache = should_crop_attention_cache(
            completed_steps=step_idx + 1,
            interval=cache_crop_interval,
        )
        if blockwise_full_attention and self.use_cls_tokens and not query_use_cache:
            raise ValueError(
                "Blockwise full attention requires cached query-token inference."
            )
        if rolling_full_attention and self.use_cls_tokens and not query_use_cache:
            raise ValueError(
                "Rolling full attention requires cached query-token inference "
                "to preserve global KDA state and its bounded attention window."
            )
        compact_tokens = int(
            getattr(
                self,
                "compact_tokens_per_step",
                getattr(self, "compact_output_tokens", 1),
            )
        )
        query_tokens_per_step = compact_tokens + (2 if self.use_cls_tokens else 1)
        tokens_per_step = self._main_stream_tokens_per_step()
        cache_tokens_per_step = self._main_stream_cache_tokens_per_step()
        camera_present = None
        if camera_layout_enabled(self):
            camera_present = camera_validity(self, dino_camera_mask_new, batch_size=B, device=device)

        # Ensure inputs on correct device
        if dino_cls_new is not None:
            dino_cls_new = dino_cls_new.to(device)
        if dino_aux_tokens_new is not None:
            dino_aux_tokens_new = dino_aux_tokens_new.to(device)
        if dino_patches_new is not None:
            dino_patches_new = dino_patches_new.to(device)
        if dino_registers_new is not None:
            dino_registers_new = dino_registers_new.to(device)
        if semantic_cls_new is not None:
            semantic_cls_new = semantic_cls_new.to(device)
        if semantic_tokens_new is not None:
            semantic_tokens_new = semantic_tokens_new.to(device)
        if proprio_new is not None:
            proprio_new = proprio_new.to(device)
        if self.use_proprio:
            if proprio_new is None:
                raise ValueError("proprio_new is required when model.config.use_proprio=True.")
            if proprio_new.ndim != 2 or int(proprio_new.shape[-1]) != int(self.config.proprio_dim):
                raise ValueError(
                    f"Expected proprio_new [B, {self.config.proprio_dim}], "
                    f"got shape={tuple(proprio_new.shape)}"
                )

        # 1. Compute the current CLS / query tokens and use the previous query
        # to form the current compact token CS_t.
        if self.use_cls_tokens:
            project_dino_cls = getattr(self, "_project_dino_cls", None)
            dino_cls_t_embed = (
                (
                    project_dino_cls(
                        dino_cls_new.to(mdtype),
                        (
                            dino_aux_tokens_new.to(mdtype)
                            if dino_aux_tokens_new is not None
                            else None
                        ),
                    )
                    if callable(project_dino_cls)
                    else self._project_dino(dino_cls_new.to(mdtype))
                )
                if self.use_dino_visual
                else None
            )
            semantic_cls_t_embed = (
                self._project_semantic(semantic_cls_new.to(mdtype)) if self.use_semantic_visual else None
            )
            cls_t_embed = self._fuse_cls(dino_cls_t_embed, semantic_cls_t_embed)
        else:
            cls_t_embed = None

        if getattr(self.config, "query_design", "legacy") == "direct_query":
            if not query_use_cache or (state.get("query_kv_cache") is None and not empty_query_prefix):
                raise ValueError("Direct queries require initialized query inference state")
            task_queries = state.get("task_queries")
            static_compacts = self._direct_static_compacts(
                dino_patches_new.unsqueeze(1),
                dino_registers_new.unsqueeze(1) if dino_registers_new is not None else None,
                task_queries, camera_mask=camera_present.unsqueeze(1),
                patch_grid_hw=dino_patch_grid_hw_new)
            query_kv_cache = state["query_kv_cache"]
            query_prefix_len = int(state["query_prefix_len"])
            query_mask_prev = state["query_attn_mask"]
            if blockwise_full_attention and step_idx > 0 and step_idx % window_steps == 0:
                self._crop_qwen_attention_cache(query_kv_cache,
                    prefix_tokens=query_prefix_len, recent_tokens=0)
                query_mask_prev = state["query_prefix_mask"]
            generated, query_hidden, query_kv_cache_next, query_attn_mask_next = self._run_direct_query_chunk(
                cls_t_embed.to(qdtype).unsqueeze(1), static_compacts,
                query_cache=query_kv_cache, query_mask=query_mask_prev,
                prefix_mask=state["query_prefix_mask"], prefix_len=query_prefix_len,
                timestep_indices=torch.full((B, 1), step_idx, device=device, dtype=torch.long),
                camera_mask=camera_present.unsqueeze(1))
            query_t, qs_t_embed = generated[:, 0], query_hidden[:, 0]
            query_for_cs_t = query_t if step_idx == 0 else state["query_last"]
            query_cache_len_next = int(query_attn_mask_next.shape[1])
            if sliding_window_attention and not blockwise_full_attention and crop_attention_cache:
                recent = history_steps * self._direct_query_stream_width()
                query_cache_len_next = self._crop_qwen_attention_cache(query_kv_cache_next,
                    prefix_tokens=query_prefix_len, recent_tokens=recent)
                query_attn_mask_next = self._crop_attention_mask(query_attn_mask_next,
                    prefix_tokens=query_prefix_len, recent_tokens=recent)
            cs_t = self._direct_complete_compacts(static_compacts,
                query_for_cs_t.unsqueeze(1), dino_patches_new.unsqueeze(1),
                dino_registers_new.unsqueeze(1) if dino_registers_new is not None else None,
                camera_mask=camera_present.unsqueeze(1), patch_grid_hw=dino_patch_grid_hw_new)[:, 0]
        elif not self.use_cls_tokens:
            query_t_batch, qs_t_batch = self._static_queries_without_cls(
                B,
                1,
                device=device,
                dtype=qdtype,
            )
            query_t = query_t_batch[:, 0]
            qs_t_embed = qs_t_batch[:, 0]
            query_for_cs_t = query_t
            query_kv_cache_next = None
            query_cache_len_next = 0
            query_attn_mask_next = None
        elif query_use_cache and (state.get("query_kv_cache", None) is not None or empty_query_prefix):
            query_kv_cache = state["query_kv_cache"]
            query_cache_len = state["query_cache_seq_len"]
            query_mask_prev = state["query_attn_mask"]
            query_prefix_len = state["query_prefix_embeds"].shape[1]
            if blockwise_full_attention and step_idx > 0 and step_idx % window_steps == 0:
                query_cache_len = self._crop_qwen_attention_cache(
                    query_kv_cache,
                    prefix_tokens=query_prefix_len,
                    recent_tokens=0,
                )
                query_mask_prev = state["query_prefix_mask"].to(
                    device=device,
                    dtype=torch.long,
                )

            if step_idx == 0:
                # Step 0 bootstraps query_0 directly from the current CLS token.
                query_for_cs_t = None
            else:
                query_for_cs_t = state["query_last"]

            # The training/no-cache query skeleton has masked QS/CS slots between
            # CLS tokens. Do not append those masked slots to a Qwen3.5 KV cache:
            # its linear-attention recurrent state ignores masks once a previous
            # cache exists, so masked dummy tokens would change future queries.
            # Keep real CLS tokens at their logical QS positions. Optional learned
            # seeds follow each CLS and participate in the recurrent history too.
            query_block = cls_t_embed.to(qdtype).unsqueeze(1)
            query_mask_step = torch.ones(B, generated_count, device=device, dtype=torch.long)
            cls_hidden_idx = 0

            query_mask_new = torch.cat([query_mask_prev, query_mask_step], dim=1)
            query_layout_pos = (
                query_prefix_len
                + query_tokens_per_step * step_idx
                + compact_tokens
                + 1
            )
            query_pos = torch.full(
                (B, 1),
                query_layout_pos,
                device=device,
                dtype=torch.long,
            )
            if generated_count > 1:
                query_block, query_pos = self._build_generated_query_inputs(
                    query_block, query_pos
                )

            query_attention_mask = (
                self._build_rolling_attention_mask(
                    query_mask_new, query_length=generated_count,
                    tokens_per_timestep=generated_count,
                    prefix_tokens=query_prefix_len, stream="query",
                ) if rolling_full_attention else query_mask_new
            )

            with self._cached_inference_attention_context():
                query_outputs = run_qwen(self,
                    inputs_embeds=query_block,
                    attention_mask=query_attention_mask,
                    position_ids=query_pos,
                    past_key_values=query_kv_cache,
                    use_cache=True,
                )
            if generated_count > 1:
                generated_queries, primary_hidden = self._project_generated_query_outputs(
                    query_outputs.last_hidden_state
                )
                query_t, qs_t_embed = generated_queries[:, 0], primary_hidden[:, 0]
            else:
                qs_t_embed = query_outputs.last_hidden_state[:, cls_hidden_idx]
                query_t = self._project_query(qs_t_embed)
            query_for_cs_t = query_t if query_for_cs_t is None else query_for_cs_t
            query_kv_cache_next = query_outputs.past_key_values
            query_cache_len_next = query_cache_len + query_block.shape[1]
            query_attn_mask_next = query_mask_new
            if sliding_window_attention and not blockwise_full_attention and crop_attention_cache:
                query_cache_len_next = self._crop_qwen_attention_cache(
                    query_kv_cache_next,
                    prefix_tokens=query_prefix_len,
                    recent_tokens=history_steps * generated_count,
                )
                recent_len = max(0, query_cache_len_next - query_prefix_len)
                query_attn_mask_next = torch.cat(
                    [
                        state["query_prefix_mask"].to(device=device, dtype=torch.long),
                        torch.ones(B, recent_len, device=device, dtype=torch.long),
                    ],
                    dim=1,
                )
        else:
            if state["cls_history"]:
                cls_history_next = torch.cat(
                    [torch.stack(state["cls_history"], dim=1), cls_t_embed.unsqueeze(1)],
                    dim=1,
                )
            else:
                cls_history_next = cls_t_embed.unsqueeze(1)
            queries_next, cls_hidden_next = self._compute_queries_from_cls(
                cls_history_next,
                prefix_embeds=state["query_prefix_embeds"],
                prefix_mask=state["query_prefix_mask"],
            )
            query_t = queries_next[:, -1]
            qs_t_embed = cls_hidden_next[:, -1]
            query_for_cs_t = query_t if step_idx == 0 else queries_next[:, -2]
            query_kv_cache_next = None
            query_cache_len_next = 0
            query_attn_mask_next = None

        if getattr(self.config, "query_design", "legacy") == "direct_query":
            pass  # Compacts were produced by the staged direct reads above.
        elif self._uses_compact_query_sources():
            task_queries = None
            if self._uses_task_query_sources():
                task_queries = state.get("task_queries")
                if task_queries is None:
                    raise RuntimeError(
                        f"{self.query_generation_mode} inference state is missing cached task queries; "
                        "initialize it with inference_reset()."
                    )
                task_queries = task_queries.to(
                    device=query_for_cs_t.device,
                    dtype=query_for_cs_t.dtype,
                )
            query_sources_t = self._compose_compact_query_sources(
                query_for_cs_t.unsqueeze(1),
                task_queries,
            )[:, 0]
            cs_t = self._compute_compact_from_query_sources(
                query_sources_t,
                dino_patches_new,
                semantic_tokens_new,
                mdtype=mdtype,
                dino_camera_mask=dino_camera_mask_new,
            )
        else:
            cs_t = self._compute_compact_from_visuals(
                query_for_cs_t.unsqueeze(1),
                dino_patches_new,
                semantic_tokens_new,
                mdtype=mdtype,
                dino_camera_mask=dino_camera_mask_new,
            )

        # 2. Sample the prior once for this plan, shared by all action endpoints.
        # Observation-only evaluation never draws latent noise.
        action_latent_for_step = None
        if getattr(self.config, "main_action_token_input", "learned") == "cvae":
            action_latent_for_step = self._action_prior_latent(
                qs_t_embed.mean(dim=-2) if getattr(self.config, "query_design", "legacy") == "direct_query" else qs_t_embed, action_latent,
                sample=decode_actions and self.config.action_cvae_inference_mode == "sample")
        elif action_latent is not None:
            raise ValueError("action_latent requires main_action_token_input='cvae'")
        # Build new tokens for the current step.
        if self.use_cls_tokens:
            new_tokens = self._assemble_main_stream_tokens(
                cs_t.to(qdtype), cls_t_embed.to(qdtype), qs_t_embed.to(qdtype),
                proprio=proprio_new,
                proprio_frame_ids=proprio_frame_ids,
                **({"engram_route_ids": engram_route_ids, "engram_state_mask": engram_state_mask}
                   if engram_route_ids is not None or engram_state_mask is not None else {}),
                action_latent=action_latent_for_step,
            )
        else:
            new_tokens = torch.cat(
                [
                    self._compact_tokens_for_layout(cs_t).to(qdtype),
                    qs_t_embed.to(qdtype).unsqueeze(1),
                ],
                dim=1,
            )

        # 3. Forward through backbone to predict the next latent state.
        if new_tokens.shape[1] != tokens_per_step:
            raise RuntimeError(
                f"Expected {tokens_per_step} tokens in the inference step block, "
                f"got {new_tokens.shape[1]}."
            )
        prev_attn_mask = state.get("attn_mask")
        if prev_attn_mask is None:
            prev_attn_mask = torch.ones(B, cache_len, device=device, dtype=torch.long)
        else:
            prev_attn_mask = prev_attn_mask.to(device=device, dtype=torch.long)

        if blockwise_full_attention and step_idx > 0 and step_idx % window_steps == 0:
            prefix_len = int(state.get("prefix_len", cache_len))
            cache_len = self._crop_qwen_attention_cache(
                kv_cache,
                prefix_tokens=prefix_len,
                recent_tokens=0,
            )
            prev_attn_mask = state["prefix_mask"].to(device=device, dtype=torch.long)

        if sliding_window_attention:
            prefix_len = int(state.get("prefix_len", cache_len))
            layout_start = prefix_len + tokens_per_step * step_idx
            token_pos = (
                torch.arange(layout_start, layout_start + tokens_per_step, device=device)
                .unsqueeze(0)
                .expand(B, -1)
            )
            mask_2d = torch.cat(
                [prev_attn_mask, torch.ones(B, tokens_per_step, device=device, dtype=torch.long)],
                dim=1,
            )
            if camera_layout_enabled(self):
                current_valid = self._main_stream_step_layout(camera_present.unsqueeze(1), 0)[0][:, 0]
                mask_2d = torch.cat((prev_attn_mask, current_valid.to(prev_attn_mask.dtype)), dim=1)
            main_attention_mask = (
                self._build_rolling_attention_mask(
                    mask_2d, query_length=tokens_per_step,
                    tokens_per_timestep=tokens_per_step, prefix_tokens=prefix_len,
                ) if rolling_full_attention else
                self._main_stream_attention_mask(
                    mask_2d, query_length=tokens_per_step,
                    tokens_per_timestep=tokens_per_step,
                )
            )
            with self._cached_inference_attention_context():
                outputs = run_qwen(self,
                    main_stream_prefix_tokens=0,
                    inputs_embeds=new_tokens,
                    attention_mask=main_attention_mask,
                    position_ids=token_pos,
                    past_key_values=kv_cache,
                    use_cache=True,
                )
            kv_cache = outputs.past_key_values
            mask_2d = self._compress_main_stream_mask(mask_2d, tokens_per_step)
            h_cs_t, h_compact_t = self._main_stream_hidden_readouts(
                outputs.last_hidden_state, camera_present
            )
            h_qs_t = outputs.last_hidden_state[:, -1]
            if blockwise_full_attention:
                cache_seq_len_next = cache_len + cache_tokens_per_step
            elif crop_attention_cache:
                cache_seq_len_next = self._crop_qwen_attention_cache(
                    kv_cache,
                    prefix_tokens=prefix_len,
                    recent_tokens=cache_tokens_per_step * history_steps,
                )
                recent_len = max(0, cache_seq_len_next - prefix_len)
                mask_2d = torch.cat(
                    [
                        state["prefix_mask"].to(device=device, dtype=torch.long),
                        torch.ones(B, recent_len, device=device, dtype=torch.long),
                    ],
                    dim=1,
                )
                if camera_layout_enabled(self):
                    mask_2d = self._crop_attention_mask(
                        self._compress_main_stream_mask(
                            torch.cat((prev_attn_mask, current_valid.to(prev_attn_mask.dtype)), dim=1),
                            tokens_per_step,
                        ),
                        prefix_tokens=prefix_len, recent_tokens=recent_len,
                    )
            else:
                cache_seq_len_next = cache_len + cache_tokens_per_step
        else:
            position_start = (int(state.get("prefix_len", cache_len)) + tokens_per_step * step_idx
                if self._uses_main_stream_action_token() else cache_len)
            new_positions = torch.arange(
                position_start, position_start + tokens_per_step, device=device
            ).unsqueeze(0).expand(B, -1)
            step_attn_mask = torch.ones(B, tokens_per_step, device=device, dtype=torch.long)
            if camera_layout_enabled(self):
                step_attn_mask = self._main_stream_step_layout(camera_present.unsqueeze(1), 0)[0][:, 0].long()
            mask_2d = torch.cat([prev_attn_mask, step_attn_mask], dim=1)

            with self._cached_inference_attention_context():
                outputs = run_qwen(self,
                    main_stream_prefix_tokens=0,
                    inputs_embeds=new_tokens,
                    attention_mask=self._main_stream_attention_mask(
                        mask_2d, query_length=tokens_per_step,
                        tokens_per_timestep=tokens_per_step,
                    ),
                    position_ids=new_positions,
                    past_key_values=kv_cache,
                    use_cache=True,
                )
            kv_cache = outputs.past_key_values
            mask_2d = self._compress_main_stream_mask(mask_2d, tokens_per_step)
            cache_seq_len_next = cache_len + cache_tokens_per_step
            h_cs_t, h_compact_t = self._main_stream_hidden_readouts(
                outputs.last_hidden_state, camera_present
            )
            h_qs_t = outputs.last_hidden_state[:, -1]
        if self.use_cls_tokens:
            pred_cls_tp1 = self._normalize_latent(
                self.cls_head(self._to_module_dtype(h_cs_t, self.cls_head))
            )
        else:
            pred_cls_tp1 = None
        pred_compact_tp1 = self._predict_main_stream_compact(h_compact_t)
        if camera_layout_enabled(self):
            pred_compact_tp1 = mask_camera_tokens(pred_compact_tp1, camera_present,
                tokens_per_camera=self.camera_compact_tokens_per_camera, layout=getattr(self, "direct_query_layout", None))

        action_hidden = self._main_stream_action_hidden(outputs.last_hidden_state)

        # 4. IDM: generate action output
        proprio_for_idm = (
            self._project_proprio(proprio_new.to(mdtype), proprio_frame_ids).to(mdtype)
            if (self.use_proprio
                    and getattr(self.config, "action_conditioning", "legacy") != "predicted_compact") and (
                not self._uses_main_stream_action_token() or self.proprio_token_placement == "action"
            )
            else None
        )
        current_compact_condition = self._adapt_compact_for_action(cs_t.to(mdtype))
        future_compact_condition = self._adapt_compact_for_action(pred_compact_tp1).to(mdtype)
        if camera_layout_enabled(self):
            current_compact_condition = mask_camera_tokens(current_compact_condition, camera_present,
                tokens_per_camera=self.camera_compact_tokens_per_camera, layout=getattr(self, "direct_query_layout", None))
            future_compact_condition = mask_camera_tokens(future_compact_condition, camera_present,
                tokens_per_camera=self.camera_compact_tokens_per_camera, layout=getattr(self, "direct_query_layout", None))
        from prism_vla.modeling.action_expert_cache import select_action_expert_kv
        idm_cond = self._build_idm_condition_list(
            backbone_kv=select_action_expert_kv(outputs, torch.ones(B, device=device, dtype=torch.bool),
                tokens_per_step=self._main_stream_tokens_per_step()),
            end_effector_type_ids=end_effector_type_ids,
            cls_t=cls_t_embed.to(mdtype),
            action_token=action_hidden,
            compact_t=current_compact_condition,
            proprio_t=proprio_for_idm,
            cls_tp1=pred_cls_tp1.to(mdtype) if pred_cls_tp1 is not None else cls_t_embed.to(mdtype),
            compact_tp1=future_compact_condition,
            compact_t_mask=camera_present,
            compact_tp1_mask=camera_present,
        )
        if not decode_actions:
            idm_output = None
        elif getattr(self.config, "engram_native", None) is not None:
            if return_all_action_heads:
                raise ValueError("Native joint inference cannot reinterpret dormant geometric heads")
            if flow_guidance_scale != 1.0:
                raise ValueError("Native inference supports guidance_scale=1")
            native_mask = engram_state_mask if engram_action_mask is None else engram_action_mask
            if self.config.engram_native.get("mode", "joint") == "joint_and_robot_eef":
                from prism_vla.modeling.engram_native import (
                    hybrid_native_inference_mask, robot_eef_action_mask, sample_robot_eef_actions,
                )
                robot_head = self.idm.heads[self.config.engram_native["robot_eef_head"]]
                eef_mask = robot_eef_action_mask(engram_route_ids, horizon=robot_head.action_horizon,
                                                action_mask=engram_robot_eef_mask)
                native_mask = hybrid_native_inference_mask(self.engram_action_head, idm_cond,
                    embodiment_ids=engram_route_ids, state_mask=native_mask, eef_mask=eef_mask)
                if isinstance(flow_initial_noise, dict):
                    if set(flow_initial_noise) != {"native_joint", "robot_eef"}:
                        raise ValueError("Hybrid flow noise requires native_joint and robot_eef tensors")
                    native_noise, eef_noise = flow_initial_noise["native_joint"], flow_initial_noise["robot_eef"]
                else:
                    native_noise = eef_noise = flow_initial_noise
                # Both returned tensors are normalized. robot_eef uses XYZ
                # chunk offsets plus relative rotation6D (not quaternions);
                # native_joint retains body/head and actuator deltas plus
                # native-arm fallback. Serving adds current hand anchors only
                # after per-route denormalization (percentage-point hand units).
                idm_output = {
                    "native_joint": self.engram_action_head.generate_actions(
                        idm_cond, embodiment_ids=engram_route_ids, action_mask=native_mask,
                        initial_noise=native_noise),
                    "robot_eef": sample_robot_eef_actions(robot_head, idm_cond,
                        embodiment_ids=engram_route_ids, action_mask=eef_mask, initial_noise=eef_noise),
                }
            else:
                idm_output = self.engram_action_head.generate_actions(
                    idm_cond, embodiment_ids=engram_route_ids, action_mask=native_mask,
                    initial_noise=flow_initial_noise)
        elif return_all_action_heads:
            idm_output = None
        elif getattr(self.idm, "idm_type", "fast_ar") in {"oft", "act_patch"}:
            idm_output = self.idm.generate_actions(idm_cond)
        elif getattr(self.idm, "idm_type", "fast_ar") == "flow_matching":
            proprio_token_for_generation = self._idm_proprio_kwargs(
                proprio_for_idm
            ).get("proprio_token")
            if flow_initial_noise is not None and flow_initial_noise.ndim == 4:
                from prism_vla.modeling.condition_mask import repeat_conditions

                n_samples, noise_batch, noise_horizon, noise_dim = flow_initial_noise.shape
                if n_samples < 1:
                    raise ValueError("flow_initial_noise must contain at least one sample")
                if noise_batch != B:
                    raise ValueError(
                        f"flow_initial_noise batch dim {noise_batch} does not match B={B}."
                    )
                # Repeat conditions only for the current sample group. The
                # observation/query/KDA work above is shared by all samples;
                # chunking does not replay it or retain decoder activations.
                sample_batch_size = n_samples if flow_sample_batch_size is None else flow_sample_batch_size
                sampled_chunks = []
                for sample_start in range(0, n_samples, sample_batch_size):
                    noise = flow_initial_noise[sample_start:sample_start + sample_batch_size]
                    count = noise.shape[0]
                    generated = self.idm.generate_actions(
                        repeat_conditions(idm_cond, count),
                        initial_noise=noise.reshape(count * B, noise_horizon, noise_dim),
                        guidance_scale=flow_guidance_scale,
                        proprio_token=(None if proprio_token_for_generation is None else
                                       proprio_token_for_generation.repeat(count, 1)),
                    )
                    sampled_chunks.append(generated.reshape(count, B, noise_horizon, noise_dim))
                idm_output = (sampled_chunks[0] if len(sampled_chunks) == 1 else
                              torch.cat(sampled_chunks, dim=0))
            else:
                idm_output = self.idm.generate_actions(
                    idm_cond,
                    initial_noise=flow_initial_noise,
                    guidance_scale=flow_guidance_scale,
                    proprio_token=proprio_token_for_generation,
                )
        else:
            idm_output = self.idm.generate(idm_cond)

        if return_all_action_heads:
            if not hasattr(self.idm, "generate_all_actions"):
                raise ValueError("return_all_action_heads requires a multi-head checkpoint")
            if flow_initial_noise is not None:
                raise ValueError("All-head inference requires individual head noise control")
            idm_output = self.idm.generate_all_actions(idm_cond)

        if action_eval_callback is not None:
            action_eval_callback(idm_cond, **self._idm_proprio_kwargs(proprio_for_idm))

        # 5. Update state
        state["kv_cache"] = kv_cache
        state["cache_seq_len"] = cache_seq_len_next
        state["attn_mask"] = mask_2d
        state["cls_t_embed"] = cls_t_embed
        state["cs_t"] = cs_t
        state["h_qs_t"] = h_qs_t
        if action_latent_for_step is not None:
            # A replay/debug receipt only; ACT remains absent from recurrent/KV memory.
            state["last_action_latent"] = action_latent_for_step.detach()
        state["query_use_cache"] = query_use_cache
        state["query_kv_cache"] = query_kv_cache_next
        state["query_cache_seq_len"] = query_cache_len_next
        state["query_attn_mask"] = query_attn_mask_next
        state["query_last"] = query_t
        if self.use_cls_tokens and getattr(self.config, "query_design", "legacy") != "direct_query":
            state["cls_history"].append(cls_t_embed)
        state["step"] += 1

        return idm_output, state
