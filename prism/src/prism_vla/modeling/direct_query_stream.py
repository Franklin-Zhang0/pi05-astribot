"""Staged image reads for the separate causal direct-query stream.

Static image compacts are computed before query generation. The main stream's
predictions are deliberately absent from this API: image reads use only states
from the separate query stream, with the existing one-observation shift.
"""
from __future__ import annotations

import torch

from prism_vla.modeling.camera_layout import camera_validity, run_qwen


class PrismVLADirectQueryMixin:
    def _uses_direct_queries(self) -> bool:
        return getattr(self.config, "query_design", "legacy") == "direct_query"

    def _direct_query_stream_width(self) -> int:
        roles = self._direct_context_roles()
        return len(self.direct_query_layout.slots_for(roles)) + 1 + self.direct_query_layout.num_generated

    def _direct_context_roles(self) -> tuple[str, ...]:
        mode = self.config.direct_query_context
        return {"cls": (), "global_cls": ("global",),
                "global_task_cls": ("global", "task")}[mode]

    def _direct_camera_features(self, patches: torch.Tensor) -> torch.Tensor:
        """Unflatten the explicit camera inventory without merging camera K/V."""
        cameras = int(self.num_dino_cameras)
        if patches.shape[-1] != self.dino_dim or patches.shape[-2] % cameras:
            raise ValueError("Direct query patches must contain equal per-camera patch grids")
        return patches.reshape(*patches.shape[:-2], cameras, patches.shape[-2] // cameras, self.dino_dim)

    def _direct_static_compacts(self, patches, registers, task_queries, *, camera_mask=None, patch_grid_hw=None):
        """Read global/task slots over all supplied frames in parallel."""
        features = self._direct_camera_features(patches)
        prefix = features.shape[:-3]
        seeds = self.global_query_tokens
        global_count = sum(self.direct_query_layout.global_counts)
        task_count = sum(self.direct_query_layout.task_counts)
        sources = []
        if global_count:
            global_queries = self._normalize_latent(seeds).expand(*prefix, global_count, self.qwen_dim)
            sources.append(global_queries)
        if task_count:
            if task_queries is None:
                raise ValueError("Instruction-conditioned direct task queries are missing")
            if tuple(task_queries.shape) != (prefix[0], task_count, self.qwen_dim):
                raise ValueError("Direct task queries must have shape [B, task_slots, D]")
            task_queries = task_queries.reshape(prefix[0], *([1] * (len(prefix) - 1)), task_count, self.qwen_dim)
            sources.append(task_queries.expand(*prefix, task_count, self.qwen_dim))
        if sources:
            queries = torch.cat(sources, dim=-2)
        else:
            queries = features.new_empty(*prefix, 0, self.qwen_dim)
        queries = self._to_module_dtype(queries, self.direct_query_resampler)
        features = self._to_module_dtype(features, self.direct_query_resampler)
        registers = (self._to_module_dtype(registers, self.direct_query_resampler)
            if registers is not None and self.config.direct_query_include_registers else None)
        result = self.direct_query_resampler(queries, features, registers,
            camera_mask=camera_mask.bool() if camera_mask is not None else None,
            roles=("global", "task"), patch_grid_hw=patch_grid_hw)
        return self._normalize_latent(result)

    def _direct_complete_compacts(self, static, generated_queries, patches, registers,
                                  *, camera_mask=None, patch_grid_hw=None,
                                  return_cross_attn_tokens=False):
        """Append generated image reads to already computed global/task slots."""
        generated_queries = self._to_module_dtype(generated_queries, self.direct_query_resampler)
        patches = self._to_module_dtype(self._direct_camera_features(patches), self.direct_query_resampler)
        registers = (self._to_module_dtype(registers, self.direct_query_resampler)
            if registers is not None and self.config.direct_query_include_registers else None)
        generated = self.direct_query_resampler(generated_queries,
            patches, registers, camera_mask=camera_mask.bool() if camera_mask is not None else None,
            roles=("generated",), patch_grid_hw=patch_grid_hw)
        raw = torch.cat((static, generated), dim=-2)
        compact = self._normalize_latent(raw)
        if self.direct_query_layout.num_compacts == 1:
            compact = compact.squeeze(-2)
        # Direct attention emits one backbone-width feature per compact query.
        # SIGReg therefore operates on these features, not legacy 4x vision queries.
        return (compact, raw) if return_cross_attn_tokens else compact

    def _build_direct_query_inputs(self, cls, static, *, prefix_len, timestep_indices,
                                   left_pad_steps=None, camera_mask=None):
        """Pack [global/task context, CLS, independent per-camera query seeds]."""
        batch, steps, dim = cls.shape
        indices = self.direct_query_layout.indices_for(self._direct_context_roles())
        index = torch.as_tensor(indices, device=static.device, dtype=torch.long)
        context = static.index_select(-2, index)
        seeds = self.direct_generated_query_seed_embeddings.to(cls).view(1, 1, -1, dim)
        seeds = self._normalize_latent(seeds).expand(batch, steps, -1, -1)
        inputs = torch.cat((context.to(cls), cls.unsqueeze(-2), seeds), dim=-2)
        width = inputs.shape[-2]
        present = camera_validity(self, camera_mask, batch_size=batch, steps=steps, device=cls.device)
        valid = torch.cat((self.direct_query_layout.camera_mask(present, roles=self._direct_context_roles()),
            torch.ones(batch, steps, 1, device=cls.device, dtype=torch.bool),
            self.direct_query_layout.camera_mask(present, roles=("generated",))), dim=-1)
        if left_pad_steps is not None:
            timestep_indices = timestep_indices + left_pad_steps[:, None]
        pos = prefix_len + width * timestep_indices.unsqueeze(-1) + torch.arange(width, device=cls.device)
        return inputs.reshape(batch, steps * width, dim), pos.reshape(batch, -1), valid.flatten(1)

    def _project_direct_query_outputs(self, hidden):
        width = self._direct_query_stream_width()
        hidden = hidden.reshape(hidden.shape[0], -1, width, self.qwen_dim)
        generated_hidden = hidden[:, :, -self.direct_query_layout.num_generated:]
        return self._project_query(generated_hidden), generated_hidden

    def _run_direct_query_chunk(self, cls, static, *, query_cache, query_mask,
                                prefix_mask, prefix_len, timestep_indices,
                                left_pad_steps=None, camera_mask=None,
                                legacy_history_steps=None, valid_timesteps=None):
        """Run an entire causal query chunk once, independent of the main pass."""
        inputs, positions, current_valid = self._build_direct_query_inputs(cls, static,
            prefix_len=prefix_len, timestep_indices=timestep_indices,
            left_pad_steps=left_pad_steps, camera_mask=camera_mask)
        width = self._direct_query_stream_width()
        if valid_timesteps is not None:
            if (valid_timesteps.dtype != torch.bool
                    or tuple(valid_timesteps.shape) != tuple(cls.shape[:2])):
                raise ValueError("Query timestep validity must be boolean [B, T]")
            current_valid = (current_valid.reshape(cls.shape[0], cls.shape[1], width)
                & valid_timesteps.to(current_valid.device).unsqueeze(-1)).flatten(1)
        token_mask = torch.cat((query_mask, current_valid.to(query_mask.dtype)), dim=1)
        attention = token_mask
        if self._uses_blockwise_full_attention() and query_mask.shape[1] == prefix_len:
            self._validate_blockwise_attention_cache(query_cache, prefix_tokens=prefix_len, label="query")
            attention = self._build_blockwise_full_attention_mask(prefix_mask=prefix_mask,
                timestep_indices=timestep_indices, tokens_per_timestep=width)
            attention = attention & token_mask[:, None, None, :].bool()
        elif self._uses_rolling_full_attention():
            attention = self._build_rolling_attention_mask(token_mask,
                query_length=inputs.shape[1], tokens_per_timestep=width,
                prefix_tokens=prefix_len, stream="query")
        elif legacy_history_steps is not None:
            # The historical per-step sliding reference retains W preceding
            # timesteps before appending the current one. Reproduce its full
            # attention visibility in a batch without unrolling the backbone.
            cached = query_mask.shape[1]
            qi = torch.arange(inputs.shape[1], device=cls.device)
            ki = torch.arange(token_mask.shape[1], device=cls.device)
            query_step = (cached - prefix_len) // width + qi // width
            key_step = (ki - prefix_len) // width
            visible = ((ki < prefix_len)[None, :] | (
                (key_step[None, :] >= query_step[:, None] - legacy_history_steps)
                & (key_step[None, :] <= query_step[:, None])))
            visible = visible & (ki[None, :] <= cached + qi[:, None])
            attention = visible[None, None] & token_mask[:, None, None, :].bool()
        with self._cached_inference_attention_context():
            outputs = run_qwen(self, inputs_embeds=inputs, attention_mask=attention,
                camera_token_mask=current_valid, position_ids=positions,
                past_key_values=query_cache, use_cache=True)
        queries, hidden = self._project_direct_query_outputs(outputs.last_hidden_state)
        generated_valid = current_valid.reshape(cls.shape[0], cls.shape[1], width)[
            :, :, -self.direct_query_layout.num_generated:].unsqueeze(-1)
        # Invalid rows are storage only. In particular, their arbitrary SDPA
        # fallback output must not become a query if the camera reappears next.
        queries = torch.where(generated_valid, queries, 0)
        hidden = torch.where(generated_valid, hidden, 0)
        return queries, hidden, outputs.past_key_values, token_mask

    def _compute_direct_queries_from_context(self, cls, static, *, prefix_embeds,
                                             prefix_mask, camera_mask=None,
                                             sliding_reference=False):
        """Training query prefill followed by one batched causal context pass."""
        batch, steps = cls.shape[:2]
        prefix_len = prefix_embeds.shape[1]
        positions = torch.arange(prefix_len, device=cls.device).expand(batch, -1)
        with self._cached_inference_attention_context():
            outputs = run_qwen(self, inputs_embeds=prefix_embeds.to(self._qwen_dtype()),
                attention_mask=prefix_mask, position_ids=positions, use_cache=True)
        queries, hidden, _, _ = self._run_direct_query_chunk(cls.to(self._qwen_dtype()), static,
            query_cache=outputs.past_key_values, query_mask=prefix_mask, prefix_mask=prefix_mask,
            prefix_len=prefix_len, timestep_indices=torch.arange(steps, device=cls.device).expand(batch, -1),
            camera_mask=camera_mask,
            legacy_history_steps=self._attention_history_steps() if sliding_reference else None)
        return queries, hidden
