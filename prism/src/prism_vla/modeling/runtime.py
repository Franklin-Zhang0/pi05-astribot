"""Shared runtime, projection, and visual-query helpers for PRISM-VLA."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn

from prism_vla.modeling.config import MULTI_QUERY_GENERATION_MODES
from prism_vla.modeling.attention_policy import resolved_full_attention_window_policy
from prism_vla.modeling.latent_norm import rms_normalize_last_dim
from prism_vla.modeling.losses import sigreg_epps_pulley_loss
from prism_vla.transition_rows import select_transition_rows
from prism_vla.modeling.camera_layout import (
    camera_layout_enabled, camera_slot_ids, camera_step_layout, camera_validity, expand_camera_presence,
    pool_camera_hidden, run_qwen,
)


class PrismVLARuntimeMixin:
    def _encode_instruction(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Build backbone-width instruction tokens using the selected text path.

        When enabled, IDs and masks must come from the frozen encoder's
        tokenizer. Its adapter stays in the training graph, while the frozen
        language model does not retain an autograd graph.
        """
        encoder = getattr(self, "instruction_encoder", None)
        if bool(getattr(getattr(self, "config", None), "text_input_deepstack", False)):
            if encoder is None:
                raise RuntimeError("Text input deepstack requires a frozen instruction encoder")
            old_ids = getattr(self, "_text_deepstack_input_ids", None)
            old_mask = getattr(self, "_text_deepstack_attention_mask", None)
            if (getattr(encoder, "_deepstack_cache", None) is None
                    or getattr(self, "_text_deepstack_memory", None) is None or old_ids is None
                    or old_ids.device != input_ids.device or not torch.equal(old_ids, input_ids)
                    or old_mask is None or not torch.equal(old_mask, attention_mask.bool())):
                self._text_deepstack_memory = encoder.deepstack_memory(input_ids, attention_mask)
                self._text_deepstack_input_ids = input_ids.detach().clone()
                self._text_deepstack_attention_mask = attention_mask.detach().bool().clone()
            return torch.empty(*input_ids.shape[:1], 0, self.qwen_dim,
                device=input_ids.device, dtype=self._qwen_dtype())
        if encoder is None:
            return self.qwen.model.embed_tokens(input_ids)
        return encoder(input_ids, attention_mask).to(dtype=self._qwen_dtype())

    def _instruction_prefix_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Keep external text memory out of the native token sequence."""
        return (attention_mask[:, :0] if bool(getattr(getattr(self, "config", None), "text_input_deepstack", False))
            else attention_mask)

    def _text_deepstack_state(self) -> dict:
        """Serialize runtime text identity only for the optional architecture."""
        if not bool(getattr(getattr(self, "config", None), "text_input_deepstack", False)):
            return {}
        return {name: getattr(self, "_" + name, None) for name in (
            "text_deepstack_memory", "text_deepstack_input_ids", "text_deepstack_attention_mask")}

    def _activate_text_deepstack_state(self, state, replacement_state=None, row_mask=None):
        """Restore immutable text memory when switching between cached streams."""
        if not bool(getattr(getattr(self, "config", None), "text_input_deepstack", False)):
            return
        memory = state.get("text_deepstack_memory")
        replacement = (None if replacement_state is None else
            replacement_state.get("text_deepstack_memory"))
        if memory is None:
            memory = replacement or getattr(self, "_text_deepstack_memory", None)
        elif replacement is not None and row_mask is not None:
            memory = memory.replace_rows(replacement, row_mask)
        if memory is None:
            raise RuntimeError("Cached text deepstack state is missing its instruction memory")
        state["text_deepstack_memory"] = memory
        self._text_deepstack_memory = memory
        for name in ("text_deepstack_input_ids", "text_deepstack_attention_mask"):
            current = state.get(name)
            replacement_value = (None if replacement_state is None else replacement_state.get(name))
            if current is None:
                current = replacement_value if replacement_value is not None else getattr(self, "_" + name, None)
            elif replacement_value is not None and row_mask is not None:
                if current.shape != replacement_value.shape:
                    raise ValueError("Text deepstack instruction refresh requires the same padded width")
                current = torch.where(row_mask.to(current.device)[:, None], replacement_value.to(current), current)
            state[name] = current
            setattr(self, "_" + name, current)

    def _text_deepstack_refresh_rows(self, state, input_ids, attention_mask, explicit_mask):
        """Automatically refresh changed instruction rows without resetting history."""
        if not bool(getattr(getattr(self, "config", None), "text_input_deepstack", False)) or not state:
            return explicit_mask
        old_ids = state.get("text_deepstack_input_ids")
        old_mask = state.get("text_deepstack_attention_mask")
        if old_ids is None or old_mask is None:
            raise ValueError("Text deepstack cached state requires its instruction identity")
        if old_ids.shape != input_ids.shape or old_mask.shape != attention_mask.shape:
            raise ValueError("Text deepstack instruction refresh requires the same padded width")
        changed = ((old_ids.to(input_ids.device) != input_ids).any(-1)
            | (old_mask.to(attention_mask.device) != attention_mask.bool()).any(-1))
        if explicit_mask is not None:
            changed = changed | self._instruction_refresh_mask(explicit_mask,
                batch_size=input_ids.shape[0], device=input_ids.device)
        return changed

    def _uses_compact_query_sources(self) -> bool:
        return getattr(self, "query_generation_mode", "cls") in (
            MULTI_QUERY_GENERATION_MODES
        )

    def _uses_task_query_sources(self) -> bool:
        return getattr(self, "query_generation_mode", "cls") in {
            "global_task_cls",
            "task_cls",
        }

    def _uses_blockwise_full_attention(self) -> bool:
        return resolved_full_attention_window_policy(self.config) == "blockwise"

    def _uses_rolling_full_attention(self) -> bool:
        return resolved_full_attention_window_policy(self.config) == "rolling"

    def _attention_history_steps(self) -> int:
        """Persist W-1 previous timesteps so the next current timestep makes W."""
        window = max(1, int(getattr(self.config, "attention_window_timesteps", 16)))
        return window - 1 if self._uses_rolling_full_attention() else window

    def _build_rolling_attention_mask(
        self,
        token_mask: torch.Tensor,
        *,
        query_length: int,
        tokens_per_timestep: int,
        prefix_tokens: int,
        stream: str = "main",
        prefix_query_tokens: int = 0,
    ) -> torch.Tensor:
        """Enforce every query's rolling lookback, including inside long chunks.

        Keys contain the instruction prefix, prior persistent timesteps, then
        current logical timesteps. Current ACT slots exist only for their own
        queries; prior ACT slots are absent. Camera/padding slots keep their
        positions and existing validity. Relative timestep identities suffice
        for contiguous retained history, including masked row-reset history.
        This mask never crops or resets DeltaNet's global recurrent state.
        """
        if token_mask.ndim != 2 or stream not in {"main", "query"}:
            raise ValueError("Rolling attention requires [B,K] validity and main/query stream")
        query_length, width, prefix, query_prefix = map(int,
            (query_length, tokens_per_timestep, prefix_tokens, prefix_query_tokens))
        keys = int(token_mask.shape[-1])
        cached = keys - query_length
        if (width < 1 or prefix < 0 or query_prefix < 0 or query_prefix > query_length
                or cached < 0 or (query_length - query_prefix) % width
                or (query_prefix and (cached != 0 or query_prefix != prefix))
                or (not query_prefix and cached < prefix)):
            raise ValueError("Rolling inputs require a static prefix and complete current timesteps")
        cached_prefix = prefix - query_prefix
        cache_width = width
        if stream == "main" and self._uses_main_stream_action_token():
            cache_width = self._main_stream_cache_tokens_per_step()
        if cache_width < 1 or (cached - cached_prefix) % cache_width:
            raise ValueError("Rolling cached history must contain complete persistent timesteps")
        cached_steps = (cached - cached_prefix) // cache_width
        window = int(getattr(self.config, "attention_window_timesteps", 16))
        if window < 1:
            raise ValueError("Rolling attention_window_timesteps must be positive")
        qi = torch.arange(query_length, device=token_mask.device)
        ki = torch.arange(keys, device=token_mask.device)
        qsteps = cached_steps + torch.div(qi - query_prefix, width, rounding_mode="floor")
        historical_steps = torch.div(ki - cached_prefix, cache_width, rounding_mode="floor")
        current_steps = cached_steps + torch.div(ki - cached - query_prefix, width, rounding_mode="floor")
        ksteps = torch.where(ki < cached, historical_steps, current_steps)
        allowed = (ksteps[None, :] <= qsteps[:, None]) & (
            ksteps[None, :] >= qsteps[:, None] - window + 1)
        prefix_keys = ki < prefix
        if not (stream == "main" and self._uses_main_stream_timestep_causal()):
            allowed &= ki[None, :] <= cached + qi[:, None]
        allowed |= prefix_keys[None, :]
        if query_prefix:
            prefix_causal = prefix_keys[None, :] & (ki[None, :] <= qi[:, None])
            allowed = torch.where((qi < query_prefix)[:, None], prefix_causal, allowed)
        result = allowed[None, None] & token_mask[:, None, None, :].bool()
        if stream == "main" and self._uses_main_stream_action_token():
            from prism_vla.modeling.act_observer import mask_action_observer_attention
            slots = (qi - query_prefix) % width
            action_end = width - self._query_tail_tokens_per_step()
            first_action = action_end - self._main_stream_action_tokens_per_step()
            observers = ((qi >= query_prefix) & (slots >= first_action) & (slots < action_end))
            result = mask_action_observer_attention(
                result, observers[None].expand(token_mask.shape[0], -1))
        return result

    def _uses_main_stream_timestep_causal(self) -> bool:
        return bool(getattr(self.config, "main_stream_timestep_causal", False))

    def _uses_main_stream_action_token(self) -> bool:
        return bool(getattr(getattr(self, "config", None), "main_stream_action_token", False))

    def _uses_action_cvae(self) -> bool:
        return getattr(getattr(self, "config", None), "main_action_token_input", "learned") == "cvae"

    def _main_stream_action_tokens_per_step(self) -> int:
        """One observer per chunk for experts, or per action step for act_patch."""
        if not self._uses_main_stream_action_token():
            return 0
        if getattr(self.config, "idm_type", "") in {"act_patch", "patchwam"}:
            horizon = int(self.config.action_horizon)
            grouping = int(getattr(self.config, "action_tokens_per_step", 1))
            if grouping < 1 or horizon % grouping:
                raise ValueError("Direct ACT horizon must contain complete physical action steps")
            return horizon // grouping
        return 1

    def _action_prior_latent(self, query: torch.Tensor, action_latent=None, *, sample: bool = False):
        if not self._uses_action_cvae():
            if action_latent is not None:
                raise ValueError("action_latent requires main_action_token_input='cvae'")
            return None
        shape = (*query.shape[:-1], int(self.config.action_cvae_latent_dim))
        if action_latent is not None:
            if tuple(action_latent.shape) != shape:
                raise ValueError(f"action_latent must have shape {shape}, got {tuple(action_latent.shape)}")
            return action_latent.to(device=query.device, dtype=torch.float32)
        if sample:
            return torch.randn(shape, device=query.device, dtype=torch.float32)
        return torch.zeros(shape, device=query.device, dtype=torch.float32)

    def _query_tail_tokens_per_step(self) -> int:
        """Number of generated query states retained in each main timestep."""
        if getattr(getattr(self, "config", None), "query_design", None) == "direct_query":
            return int(self.direct_query_layout.num_generated)
        return 1

    def _main_stream_tokens_per_step(self) -> int:
        """Return the main stride, independently of the query stream's stride."""
        return (int(self.compact_tokens_per_step)
                + int(bool(getattr(self, "use_cls_tokens", True)))
                + self._query_tail_tokens_per_step()
                + self._main_stream_action_tokens_per_step())

    def _main_stream_cache_tokens_per_step(self) -> int:
        """Return physical KV width; ACT is an ephemeral observer query."""
        return self._main_stream_tokens_per_step() - self._main_stream_action_tokens_per_step()

    def _compress_main_stream_mask(
        self, mask: torch.Tensor, query_length: int, prefix_query_tokens: int = 0,
    ) -> torch.Tensor:
        """Remove this call's ACT key columns after its physical KV update.

        Cached columns already exclude ACT. The current call can include a
        static prefix followed by complete logical timesteps. Missing camera
        slots retain their existing masked storage and positional gaps.
        """
        if not self._uses_main_stream_action_token():
            return mask
        query_length, prefix_query_tokens = int(query_length), int(prefix_query_tokens)
        width = self._main_stream_tokens_per_step()
        if (mask.ndim != 2 or query_length < prefix_query_tokens or prefix_query_tokens < 0
                or query_length > mask.shape[-1] or (query_length - prefix_query_tokens) % width):
            raise ValueError("Main cache mask requires a prefix and complete current timesteps")
        cached = int(mask.shape[-1]) - query_length
        prefix_end = cached + prefix_query_tokens
        steps = mask[:, prefix_end:].reshape(mask.shape[0], -1, width)
        # Fixed slices avoid CUDA boolean-index shape synchronization.
        action_end = width - self._query_tail_tokens_per_step()
        first_action = action_end - self._main_stream_action_tokens_per_step()
        compact = torch.cat((steps[..., :first_action], steps[..., action_end:]), dim=-1).flatten(1)
        return torch.cat((mask[:, :prefix_end], compact), dim=1)

    def _assemble_main_stream_tokens(
        self, compact: torch.Tensor, cls: torch.Tensor, query: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        action_latent: Optional[torch.Tensor] = None,
        proprio_frame_ids: Optional[torch.Tensor] = None,
        engram_route_ids: Optional[torch.Tensor] = None,
        engram_state_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pack [compacts, CLS, (ACT), Q] with an isolated optional CVAE input.

        Condition-placed proprio is fused into ACT before the backbone, keeping
        it available through the sole expert condition. Action-placed proprio
        continues to use the expert's existing action-sequence input.
        """
        direct = getattr(getattr(self, "config", None), "query_design", None) == "direct_query"
        reference = cls if direct else query
        if direct and tuple(query.shape) != (*cls.shape[:-1], self._query_tail_tokens_per_step(), cls.shape[-1]):
            raise ValueError("Direct-query tail must match CLS batch dimensions and configured query count")
        if direct:
            expected = (*cls.shape[:-1], int(self.compact_tokens_per_step), self.qwen_dim)
            if self.compact_tokens_per_step == 1 and tuple(compact.shape) == tuple(cls.shape):
                compact = compact.unsqueeze(-2)
            if tuple(compact.shape) != expected:
                raise ValueError(f"Direct-query compacts must have shape {expected}, got {tuple(compact.shape)}")
            compact_layout = compact
        else:
            compact_layout = self._compact_tokens_for_layout(compact)
        if (getattr(getattr(self, "config", None), "action_conditioning", "legacy") == "predicted_compact"
                and getattr(self, "use_proprio", False)):
            # The legacy route injects state directly into action decoders.
            # With the compact bottleneck, state must enter *before* Qwen so
            # neither continuous nor FAST heads get a separate bypass.
            expected = (*cls.shape[:-1], int(self.config.proprio_dim))
            if proprio is None or tuple(proprio.shape) != expected:
                raise ValueError(f"Predicted-compact backbone proprio must have shape {expected}")
            projected = self._project_proprio(proprio.reshape(-1, expected[-1]).to(reference.device),
                None if proprio_frame_ids is None else proprio_frame_ids.reshape(-1),
                **({"engram_route_ids": None if engram_route_ids is None else engram_route_ids.reshape(-1),
                    "engram_state_mask": None if engram_state_mask is None else engram_state_mask.reshape(-1, expected[-1])}
                   if engram_route_ids is not None or engram_state_mask is not None else {}))
            projected = projected.reshape(*cls.shape[:-1], self.qwen_dim).to(query)
            query = query + (projected.unsqueeze(-2) if direct else projected)
        parts = [compact_layout.to(reference)]
        if getattr(self, "use_cls_tokens", True):
            parts.append(cls.to(reference).unsqueeze(-2))
        if self._uses_main_stream_action_token():
            if self._uses_action_cvae():
                latent = self._action_prior_latent(reference, action_latent)
                action = self.action_cvae_projection(
                    latent.to(self.action_cvae_projection.weight.dtype)).to(reference)
            else:
                if action_latent is not None:
                    raise ValueError("action_latent requires main_action_token_input='cvae'")
                action = self.main_action_token.to(reference).expand_as(reference)
            if (getattr(self, "use_proprio", False)
                    and self.proprio_token_placement == "condition"):
                if proprio is None:
                    raise ValueError("ACT input requires current proprio when proprio_token_placement='condition'")
                expected = (*reference.shape[:-1], int(self.config.proprio_dim))
                if tuple(proprio.shape) != expected:
                    raise ValueError(f"ACT proprio must have shape {expected}, got {tuple(proprio.shape)}")
                proprio_rows = proprio.reshape(-1, expected[-1]).to(reference.device)
                if self._uses_action_cvae():
                    # Inactive padded rows may have no measured state. Keep
                    # their private observer and zero gradients finite too.
                    proprio_rows = torch.nan_to_num(proprio_rows, nan=0.0, posinf=0.0, neginf=0.0)
                projected = self._project_proprio(proprio_rows,
                    None if proprio_frame_ids is None else proprio_frame_ids.reshape(-1),
                    **({"engram_route_ids": None if engram_route_ids is None else engram_route_ids.reshape(-1),
                        "engram_state_mask": None if engram_state_mask is None else engram_state_mask.reshape(-1, expected[-1])}
                       if engram_route_ids is not None or engram_state_mask is not None else {}))
                action = action + projected.reshape_as(reference).to(reference)
            if getattr(self, "main_action_step_embeddings", None) is not None:
                action = action.unsqueeze(-2) + self.main_action_step_embeddings.to(reference)
            else:
                action = action.unsqueeze(-2)
            parts.append(action)
        parts.append(query.to(reference) if direct else query.unsqueeze(-2))
        return torch.cat(parts, dim=-2)

    def _main_stream_step_layout(self, present: torch.Tensor, start_position):
        if getattr(getattr(self, "config", None), "query_design", None) == "direct_query":
            if present.ndim != 3:
                raise ValueError("Camera inventory must be [B, T, C]")
            batch, steps, _ = present.shape
            layout = self.direct_query_layout
            compact = layout.camera_mask(present)
            query_tail = layout.camera_mask(present, roles=("generated",))
            special = torch.ones(batch, steps,
                int(bool(getattr(self, "use_cls_tokens", True))) + self._main_stream_action_tokens_per_step(),
                device=present.device, dtype=torch.bool)
            valid = torch.cat((compact, special, query_tail), dim=-1)
            start = torch.as_tensor(start_position, device=present.device, dtype=torch.long)
            if start.ndim == 0:
                start = start.expand(batch)
            if tuple(start.shape) != (batch,):
                raise ValueError("Each stream needs one starting logical token position")
            flat = valid.flatten(1)
            positions = (start[:, None] + torch.arange(flat.shape[1], device=present.device)[None]).reshape_as(valid)
            return valid, positions, start + flat.shape[1]
        return camera_step_layout(present, start_position,
            tokens_per_camera=self.camera_compact_tokens_per_camera,
            special_tokens=2 + self._main_stream_action_tokens_per_step())

    def _main_stream_action_hidden(self, hidden: torch.Tensor) -> Optional[torch.Tensor]:
        """Read only the ACT position; visual prediction maps never process it."""
        if not self._uses_main_stream_action_token():
            return None
        if hidden.shape[-2:] != (self._main_stream_tokens_per_step(), self.qwen_dim):
            raise ValueError("ACT readout requires a complete main-stream timestep")
        first_action = int(self.compact_tokens_per_step) + int(bool(getattr(self, "use_cls_tokens", True)))
        if getattr(self, "main_action_step_embeddings", None) is not None:
            action = hidden[..., first_action:first_action + self._main_stream_action_tokens_per_step(), :]
        else:
            action = hidden[..., first_action, :]
        if bool(getattr(getattr(self, "idm", None), "requires_raw_action_token", False)):
            return action
        return self._normalize_latent(action)

    def _main_stream_attention_mask(
        self,
        token_mask: torch.Tensor,
        *,
        query_length: int,
        tokens_per_timestep: int,
        prefix_query_tokens: int = 0,
    ) -> torch.Tensor:
        """Expose whole current timesteps in main-stream full attention only.

        Cached keys retain their existing window/block policy and are all
        earlier than the supplied timesteps. Static prefix queries keep their
        ordinary token-causal visibility. This helper never handles the
        separate CLS/query-generation stream or changes recurrent state.
        """
        if not self._uses_main_stream_timestep_causal():
            return token_mask
        if token_mask.ndim != 2:
            raise ValueError("Main-stream token_mask must have shape [B, K]")
        query_length = int(query_length)
        tokens_per_timestep = int(tokens_per_timestep)
        prefix_query_tokens = int(prefix_query_tokens)
        if (tokens_per_timestep <= 0 or prefix_query_tokens < 0
                or query_length < prefix_query_tokens
                or (query_length - prefix_query_tokens) % tokens_per_timestep
                or query_length > token_mask.shape[-1]):
            raise ValueError("Main-stream inputs must contain a prefix and complete timesteps")
        cached_tokens = int(token_mask.shape[-1]) - query_length
        query_indices = torch.arange(query_length, device=token_mask.device)
        key_indices = torch.arange(token_mask.shape[-1], device=token_mask.device)
        current_key_indices = key_indices - cached_tokens
        query_steps = torch.div(
            query_indices - prefix_query_tokens, tokens_per_timestep, rounding_mode="floor"
        )
        key_steps = torch.div(
            current_key_indices - prefix_query_tokens, tokens_per_timestep, rounding_mode="floor"
        )
        step_allowed = query_steps[:, None] >= key_steps[None, :]
        prefix_allowed = key_indices[None, :] <= (cached_tokens + query_indices[:, None])
        allowed = torch.where(
            (query_indices < prefix_query_tokens)[:, None], prefix_allowed, step_allowed
        )
        mask = allowed[None, None] & token_mask[:, None, None, :].bool()
        if self._uses_main_stream_action_token():
            from prism_vla.modeling.act_observer import mask_action_observer_attention
            slots = (query_indices - prefix_query_tokens) % tokens_per_timestep
            action_end = tokens_per_timestep - self._query_tail_tokens_per_step()
            first_action = action_end - self._main_stream_action_tokens_per_step()
            observer = ((query_indices >= prefix_query_tokens)
                & (slots >= first_action) & (slots < action_end))
            mask = mask_action_observer_attention(mask, observer.unsqueeze(0).expand(mask.shape[0], -1))
        return mask

    def _is_blockwise_full_attention_boundary(self, timestep: int) -> bool:
        block_steps = max(
            1,
            int(getattr(self.config, "attention_window_timesteps", 16)),
        )
        timestep = int(timestep)
        return self._uses_blockwise_full_attention() and timestep > 0 and timestep % block_steps == 0

    def _build_blockwise_full_attention_mask(
        self,
        *,
        prefix_mask: torch.Tensor,
        timestep_indices: torch.Tensor,
        tokens_per_timestep: int,
        timestep_causal: bool = False,
    ):
        """Build a causal block-diagonal mask for cached full-attention layers.

        The cached keys must contain only the static prefix. Current tokens can
        see valid prefix tokens and earlier tokens from their own temporal
        block. Qwen3.5 DeltaNet layers ignore this mask once their recurrent
        cache exists, so their state remains global across blocks.
        """
        if prefix_mask.ndim != 2:
            raise ValueError(
                f"prefix_mask must have shape [B, P], got {tuple(prefix_mask.shape)}"
            )
        if timestep_indices.ndim != 2:
            raise ValueError(
                "timestep_indices must have shape [B, T], "
                f"got {tuple(timestep_indices.shape)}"
            )
        batch_size, num_steps = timestep_indices.shape
        if int(prefix_mask.shape[0]) != int(batch_size):
            raise ValueError(
                "prefix_mask and timestep_indices batch sizes differ: "
                f"{prefix_mask.shape[0]} != {batch_size}"
            )
        tokens_per_timestep = int(tokens_per_timestep)
        if tokens_per_timestep <= 0:
            raise ValueError(
                f"tokens_per_timestep must be positive, got {tokens_per_timestep}"
            )
        block_steps = int(getattr(self.config, "attention_window_timesteps", 16))
        if block_steps <= 0:
            raise ValueError(
                "attention_window_timesteps must be positive for blockwise full attention, "
                f"got {block_steps}"
            )

        token_block_ids = torch.div(
            timestep_indices.to(dtype=torch.long),
            block_steps,
            rounding_mode="floor",
        ).repeat_interleave(tokens_per_timestep, dim=1)
        query_length = int(num_steps) * tokens_per_timestep
        if timestep_causal:
            token_steps = timestep_indices.repeat_interleave(tokens_per_timestep, dim=1)
            causal = token_steps.unsqueeze(2) >= token_steps.unsqueeze(1)
        else:
            token_indices = torch.arange(query_length, device=timestep_indices.device)
            causal = token_indices.view(1, query_length, 1) >= token_indices.view(1, 1, query_length)
        same_block = token_block_ids.unsqueeze(2) == token_block_ids.unsqueeze(1)
        current_allowed = same_block & causal
        prefix_allowed = prefix_mask.to(device=timestep_indices.device, dtype=torch.bool)
        prefix_allowed = prefix_allowed.unsqueeze(1).expand(batch_size, query_length, -1)
        dense_mask = torch.cat([prefix_allowed, current_allowed], dim=-1).unsqueeze(1)
        if timestep_causal and self._uses_main_stream_action_token():
            from prism_vla.modeling.act_observer import mask_action_observer_attention
            indices = torch.arange(query_length, device=timestep_indices.device)
            slots = indices % tokens_per_timestep
            action_end = tokens_per_timestep - self._query_tail_tokens_per_step()
            first_action = action_end - self._main_stream_action_tokens_per_step()
            observer = (slots >= first_action) & (slots < action_end)
            dense_mask = mask_action_observer_attention(
                dense_mask, observer.unsqueeze(0).expand(batch_size, -1))

        backend = str(
            getattr(self.config, "blockwise_full_attention_backend", "dense_sdpa")
        ).strip().lower()
        if backend != "dense_sdpa":
            raise ValueError(
                "This branch only supports blockwise_full_attention_backend='dense_sdpa', "
                f"got {backend!r}."
            )
        return dense_mask

    def _validate_blockwise_attention_cache(self, cache, *, prefix_tokens: int, label: str) -> None:
        if (cache is None and prefix_tokens == 0
                and bool(getattr(getattr(self, "config", None), "text_input_deepstack", False))):
            return
        if cache is None or not bool(getattr(cache, "has_previous_state", False)):
            raise RuntimeError(
                f"{label} blockwise attention requires a primed DeltaNet recurrent cache."
            )
        cache_length = int(cache.get_seq_length())
        if cache_length != int(prefix_tokens):
            raise RuntimeError(
                f"{label} blockwise attention requires prefix-only full-attention KV, "
                f"got cache length {cache_length} and prefix length {prefix_tokens}."
            )

    def _model_dtype(self) -> torch.dtype:
        if isinstance(self.visual_proj, nn.Linear):
            return self.visual_proj.weight.dtype
        return self.qwen.model.embed_tokens.weight.dtype

    def _qwen_dtype(self) -> torch.dtype:
        return self.qwen.model.embed_tokens.weight.dtype

    def _model_device(self) -> torch.device:
        if isinstance(self.visual_proj, nn.Linear):
            return self.visual_proj.weight.device
        return self.qwen.model.embed_tokens.weight.device

    def _cached_inference_attention_context(self):
        """Select the configured SDPA backend for existing KV-cache calls."""
        if str(getattr(self.config, "attn_implementation", "")) != "sdpa":
            return nullcontext()

        backend_name = str(
            getattr(self.config, "cached_sdpa_backend", "math")
        ).strip().lower()
        supported_backends = {"math", "efficient", "cudnn"}
        if backend_name not in supported_backends:
            raise ValueError(
                f"Unsupported cached_sdpa_backend={backend_name!r}; expected one of "
                f"{sorted(supported_backends)}."
            )
        if not torch.cuda.is_available():
            return nullcontext()

        enum_names = {
            "math": "MATH",
            "efficient": "EFFICIENT_ATTENTION",
            "cudnn": "CUDNN_ATTENTION",
        }
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
        except (ImportError, AttributeError):
            SDPBackend = None
            sdpa_kernel = None

        if SDPBackend is not None and sdpa_kernel is not None:
            backend = getattr(SDPBackend, enum_names[backend_name], None)
            if backend is None:
                raise RuntimeError(
                    f"PyTorch does not expose the requested cached SDPA backend "
                    f"{backend_name!r}."
                )
            return sdpa_kernel([backend])

        backend_flags = {
            "enable_flash": False,
            "enable_math": backend_name == "math",
            "enable_mem_efficient": backend_name == "efficient",
            "enable_cudnn": backend_name == "cudnn",
        }
        try:
            return torch.backends.cuda.sdp_kernel(**backend_flags)
        except TypeError as exc:
            if backend_name == "cudnn":
                raise RuntimeError(
                    "This PyTorch version cannot explicitly select cuDNN SDPA."
                ) from exc
            backend_flags.pop("enable_cudnn")
            return torch.backends.cuda.sdp_kernel(**backend_flags)

    @staticmethod
    def _crop_qwen_attention_cache(
        cache,
        *,
        prefix_tokens: int,
        recent_tokens: int,
        detach: bool = False,
    ) -> int:
        """Crop full-attention KV cache while preserving DeltaNet memory state.

        Qwen3.5's cache stores full-attention keys/values in tensors with a
        sequence dimension, and DeltaNet memory in fixed-shape conv/recurrent
        tensors. This helper trims only the attention KV tensors to
        ``prefix + recent`` tokens and keeps conv/recurrent states so older
        tokens remain available to DeltaNet memory. When ``detach`` is true,
        all cached tensors retain their values for future chunks but stop
        gradients at the chunk boundary.
        """
        if cache is None:
            return 0
        if not hasattr(cache, "key_cache") or not hasattr(cache, "value_cache"):
            return int(cache.get_seq_length())

        prefix_tokens = max(0, int(prefix_tokens))
        recent_tokens = max(0, int(recent_tokens))
        new_seq_len = 0
        for layer_idx, key in enumerate(cache.key_cache):
            value = cache.value_cache[layer_idx]
            if key is None or value is None:
                continue
            if key.ndim < 3 or value.ndim < 3:
                continue
            seq_len = int(key.shape[-2])
            if seq_len <= prefix_tokens + recent_tokens:
                if detach:
                    cache.key_cache[layer_idx] = key.detach()
                    cache.value_cache[layer_idx] = value.detach()
                new_seq_len = max(new_seq_len, seq_len)
                continue

            keep_prefix = min(prefix_tokens, seq_len)
            recent_start = max(keep_prefix, seq_len - recent_tokens)
            if keep_prefix > 0:
                cropped_key = torch.cat(
                    [key[..., :keep_prefix, :], key[..., recent_start:, :]],
                    dim=-2,
                )
                cropped_value = torch.cat(
                    [value[..., :keep_prefix, :], value[..., recent_start:, :]],
                    dim=-2,
                )
            else:
                cropped_key = key[..., recent_start:, :]
                cropped_value = value[..., recent_start:, :]
            if detach:
                cropped_key = cropped_key.detach()
                cropped_value = cropped_value.detach()
            cache.key_cache[layer_idx] = cropped_key
            cache.value_cache[layer_idx] = cropped_value
            new_seq_len = max(new_seq_len, int(cache.key_cache[layer_idx].shape[-2]))
        if detach:
            PrismVLARuntimeMixin._detach_qwen_state_cache(cache)
        return new_seq_len or int(cache.get_seq_length())

    @staticmethod
    def _detach_qwen_state_cache(cache) -> None:
        """Detach non-attention cached tensors, including DeltaNet states."""

        def detach_value(value):
            if torch.is_tensor(value):
                return value.detach()
            if isinstance(value, list):
                return [detach_value(item) for item in value]
            if isinstance(value, tuple):
                return tuple(detach_value(item) for item in value)
            if isinstance(value, dict):
                return {key: detach_value(item) for key, item in value.items()}
            return value

        for attr in ("conv_states", "recurrent_states"):
            if hasattr(cache, attr):
                value = getattr(cache, attr)
                if value is not None:
                    setattr(cache, attr, detach_value(value))

    @staticmethod
    def _clone_qwen_cache(cache, *, detach: bool = False):
        """Clone a Qwen cache object so in-place cache updates cannot mutate the source."""
        if cache is None:
            return None

        def clone_value(value):
            if torch.is_tensor(value):
                value = value.detach() if detach else value
                return value.clone()
            if isinstance(value, list):
                return [clone_value(item) for item in value]
            if isinstance(value, tuple):
                return tuple(clone_value(item) for item in value)
            if isinstance(value, dict):
                return {key: clone_value(item) for key, item in value.items()}
            return value

        cloned = cache.__class__.__new__(cache.__class__)
        for attr, value in vars(cache).items():
            setattr(cloned, attr, clone_value(value))
        return cloned

    @staticmethod
    def _restore_qwen_cache(cache, snapshot):
        """Restore cache attributes from a snapshot clone."""
        if cache is None or snapshot is None:
            return cache
        for attr, value in vars(snapshot).items():
            setattr(cache, attr, value)
        return cache

    @staticmethod
    def _replace_tensor_rows(
        current: torch.Tensor,
        replacement: torch.Tensor,
        reset_mask: torch.Tensor,
    ) -> torch.Tensor:
        if current.numel() == 0 and tuple(current.shape) == tuple(replacement.shape):
            return current
        replacement = replacement.to(device=current.device, dtype=current.dtype)
        if current.ndim >= 3 and replacement.ndim >= 3 and current.shape[-2] != replacement.shape[-2]:
            current_seq = int(current.shape[-2])
            replacement_seq = int(replacement.shape[-2])
            if replacement_seq < current_seq:
                pad_shape = (*replacement.shape[:-2], current_seq - replacement_seq, replacement.shape[-1])
                pad = torch.zeros(pad_shape, device=replacement.device, dtype=replacement.dtype)
                replacement = torch.cat([replacement, pad], dim=-2)
            else:
                replacement = replacement[..., :current_seq, :]
        elif tuple(current.shape[1:]) != tuple(replacement.shape[1:]):
            raise ValueError(
                "Cannot reset cache rows with incompatible tensor shapes: "
                f"current={tuple(current.shape)}, replacement={tuple(replacement.shape)}"
            )

        mask_shape = (reset_mask.shape[0],) + (1,) * (current.ndim - 1)
        mask = reset_mask.to(device=current.device, dtype=torch.bool).view(mask_shape)
        return torch.where(mask, replacement, current)

    @classmethod
    def _reset_qwen_cache_rows(cls, cache, replacement_cache, reset_mask: torch.Tensor,
                               *, empty_prefix_reset: bool = False):
        """Replace selected batch rows in Qwen3.5 attention and DeltaNet caches."""
        if empty_prefix_reset and cache is not None and replacement_cache is None:
            # An empty native prefix has no primed cache to copy. Clear the
            # selected episode rows, including persistent KDA memory.
            for attr in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
                values = getattr(cache, attr, None)
                if values is None:
                    continue
                for index, value in enumerate(values):
                    if torch.is_tensor(value) and value.ndim and value.shape[0] == reset_mask.shape[0]:
                        rows = reset_mask.to(value.device).view(-1, *([1] * (value.ndim - 1)))
                        values[index] = torch.where(rows, torch.zeros_like(value), value)
            return cache
        if cache is None or replacement_cache is None:
            return replacement_cache if cache is None else cache
        for attr in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
            current_list = getattr(cache, attr, None)
            replacement_list = getattr(replacement_cache, attr, None)
            if current_list is None or replacement_list is None:
                continue
            for layer_idx, replacement in enumerate(replacement_list):
                if replacement is None:
                    continue
                current = current_list[layer_idx]
                if current is None:
                    current_list[layer_idx] = replacement
                    continue
                current_list[layer_idx] = cls._replace_tensor_rows(
                    current,
                    replacement,
                    reset_mask,
                )
        return cache

    @staticmethod
    def _instruction_refresh_mask(mask: torch.Tensor, *, batch_size: int, device) -> torch.Tensor:
        """Validate an explicit per-row refresh without treating it as a reset."""
        if (not torch.is_tensor(mask) or mask.dtype != torch.bool
                or mask.ndim != 1 or mask.shape[0] != batch_size):
            raise ValueError(f"prefix refresh mask must be boolean [B={batch_size}]")
        return mask.to(device=device)

    @staticmethod
    def _refresh_qwen_instruction_rows(cache, replacement_cache, refresh_mask: torch.Tensor,
                                      *, instruction_tokens: int):
        """Replace instruction full-attention K/V, retaining all episode memory.

        The replacement is encoded from a fresh prefix. Its recurrent/conv
        states are deliberately unused: old instructions remain represented in
        the ongoing KDA state and already encoded visual history. Removing that
        influence would require discarding/replaying the episode's memory.
        """
        if cache is None or replacement_cache is None:
            raise ValueError("Instruction refresh requires initialized current and replacement caches")
        if type(instruction_tokens) is not int or instruction_tokens < 1:
            raise ValueError("Instruction refresh requires a positive fixed prefix width")
        for attr in ("key_cache", "value_cache"):
            current_list = getattr(cache, attr, None)
            replacement_list = getattr(replacement_cache, attr, None)
            if current_list is None or replacement_list is None or len(current_list) != len(replacement_list):
                raise ValueError("Instruction refresh requires compatible native attention cache layers")
            for layer, replacement in enumerate(replacement_list):
                current = current_list[layer]
                if replacement is None:
                    if current is not None:
                        raise ValueError("Instruction refresh attention layer inventory differs")
                    continue
                if current is None or not torch.is_tensor(current) or not torch.is_tensor(replacement):
                    raise ValueError("Instruction refresh requires primed attention tensors")
                if current.numel() == replacement.numel() == 0:
                    continue
                if (current.ndim < 3 or replacement.ndim != current.ndim
                        or current.shape[0] != refresh_mask.shape[0]
                        or current.shape[:-2] != replacement.shape[:-2]
                        or current.shape[-1] != replacement.shape[-1]
                        or current.shape[-2] < instruction_tokens
                        or replacement.shape[-2] < instruction_tokens):
                    raise ValueError("Instruction refresh attention tensor shapes differ")
                replacement = replacement.to(current)
                rows = refresh_mask.to(current.device).view(
                    refresh_mask.shape[0], *([1] * (current.ndim - 1)))
                prefix = torch.where(rows, replacement[..., :instruction_tokens, :],
                    current[..., :instruction_tokens, :])
                current_list[layer] = torch.cat((prefix, current[..., instruction_tokens:, :]), dim=-2)
        return cache

    @staticmethod
    def _refresh_instruction_mask_rows(current: torch.Tensor, replacement: torch.Tensor,
                                       refresh_mask: torch.Tensor, *, instruction_tokens: int):
        """Change only instruction visibility; keep visual/padding history intact."""
        if (current.ndim != 2 or replacement.ndim != 2
                or current.shape[0] != replacement.shape[0]
                or current.shape[0] != refresh_mask.shape[0]
                or current.shape[1] < instruction_tokens or replacement.shape[1] < instruction_tokens):
            raise ValueError("Instruction refresh mask shapes differ")
        prefix = torch.where(refresh_mask.to(current.device).unsqueeze(1),
            replacement[:, :instruction_tokens].to(current), current[:, :instruction_tokens])
        return torch.cat((prefix, current[:, instruction_tokens:]), dim=1)

    def _refresh_stream_instruction_prefix(self, state, replacement_state,
                                           refresh_mask: torch.Tensor, *, instruction_tokens: int):
        """Apply an explicit instruction update to both cached training streams."""
        if bool(getattr(getattr(self, "config", None), "text_input_deepstack", False)):
            self._activate_text_deepstack_state(state, replacement_state, refresh_mask)
            return state
        if int(state["prefix_len"]) != int(replacement_state["prefix_len"]):
            raise ValueError("Instruction refresh requires the same padded prefix width")
        if int(state.get("instruction_prefix_len", instruction_tokens)) != instruction_tokens:
            raise ValueError("Instruction refresh requires the same padded instruction width")
        for key in ("main_cache", "query_cache"):
            state[key] = self._refresh_qwen_instruction_rows(state[key], replacement_state[key],
                refresh_mask, instruction_tokens=instruction_tokens)
        for key in ("main_mask", "query_mask", "prefix_mask"):
            state[key] = self._refresh_instruction_mask_rows(state[key], replacement_state["prefix_mask"],
                refresh_mask, instruction_tokens=instruction_tokens)
        return state

    @staticmethod
    def _reset_attention_mask_rows(
        current_mask: torch.Tensor,
        prefix_mask: torch.Tensor,
        reset_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_len = int(current_mask.shape[1])
        prefix = prefix_mask.to(device=current_mask.device, dtype=current_mask.dtype)
        if int(prefix.shape[1]) < target_len:
            pad = torch.zeros(
                (prefix.shape[0], target_len - int(prefix.shape[1])),
                device=prefix.device,
                dtype=prefix.dtype,
            )
            prefix = torch.cat([prefix, pad], dim=1)
        elif int(prefix.shape[1]) > target_len:
            prefix = prefix[:, :target_len]
        return torch.where(
            reset_mask.to(device=current_mask.device, dtype=torch.bool).unsqueeze(1),
            prefix,
            current_mask,
        )

    @staticmethod
    def _crop_attention_mask(
        mask: torch.Tensor,
        *,
        prefix_tokens: int,
        recent_tokens: int,
    ) -> torch.Tensor:
        prefix_tokens = max(0, int(prefix_tokens))
        recent_tokens = max(0, int(recent_tokens))
        seq_len = int(mask.shape[1])
        if seq_len <= prefix_tokens + recent_tokens:
            return mask
        keep_prefix = min(prefix_tokens, seq_len)
        recent_start = max(keep_prefix, seq_len - recent_tokens)
        if keep_prefix > 0:
            return torch.cat([mask[:, :keep_prefix], mask[:, recent_start:]], dim=1)
        return mask[:, recent_start:]

    @staticmethod
    def _retain_real_qwen_timesteps(cache, mask: torch.Tensor, *, prefix_tokens: int,
                                    tokens_per_timestep: int, history_timesteps: int,
                                    detach: bool = True):
        """Keep whole real timestep groups, right-aligned independently per row.

        A suffix crop would retain storage padding after a short continuing
        chunk and discard actual visual history. Group selection preserves
        camera/slot identities, original RoPE K/V and global KDA memory.
        Only CPU shape metadata chooses the output width; row selections use
        tensor operations, without per-layer device reads.
        """
        if (mask.ndim != 2 or type(tokens_per_timestep) is not int or tokens_per_timestep < 1
                or prefix_tokens < 0 or mask.shape[1] < prefix_tokens
                or (mask.shape[1] - prefix_tokens) % tokens_per_timestep):
            raise ValueError("Real-history retention requires complete timestep groups after the prefix")
        batch = mask.shape[0]
        steps = (mask.shape[1] - prefix_tokens) // tokens_per_timestep
        keep = min(max(0, int(history_timesteps)), steps)
        temporal = mask[:, prefix_tokens:].reshape(batch, steps, tokens_per_timestep)
        if keep:
            position = torch.arange(steps, device=mask.device).expand(batch, -1)
            selected = torch.where(temporal.bool().any(dim=-1), position, -1).sort(dim=1).values[:, -keep:]
            selected_valid = selected >= 0
            selected = selected.clamp_min(0)
            retained = temporal.gather(1, selected[..., None].expand(batch, keep, tokens_per_timestep))
            retained = torch.where(selected_valid[..., None], retained, 0)
        else:
            selected = torch.empty(batch, 0, device=mask.device, dtype=torch.long)
            selected_valid = torch.empty(batch, 0, device=mask.device, dtype=torch.bool)
            retained = temporal[:, :0]
        for name in ("key_cache", "value_cache"):
            values = getattr(cache, name, None)
            if values is None:
                raise ValueError("Real-history retention requires native attention K/V lists")
            for index, value in enumerate(values):
                if value is None or value.numel() == 0:
                    continue
                if value.shape[0] != batch or value.shape[-2] != mask.shape[1]:
                    raise ValueError("Real-history cache and token-mask lengths differ")
                grouped = value[..., prefix_tokens:, :].reshape(
                    *value.shape[:-2], steps, tokens_per_timestep, value.shape[-1])
                shape = (batch, *([1] * (value.ndim - 3)), keep, 1, 1)
                gather = selected.to(value.device).view(shape).expand(
                    *value.shape[:-2], keep, tokens_per_timestep, value.shape[-1])
                recent = grouped.gather(-3, gather)
                valid = selected_valid.to(value.device).view(shape)
                recent = torch.where(valid, recent, 0).reshape(
                    *value.shape[:-2], keep * tokens_per_timestep, value.shape[-1])
                result = torch.cat((value[..., :prefix_tokens, :], recent), dim=-2)
                values[index] = result.detach() if detach else result
        if detach:
            PrismVLARuntimeMixin._detach_qwen_state_cache(cache)
        retained_mask = torch.cat((mask[:, :prefix_tokens], retained.flatten(1)), dim=1)
        return retained_mask.shape[1], retained_mask

    @staticmethod
    def _pad_or_trim_attention_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
        target_len = int(target_len)
        current_len = int(mask.shape[1])
        if current_len == target_len:
            return mask
        if current_len > target_len:
            return mask[:, :target_len]
        pad = torch.zeros(
            (mask.shape[0], target_len - current_len),
            device=mask.device,
            dtype=mask.dtype,
        )
        return torch.cat([mask, pad], dim=1)

    def _configure_training_attention_backends(self) -> None:
        """Disable cuDNN fused SDPA globally so checkpoint recompute matches."""
        if str(getattr(self.config, "attn_implementation", "")) != "sdpa":
            return
        if not bool(getattr(self.config, "disable_cudnn_sdpa", True)):
            return
        if not torch.cuda.is_available():
            return
        enable_cudnn_sdp = getattr(torch.backends.cuda, "enable_cudnn_sdp", None)
        if enable_cudnn_sdp is None:
            return
        enable_cudnn_sdp(False)

    def _normalize_latent(self, features: torch.Tensor) -> torch.Tensor:
        return rms_normalize_last_dim(features)

    def _build_idm_condition_list(
        self,
        *,
        cls_t: torch.Tensor,
        compact_t: torch.Tensor,
        proprio_t: Optional[torch.Tensor],
        cls_tp1: torch.Tensor,
        compact_tp1: torch.Tensor,
        compact_t_mask: Optional[torch.Tensor] = None,
        compact_tp1_mask: Optional[torch.Tensor] = None,
        action_token: Optional[torch.Tensor] = None,
        end_effector_type_ids: Optional[torch.Tensor] = None,
        backbone_kv=None,
    ) -> list[torch.Tensor]:
        if self._uses_main_stream_action_token():
            if bool(getattr(self.idm, "requires_raw_action_token", False)):
                if (action_token is None or action_token.ndim != 3
                        or action_token.shape[1:] != (self._main_stream_action_tokens_per_step(), self.qwen_dim)):
                    raise ValueError("act_patch requires one [N, action_horizon, backbone_dim] ACT block")
                return [action_token]
            if action_token is None or action_token.ndim != 2 or action_token.shape[-1] != self.qwen_dim:
                raise ValueError("main_stream_action_token requires one ACT hidden [N, backbone_dim] per action row")
            from prism_vla.modeling.condition_mask import ConditionList
            return ConditionList([action_token], valid_mask=torch.ones(
                action_token.shape[0], 1, device=action_token.device, dtype=torch.bool),
                end_effector_type_ids=end_effector_type_ids)
        cond_list: list[torch.Tensor] = []
        condition_masks: list[torch.Tensor] = []
        compact_tokens = int(
            getattr(
                self,
                "compact_tokens_per_step",
                getattr(self, "compact_output_tokens", 1),
            )
        )

        direct_layout = (getattr(self, "direct_query_layout", None)
            if getattr(getattr(self, "config", None), "query_design", None) == "direct_query" else None)

        def append_condition(condition: torch.Tensor, camera_mask=None, *, compact_condition=False) -> None:
            if direct_layout is not None and compact_condition:
                if compact_tokens == 1 and condition.ndim == 2:
                    condition = condition.unsqueeze(-2)
                if condition.ndim != 3 or condition.shape[-2:] != (compact_tokens, self.qwen_dim):
                    raise ValueError("Direct-query action condition requires role-major compact slots")
                cond_list.extend(condition.unbind(dim=-2))
                validity = (torch.ones(condition.shape[0], compact_tokens, device=condition.device, dtype=torch.bool)
                    if camera_mask is None else direct_layout.camera_mask(camera_mask).to(condition.device))
                condition_masks.extend(validity.unbind(-1))
                return
            if (
                compact_tokens > 1
                and condition.ndim >= 3
                and condition.shape[-2:] == (compact_tokens, self.qwen_dim)
            ):
                cond_list.extend(condition.unbind(dim=-2))
                if camera_layout_enabled(self):
                    if camera_mask is None:
                        camera_mask = torch.ones(condition.shape[0], compact_tokens, device=condition.device, dtype=torch.bool)
                    camera_mask = expand_camera_presence(camera_mask, compact_tokens,
                        tokens_per_camera=self.camera_compact_tokens_per_camera)
                    condition_masks.extend(camera_mask.to(condition.device).unbind(-1))
                return
            cond_list.append(condition)
            if camera_layout_enabled(self):
                condition_masks.append(torch.ones(condition.shape[0], device=condition.device, dtype=torch.bool)
                    if camera_mask is None else camera_mask.to(condition.device).bool().reshape(condition.shape[0]))

        if bool(getattr(self, "use_current_cls_condition_for_idm", True)):
            append_condition(cls_t)
        if bool(getattr(self, "use_current_compact_condition_for_idm", True)):
            append_condition(compact_t, compact_t_mask, compact_condition=True)
        if (
            proprio_t is not None
            and str(getattr(self, "proprio_token_placement", "condition")) == "condition"
            and getattr(getattr(self, "config", None), "action_conditioning", "legacy") != "predicted_compact"
        ):
            append_condition(proprio_t)
        if bool(getattr(self, "use_future_cls_condition_for_idm", True)):
            append_condition(cls_tp1)
        if bool(getattr(self, "use_future_compact_condition_for_idm", True)):
            append_condition(compact_tp1, compact_tp1_mask, compact_condition=True)
        if camera_layout_enabled(self):
            from prism_vla.modeling.condition_mask import ConditionList
            return ConditionList(cond_list, valid_mask=torch.stack(condition_masks, dim=-1),
                end_effector_type_ids=end_effector_type_ids, backbone_kv=backbone_kv)
        if end_effector_type_ids is not None or backbone_kv is not None:
            from prism_vla.modeling.condition_mask import ConditionList
            return ConditionList(cond_list, valid_mask=torch.ones(
                cond_list[0].shape[0], len(cond_list), device=cond_list[0].device, dtype=torch.bool),
                end_effector_type_ids=end_effector_type_ids, backbone_kv=backbone_kv)
        return cond_list

    def _idm_proprio_kwargs(
        self,
        proprio_t: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Route proprio separately when it belongs to the action-token sequence."""
        if str(getattr(self, "proprio_token_placement", "condition")) != "action":
            return {}
        if proprio_t is None:
            raise ValueError(
                "Projected proprio is required when proprio_token_placement='action'."
            )
        return {"proprio_token": proprio_t}

    @staticmethod
    def _to_module_dtype(features: torch.Tensor, module: nn.Module) -> torch.Tensor:
        param = next(module.parameters(), None)
        if param is None:
            return features
        return features.to(device=param.device, dtype=param.dtype)

    def _project_dino(self, features: torch.Tensor) -> torch.Tensor:
        return self._normalize_latent(self.visual_proj(self._to_module_dtype(features, self.visual_proj)))

    def _project_dino_cls(
        self,
        features: torch.Tensor,
        aux_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project CLS and optionally add gated information from global aux tokens."""
        projected_cls = self._project_dino(features)
        if not bool(getattr(self, "use_cls_register_fusion", False)):
            return projected_cls
        if aux_tokens is None:
            raise ValueError(
                "Auxiliary visual tokens are required when use_cls_register_fusion=true."
            )
        if aux_tokens.ndim != features.ndim + 1:
            raise ValueError(
                "Expected auxiliary tokens to add one token axis to CLS features; "
                f"got CLS={tuple(features.shape)}, aux={tuple(aux_tokens.shape)}."
            )
        if aux_tokens.shape[:-2] != features.shape[:-1]:
            raise ValueError(
                "CLS and auxiliary-token leading dimensions must match; "
                f"got CLS={tuple(features.shape)}, aux={tuple(aux_tokens.shape)}."
            )
        if aux_tokens.shape[-1] != features.shape[-1]:
            raise ValueError(
                "CLS and auxiliary-token feature dimensions must match; "
                f"got {features.shape[-1]} and {aux_tokens.shape[-1]}."
            )
        if aux_tokens.shape[-2] <= 0:
            raise ValueError("At least one auxiliary visual token is required.")
        if self.cls_register_attn is None or self.cls_register_query is None:
            raise RuntimeError("CLS register-fusion modules were not initialized.")

        projected_aux = self._project_dino(aux_tokens)
        flat_aux = projected_aux.reshape(-1, projected_aux.shape[-2], self.qwen_dim)
        query = self._normalize_latent(self.cls_register_query)
        query = self._to_module_dtype(query, self.cls_register_attn)
        query = query.expand(flat_aux.shape[0], -1, -1)
        flat_aux = self._to_module_dtype(flat_aux, self.cls_register_attn)
        attended, _ = self.cls_register_attn(
            query,
            flat_aux,
            flat_aux,
            need_weights=False,
        )
        attended = attended[:, 0].reshape(*projected_cls.shape[:-1], self.qwen_dim)
        gate = self.cls_register_gate.to(
            device=attended.device,
            dtype=attended.dtype,
        )
        return self._normalize_latent(projected_cls + gate * attended)

    def _project_semantic(self, features: torch.Tensor) -> torch.Tensor:
        if self.semantic_visual_proj is None:
            raise RuntimeError("Semantic visual stream is disabled.")
        return self._normalize_latent(
            self.semantic_visual_proj(self._to_module_dtype(features, self.semantic_visual_proj))
        )

    def _project_proprio(self, proprio: torch.Tensor,
                         proprio_frame_ids: Optional[torch.Tensor] = None, *,
                         engram_route_ids: Optional[torch.Tensor] = None,
                         engram_state_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.proprio_encoder is None:
            raise RuntimeError("Proprio stream is disabled.")
        if getattr(getattr(self, "config", None), "engram_native", None) is not None:
            if proprio_frame_ids is None or tuple(proprio_frame_ids.shape) != tuple(proprio.shape[:-1]):
                raise ValueError("Native joint proprio requires explicit joint frame IDs")
            valid = (proprio_frame_ids == 9).all()
            if proprio.device.type == "cuda":
                torch._assert_async(valid, "Native Engram proprio must use joint frame ID9")
            elif not bool(valid):
                raise ValueError("Native Engram proprio must use joint frame ID9")
            if engram_route_ids is None or engram_state_mask is None:
                raise ValueError("Native joint proprio requires route IDs and explicit state availability")
            values = self._to_module_dtype(proprio, self.engram_proprio_encoder)
            projected = self.engram_proprio_encoder(values, engram_state_mask, engram_route_ids)
        else:
            if engram_route_ids is not None or engram_state_mask is not None:
                raise ValueError("Native proprio metadata requires model.config.engram_native")
            from prism_vla.modeling.proprio import FrameSpecificProprioEncoder
            values = self._to_module_dtype(proprio, self.proprio_encoder)
            projected = (self.proprio_encoder(values, proprio_frame_ids)
                if isinstance(self.proprio_encoder, FrameSpecificProprioEncoder)
                else self.proprio_encoder(values))
        projected = self._normalize_latent(projected)
        probability = float(getattr(getattr(self, "config", None), "proprio_token_dropout", 0.0))
        if self.training and probability:
            keep = torch.rand(*projected.shape[:-1], 1, device=projected.device) >= probability
            projected = torch.where(keep, projected, 0.0)
        return projected

    def _project_query(self, features: torch.Tensor) -> torch.Tensor:
        return self._normalize_latent(
            self.query_proj_head(self._to_module_dtype(features, self.query_proj_head))
        )

    def _predict_compact_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if camera_layout_enabled(self):
            compact = self.camera_compact_head(self._to_module_dtype(hidden, self.camera_compact_head))
            compact = self._normalize_latent(compact)
            return compact.squeeze(-2) if self.compact_tokens_per_step == 1 else compact
        compact = self.compact_head(self._to_module_dtype(hidden, self.compact_head))
        output_tokens = int(
            getattr(
                self,
                "compact_tokens_per_step",
                getattr(self, "compact_output_tokens", 1),
            )
        )
        if output_tokens == 1:
            return self._normalize_latent(compact)
        compact = compact.reshape(*hidden.shape[:-1], output_tokens, self.qwen_dim)
        return self._normalize_latent(compact)

    def _main_stream_hidden_readouts(
        self, hidden: torch.Tensor, camera_present: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select CLS and compact predictor inputs without adding a Q head."""
        compact_tokens = int(self.compact_tokens_per_step)
        direct = getattr(getattr(self, "config", None), "query_design", None) == "direct_query"
        if self._uses_main_stream_timestep_causal() or direct:
            return hidden[..., compact_tokens, :], hidden[..., :compact_tokens, :]
        compact_hidden = hidden[..., :compact_tokens, :]
        cls_hidden = (pool_camera_hidden(compact_hidden, camera_present,
            tokens_per_camera=self.camera_compact_tokens_per_camera)
                      if camera_layout_enabled(self) else compact_hidden.mean(dim=-2))
        query_offset = (compact_tokens + (1 if getattr(self, "use_cls_tokens", True) else 0)
                        + self._main_stream_action_tokens_per_step())
        return cls_hidden, hidden[..., query_offset, :]

    def _predict_main_stream_compact(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply each existing output slot's map to its corresponding hidden."""
        direct = getattr(getattr(self, "config", None), "query_design", None) == "direct_query"
        if not self._uses_main_stream_timestep_causal() and not direct:
            return self._predict_compact_from_hidden(hidden)
        output_tokens = int(self.compact_tokens_per_step)
        if hidden.shape[-2:] != (output_tokens, self.qwen_dim):
            raise ValueError("Positional compact readout requires one hidden per compact slot")
        if camera_layout_enabled(self):
            layout = (getattr(self, "direct_query_layout", None)
                if getattr(getattr(self, "config", None), "query_design", None) == "direct_query" else None)
            if layout is not None:
                camera_order = torch.as_tensor(layout.camera_major_indices(), dtype=torch.long, device=hidden.device)
                hidden = hidden.index_select(-2, camera_order)
            compact = self.camera_compact_head(
                self._to_module_dtype(hidden, self.camera_compact_head), positional=True
            )
            if layout is not None:
                role_order = torch.as_tensor(layout.role_major_indices(), dtype=torch.long, device=compact.device)
                compact = compact.index_select(-2, role_order)
        else:
            # Preserve the legacy output-row parameterization for source layouts.
            compact = self.compact_head(self._to_module_dtype(hidden, self.compact_head))
            compact = compact.reshape(*hidden.shape[:-2], output_tokens, output_tokens, self.qwen_dim)
            compact = compact.diagonal(dim1=-3, dim2=-2).movedim(-1, -2)
        compact = self._normalize_latent(compact)
        return compact.squeeze(-2) if output_tokens == 1 else compact

    def _compact_tokens_for_layout(self, compact: torch.Tensor) -> torch.Tensor:
        output_tokens = int(
            getattr(
                self,
                "compact_tokens_per_step",
                getattr(self, "compact_output_tokens", 1),
            )
        )
        if compact.ndim >= 3 and compact.shape[-2:] == (output_tokens, self.qwen_dim):
            return compact
        if output_tokens == 1:
            return compact.unsqueeze(-2)
        raise ValueError(
            "Expected multi-token compact shape ending in "
            f"({output_tokens}, {self.qwen_dim}), got {tuple(compact.shape)}"
        )

    def _pool_compact_tokens(self, compact: torch.Tensor) -> torch.Tensor:
        output_tokens = int(
            getattr(
                self,
                "compact_tokens_per_step",
                getattr(self, "compact_output_tokens", 1),
            )
        )
        if compact.ndim >= 3 and compact.shape[-2:] == (output_tokens, self.qwen_dim):
            return self._normalize_latent(compact.mean(dim=-2))
        if output_tokens == 1:
            return compact
        raise ValueError(
            "Expected multi-token compact shape ending in "
            f"({output_tokens}, {self.qwen_dim}), got {tuple(compact.shape)}"
        )

    def _compact_loss_mean_dim(self):
        compact_tokens = int(
            getattr(
                self,
                "compact_tokens_per_step",
                getattr(self, "compact_output_tokens", 1),
            )
        )
        return (-1, -2) if compact_tokens > 1 else -1

    def _compact_norm_metric(
        self,
        compact: torch.Tensor,
        valid_float: torch.Tensor,
        n_valid: torch.Tensor,
    ) -> torch.Tensor:
        norm = torch.linalg.vector_norm(compact.float(), dim=-1)
        while norm.ndim > valid_float.ndim:
            norm = norm.mean(dim=-1)
        return (norm * valid_float).sum() / n_valid

    def _static_queries_without_cls(
        self,
        batch_size: int,
        steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.no_cls_query_token is None:
            raise RuntimeError("No-CLS query token is only available when use_cls_tokens=False.")
        hidden = self._normalize_latent(
            self.no_cls_query_token.to(device=device, dtype=dtype).view(1, 1, self.qwen_dim)
        )
        hidden = hidden.expand(batch_size, steps, self.qwen_dim)
        return self._project_query(hidden), hidden

    def _global_queries(
        self,
        batch_size: int,
        steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.global_query_tokens is None:
            raise RuntimeError("Global query tokens are not enabled.")
        hidden = self._normalize_latent(
            self.global_query_tokens.to(device=device, dtype=dtype)
        )
        hidden = hidden.view(1, 1, self.num_global_query_tokens, self.qwen_dim)
        hidden = hidden.expand(
            batch_size,
            steps,
            self.num_global_query_tokens,
            self.qwen_dim,
        )
        return self._project_query(hidden)

    def _compute_task_queries(
        self,
        instruction_embeds: torch.Tensor,
        instruction_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Generate all task queries in one instruction-only backbone call."""
        if self.task_query_seed_tokens is None:
            raise RuntimeError("Task query tokens are not enabled.")
        if instruction_embeds.ndim != 3:
            raise ValueError(
                "Expected instruction_embeds [B, L, D], got "
                f"{tuple(instruction_embeds.shape)}."
            )
        batch_size, instruction_len, _ = instruction_embeds.shape
        instruction_mask = self._instruction_prefix_mask(instruction_mask)
        device = instruction_embeds.device
        qdtype = self._qwen_dtype()
        instruction = instruction_embeds.to(qdtype)
        seeds = self.task_query_seed_tokens.to(device=device, dtype=qdtype)
        seeds = seeds.unsqueeze(0).expand(batch_size, -1, -1)
        task_inputs = torch.cat([instruction, seeds], dim=1)
        seed_mask = torch.ones(
            batch_size,
            self.num_task_query_tokens,
            device=device,
            dtype=instruction_mask.dtype,
        )
        task_mask = torch.cat(
            [instruction_mask.to(device=device), seed_mask],
            dim=1,
        )
        position_ids = torch.arange(
            instruction_len + self.num_task_query_tokens,
            device=device,
        ).unsqueeze(0).expand(batch_size, -1)
        outputs = run_qwen(self,
            inputs_embeds=task_inputs,
            attention_mask=task_mask,
            position_ids=position_ids,
            output_hidden_states=False,
            use_cache=False,
        )
        task_hidden = outputs.last_hidden_state[:, -self.num_task_query_tokens :]
        return self._project_query(task_hidden)

    def _build_generated_query_inputs(
        self,
        cls_tokens: torch.Tensor,
        cls_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append causal learned seeds after each CLS in the query-only pass."""
        count = int(getattr(self, "num_generated_query_tokens", 1))
        if count == 1:
            return cls_tokens, cls_positions
        batch_size, steps, dim = cls_tokens.shape
        seeds = self.generated_query_seed_embeddings.to(
            device=cls_tokens.device, dtype=cls_tokens.dtype
        ).view(1, 1, count - 1, dim).expand(batch_size, steps, -1, -1)
        inputs = torch.cat([cls_tokens.unsqueeze(2), seeds], dim=2)
        positions = cls_positions.unsqueeze(-1) + torch.arange(
            count, device=cls_positions.device, dtype=cls_positions.dtype
        )
        return inputs.reshape(batch_size, steps * count, dim), positions.reshape(
            batch_size, steps * count
        )

    def _project_generated_query_outputs(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return generated sources [B,T,G,D] and the primary CLS hidden [B,T,D]."""
        count = int(getattr(self, "num_generated_query_tokens", 1))
        generated_hidden = hidden.reshape(hidden.shape[0], -1, count, self.qwen_dim)
        return self._project_query(generated_hidden), generated_hidden[:, :, 0]

    def _compose_compact_query_sources(
        self,
        cls_queries: torch.Tensor,
        task_queries: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Order sources as optional global, task, then generated CLS/seed outputs."""
        count = int(getattr(self, "num_generated_query_tokens", 1))
        if cls_queries.ndim == 3 and count == 1:
            cls_queries = cls_queries.unsqueeze(2)
        if cls_queries.ndim != 4 or cls_queries.shape[2:] != (count, self.qwen_dim):
            raise ValueError(
                f"Expected generated queries [B, T, {count}, {self.qwen_dim}], "
                f"got {tuple(cls_queries.shape)}."
            )
        batch_size, steps = cls_queries.shape[:2]
        sources = []
        if self.query_generation_mode in {"global_cls", "global_task_cls"}:
            global_queries = self._global_queries(
                batch_size,
                steps,
                device=cls_queries.device,
                dtype=cls_queries.dtype,
            )
            sources.append(global_queries)
        if self._uses_task_query_sources():
            if task_queries is None or task_queries.ndim != 3:
                shape = None if task_queries is None else tuple(task_queries.shape)
                raise ValueError(f"Expected task_queries [B, M, D], got {shape}.")
            if task_queries.shape[:2] != (batch_size, self.num_task_query_tokens):
                raise ValueError(
                    "Task query shape mismatch: expected "
                    f"({batch_size}, {self.num_task_query_tokens}, {self.qwen_dim}), "
                    f"got {tuple(task_queries.shape)}."
                )
            sources.append(
                task_queries.unsqueeze(1).expand(
                    batch_size,
                    steps,
                    self.num_task_query_tokens,
                    self.qwen_dim,
                )
            )
        elif task_queries is not None:
            raise ValueError(
                f"Task queries are disabled in {self.query_generation_mode!r} mode."
            )
        sources.append(cls_queries)
        return torch.cat(sources, dim=2)

    def _project_dino_queries(self, query: torch.Tensor) -> torch.Tensor:
        projected = self._project_dino_query_sources(query)
        if projected.shape[1] != 1:
            raise ValueError(f"Expected one source query, got {projected.shape[1]}.")
        return projected[:, 0]

    def _project_dino_query_sources(self, queries: torch.Tensor) -> torch.Tensor:
        if self.dino_query_proj_head is None:
            raise RuntimeError("DINO query projection is disabled.")
        if queries.ndim != 3 or queries.shape[-1] != self.qwen_dim:
            raise ValueError(
                f"Expected query sources [B, S, {self.qwen_dim}], got {tuple(queries.shape)}"
            )
        projected = self.dino_query_proj_head(
            self._to_module_dtype(queries, self.dino_query_proj_head)
        )
        projected = projected.reshape(
            queries.shape[0],
            queries.shape[1],
            self.num_query_tokens,
            self.dino_dim,
        )
        return self._normalize_latent(projected)

    def _dino_camera_fusion_weights(self, logits: torch.Tensor, camera_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Softmax only over real camera slots; all-missing rows remain zero."""
        scaled = logits / self.dino_camera_fusion_temperature
        if camera_mask is None:
            return torch.softmax(scaled, dim=-1)
        if camera_mask.ndim != 2 or tuple(camera_mask.shape) != (logits.shape[0], logits.shape[-1]):
            raise ValueError("dino_camera_mask must have shape [B, num_dino_cameras]")
        valid = camera_mask.to(device=logits.device, dtype=torch.bool)
        if logits.ndim == 3:
            valid = valid[:, None]
        weights = torch.softmax(scaled.masked_fill(~valid, torch.finfo(scaled.dtype).min), dim=-1)
        weights = torch.where(valid, weights, 0)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)

    def _compute_camera_weighted_dino_compact_sources(
        self,
        queries: torch.Tensor,
        dino_patches: torch.Tensor,
        *,
        dino_camera_mask: Optional[torch.Tensor] = None,
        return_cross_attn_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Share each camera's patch K/V across all source-level queries."""
        if (
            self.dino_native_cross_attn is None
            or self.dino_camera_compact_out_proj is None
            or self.dino_camera_weight_head is None
        ):
            raise RuntimeError("Camera-weighted native-DINO compact modules are missing.")
        batch_size, source_count, _ = queries.shape
        num_cameras = int(self.num_dino_cameras)
        if dino_patches.ndim != 3 or dino_patches.shape[0] != batch_size:
            raise ValueError(
                f"Expected dino_patches [B, P, D], got {tuple(dino_patches.shape)}"
            )
        if dino_patches.shape[1] % num_cameras != 0:
            raise ValueError(
                f"DINO patch count {dino_patches.shape[1]} is not divisible by "
                f"num_dino_cameras={num_cameras}."
            )
        if self.num_query_tokens % num_cameras != 0:
            raise ValueError(
                f"num_query_tokens={self.num_query_tokens} is not divisible by "
                f"num_dino_cameras={num_cameras}."
            )

        dino_queries = self._project_dino_query_sources(queries)
        dino_dim = dino_queries.shape[-1]
        queries_per_camera = self.num_query_tokens // num_cameras
        patches_per_camera = dino_patches.shape[1] // num_cameras
        queries_by_camera = dino_queries.reshape(
            batch_size,
            source_count,
            num_cameras,
            queries_per_camera,
            dino_dim,
        ).permute(0, 2, 1, 3, 4)
        patches_by_camera = dino_patches.to(
            device=dino_queries.device,
            dtype=dino_queries.dtype,
        ).reshape(batch_size, num_cameras, patches_per_camera, dino_dim)
        if dino_camera_mask is not None:
            valid = dino_camera_mask.to(device=patches_by_camera.device, dtype=torch.bool)
            patches_by_camera = torch.where(valid[:, :, None, None], patches_by_camera, 0)
        attended = self._run_cross_attn_stack(
            queries_by_camera.reshape(
                batch_size * num_cameras,
                source_count * queries_per_camera,
                dino_dim,
            ),
            patches_by_camera.reshape(
                batch_size * num_cameras,
                patches_per_camera,
                dino_dim,
            ),
            first_attn=self.dino_native_cross_attn,
            extra_attns=self.dino_native_cross_attn_extra,
            extra_norms=self.dino_native_cross_attn_extra_norms,
        )
        attended = attended.reshape(
            batch_size,
            num_cameras,
            source_count,
            queries_per_camera,
            dino_dim,
        )
        if dino_camera_mask is not None:
            attended = torch.where(valid[:, :, None, None, None], attended, 0)
        cross_attn_tokens = attended.permute(0, 2, 1, 3, 4).reshape(
            batch_size,
            source_count * self.num_query_tokens,
            dino_dim,
        )
        camera_compacts = self._normalize_latent(
            self.dino_camera_compact_out_proj(
                self._to_module_dtype(
                    attended.flatten(-2),
                    self.dino_camera_compact_out_proj,
                )
            )
        ).permute(0, 2, 1, 3)
        weight_logits = self.dino_camera_weight_head(
            self._to_module_dtype(queries, self.dino_camera_weight_head)
        )
        camera_weights = self._dino_camera_fusion_weights(weight_logits, dino_camera_mask).to(camera_compacts.dtype)
        compact = self._normalize_latent(
            (camera_compacts * camera_weights.unsqueeze(-1)).sum(dim=2)
        )
        return (compact, cross_attn_tokens) if return_cross_attn_tokens else compact

    def _compute_camera_weighted_dino_compact(
        self,
        query: torch.Tensor,
        dino_patches: torch.Tensor,
        *,
        dino_camera_mask: Optional[torch.Tensor] = None,
        return_cross_attn_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if (
            self.dino_native_cross_attn is None
            or self.dino_camera_compact_out_proj is None
            or self.dino_camera_weight_head is None
        ):
            raise RuntimeError("Camera-weighted native-DINO compact modules are missing.")

        num_cameras = int(self.num_dino_cameras)
        if num_cameras <= 0:
            raise RuntimeError(f"Invalid num_dino_cameras={num_cameras}.")
        if dino_patches.ndim != 3:
            raise ValueError(f"Expected dino_patches [B, P, D], got {tuple(dino_patches.shape)}")
        if dino_patches.shape[1] % num_cameras != 0:
            raise ValueError(
                f"DINO patch count {dino_patches.shape[1]} is not divisible by num_dino_cameras={num_cameras}."
            )
        if self.num_query_tokens % num_cameras != 0:
            raise ValueError(
                f"num_query_tokens={self.num_query_tokens} is not divisible by num_dino_cameras={num_cameras}."
            )

        dino_query = self._project_dino_queries(query)
        batch_size, _, dino_dim = dino_query.shape
        queries_per_camera = self.num_query_tokens // num_cameras
        patches_per_camera = dino_patches.shape[1] // num_cameras

        dino_query_by_camera = dino_query.reshape(
            batch_size,
            num_cameras,
            queries_per_camera,
            dino_dim,
        )
        dino_tokens_by_camera = dino_patches.to(
            device=dino_query.device,
            dtype=dino_query.dtype,
        ).reshape(batch_size, num_cameras, patches_per_camera, dino_dim)
        if dino_camera_mask is not None:
            valid = dino_camera_mask.to(device=dino_tokens_by_camera.device, dtype=torch.bool)
            dino_tokens_by_camera = torch.where(valid[:, :, None, None], dino_tokens_by_camera, 0)

        attended = self._run_cross_attn_stack(
            dino_query_by_camera.reshape(batch_size * num_cameras, queries_per_camera, dino_dim),
            dino_tokens_by_camera.reshape(batch_size * num_cameras, patches_per_camera, dino_dim),
            first_attn=self.dino_native_cross_attn,
            extra_attns=self.dino_native_cross_attn_extra,
            extra_norms=self.dino_native_cross_attn_extra_norms,
        )
        if dino_camera_mask is not None:
            attended = torch.where(valid[:, :, None, None], attended.reshape(batch_size, num_cameras, queries_per_camera, dino_dim), 0)
        cross_attn_tokens = attended.reshape(
            batch_size,
            num_cameras * queries_per_camera,
            dino_dim,
        )
        if int(getattr(self, "compact_output_tokens", 1)) > 1:
            if self.dino_token_compact_out_proj is None:
                raise RuntimeError("DINO token compact projection is missing.")
            output_tokens = int(self.compact_output_tokens)
            attended = attended.reshape(batch_size, num_cameras, queries_per_camera, dino_dim)
            attended = attended[:, :, :output_tokens]
            camera_compacts = self._normalize_latent(
                self.dino_token_compact_out_proj(
                    self._to_module_dtype(attended, self.dino_token_compact_out_proj)
                )
            )
            weight_logits = self.dino_camera_weight_head(
                self._to_module_dtype(query.squeeze(1), self.dino_camera_weight_head)
            )
            camera_weights = self._dino_camera_fusion_weights(weight_logits, dino_camera_mask).to(camera_compacts.dtype)
            fused = (camera_compacts * camera_weights.view(batch_size, num_cameras, 1, 1)).sum(dim=1)
            fused = self._normalize_latent(fused)
            return (fused, cross_attn_tokens) if return_cross_attn_tokens else fused

        attended = attended.reshape(batch_size, num_cameras, queries_per_camera * dino_dim)
        camera_compacts = self._normalize_latent(
            self.dino_camera_compact_out_proj(
                self._to_module_dtype(attended, self.dino_camera_compact_out_proj)
            )
        )
        weight_logits = self.dino_camera_weight_head(
            self._to_module_dtype(query.squeeze(1), self.dino_camera_weight_head)
        )
        camera_weights = self._dino_camera_fusion_weights(weight_logits, dino_camera_mask).to(camera_compacts.dtype)
        fused = (camera_compacts * camera_weights.unsqueeze(-1)).sum(dim=1)
        fused = self._normalize_latent(fused)
        return (fused, cross_attn_tokens) if return_cross_attn_tokens else fused

    def _adapt_compact_for_action(self, compact_token: torch.Tensor) -> torch.Tensor:
        if self._uses_main_stream_action_token():
            # Legacy endpoint plumbing may still materialize these values;
            # they are never action conditions when ACT is enabled.
            return compact_token
        return self._normalize_latent(
            self.compact_action_adapter(self._to_module_dtype(compact_token, self.compact_action_adapter))
        )

    def _fuse_cls(
        self,
        dino_cls: Optional[torch.Tensor],
        semantic_cls: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.visual_encoder_mode == "dino":
            if dino_cls is None:
                raise ValueError("DINO CLS is required when visual_encoder_mode='dino'.")
            return dino_cls
        if self.visual_encoder_mode == "semantic":
            if semantic_cls is None:
                raise ValueError("Semantic CLS is required when visual_encoder_mode='semantic'.")
            return semantic_cls
        if dino_cls is None or semantic_cls is None or self.cls_fusion is None:
            raise ValueError("Both DINO and semantic CLS are required when visual_encoder_mode='both'.")
        return self._normalize_latent(self.cls_fusion(dino_cls, semantic_cls))

    def _fuse_compact(
        self,
        dino_compact: Optional[torch.Tensor],
        semantic_compact: Optional[torch.Tensor],
    ) -> torch.Tensor:
        streams: list[torch.Tensor] = []
        if self.use_dino_visual:
            if dino_compact is None:
                raise ValueError("DINO compact is required by this visual_encoder_mode.")
            streams.append(dino_compact)
        if self.use_semantic_visual:
            if semantic_compact is None:
                raise ValueError("Semantic compact is required by this visual_encoder_mode.")
            streams.append(semantic_compact)

        if not streams:
            raise RuntimeError("At least one compact stream must be enabled.")
        if len(streams) == 1:
            return streams[0]
        if self.compact_fusion is None:
            raise RuntimeError("compact_fusion is missing despite multiple compact streams.")
        return self._normalize_latent(self.compact_fusion(*streams))

    def _build_visual_prefix(
        self,
        dino_patches_0: Optional[torch.Tensor],
        semantic_tokens_0: Optional[torch.Tensor],
        *,
        mdtype: torch.dtype,
        qdtype: torch.dtype,
    ) -> torch.Tensor:
        if not self.use_start_patch_visual_tokens:
            source = dino_patches_0 if dino_patches_0 is not None else semantic_tokens_0
            if source is None:
                raise ValueError("At least one visual tensor is required to infer prefix batch size.")
            return torch.empty(
                source.shape[0],
                0,
                self.qwen_dim,
                device=source.device,
                dtype=qdtype,
            )

        prefix_parts: list[torch.Tensor] = []
        if self.use_dino_visual:
            if dino_patches_0 is None:
                raise ValueError("DINO patch tokens are required by this visual_encoder_mode.")
            prefix_parts.append(self._project_dino(dino_patches_0.to(mdtype)).to(qdtype))
        if self.use_semantic_visual:
            if semantic_tokens_0 is None:
                raise ValueError("Semantic visual tokens are required by this visual_encoder_mode.")
            prefix_parts.append(self._project_semantic(semantic_tokens_0.to(mdtype)).to(qdtype))
        if not prefix_parts:
            raise RuntimeError("At least one visual encoder stream must be enabled.")
        return torch.cat(prefix_parts, dim=1)

    def _compute_compact_from_visuals(
        self,
        query: torch.Tensor,
        dino_patches: Optional[torch.Tensor],
        semantic_tokens: Optional[torch.Tensor],
        *,
        mdtype: torch.dtype,
        dino_camera_mask: Optional[torch.Tensor] = None,
        return_cross_attn_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if camera_layout_enabled(self):
            return self._compute_compact_from_query_sources(
                query, dino_patches, semantic_tokens, mdtype=mdtype,
                dino_camera_mask=dino_camera_mask,
                return_cross_attn_tokens=return_cross_attn_tokens,
            )
        dino_compact = None
        cross_attn_tokens = None
        if self.use_dino_visual:
            if dino_patches is None:
                raise ValueError("DINO patch tokens are required by this visual_encoder_mode.")
            if self.compact_query_mode == "native_dino":
                if self.dino_native_cross_attn is None or self.dino_compact_out_proj is None:
                    raise RuntimeError("Native-DINO compact modules are missing.")
                if self.dino_camera_fusion == "weighted_average":
                    camera_result = self._compute_camera_weighted_dino_compact(
                        query,
                        dino_patches,
                        dino_camera_mask=dino_camera_mask,
                        return_cross_attn_tokens=return_cross_attn_tokens,
                    )
                    if return_cross_attn_tokens:
                        dino_compact, cross_attn_tokens = camera_result
                    else:
                        dino_compact = camera_result
                else:
                    dino_query = self._project_dino_queries(query)
                    dino_tokens = dino_patches.to(
                        device=dino_query.device,
                        dtype=dino_query.dtype,
                    )
                    dino_attended = self._run_cross_attn_stack(
                        dino_query,
                        dino_tokens,
                        first_attn=self.dino_native_cross_attn,
                        extra_attns=self.dino_native_cross_attn_extra,
                        extra_norms=self.dino_native_cross_attn_extra_norms,
                    )
                    cross_attn_tokens = dino_attended
                    if int(getattr(self, "compact_output_tokens", 1)) > 1:
                        if self.dino_token_compact_out_proj is None:
                            raise RuntimeError("DINO token compact projection is missing.")
                        dino_attended = dino_attended[:, : int(self.compact_output_tokens)]
                        dino_compact = self._normalize_latent(
                            self.dino_token_compact_out_proj(
                                self._to_module_dtype(dino_attended, self.dino_token_compact_out_proj)
                            )
                        )
                    else:
                        dino_attended = dino_attended.flatten(1)
                        dino_compact = self._normalize_latent(
                            self.dino_compact_out_proj(
                                self._to_module_dtype(dino_attended, self.dino_compact_out_proj)
                            )
                        )
            else:
                patches_proj = self._project_dino(dino_patches.to(mdtype))
                dino_compact = self._run_cross_attn_stack(
                    query,
                    patches_proj,
                    first_attn=self.cross_attn,
                    extra_attns=self.cross_attn_extra,
                    extra_norms=self.cross_attn_extra_norms,
                )
                dino_compact = self._normalize_latent(dino_compact.squeeze(1))

        semantic_compact = None
        if self.use_semantic_visual:
            if self.semantic_cross_attn is None:
                raise RuntimeError("Semantic visual stream is enabled but semantic_cross_attn is missing.")
            if semantic_tokens is None:
                raise ValueError("Semantic visual tokens are required by this visual_encoder_mode.")
            semantic_proj = self._project_semantic(semantic_tokens.to(mdtype))
            semantic_compact = self._run_cross_attn_stack(
                query,
                semantic_proj,
                first_attn=self.semantic_cross_attn,
                extra_attns=self.semantic_cross_attn_extra,
                extra_norms=self.semantic_cross_attn_extra_norms,
            )
            semantic_compact = self._normalize_latent(semantic_compact.squeeze(1))

        fused_compact = self._fuse_compact(dino_compact, semantic_compact)
        if not return_cross_attn_tokens:
            return fused_compact
        if cross_attn_tokens is None:
            raise RuntimeError(
                "Cross-attention token return requires native-DINO compact querying."
            )
        return fused_compact, cross_attn_tokens

    def _sigreg_enabled(self) -> bool:
        return float(getattr(self.config, "w_sigreg", 0.0)) > 0.0

    def _set_sigreg_projection_step(self, step: int) -> None:
        """Select deterministic projections without touching the global RNG."""
        self._sigreg_projection_step = max(int(step), 0)

    def _sigreg_generator(self, device: torch.device) -> torch.Generator:
        base_seed = int(getattr(self.config, "sigreg_seed", 0))
        step = int(getattr(self, "_sigreg_projection_step", 0))
        return torch.Generator(device=device).manual_seed(base_seed + step)

    def _cross_attn_sigreg(
        self,
        cross_attn_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        dino_camera_mask: Optional[torch.Tensor] = None,
        *,
        row_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Regularize each valid frame's query-token set and measure collapse."""
        if cross_attn_tokens.ndim < 3:
            raise ValueError(
                "Expected cross-attention tokens ending in [N, D], got "
                f"shape={tuple(cross_attn_tokens.shape)}"
            )
        token_sets = cross_attn_tokens.reshape(
            -1,
            cross_attn_tokens.shape[-2],
            cross_attn_tokens.shape[-1],
        )
        valid = valid_mask.to(device=token_sets.device, dtype=torch.bool).reshape(-1)
        if valid.numel() != token_sets.shape[0]:
            raise ValueError(
                f"SIGReg mask has {valid.numel()} entries for {token_sets.shape[0]} token sets."
            )
        token_sets = select_transition_rows(token_sets, valid, row_indices)
        if token_sets.shape[0] == 0:
            zero = cross_attn_tokens.sum() * 0.0
            return zero, zero.detach()

        token_mask = None
        if dino_camera_mask is not None:
            cameras = dino_camera_mask.to(device=token_sets.device, dtype=torch.bool).reshape(-1, self.num_dino_cameras)
            if getattr(self.config, "query_design", "legacy") == "direct_query":
                if cameras.shape[0] != valid.numel() or token_sets.shape[-2] != self.direct_query_layout.num_compacts:
                    raise ValueError("SIGReg direct camera mask must match frames and role-major slots")
                token_mask = self.direct_query_layout.camera_mask(select_transition_rows(cameras, valid, row_indices))
            else:
                per_source_queries = self.num_query_tokens
                if camera_layout_enabled(self):
                    per_source_queries = self.num_dino_cameras * int(self.config.camera_compact_queries_per_source)
                if cameras.shape[0] != valid.numel() or token_sets.shape[-2] % per_source_queries:
                    raise ValueError("SIGReg camera mask must match frames and source/camera query groups")
                queries_per_camera = per_source_queries // self.num_dino_cameras
                source_count = token_sets.shape[-2] // per_source_queries
                token_mask = select_transition_rows(cameras, valid, row_indices).repeat_interleave(queries_per_camera, dim=-1).repeat(1, source_count)
            token_sets = torch.where(token_mask.unsqueeze(-1), token_sets, 0)

        loss = sigreg_epps_pulley_loss(
            token_sets,
            sample_mask=token_mask,
            num_slices=int(self.config.sigreg_num_slices),
            num_knots=int(self.config.sigreg_num_knots),
            t_max=float(self.config.sigreg_t_max),
            generator=self._sigreg_generator(token_sets.device),
        )
        with torch.no_grad():
            normalized = torch.nn.functional.normalize(token_sets.float(), dim=-1)
            similarities = normalized @ normalized.transpose(-1, -2)
            num_tokens = int(token_sets.shape[-2])
            off_diagonal = ~torch.eye(
                num_tokens,
                device=similarities.device,
                dtype=torch.bool,
            )
            if token_mask is None:
                pairwise_cosine = similarities[..., off_diagonal].mean()
            else:
                pairs = token_mask[:, :, None] & token_mask[:, None, :] & off_diagonal
                counts = pairs.sum(dim=(-2, -1))
                per_frame = (similarities * pairs).sum(dim=(-2, -1)) / counts.clamp_min(1)
                supported = counts > 0
                pairwise_cosine = (per_frame * supported).sum() / supported.sum().clamp_min(1)
        return loss, pairwise_cosine

    def _compute_compact_from_query_sources(
        self,
        queries: torch.Tensor,
        dino_patches: Optional[torch.Tensor],
        semantic_tokens: Optional[torch.Tensor],
        *,
        mdtype: torch.dtype,
        dino_camera_mask: Optional[torch.Tensor] = None,
        return_cross_attn_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Query each frame's patch tokens once per source query latent."""
        if camera_layout_enabled(self):
            if queries.ndim == 2:
                queries = queries.unsqueeze(1)
            if queries.ndim != 3 or queries.shape[1:] != (self.query_source_count, self.qwen_dim):
                raise ValueError('Camera compact source inventory differs from configured query sources')
            if dino_patches is None or dino_patches.ndim != 3:
                raise ValueError('Camera compact layout requires dense camera patches [B, C*P, d]')
            batch, total_patches, width = dino_patches.shape
            cameras = int(self.num_dino_cameras)
            if total_patches % cameras:
                raise ValueError('Dense patches must use equal-sized configured camera storage slots')
            result = self.camera_compact_resampler(
                queries, dino_patches.reshape(batch, cameras, total_patches // cameras, width),
                camera_mask=camera_validity(self, dino_camera_mask, batch_size=batch, device=queries.device),
                camera_ids=camera_slot_ids(self, queries.device),
            )
            compact = self._normalize_latent(result.tokens)
            compact = compact.reshape(batch, self.compact_tokens_per_step, self.qwen_dim)
            present = expand_camera_presence(result.camera_mask, self.compact_tokens_per_step,
                tokens_per_camera=self.camera_compact_tokens_per_camera)
            compact = torch.where(present.unsqueeze(-1), compact, 0)
            if self.compact_tokens_per_step == 1:
                compact = compact[:, 0]
            if not return_cross_attn_tokens:
                return compact
            source_count = queries.shape[1]
            count = int(self.config.camera_compact_queries_per_source)
            refined = result.refined_tokens.reshape(batch, cameras, source_count, count, width)
            # Preserve SIGReg's existing source-major camera-mask convention.
            refined = refined.permute(0, 2, 1, 3, 4).reshape(batch, source_count * cameras * count, width)
            return compact, refined
        if not self._uses_compact_query_sources():
            return self._compute_compact_from_visuals(
                queries,
                dino_patches,
                semantic_tokens,
                mdtype=mdtype,
                dino_camera_mask=dino_camera_mask,
                return_cross_attn_tokens=return_cross_attn_tokens,
            )
        if queries.ndim != 3 or queries.shape[1:] != (
            self.compact_tokens_per_step,
            self.qwen_dim,
        ):
            raise ValueError(
                "Expected source queries ending in "
                f"({self.compact_tokens_per_step}, {self.qwen_dim}), got "
                f"{tuple(queries.shape)}."
            )
        batch_size, source_count, _ = queries.shape
        dino_compact = None
        cross_attn_tokens = None
        if self.use_dino_visual:
            if dino_patches is None:
                raise ValueError("DINO patch tokens are required by this visual_encoder_mode.")
            if self.compact_query_mode == "native_dino":
                if self.dino_camera_fusion == "weighted_average":
                    camera_result = self._compute_camera_weighted_dino_compact_sources(
                        queries,
                        dino_patches,
                        dino_camera_mask=dino_camera_mask,
                        return_cross_attn_tokens=return_cross_attn_tokens,
                    )
                    if return_cross_attn_tokens:
                        dino_compact, cross_attn_tokens = camera_result
                    else:
                        dino_compact = camera_result
                else:
                    dino_queries = self._project_dino_query_sources(queries)
                    dino_tokens = dino_patches.to(
                        device=dino_queries.device,
                        dtype=dino_queries.dtype,
                    )
                    attended = self._run_cross_attn_stack(
                        dino_queries.reshape(
                            batch_size,
                            source_count * self.num_query_tokens,
                            self.dino_dim,
                        ),
                        dino_tokens,
                        first_attn=self.dino_native_cross_attn,
                        extra_attns=self.dino_native_cross_attn_extra,
                        extra_norms=self.dino_native_cross_attn_extra_norms,
                    ).reshape(
                        batch_size,
                        source_count,
                        self.num_query_tokens,
                        self.dino_dim,
                    )
                    cross_attn_tokens = attended.reshape(
                        batch_size,
                        source_count * self.num_query_tokens,
                        self.dino_dim,
                    )
                    dino_compact = self._normalize_latent(
                        self.dino_compact_out_proj(
                            self._to_module_dtype(
                                attended.flatten(-2),
                                self.dino_compact_out_proj,
                            )
                        )
                    )
            else:
                projected_patches = self._project_dino(dino_patches.to(mdtype))
                dino_compact = self._normalize_latent(
                    self._run_cross_attn_stack(
                        queries,
                        projected_patches,
                        first_attn=self.cross_attn,
                        extra_attns=self.cross_attn_extra,
                        extra_norms=self.cross_attn_extra_norms,
                    )
                )

        semantic_compact = None
        if self.use_semantic_visual:
            if semantic_tokens is None:
                raise ValueError("Semantic visual tokens are required by this visual_encoder_mode.")
            semantic_projected = self._project_semantic(semantic_tokens.to(mdtype))
            semantic_compact = self._normalize_latent(
                self._run_cross_attn_stack(
                    queries,
                    semantic_projected,
                    first_attn=self.semantic_cross_attn,
                    extra_attns=self.semantic_cross_attn_extra,
                    extra_norms=self.semantic_cross_attn_extra_norms,
                )
            )

        compact = self._fuse_compact(dino_compact, semantic_compact)
        if compact.shape != (batch_size, source_count, self.qwen_dim):
            raise RuntimeError(
                "Each global/task/CLS source query must produce exactly one compact token; "
                f"got {tuple(compact.shape)}."
            )
        if not return_cross_attn_tokens:
            return compact
        if cross_attn_tokens is None:
            raise RuntimeError(
                "Cross-attention token return requires native-DINO compact querying."
            )
        return compact, cross_attn_tokens

    @staticmethod
    def _run_cross_attn_stack(
        query: torch.Tensor,
        key_value: torch.Tensor,
        *,
        first_attn: nn.MultiheadAttention,
        extra_attns: Optional[nn.ModuleList],
        extra_norms: Optional[nn.ModuleList],
    ) -> torch.Tensor:
        attended, _ = first_attn(
            query,
            key_value,
            key_value,
            need_weights=False,
        )
        if not extra_attns:
            return attended
        if extra_norms is None or len(extra_norms) != len(extra_attns):
            raise RuntimeError("Cross-attention stack is missing matching normalization layers.")
        for attn, norm in zip(extra_attns, extra_norms):
            residual = attended
            delta, _ = attn(
                attended,
                key_value,
                key_value,
                need_weights=False,
            )
            attended = norm(residual + delta)
        return attended

    def _compute_queries_from_cls(
        self,
        cls_tokens: torch.Tensor,
        *,
        prefix_embeds: Optional[torch.Tensor] = None,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute queries with their original CLS/query logical skeleton.

        Layout skeleton:
            [INST | PATCH_0 | CS_0 | CLS_0 | QS_0 | CS_1 | CLS_1 | QS_1 | ...]
        The query-only pass feeds the current CLS embedding into the QS slot so
        enabling the separate main-stream ACT slot does not move this stream.
        """
        if int(getattr(self, "num_generated_query_tokens", 1)) > 1:
            raise ValueError(
                "num_generated_query_tokens > 1 requires cached/sliding query "
                "generation; the uncached masked skeleton is unsupported."
            )
        B, T, _ = cls_tokens.shape
        device = cls_tokens.device
        qdtype = self._qwen_dtype()
        cls_inputs = cls_tokens.to(qdtype)

        if prefix_embeds is None:
            prefix_len = 0
            prefix = None
            p_mask = None
        else:
            prefix = prefix_embeds.to(qdtype)
            prefix_len = prefix.shape[1]
            if prefix_mask is None:
                p_mask = torch.ones(B, prefix_len, device=device, dtype=torch.long)
            else:
                p_mask = prefix_mask.to(device=device, dtype=torch.long)

        compact_tokens = int(
            getattr(
                self,
                "compact_tokens_per_step",
                getattr(self, "compact_output_tokens", 1),
            )
        )
        tokens_per_step = compact_tokens + 2
        interleaved_len = tokens_per_step * T
        seq_len = prefix_len + interleaved_len

        query_inputs = torch.zeros(B, seq_len, self.qwen_dim, device=device, dtype=qdtype)
        query_mask = torch.zeros(B, seq_len, device=device, dtype=torch.long)

        if prefix is not None:
            query_inputs[:, :prefix_len] = prefix
            query_mask[:, :prefix_len] = p_mask

        # Keep the query stream's original QS position, independently of ACT in
        # the main stream, rather than using the block's first compact position.
        query_pos = (
            prefix_len
            + tokens_per_step * torch.arange(T, device=device)
            + compact_tokens
            + 1
        )
        query_inputs[:, query_pos] = cls_inputs
        query_mask[:, query_pos] = 1

        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        query_outputs = run_qwen(self,
            inputs_embeds=query_inputs,
            attention_mask=query_mask,
            position_ids=pos_ids,
            output_hidden_states=False,
            use_cache=False,
        )
        cls_hidden = query_outputs.last_hidden_state[:, query_pos]  # [B, T, D]
        queries = self._project_query(cls_hidden)
        return queries, cls_hidden
