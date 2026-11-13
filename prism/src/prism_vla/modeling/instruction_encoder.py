"""Frozen, independently configured Qwen instruction features and their adapter."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM


def load_qwen_text_causal_lm(
    model_name: str, *, revision: str | None = None,
    attn_implementation: str = "sdpa", dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Construct the trained Qwen graph from verified local configuration.

    The complete final state is strictly restored by load_bundle. No base
    checkpoint is loaded, and no network or private cache path is consulted.
    """
    from prism_vla.modeling.release_assets import asset_path
    config = AutoConfig.from_pretrained(str(asset_path("qwen")), local_files_only=True)
    text_config = getattr(config, "text_config", config)
    return AutoModelForCausalLM.from_config(
        text_config, dtype=dtype, attn_implementation=attn_implementation,
    )


def _encoder_hidden_size(encoder: nn.Module) -> int:
    """Resolve the text width for text-only and composite Qwen configurations."""
    config = getattr(encoder, "config", None)
    text_config = getattr(config, "text_config", config)
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        get_embeddings = getattr(encoder, "get_input_embeddings", None)
        embeddings = get_embeddings() if callable(get_embeddings) else None
        if embeddings is None:
            embeddings = getattr(encoder, "embed_tokens", None)
        hidden_size = getattr(embeddings, "embedding_dim", None)
    if hidden_size is None or int(hidden_size) <= 0:
        raise ValueError("Cannot infer the frozen instruction encoder's text hidden size.")
    return int(hidden_size)


class FrozenQwenInstructionEncoder(nn.Module):
    """Map a separate frozen Qwen's final text states to backbone input vectors.

    The encoder checkpoint, vocabulary, and hidden size are independent of the
    trainable backbone. Callers must tokenize instructions using this encoder's
    tokenizer. Only the optional width projection, two Transformer blocks, and
    output normalization train. ``encoder`` allows CPU tests with a tiny local
    text model without downloading a checkpoint.
    """

    def __init__(
        self,
        model_name: str,
        backbone_dim: int,
        *,
        revision: str | None = None,
        nhead: int = 8,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        attn_implementation: str = "sdpa",
        encoder: nn.Module | None = None,
        text_input_deepstack: bool = False,
        deepstack_layer_indices: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        if backbone_dim <= 0 or (not text_input_deepstack and (nhead <= 0 or backbone_dim % nhead != 0)):
            raise ValueError("backbone_dim must be positive and hidden-state adapters require divisible positive nhead.")
        if dim_feedforward is None:
            dim_feedforward = 4 * backbone_dim
        if dim_feedforward <= 0:
            raise ValueError("dim_feedforward must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.model_name = model_name
        self.revision = revision
        self.backbone_dim = int(backbone_dim)

        if encoder is None:
            # Retain only the base text network, so no LM logits, vision
            # weights, or duplicate output head are needed.
            causal_lm = load_qwen_text_causal_lm(
                model_name,
                revision=revision,
                attn_implementation=attn_implementation,
            )
            encoder = causal_lm.model
            if hasattr(encoder, "language_model"):
                encoder = encoder.language_model
            del causal_lm
        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.encoder_dim = _encoder_hidden_size(self.encoder)
        self.text_input_deepstack = bool(text_input_deepstack)
        self.deepstack_layer_indices = tuple(deepstack_layer_indices)
        self._deepstack_cache = None
        self._deepstack_input_ids = None
        self._deepstack_attention_mask = None
        if self.text_input_deepstack:
            # No unused hidden-state adapter parameters in this architecture.
            return

        self.projection = (
            nn.Linear(self.encoder_dim, self.backbone_dim)
            if self.encoder_dim != self.backbone_dim
            else nn.Identity()
        )
        # Construct independent layers rather than cloning a shared initial
        # layer, and use pre-LN for stable adaptation of pretrained features.
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=self.backbone_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(2)
        )
        self.norm = nn.LayerNorm(self.backbone_dim)

    def deepstack_memory(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Reuse detached text-only K/V until instruction IDs or padding change.

        Only raw frozen caches are retained. Learned geometry mappings execute
        afresh inside PRISM attention so optimizer steps never reuse old graphs.
        """
        if not self.text_input_deepstack:
            raise RuntimeError("Text input deepstack is disabled")
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape or input_ids.shape[1] == 0:
            raise ValueError("Instructions must have matching nonempty [batch, length] IDs and masks")
        valid = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        if not bool(valid.any(-1).all()):
            raise ValueError("Every instruction must contain a non-padding token")
        reusable = (self._deepstack_cache is not None
            and self._deepstack_input_ids.device == input_ids.device
            and self._deepstack_input_ids.shape == input_ids.shape)
        rows = None
        if reusable:
            changed = ((input_ids != self._deepstack_input_ids).any(-1)
                | (valid != self._deepstack_attention_mask).any(-1))
            if not bool(changed.any()):
                return self._deepstack_cache
            rows = changed.nonzero(as_tuple=False).flatten()
        from prism_vla.modeling.text_deepstack import extract_text_deepstack_memory
        self.encoder.eval()
        selected_ids = input_ids if rows is None else input_ids.index_select(0, rows)
        selected_mask = valid if rows is None else valid.index_select(0, rows)
        memory = extract_text_deepstack_memory(
            self.encoder, input_ids=selected_ids, attention_mask=selected_mask,
            position_ids=selected_mask.long().cumsum(-1).sub(1).clamp_min(0),
            layer_indices=self.deepstack_layer_indices,
        )
        if rows is not None:
            memory = self._deepstack_cache.update_rows(rows, memory)
        self._deepstack_input_ids = input_ids.detach().clone()
        self._deepstack_attention_mask = valid.detach().clone()
        self._deepstack_cache = memory
        return memory

    def _load_from_state_dict(self, *args, **kwargs):
        self._deepstack_cache = None
        self._deepstack_input_ids = None
        self._deepstack_attention_mask = None
        return super()._load_from_state_dict(*args, **kwargs)

    def _apply(self, fn, recurse=True):
        # Cached tensors are runtime-only and must not survive device/dtype moves.
        result = super()._apply(fn, recurse=recurse)
        self._deepstack_cache = None
        self._deepstack_input_ids = None
        self._deepstack_attention_mask = None
        return result

    def train(self, mode: bool = True) -> FrozenQwenInstructionEncoder:
        """Train the adapter while keeping frozen encoder dropout disabled."""
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return ``[batch, length, backbone_dim]`` vectors with zero padding.

        At least one unmasked token is required per instruction. The adapter
        attends bidirectionally within the instruction; the frozen Qwen retains
        its pretrained causal text attention. ``no_grad`` applies only to the
        frozen encoder, so adapter parameters receive normal gradients.
        """
        if self.text_input_deepstack:
            self.deepstack_memory(input_ids, attention_mask)
            return torch.empty(input_ids.shape[0], 0, self.backbone_dim,
                device=input_ids.device, dtype=next(self.encoder.parameters()).dtype)
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have the same [batch, length] shape.")
        if input_ids.shape[1] == 0:
            raise ValueError("Instructions must contain at least one token.")
        valid_tokens = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        if not bool(valid_tokens.any(dim=1).all()):
            raise ValueError("Every instruction must contain at least one non-padding token.")
        # Keep the text positions independent of left or right batch padding.
        position_ids = valid_tokens.long().cumsum(dim=-1).sub(1).clamp_min(0)
        # no_grad (rather than inference_mode) produces ordinary tensors that
        # autograd may save for the trainable width projection and adapter.
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=valid_tokens,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state
        expected_shape = (*input_ids.shape, self.encoder_dim)
        if hidden_states.shape != expected_shape:
            raise ValueError(
                "Frozen instruction encoder returned hidden states with shape "
                f"{tuple(hidden_states.shape)}, expected {expected_shape}."
            )
        hidden_states = hidden_states.to(dtype=self.norm.weight.dtype)
        padding_mask = ~valid_tokens
        hidden_states = hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        hidden_states = self.projection(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=padding_mask)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.to(dtype=self.norm.weight.dtype)
        return hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)
