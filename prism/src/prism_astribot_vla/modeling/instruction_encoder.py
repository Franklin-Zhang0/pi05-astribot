"""Frozen, independently configured Qwen instruction features and their adapter."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM


def load_qwen_text_causal_lm(
    model_name: str,
    *,
    revision: str | None = None,
    attn_implementation: str = "sdpa",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Construct the exact text graph from the pinned Qwen configuration.

    The Astribot checkpoint already contains both Qwen text networks' weights.
    """
    config = AutoConfig.from_pretrained(model_name, revision=revision, local_files_only=True)
    text_config = getattr(config, "text_config", config)
    return AutoModelForCausalLM.from_config(
        text_config, dtype=dtype, attn_implementation=attn_implementation,
    )


def load_qwen_snapshot_encoder(model_name: str, *, revision: str | None, attn_implementation: str):
    """Load both towers of the native Qwen3.5 base without an LM head."""
    from transformers import AutoImageProcessor, Qwen3_5Model

    config = AutoConfig.from_pretrained(model_name, revision=revision)
    if config.model_type != "qwen3_5" or getattr(config, "vision_config", None) is None:
        raise ValueError("VLM image input requires a multimodal Qwen3.5 instruction checkpoint")
    model, info = Qwen3_5Model.from_pretrained(
        model_name, revision=revision, config=config, dtype=torch.bfloat16,
        attn_implementation=attn_implementation, output_loading_info=True,
        # User mappings run before Transformers' global qwen3_5_text
        # conversion. Strip the outer conditional-generation prefix first,
        # preserving language_model.* on the composite base model.
        # HF's save_pretrained may also export nested language_model/visual
        # names. Those follow the library's own inverse conversion path.
        key_mapping={r"^model\.language_model\.(?!(?:language_model|visual)\.)": "language_model.",
                     r"^model\.visual\.": "visual."},
    )
    missing = sorted(info.get("missing_keys", ()))
    mismatched = sorted(info.get("mismatched_keys", ()), key=str)
    errors = info.get("error_msgs", ())
    if missing or mismatched or errors:
        raise RuntimeError(f"Incomplete pretrained Qwen3.5 visual or text weights for {model_name!r}: "
                           f"missing={missing[:8]!r}, mismatched={mismatched[:8]!r}, errors={errors!r}")
    processor = AutoImageProcessor.from_pretrained(model_name, revision=revision)
    return model, processor


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
        vlm_image_input: bool = False,
        vlm_image_size: int = 256,
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

        self.vlm_image_input = vlm_image_input
        if vlm_image_input:
            if not text_input_deepstack:
                raise ValueError("VLM image input requires text_input_deepstack")
            if encoder is not None:
                raise ValueError("VLM image input must load a native composite Qwen checkpoint")
            native, processor = load_qwen_snapshot_encoder(
                model_name, revision=revision, attn_implementation=attn_implementation,
            )
            config = native.config
            stride = config.vision_config.patch_size * config.vision_config.spatial_merge_size
            if vlm_image_size <= 0 or vlm_image_size % stride:
                raise ValueError(f"vlm_image_size must be a positive multiple of {stride}")
            self.vlm_image_size = vlm_image_size
            self.snapshot_grid_side = vlm_image_size // stride
            self.snapshot_image_token_count = self.snapshot_grid_side ** 2
            self.snapshot_start_id = config.vision_start_token_id
            self.snapshot_end_id = config.vision_end_token_id
            self.snapshot_image_id = config.image_token_id
            self.image_processor = processor
            encoder = native.language_model
            # Keep existing text state-dict names and include the tower in the
            # existing replicated, frozen BF16 source partition under HSDP.
            encoder.add_module("vlm_visual", native.visual)
        elif encoder is None:
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

    @torch.inference_mode(False)
    @torch.no_grad()
    def encode_snapshot_images(self, images: torch.Tensor) -> torch.Tensor:
        """Return reusable native merged tokens [B,N,D] for unnormalized VLA RGB."""
        if not self.vlm_image_input:
            raise ValueError("VLM image input is disabled")
        if images.ndim != 4 or images.shape[1] != 3 or not images.shape[0]:
            raise ValueError("Snapshot images must be nonempty [B,3,H,W] RGB tensors")
        if not images.is_floating_point() and images.dtype != torch.uint8:
            raise ValueError("Expected uint8 or floating-point raw RGB")
        visual = self.encoder.vlm_visual
        parameter = next(visual.parameters())
        rgb = images.to(device=parameter.device, dtype=torch.float32)
        if images.dtype == torch.uint8:
            rgb = rgb / 255.
        else:
            if not bool(((rgb >= 0) & (rgb <= 255)).all()):
                raise ValueError("Expected finite raw RGB in [0,1] or [0,255]")
            scales = torch.where(rgb.amax(dim=(1, 2, 3), keepdim=True) > 1.5, 255., 1.)
            rgb = rgb / scales
        # Retain the complete VLA crop and its aspect ratio across buckets.
        # A fixed canvas gives every snapshot the same reserved prefix width.
        height, width = rgb.shape[-2:]
        side = self.vlm_image_size
        if (height, width) != (side, side):
            scale = side / max(height, width)
            out_h, out_w = max(1, round(height * scale)), max(1, round(width * scale))
            rgb = torch.nn.functional.interpolate(rgb, size=(out_h, out_w), mode="bicubic",
                align_corners=False, antialias=True).clamp_(0., 1.)
            top, left = (side - out_h) // 2, (side - out_w) // 2
            rgb = torch.nn.functional.pad(rgb, (left, side-out_w-left, top, side-out_h-top), value=0.5)
        # The checkpoint processor owns normalization, temporal replication
        # and merge-aware patch ordering; RGB is already unit-scaled.
        processed = self.image_processor(
            images=rgb, do_resize=False, do_rescale=False,
            return_tensors="pt", input_data_format="channels_first",
        )
        pixels = processed.pixel_values.to(device=rgb.device, dtype=parameter.dtype)
        grid = processed.image_grid_thw.to(device=rgb.device)
        visual.eval()
        merged = visual(pixels, grid_thw=grid, return_dict=True).pooler_output
        return merged.reshape(images.shape[0], self.snapshot_image_token_count, self.encoder_dim).detach()

    @torch.inference_mode(False)
    @torch.no_grad()
    def snapshot_deepstack_memory(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        image_features: torch.Tensor | None, image_present: torch.Tensor,
    ):
        """Encode [vision-start, native merged image tokens, vision-end, text]."""
        from prism_astribot_vla.modeling.text_deepstack import extract_text_deepstack_memory

        if not self.vlm_image_input or input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise ValueError("VLM snapshots require image mode and matching [B,T] text IDs/mask")
        batch = input_ids.shape[0]
        if image_present.shape != (batch,):
            raise ValueError("Snapshot availability must have shape [B]")
        present = image_present.to(device=input_ids.device, dtype=torch.bool)
        valid = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        if not bool(valid.any(dim=-1).all()):
            raise ValueError("Every snapshot needs a non-padding instruction")
        # Remove gaps between image and text even for left-padded instructions.
        order = valid.to(torch.int32).argsort(dim=-1, descending=True, stable=True)
        input_ids = input_ids.gather(1, order)
        valid = valid.gather(1, order)
        side = self.snapshot_grid_side
        count = self.snapshot_image_token_count
        prefix = input_ids.new_full((batch, count + 2), self.snapshot_image_id)
        prefix[:, 0], prefix[:, -1] = self.snapshot_start_id, self.snapshot_end_id
        ids = torch.cat((prefix, input_ids), dim=-1)
        mask = torch.cat((present[:, None].expand_as(prefix), valid), dim=-1)

        self.encoder.eval()
        embeddings = self.encoder.get_input_embeddings()(ids)
        if image_features is None:
            if bool(present.any()):
                raise ValueError("Missing merged tokens for a present VLM image")
        else:
            if image_features.shape != (batch, count, self.encoder_dim):
                raise ValueError("Snapshot image token geometry does not match the frozen encoder")
            embeddings[:, 1:1+count] = image_features.to(embeddings)
        embeddings = embeddings.masked_fill(~mask[..., None], 0)

        # Equivalent to native Qwen3.5 get_rope_index for one image grid (1,H,W).
        # Each text token advances along all three axes; the first text token
        # follows vision-end at 2 + side. Padding is never attended.
        image_slots = torch.arange(count, device=ids.device)
        image_thw = torch.stack((torch.zeros_like(image_slots), image_slots // side, image_slots % side)) + 1
        prefix_thw = torch.nn.functional.pad(image_thw, (1, 0), value=0)
        prefix_thw = torch.nn.functional.pad(prefix_thw, (0, 1), value=1 + side)
        text_positions = valid.long().cumsum(-1) - 1 + present.long()[:, None] * (2 + side)
        source_positions = torch.cat((prefix_thw[:, None].expand(-1, batch, -1),
            text_positions[None].expand(3, -1, -1)), dim=-1).masked_fill(~mask[None], 0)
        return extract_text_deepstack_memory(
            self.encoder, input_ids=ids, inputs_embeds=embeddings, attention_mask=mask,
            source_position_ids=source_positions, layer_indices=self.deepstack_layer_indices,
        )

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
        from prism_astribot_vla.modeling.text_deepstack import extract_text_deepstack_memory
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
