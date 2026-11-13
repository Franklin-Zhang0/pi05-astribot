"""C-RADIO visual encoder wrapper for PRISM-VLA."""

from __future__ import annotations

import ast
from contextlib import contextmanager
from functools import lru_cache
import inspect
import textwrap
import threading
from types import MethodType
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


_REENTRANT_PAIRS_LOCK = threading.Lock()


@contextmanager
def _cradio_reentrant_checkpoint_pairs(backbone, *, enabled: bool = False):
    """Keep block0 separate and checkpoint blocks1..26 in native timm pairs.

    The temporary global alias is restricted to this exact backbone's blocks
    during one serialized forward. Checkpoint closures retain their functions
    after restoration; no module, tensor cast, RNG or backward API is replaced.
    """
    if type(enabled) is not bool:
        raise ValueError("cradio_reentrant_checkpoint_pairs must be bool")
    receipt = {"enabled": enabled, "selected_forward_calls": 0}
    if not enabled:
        yield receipt
        return
    from timm.layers import use_reentrant_ckpt
    from timm.models._manipulate import checkpoint_seq

    blocks = getattr(backbone, "blocks", None)
    function = getattr(getattr(backbone, "forward_features", None), "__func__", None)
    namespace = getattr(function, "__globals__", None)
    if (threading.current_thread() is not threading.main_thread()
            or not isinstance(blocks, (nn.Sequential, nn.ModuleList)) or len(blocks) != 27
            or not getattr(backbone, "training", False) or not getattr(backbone, "grad_checkpointing", False)
            or not torch.is_grad_enabled() or not use_reentrant_ckpt()
            or not isinstance(namespace, dict) or namespace.get("checkpoint_seq") is not checkpoint_seq
            or "every" not in inspect.signature(checkpoint_seq).parameters):
        raise RuntimeError("C-RADIO pairing requires the serialized27-block native reentrant training path")
    if not _REENTRANT_PAIRS_LOCK.acquire(blocking=False):
        raise RuntimeError("A C-RADIO checkpoint pairing context is already active")
    ordered = tuple(blocks)

    def selected(functions, x, *args, **kwargs):
        if functions is not blocks:
            return checkpoint_seq(functions, x, *args, **kwargs)
        if args or kwargs or not use_reentrant_ckpt():
            raise RuntimeError("C-RADIO native checkpoint call policy changed during pairing")
        receipt["selected_forward_calls"] += 1
        x = checkpoint_seq(ordered[:1], x, every=1, use_reentrant=True)
        return checkpoint_seq(ordered[1:], x, every=2, use_reentrant=True)

    namespace["checkpoint_seq"] = selected
    try:
        yield receipt
        if receipt["selected_forward_calls"] != 1:
            raise RuntimeError("C-RADIO pairing did not select exactly one native block sequence")
    finally:
        owned = namespace.get("checkpoint_seq") is selected
        if owned:
            namespace["checkpoint_seq"] = checkpoint_seq
        _REENTRANT_PAIRS_LOCK.release()
        if not owned:
            raise RuntimeError("C-RADIO checkpoint pairing function ownership changed")


def _cradio_patch_generator(encoder: nn.Module) -> nn.Module:
    """Resolve the pinned C-RADIO patch-generator topology."""
    hf_model = getattr(encoder, "model", None)
    radio_model = getattr(hf_model, "radio_model", None)
    backbone = getattr(radio_model, "model", None)
    generator = getattr(backbone, "patch_generator", None)
    if not isinstance(generator, nn.Module):
        raise RuntimeError(
            "C-RADIO positional chunking requires "
            "encoder.model.radio_model.model.patch_generator."
        )
    return generator


@lru_cache(maxsize=256)
def _forward_contains_random_calls(forward: Callable[..., Any]) -> bool:
    """Recognize direct RNG/dropout calls in Python module forwards."""
    try:
        source = ast.parse(textwrap.dedent(inspect.getsource(forward)))
    except (OSError, TypeError, IndentationError, SyntaxError):
        return False
    random_calls = {
        "rand", "rand_like", "randn", "randn_like", "randint", "randint_like",
        "randperm", "bernoulli", "bernoulli_", "multinomial", "normal",
        "normal_", "uniform_", "random_", "dropout", "drop_path",
        "feature_dropout", "alpha_dropout", "feature_alpha_dropout", "rrelu",
        "sample", "rsample",
    }
    for node in ast.walk(source):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.attr if isinstance(node.func, ast.Attribute)
            else node.func.id if isinstance(node.func, ast.Name)
            else None
        )
        if name in random_calls:
            return True
    return False


def cradio_stochastic_module_names(
    encoder: nn.Module,
    *,
    exclude_patch_generator: bool = False,
) -> list[str]:
    """List active known randomness/batch-statistics paths in an encoder.

    C-RADIO's CPE viewport and positional dropout use direct RNG calls rather
    than standard Dropout modules. This also recognizes direct known random
    calls in custom Python forwards; it is not an audit of opaque extensions.
    """
    excluded = (
        {id(module) for module in _cradio_patch_generator(encoder).modules()}
        if exclude_patch_generator else set()
    )
    names = []
    for name, module in encoder.named_modules():
        if id(module) in excluded or not module.training:
            continue
        if isinstance(module, nn.modules.dropout._DropoutNd):
            stochastic = module.p > 0
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            stochastic = True
        elif type(module).__name__ == "DropPath" and hasattr(module, "drop_prob"):
            stochastic = float(module.drop_prob or 0) > 0
        elif type(module).__name__.startswith("PatchDropout") and hasattr(module, "prob"):
            stochastic = float(module.prob or 0) > 0
        else:
            custom_positional = bool(getattr(module, "abs_pos", True)) and (
                bool(getattr(module, "cpe_mode", False))
                or float(getattr(module, "pos_dropout", 0) or 0) > 0
            )
            probability = getattr(module, "drop_prob", 0)
            functional_dropout = getattr(module, "dropout", 0)
            forward = getattr(module.forward, "__func__", module.forward)
            stochastic = (
                custom_positional
                or isinstance(probability, (int, float)) and probability > 0
                or isinstance(functional_dropout, (int, float)) and functional_dropout > 0
                or _forward_contains_random_calls(forward)
            )
        if stochastic:
            names.append(name or "<encoder>")
    return names


def _forward_cradio_positional_chunks(
    generator: nn.Module,
    images: torch.Tensor,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Execute the original patch generator in ordered frame groups."""
    original = generator._prism_positional_original_forward
    frames = generator._prism_positional_chunk_frames
    if frames == 0:
        return original(images)
    if generator.num_video_frames is not None:
        raise RuntimeError("C-RADIO positional chunking does not support cpe_video_mode.")
    if not torch.is_tensor(images) or images.ndim != 4:
        raise ValueError("C-RADIO positional chunking expects images [N, C, H, W].")
    sizes = []
    outputs = []
    tuple_output = bool(generator.return_pos_enc)
    for start in range(0, max(images.shape[0], 1), frames):
        chunk = images[start : start + frames]
        output = original(chunk)
        expected = isinstance(output, tuple) and len(output) == 2 if tuple_output else torch.is_tensor(output)
        if not expected:
            raise RuntimeError("Unexpected C-RADIO patch-generator output structure.")
        values = output if tuple_output else (output,)
        if any(not torch.is_tensor(value) or value.ndim < 1 for value in values):
            raise RuntimeError("C-RADIO patch-generator outputs must be batched tensors.")
        if values[0].shape[0] != chunk.shape[0]:
            raise RuntimeError("C-RADIO patch-generator output changed the frame batch size.")
        sizes.append(chunk.shape[0])
        outputs.append(values)

    patches = (
        outputs[0][0] if len(outputs) == 1
        else torch.cat([output[0] for output in outputs], dim=0)
    )
    if not tuple_output:
        return patches
    positions = [output[1] for output in outputs]
    # The pinned generator broadcasts one positional tensor in evaluation or
    # when the input already spans its full position grid. Keep that shape.
    input_grid = tuple(int(size) // generator.patch_size for size in images.shape[-2:])
    batched_positions = (
        generator.training and generator.cpe_mode
        and input_grid != (generator.num_rows, generator.num_cols)
    )
    if not batched_positions and all(position.shape[0] == 1 for position in positions):
        if any(not torch.equal(positions[0], position) for position in positions[1:]):
            raise RuntimeError("Broadcast C-RADIO position tensors changed between chunks.")
        return patches, positions[0]
    if any(position.shape[0] != size for position, size in zip(positions, sizes)):
        raise RuntimeError("C-RADIO position tensors must match their frame batches.")
    return patches, positions[0] if len(positions) == 1 else torch.cat(positions, dim=0)


def configure_cradio_positional_chunking(
    encoder: nn.Module,
    positional_chunk_frames: int,
) -> None:
    """Keep positional RNG grouping fixed while changing backbone batching.

    Configure after selecting the encoder's training mode. Only the existing
    patch-generator forward is wrapped; parameter identities, precision,
    autograd, and checkpoint keys are unchanged. Zero restores the original
    whole-batch path. Active stochastic operations elsewhere are rejected.
    """
    if isinstance(positional_chunk_frames, bool) or not isinstance(positional_chunk_frames, int):
        raise ValueError("positional_chunk_frames must be a nonnegative integer.")
    if positional_chunk_frames < 0:
        raise ValueError("positional_chunk_frames must be a nonnegative integer.")
    generator = _cradio_patch_generator(encoder)
    required = ("num_video_frames", "return_pos_enc", "cpe_mode", "patch_size", "num_rows", "num_cols")
    if any(not hasattr(generator, field) for field in required) or not callable(generator.forward):
        raise RuntimeError("Unsupported pinned C-RADIO patch-generator interface.")
    original = getattr(generator, "_prism_positional_original_forward", generator.forward)
    try:
        parameters = list(inspect.signature(original).parameters.values())
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Cannot validate C-RADIO patch-generator forward.") from exc
    if len(parameters) != 1 or parameters[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise RuntimeError("C-RADIO patch-generator forward must accept one image tensor.")
    if positional_chunk_frames:
        if generator.num_video_frames is not None:
            raise RuntimeError("C-RADIO positional chunking does not support cpe_video_mode.")
        stochastic = cradio_stochastic_module_names(encoder, exclude_patch_generator=True)
        if stochastic:
            raise RuntimeError(
                "C-RADIO positional chunking cannot preserve active stochastic "
                "operations outside the patch generator: " + repr(stochastic)
            )
    if hasattr(generator, "_prism_positional_original_forward"):
        if getattr(generator.forward, "__func__", None) is not _forward_cradio_positional_chunks:
            raise RuntimeError("C-RADIO patch-generator forward changed after positional wrapping.")
    elif positional_chunk_frames:
        generator._prism_positional_original_forward = original
        generator.forward = MethodType(_forward_cradio_positional_chunks, generator)
    generator._prism_positional_chunk_frames = positional_chunk_frames


def _flatten_cradio_blocks(value: Any) -> list[nn.Module]:
    if not isinstance(value, (nn.ModuleList, nn.Sequential)):
        raise RuntimeError(
            "C-RADIO block compilation requires "
            "encoder.model.radio_model.model.blocks to be a ModuleList/Sequential."
        )
    blocks: list[nn.Module] = []
    for item in value:
        if isinstance(item, (nn.ModuleList, nn.Sequential)):
            blocks.extend(_flatten_cradio_blocks(item))
        elif isinstance(item, nn.Module):
            blocks.append(item)
        else:
            raise RuntimeError(
                "C-RADIO block compilation found a non-module block value "
                f"{type(item).__name__}."
            )
    if not blocks:
        raise RuntimeError("C-RADIO block compilation found no transformer blocks.")
    if len({id(block) for block in blocks}) != len(blocks):
        raise RuntimeError("C-RADIO block compilation found duplicate module objects.")
    return blocks


def compile_cradio_backbone_blocks(
    encoder: nn.Module,
    *,
    backend: str = "inductor",
    mode: str = "default",
    dynamic: bool = False,
    compiler: Optional[Callable[..., Callable[..., Any]]] = None,
) -> int:
    """Compile C-RADIO transformer blocks without wrapping the encoder module."""

    hf_model = getattr(encoder, "model", None)
    radio_model = getattr(hf_model, "radio_model", None)
    backbone = getattr(radio_model, "model", None)
    blocks = _flatten_cradio_blocks(getattr(backbone, "blocks", None))
    compile_fn = torch.compile if compiler is None else compiler
    if not callable(compile_fn):
        raise RuntimeError("C-RADIO block compiler is not callable.")

    requested = (str(backend), str(mode), bool(dynamic))
    compiled_count = 0
    for index, block in enumerate(blocks):
        installed = getattr(block, "_prism_cradio_block_compile_spec", None)
        if installed is not None:
            if tuple(installed) != requested:
                raise RuntimeError(
                    "C-RADIO block compiler options changed after installation at "
                    f"block {index}: installed={tuple(installed)!r}, "
                    f"requested={requested!r}."
                )
            continue
        eager_forward = block.forward
        try:
            compiled_forward = compile_fn(
                eager_forward,
                backend=requested[0],
                mode=requested[1],
                dynamic=requested[2],
                fullgraph=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to install C-RADIO block compiler at block {index}."
            ) from exc
        if not callable(compiled_forward):
            raise RuntimeError(
                f"C-RADIO compiler returned a non-callable value at block {index}."
            )
        block.forward = compiled_forward
        block._prism_cradio_block_compile_spec = requested
        compiled_count += 1
    return compiled_count


CRADIO_TUNING_MODES = ("full", "summary_tokens")
CRADIO_SUMMARY_TOKEN_PARAMETER = (
    "model.radio_model.model.patch_generator.cls_token.token"
)
CRADIO_SO400M_SUMMARY_TOKEN_SHAPE = (10, 1152)
CRADIO_SO400M_CLS_TOKENS = 3
CRADIO_SO400M_REGISTER_TOKENS = 7


class CRADIOEncoder(nn.Module):
    """Frozen-compatible Hugging Face wrapper for NVIDIA C-RADIOv4.

    C-RADIO exposes multiple global summary tokens flattened into one vector.
    PRISM-VLA consumes one CLS-equivalent vector with the same width as spatial
    tokens, so this wrapper reshapes and mean-pools those summary tokens.
    """

    def __init__(
        self,
        model_name: str = "nvidia/C-RADIOv4-SO400M",
        image_size: int = 256,
        pretrained: bool = True,
        trainable: bool = False,
        tuning_mode: str = "full",
        gradient_checkpointing: bool = False,
        cradio_reentrant_checkpoint_pairs: bool = False,
        revision: str = "",
        summary_pooling: str = "mean",
        trust_remote_code: bool = True,
        dynamic_resolution: bool = False,
    ):
        super().__init__()
        if type(cradio_reentrant_checkpoint_pairs) is not bool:
            raise ValueError("cradio_reentrant_checkpoint_pairs must be bool")
        self.cradio_reentrant_checkpoint_pairs = cradio_reentrant_checkpoint_pairs
        if not pretrained:
            raise ValueError("CRADIOEncoder currently requires pretrained=True.")
        if image_size < 256 or image_size % 16 != 0:
            raise ValueError(
                "C-RADIOv4 image_size must be at least 256 and divisible by 16, "
                f"got {image_size}."
            )

        from transformers import AutoModel

        summary_pooling = str(summary_pooling).strip().lower()
        if summary_pooling not in {"mean", "first"}:
            raise ValueError(
                "C-RADIO summary_pooling must be 'mean' or 'first', "
                f"got {summary_pooling!r}."
            )
        tuning_mode = str(tuning_mode or "full").strip().lower().replace("-", "_")
        if tuning_mode not in CRADIO_TUNING_MODES:
            raise ValueError(
                "C-RADIO tuning_mode must be one of "
                f"{list(CRADIO_TUNING_MODES)!r}, got {tuning_mode!r}."
            )
        if not trainable and tuning_mode != "full":
            raise ValueError(
                "A frozen C-RADIO encoder cannot request a trainable tuning mode; "
                f"got tuning_mode={tuning_mode!r}."
            )
        load_kwargs = {"trust_remote_code": trust_remote_code}
        if revision:
            load_kwargs["revision"] = str(revision)
        from prism_vla.modeling.release_assets import build_cradio_architecture
        self.model = build_cradio_architecture()
        self.image_size = int(image_size)
        self.model_name = str(model_name)
        self.revision = str(revision)
        self.summary_pooling = summary_pooling
        self.tuning_mode = tuning_mode
        self.dynamic_resolution = bool(dynamic_resolution)
        self.gradient_checkpointing = bool(trainable and gradient_checkpointing)
        self.patch_size = int(getattr(self.model.config, "patch_size", 16))
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size={self.image_size} is not divisible by "
                f"C-RADIO patch_size={self.patch_size}."
            )

        radio_model = getattr(self.model, "radio_model", None)
        embed_dim = getattr(radio_model, "embed_dim", None)
        if embed_dim is None:
            backbone = getattr(self.model, "model", None)
            embed_dim = getattr(backbone, "embed_dim", None)
        if embed_dim is None:
            raise ValueError("Could not infer C-RADIO spatial feature dimension.")
        self.embed_dim = int(embed_dim)
        self.summary_dim = int(
            getattr(radio_model, "summary_dim", self.embed_dim)
        )
        if self.summary_dim % self.embed_dim != 0:
            raise ValueError(
                "C-RADIO summary dimension must be a multiple of its spatial "
                f"dimension, got summary_dim={self.summary_dim}, "
                f"embed_dim={self.embed_dim}."
            )
        self.num_summary_tokens = self.summary_dim // self.embed_dim
        # Keep the nominal square count stable for checkpoint/config contracts.
        # Dynamic forwards validate their actual patch count from runtime H/W.
        self.num_patches = (self.image_size // self.patch_size) ** 2
        nearest_resolution = getattr(self.model, "get_nearest_supported_resolution", None)
        if callable(nearest_resolution):
            nearest = nearest_resolution(self.image_size, self.image_size)
            nearest_hw = tuple(int(value) for value in nearest)
            if nearest_hw != (self.image_size, self.image_size):
                raise ValueError(
                    f"C-RADIO does not support image_size={self.image_size} exactly; "
                    f"nearest supported resolution is {nearest_hw}."
                )

        for param in self.model.parameters():
            param.requires_grad = bool(trainable and tuning_mode == "full")
        if trainable and tuning_mode == "summary_tokens":
            summary_parameter = self._resolve_summary_token_parameter()
            summary_parameter.requires_grad_(True)
            self.register_buffer(
                "_initial_summary_tokens",
                summary_parameter.detach().clone(),
                persistent=False,
            )
            self._validate_summary_token_trainability()
        if self.gradient_checkpointing:
            self._enable_gradient_checkpointing()
        self.model.train(mode=bool(trainable))

    def _enable_gradient_checkpointing(self) -> None:
        from timm.layers import set_reentrant_ckpt

        backbone = getattr(getattr(self.model, "radio_model", None), "model", None)
        set_checkpointing = getattr(backbone, "set_grad_checkpointing", None)
        if backbone is None or not callable(set_checkpointing):
            raise RuntimeError(
                "Pinned C-RADIO backbone does not expose timm's "
                "set_grad_checkpointing API."
            )
        # Retained visual micro-batches coexist until policy backward finishes.
        # Reentrant checkpointing saves only each block input and avoids mutable
        # non-reentrant autograd metadata being shared across those forwards.
        set_reentrant_ckpt(True)
        set_checkpointing(True)
        if not bool(getattr(backbone, "grad_checkpointing", False)):
            raise RuntimeError(
                "C-RADIO set_grad_checkpointing(True) did not enable the "
                "backbone's native checkpointed block path."
            )

    def _resolve_summary_token_parameter(self) -> nn.Parameter:
        value: Any = self
        for component in CRADIO_SUMMARY_TOKEN_PARAMETER.split("."):
            value = getattr(value, component, None)
            if value is None:
                raise RuntimeError(
                    "Pinned C-RADIO summary-token topology is missing "
                    f"{CRADIO_SUMMARY_TOKEN_PARAMETER!r} at component {component!r}."
                )
        if not isinstance(value, nn.Parameter):
            raise RuntimeError(
                "Pinned C-RADIO summary-token topology must resolve to an "
                f"nn.Parameter, got {type(value).__name__}."
            )
        expected_shape = CRADIO_SO400M_SUMMARY_TOKEN_SHAPE
        if tuple(value.shape) != expected_shape:
            raise RuntimeError(
                "Pinned C-RADIO summary-token parameter shape changed: "
                f"expected={expected_shape}, actual={tuple(value.shape)}."
            )
        if value.dtype != torch.float32:
            raise RuntimeError(
                "Pinned C-RADIO summary-token parameter must retain FP32 "
                f"storage, got {value.dtype}."
            )
        if dict(self.named_parameters()).get(CRADIO_SUMMARY_TOKEN_PARAMETER) is not value:
            raise RuntimeError(
                "Pinned C-RADIO summary-token parameter is not registered at "
                f"the canonical name {CRADIO_SUMMARY_TOKEN_PARAMETER!r}."
            )
        patch_generator = getattr(
            getattr(getattr(self.model, "radio_model", None), "model", None),
            "patch_generator",
            None,
        )
        topology = {
            "num_cls_tokens": CRADIO_SO400M_CLS_TOKENS,
            "num_registers": CRADIO_SO400M_REGISTER_TOKENS,
            "num_skip": CRADIO_SO400M_CLS_TOKENS + CRADIO_SO400M_REGISTER_TOKENS,
        }
        for attribute, expected in topology.items():
            actual = getattr(patch_generator, attribute, None)
            if actual is None or int(actual) != expected:
                raise RuntimeError(
                    "Pinned C-RADIO summary-token topology changed: "
                    f"{attribute} expected={expected}, actual={actual!r}."
                )
        return value

    def _validate_summary_token_trainability(self) -> None:
        trainable = [
            (name, tuple(parameter.shape), int(parameter.numel()))
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        expected = [
            (
                CRADIO_SUMMARY_TOKEN_PARAMETER,
                CRADIO_SO400M_SUMMARY_TOKEN_SHAPE,
                int(np.prod(CRADIO_SO400M_SUMMARY_TOKEN_SHAPE)),
            )
        ]
        if trainable != expected:
            raise RuntimeError(
                "summary_tokens tuning must expose exactly the canonical "
                f"C-RADIO token tensor: expected={expected!r}, actual={trainable!r}."
            )

    def summary_token_parameter(self) -> nn.Parameter:
        """Return the canonical C-RADIO CLS/register token tensor."""
        return self._resolve_summary_token_parameter()

    def summary_token_similarity_loss(self) -> torch.Tensor:
        """Penalize angular drift from the pretrained summary tokens."""
        reference = getattr(self, "_initial_summary_tokens", None)
        if reference is None:
            raise RuntimeError(
                "Summary-token similarity requires trainable summary_tokens mode."
            )
        current = self.summary_token_parameter().float()
        reference = reference.to(device=current.device, dtype=current.dtype)
        return (1.0 - F.cosine_similarity(current, reference, dim=-1)).mean()

    def _summary_tokens(self, summary: torch.Tensor) -> torch.Tensor:
        if summary.ndim == 3:
            if summary.shape[-1] != self.embed_dim:
                raise ValueError(
                    "Unexpected C-RADIO summary token width: "
                    f"{tuple(summary.shape)} versus embed_dim={self.embed_dim}."
                )
            return summary
        if summary.ndim != 2:
            raise ValueError(
                f"Expected C-RADIO summary rank 2/3, got {tuple(summary.shape)}."
            )
        if summary.shape[-1] % self.embed_dim != 0:
            raise ValueError(
                "C-RADIO flattened summary width is not divisible by the spatial "
                f"feature width: {tuple(summary.shape)}, embed_dim={self.embed_dim}."
            )
        return summary.reshape(summary.shape[0], -1, self.embed_dim)

    def _collapse_summary(self, summary: torch.Tensor) -> torch.Tensor:
        summary_tokens = self._summary_tokens(summary)
        return (
            summary_tokens.mean(dim=1)
            if self.summary_pooling == "mean"
            else summary_tokens[:, 0]
        )

    def forward_with_aux_tokens(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return pooled, unpooled-global, and flattened spatial tokens.

        This wrapper does not know how flattened frames are grouped temporally,
        so it intentionally does not enter C-RADIO's ``cpe_video_mode`` here.
        Callers that know the exact ``B x T`` layout may apply that context to
        keep CPE viewports consistent across a temporal group.
        """
        expected_patches = self.num_patches_for_resolution(images.shape[-2:])
        paired = (getattr(self, "cradio_reentrant_checkpoint_pairs", False)
            and self.gradient_checkpointing and self.training and torch.is_grad_enabled())
        backbone = getattr(getattr(self.model, "radio_model", None), "model", None) if paired else None
        with _cradio_reentrant_checkpoint_pairs(backbone, enabled=paired):
            outputs = self.model(images)
        if isinstance(outputs, dict):
            outputs = outputs.get("backbone", outputs)
        if hasattr(outputs, "summary") and hasattr(outputs, "features"):
            summary = outputs.summary
            spatial = outputs.features
        elif isinstance(outputs, (tuple, list)) and len(outputs) == 2:
            summary, spatial = outputs
        else:
            raise TypeError(
                "C-RADIO forward must return (summary, spatial_features), got "
                f"{type(outputs)!r}."
            )

        if spatial.ndim == 4:
            spatial = spatial.flatten(2).transpose(1, 2)
        if spatial.ndim != 3:
            raise ValueError(
                f"Expected C-RADIO spatial features [B, P, D], got {tuple(spatial.shape)}."
            )
        if spatial.shape[1] != expected_patches or spatial.shape[2] != self.embed_dim:
            raise ValueError(
                "C-RADIO output shape disagrees with the runtime encoder contract: "
                f"got {tuple(spatial.shape)}, expected [B, {expected_patches}, "
                f"{self.embed_dim}]."
            )
        aux_global_tokens = self._summary_tokens(summary)
        return self._collapse_summary(aux_global_tokens), aux_global_tokens, spatial

    def forward_with_register_tokens(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return CLS summaries, spatial features and post-ViT register activations.

        The pinned RADIO public output drops registers before returning features.
        Run the same input conditioner and backbone once, retaining its register
        slice before using RADIO's own output extraction for CLS and patches.
        """
        expected_patches = self.num_patches_for_resolution(images.shape[-2:])
        radio = getattr(self.model, "radio_model", None)
        backbone = getattr(radio, "model", None)
        generator = getattr(backbone, "patch_generator", None)
        if (radio is None or backbone is None or generator is None
                or not callable(getattr(radio, "input_conditioner", None))
                or not callable(getattr(backbone, "forward_features", None))
                or not callable(getattr(radio, "_extract_final", None))):
            raise RuntimeError("Register extraction requires the pinned C-RADIO topology")
        n_cls = int(getattr(generator, "num_cls_tokens", -1))
        n_registers = int(getattr(generator, "num_registers", -1))
        n_skip = int(getattr(generator, "num_skip", -1))
        if (n_cls, n_registers, n_skip) != (
            CRADIO_SO400M_CLS_TOKENS, CRADIO_SO400M_REGISTER_TOKENS,
            CRADIO_SO400M_CLS_TOKENS + CRADIO_SO400M_REGISTER_TOKENS,
        ):
            raise RuntimeError("Pinned C-RADIO CLS/register topology changed")
        paired = (getattr(self, "cradio_reentrant_checkpoint_pairs", False)
            and self.gradient_checkpointing and self.training and torch.is_grad_enabled())
        with _cradio_reentrant_checkpoint_pairs(backbone, enabled=paired):
            conditioned = radio.input_conditioner(images)
            tokens = backbone.forward_features(conditioned)
            outputs = radio._extract_final(conditioned, tokens, feature_fmt="NLC")
        if (not torch.is_tensor(tokens) or tokens.ndim != 3
                or tokens.shape[1:] != (n_skip + expected_patches, self.embed_dim)):
            raise RuntimeError("Unexpected C-RADIO full-token output shape")
        if isinstance(outputs, dict):
            outputs = outputs.get("backbone", outputs)
        if not hasattr(outputs, "summary") or not hasattr(outputs, "features"):
            raise TypeError("C-RADIO output must expose summary and spatial features")
        spatial = outputs.features
        if spatial.shape != (images.shape[0], expected_patches, self.embed_dim):
            raise RuntimeError("Unexpected C-RADIO spatial feature shape")
        aux = self._summary_tokens(outputs.summary)
        return self._collapse_summary(aux), aux, spatial, tokens[:, n_cls:n_skip]

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one global token and flattened spatial tokens."""
        cls_tokens, _, spatial = self.forward_with_aux_tokens(images)
        return cls_tokens, spatial

    @staticmethod
    def _array_to_tensor(array: np.ndarray) -> torch.Tensor:
        if array.ndim == 3:
            array = array[None]
        if array.ndim != 4:
            raise ValueError(f"Expected image array rank 3/4, got shape={array.shape}.")
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2)
        elif tensor.shape[1] != 3:
            raise ValueError(f"Expected RGB images, got shape={array.shape}.")
        return tensor

    @classmethod
    def _to_tensor(
        cls,
        images: list | np.ndarray | Image.Image | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            tensor = images.detach()
            if tensor.ndim == 5:
                if tensor.shape[-1] == 3:
                    tensor = tensor.reshape(-1, *tensor.shape[-3:])
                elif tensor.shape[2] == 3:
                    tensor = tensor.reshape(-1, *tensor.shape[2:])
                else:
                    raise ValueError(f"Expected RGB images, got shape={tuple(tensor.shape)}.")
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 4:
                raise ValueError(
                    f"Expected image tensor rank 3/4/5, got shape={tuple(tensor.shape)}."
                )
            if tensor.shape[-1] == 3:
                tensor = tensor.permute(0, 3, 1, 2)
            elif tensor.shape[1] != 3:
                raise ValueError(f"Expected RGB images, got shape={tuple(tensor.shape)}.")
            return tensor
        if isinstance(images, np.ndarray):
            return cls._array_to_tensor(images)
        if isinstance(images, Image.Image):
            return cls._array_to_tensor(np.asarray(images.convert("RGB")))
        if isinstance(images, (list, tuple)):
            tensors = [cls._to_tensor(image) for image in images]
            return torch.cat(tensors, dim=0)
        raise TypeError(f"Unsupported image input type: {type(images)!r}.")

    def preprocess_image_tensor(
        self,
        images: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Convert raw RGB images to C-RADIO's [0, 1] input.

        Dynamic mode treats H/W as an already selected resolution bucket and
        never changes its geometry. Fixed mode retains the legacy square resize.
        """
        if device is None:
            device = self.device
        images = self._to_tensor(images).contiguous()
        unsigned_rgb = images.dtype == torch.uint8
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        if unsigned_rgb:
            if images.numel():
                # uint8 cannot be negative. Keep the legacy whole-call scaling
                # decision on the device, including all-zero/all-one images,
                # rather than synchronizing CUDA for two Python conditions.
                images = torch.where(torch.amax(images) > 1.5, images.div(255.0), images)
        else:
            if images.numel() and torch.amin(images) < 0:
                raise ValueError("C-RADIO preprocessing expects raw unnormalized images.")
            if images.numel() and torch.amax(images) > 1.5:
                images = images.div(255.0)
        if self.dynamic_resolution:
            self.num_patches_for_resolution(images.shape[-2:])
        elif images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return images.clamp_(0.0, 1.0)

    def num_patches_for_resolution(self, resolution: tuple[int, int]) -> int:
        """Validate a runtime H/W bucket and return its spatial token count."""
        if len(resolution) != 2:
            raise ValueError(f"Expected image resolution (H, W), got {resolution!r}.")
        height, width = (int(resolution[0]), int(resolution[1]))
        if height <= 0 or width <= 0:
            raise ValueError(f"C-RADIO image dimensions must be positive, got {(height, width)}.")
        resolution_step = int(
            getattr(self.model, "min_resolution_step", self.patch_size)
            or self.patch_size
        )
        if height % resolution_step != 0 or width % resolution_step != 0:
            raise ValueError(
                "C-RADIO dynamic resolution must be aligned to "
                f"min_resolution_step={resolution_step}, got HxW={height}x{width}."
            )
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"C-RADIO H/W must be divisible by patch_size={self.patch_size}, "
                f"got {height}x{width}."
            )
        nearest_resolution = getattr(self.model, "get_nearest_supported_resolution", None)
        if callable(nearest_resolution):
            nearest_hw = tuple(
                int(value) for value in nearest_resolution(height, width)
            )
            if nearest_hw != (height, width):
                raise ValueError(
                    "C-RADIO does not support the requested dynamic resolution "
                    f"{(height, width)} exactly; nearest is {nearest_hw}."
                )
        return (height // self.patch_size) * (width // self.patch_size)

    def preprocess_images(
        self,
        images: list | np.ndarray | Image.Image | torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        return self.preprocess_image_tensor(self._to_tensor(images), device=device)

    def extract_features(
        self,
        images: list | np.ndarray | Image.Image | torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cls_tokens, _, spatial = self.extract_features_with_aux_tokens(
            images,
            device=device,
        )
        return cls_tokens, spatial

    def extract_features_with_aux_tokens(
        self,
        images: list | np.ndarray | Image.Image | torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract pooled summary, raw summary tokens, and spatial features."""
        return self.forward_with_aux_tokens(
            self.preprocess_images(images, device=device)
        )

    def extract_features_with_register_tokens(
        self,
        images: list | np.ndarray | Image.Image | torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_with_register_tokens(self.preprocess_images(images, device=device))

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype
