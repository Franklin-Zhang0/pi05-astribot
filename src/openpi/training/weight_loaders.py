import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import jax
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class ExpandedActionDimCheckpointWeightLoader(WeightLoader):
    """Load a checkpoint while expanding only the action input/output projections.

    Released pi0/pi0.5 checkpoints use 32 action dimensions. Robots with more
    actuators need wider projections, but all other pretrained parameters should
    still load exactly. The overlapping projection block is copied from the
    checkpoint and newly added rows/columns keep their initialized values.
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _merge_params_with_expanded_action_dim(loaded_params, params)


@dataclasses.dataclass(frozen=True)
class MultiEmbodimentCheckpointWeightLoader(WeightLoader):
    """Initialize routed action projections from pi05_base's native projections.

    Backbone parameters must all be present with exactly matching shapes. Only
    the physical action axes may grow, and every routed head receives the same
    pretrained 32-channel overlap. Checkpoint restore/resume uses the ordinary
    complete multi-head parameter tree instead of this base initialization path.
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _merge_params_with_action_heads(loaded_params, params)


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _merge_params_with_expanded_action_dim(loaded_params: at.Params, params: at.Params) -> at.Params:
    """Merge pretrained params, allowing shape changes only in action projections."""

    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    result = dict(flat_ref)

    allowed_shape_change = re.compile(r"(?:^|/)(?:action_in_proj|action_out_proj)/(?:kernel|bias)(?:/value)?$")
    for key, loaded_value in flat_loaded.items():
        if key not in flat_ref:
            continue

        ref_value = flat_ref[key]
        if loaded_value.shape == ref_value.shape:
            result[key] = (
                loaded_value.astype(ref_value.dtype) if loaded_value.dtype != ref_value.dtype else loaded_value
            )
            continue

        if not allowed_shape_change.search(key) or loaded_value.ndim != ref_value.ndim:
            raise ValueError(
                f"Checkpoint parameter shape mismatch outside action projections: {key}: "
                f"checkpoint={loaded_value.shape}, initialized={ref_value.shape}"
            )

        # Training passes ShapeDtypeStruct leaves from jax.eval_shape, not the
        # concrete randomly initialized arrays. There is no initialized value
        # to preserve for the newly added action dimensions in that case, so
        # zero-initialize only the expanded region before copying the complete
        # pretrained overlap. Concrete callers still preserve their values.
        if isinstance(ref_value, jax.ShapeDtypeStruct):
            expanded = np.zeros(ref_value.shape, dtype=ref_value.dtype)
        else:
            expanded = np.array(ref_value, copy=True)
        overlap = tuple(slice(0, min(old, new)) for old, new in zip(loaded_value.shape, ref_value.shape, strict=True))
        expanded[overlap] = loaded_value[overlap].astype(expanded.dtype, copy=False)
        result[key] = expanded

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _merge_params_with_action_heads(loaded_params: at.Params, params: at.Params) -> at.Params:
    """Expand pi05 action axes into named heads while loading the full backbone."""
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    head_pattern = re.compile(r"^(.*?)(action_in_projs|action_out_projs)/([^/]+)/(kernel|bias)(/value)?$")
    result = {}
    used = set()
    head_names = set()
    for key, reference in flat_ref.items():
        match = head_pattern.fullmatch(key)
        if match is None:
            if key not in flat_loaded:
                if "lora" in key:
                    if isinstance(reference, jax.ShapeDtypeStruct):
                        raise ValueError(f"Cannot initialize missing LoRA shape without values: {key}")
                    result[key] = reference
                    continue
                raise ValueError(f"Checkpoint is missing backbone parameter: {key}")
            loaded = flat_loaded[key]
            if loaded.shape != reference.shape:
                raise ValueError(f"Checkpoint backbone shape mismatch: {key}: {loaded.shape} != {reference.shape}")
            result[key] = np.asarray(loaded, dtype=reference.dtype)
            used.add(key)
            continue

        prefix, projection, name, parameter, suffix = match.groups()
        head_names.add(name)
        source = f"{prefix}{projection[:-1]}/{parameter}{suffix or ''}"
        if source not in flat_loaded:
            raise ValueError(f"Checkpoint is missing native action projection parameter: {source}")
        loaded = flat_loaded[source]
        used.add(source)
        if loaded.ndim != len(reference.shape):
            raise ValueError(f"Action projection rank mismatch: {key}")
        # The expert hidden width must match exactly. Only the incoming action
        # dimension / outgoing velocity dimension may be expanded.
        axis = 0 if parameter == "bias" or projection == "action_in_projs" else 1
        if projection == "action_in_projs" and parameter == "bias":
            allowed_shape = loaded.shape == reference.shape
        else:
            allowed_shape = (
                all(old == new for i, (old, new) in enumerate(zip(loaded.shape, reference.shape, strict=True)) if i != axis)
                and reference.shape[axis] >= loaded.shape[axis]
            )
        if not allowed_shape:
            raise ValueError(f"Action projection shape mismatch outside expanded action axis: {key}")
        if isinstance(reference, jax.ShapeDtypeStruct):
            expanded = np.zeros(reference.shape, dtype=reference.dtype)
        else:
            expanded = np.array(reference, copy=True)
        overlap = tuple(slice(0, size) for size in loaded.shape)
        expanded[overlap] = np.asarray(loaded, dtype=expanded.dtype)
        result[key] = expanded
    if not head_names:
        raise ValueError("Multi-embodiment initialization requires named action heads")
    if unexpected := set(flat_loaded) - used:
        raise ValueError(f"Checkpoint has unexpected parameters: {sorted(unexpected)[:5]}")
    return flax.traverse_util.unflatten_dict(result, sep="/")
