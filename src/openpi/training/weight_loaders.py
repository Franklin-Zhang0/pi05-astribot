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
