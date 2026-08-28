import jax
import numpy as np
import pytest

from openpi.training import weight_loaders


def test_merge_params_with_expanded_action_dim_preserves_new_projection_region() -> None:
    loaded = {
        "action_in_proj": {"kernel": np.arange(6, dtype=np.float32).reshape(2, 3)},
        "action_out_proj": {
            "kernel": np.arange(6, dtype=np.float32).reshape(3, 2),
            "bias": np.arange(2, dtype=np.float32),
        },
        "encoder": {"kernel": np.full((2, 2), 7, dtype=np.float32)},
    }
    initialized = {
        "action_in_proj": {"kernel": np.full((4, 3), -1, dtype=np.float32)},
        "action_out_proj": {
            "kernel": np.full((3, 4), -2, dtype=np.float32),
            "bias": np.full((4,), -3, dtype=np.float32),
        },
        "encoder": {"kernel": np.zeros((2, 2), dtype=np.float32)},
    }

    merged = weight_loaders._merge_params_with_expanded_action_dim(loaded, initialized)  # noqa: SLF001

    np.testing.assert_array_equal(merged["action_in_proj"]["kernel"][:2], loaded["action_in_proj"]["kernel"])
    np.testing.assert_array_equal(merged["action_in_proj"]["kernel"][2:], -1)
    np.testing.assert_array_equal(merged["action_out_proj"]["kernel"][:, :2], loaded["action_out_proj"]["kernel"])
    np.testing.assert_array_equal(merged["action_out_proj"]["kernel"][:, 2:], -2)
    np.testing.assert_array_equal(merged["action_out_proj"]["bias"][:2], loaded["action_out_proj"]["bias"])
    np.testing.assert_array_equal(merged["action_out_proj"]["bias"][2:], -3)
    np.testing.assert_array_equal(merged["encoder"]["kernel"], 7)


def test_merge_params_with_expanded_action_dim_rejects_other_shape_changes() -> None:
    loaded = {"encoder": {"kernel": np.zeros((2, 2), dtype=np.float32)}}
    initialized = {"encoder": {"kernel": np.zeros((3, 2), dtype=np.float32)}}

    with pytest.raises(ValueError, match="outside action projections"):
        weight_loaders._merge_params_with_expanded_action_dim(loaded, initialized)  # noqa: SLF001


def test_merge_params_with_expanded_action_dim_supports_shape_struct_reference() -> None:
    loaded = {
        "action_in_proj": {"kernel": np.arange(6, dtype=np.float32).reshape(2, 3)},
        "action_out_proj": {"bias": np.arange(2, dtype=np.float32)},
    }
    initialized_shapes = {
        "action_in_proj": {"kernel": jax.ShapeDtypeStruct((4, 3), np.float32)},
        "action_out_proj": {"bias": jax.ShapeDtypeStruct((4,), np.float32)},
    }

    merged = weight_loaders._merge_params_with_expanded_action_dim(loaded, initialized_shapes)  # noqa: SLF001

    np.testing.assert_array_equal(merged["action_in_proj"]["kernel"][:2], loaded["action_in_proj"]["kernel"])
    np.testing.assert_array_equal(merged["action_in_proj"]["kernel"][2:], 0)
    np.testing.assert_array_equal(merged["action_out_proj"]["bias"][:2], loaded["action_out_proj"]["bias"])
    np.testing.assert_array_equal(merged["action_out_proj"]["bias"][2:], 0)
