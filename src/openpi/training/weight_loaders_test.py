import jax
import numpy as np
import pytest

from openpi.training import weight_loaders


def native_and_routed_params(*, shapes=False):
    loaded = {
        "action_in_proj": {"kernel": np.arange(96, dtype=np.float32).reshape(32, 3), "bias": np.ones(3)},
        "action_out_proj": {"kernel": np.arange(96, dtype=np.float32).reshape(3, 32), "bias": np.arange(32)},
        "backbone": {"kernel": np.full((2, 3), 7, np.float32)},
    }
    reference = {"action_in_projs": {}, "action_out_projs": {}, "backbone": {"kernel": np.zeros((2, 3))}}
    for name in ("gripper", "l6", "l20", "inspire"):
        reference["action_in_projs"][name] = {"kernel": np.full((64, 3), -2.0), "bias": np.full(3, -2.0)}
        reference["action_out_projs"][name] = {"kernel": np.full((3, 64), -3.0), "bias": np.full(64, -3.0)}
    if shapes:
        reference = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), reference)
    return loaded, reference


@pytest.mark.parametrize("shapes", [False, True])
def test_multi_head_initialization_copies_native_overlap_and_complete_backbone(shapes):
    loaded, reference = native_and_routed_params(shapes=shapes)
    merged = weight_loaders._merge_params_with_action_heads(loaded, reference)  # noqa: SLF001
    np.testing.assert_array_equal(merged["backbone"]["kernel"], loaded["backbone"]["kernel"])
    for name in ("gripper", "l6", "l20", "inspire"):
        np.testing.assert_array_equal(merged["action_in_projs"][name]["kernel"][:32], loaded["action_in_proj"]["kernel"])
        np.testing.assert_array_equal(merged["action_out_projs"][name]["kernel"][:, :32], loaded["action_out_proj"]["kernel"])
        np.testing.assert_array_equal(merged["action_out_projs"][name]["bias"][:32], loaded["action_out_proj"]["bias"])
        np.testing.assert_array_equal(merged["action_in_projs"][name]["kernel"][32:], 0 if shapes else -2)
        np.testing.assert_array_equal(merged["action_out_projs"][name]["kernel"][:, 32:], 0 if shapes else -3)


def test_multi_head_initialization_rejects_partial_or_incompatible_backbone():
    loaded, reference = native_and_routed_params()
    del loaded["backbone"]["kernel"]
    with pytest.raises(ValueError, match="missing backbone"):
        weight_loaders._merge_params_with_action_heads(loaded, reference)  # noqa: SLF001
    loaded, reference = native_and_routed_params()
    reference["backbone"]["kernel"] = np.zeros((3, 3))
    with pytest.raises(ValueError, match="backbone shape mismatch"):
        weight_loaders._merge_params_with_action_heads(loaded, reference)  # noqa: SLF001
    loaded, reference = native_and_routed_params()
    loaded["unused_backbone"] = {"kernel": np.zeros((1,))}
    with pytest.raises(ValueError, match="unexpected parameters"):
        weight_loaders._merge_params_with_action_heads(loaded, reference)  # noqa: SLF001


def test_multi_head_initialization_only_expands_action_axis():
    loaded, reference = native_and_routed_params()
    reference["action_in_projs"]["l6"]["kernel"] = np.zeros((64, 4))
    with pytest.raises(ValueError, match="outside expanded action axis"):
        weight_loaders._merge_params_with_action_heads(loaded, reference)  # noqa: SLF001


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
