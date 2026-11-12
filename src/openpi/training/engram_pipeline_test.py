"""A real synthetic parquet batch must reach the configured training model."""

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from openpi.models import tokenizer
from openpi.shared.engram_multi import EMBODIMENTS
from openpi.shared.engram_multi import ROUTE_NAMES
from openpi.shared.engram_multi import physical_mask
from openpi.shared.engram_multi import save_route_stats
from openpi.training import config
from openpi.training import data_loader
from openpi.training import engram_multi_dataset as multi
from openpi.training.engram_multi_dataset_test import make_source


def test_configured_parquet_loader_forwards_routes_masks_and_token_state(tmp_path, monkeypatch):
    token_model = Path("/tmp/engram-openpi-cache/big_vision/paligemma_tokenizer.model")
    if not token_model.is_file():
        pytest.skip("Cached PaliGemma sentencepiece model is unavailable")
    monkeypatch.setattr(tokenizer.download, "maybe_download", lambda *_a, **_kw: token_model)
    captured = []
    original_tokenize = tokenizer.PaligemmaTokenizer.tokenize

    def record_token_state(self, prompt, state=None):
        captured.append(np.asarray(state).copy())
        return original_tokenize(self, prompt, state)

    monkeypatch.setattr(tokenizer.PaligemmaTokenizer, "tokenize", record_token_state)
    sources = [make_source(tmp_path, index, hands=index != 3, wrists=index != 2) for index in range(4)]
    manifest_path = tmp_path / "dataset.json"
    manifest_path.write_text(json.dumps({"format": multi.MANIFEST_FORMAT, "sources": sources}))
    raw = multi.EngramMultiEmbodimentDataset(manifest_path, 3, split="train", decode_images=False)
    stats, counts = multi.compute_training_stats(raw, seed=0, batch_size=5)
    assets = tmp_path / "assets" / "engram/multi_embodiment_train"
    stats_path = assets / "engram_multi_stats.json"
    save_route_stats(stats_path, stats)
    recipe = config.get_config("pi05_engram_multi")
    model_config = dataclasses.replace(recipe.model, action_horizon=3)
    identity = multi.stats_identity(manifest_path, "train", 3, 0.01, 0.0)
    identity.update({
        **counts, "sampled_anchors": counts["used_anchors"], "max_frames": None,
        "stats_filename": stats_path.name, "stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        "max_token_len": model_config.max_token_len, "action_head_names": list(ROUTE_NAMES),
        "structured_prompt_probability": recipe.data.structured_prompt_probability,
    })
    (assets / "engram_multi_manifest.json").write_text(json.dumps(identity))
    factory = dataclasses.replace(
        recipe.data, data_root=str(manifest_path), video_mode="original",
        assets=dataclasses.replace(recipe.data.assets, assets_dir=str(tmp_path / "assets")),
    )
    data_config = dataclasses.replace(factory.create(tmp_path, model_config), engram_decode_images=False)
    loader = data_loader.create_torch_data_loader(
        data_config, model_config, 3, len(raw), num_workers=0, num_batches=1, shuffle=False,
    )
    observation, actions = next(iter(loader))
    assert observation.state.shape == (24, 64)
    assert observation.tokenized_prompt.shape == (24, model_config.max_token_len)
    assert actions.shape == (24, 3, 64)
    np.testing.assert_allclose(observation.state, np.stack(captured))
    for index, spec in enumerate(EMBODIMENTS):
        rows = np.flatnonzero(np.asarray(observation.embodiment_id) == index)
        assert len(rows) == 6
        mask = physical_mask(index, index != 3)
        np.testing.assert_array_equal(observation.action_mask[rows], np.broadcast_to(mask, (6, 64)))
        assert np.isfinite(np.asarray(actions)[rows]).all()
        assert not np.asarray(observation.state)[rows][:, ~mask].any()
        assert not np.asarray(actions)[rows][:, :, ~mask].any()
        assert np.asarray(observation.action_mask)[rows, :spec.joint_dim].all()
        np.testing.assert_array_equal(observation.image_masks["left_wrist_0_rgb"][rows], index != 2)
    assert loader.data_config().engram_video_identity == raw.video_identity
    raw.close()
    # Model and prompt changes must fail before constructing a training dataset.
    def unexpected_dataset_creation(*_args, **_kwargs):
        raise AssertionError("Incompatible contracts must be rejected before dataset construction")

    monkeypatch.setattr(data_loader, "create_torch_dataset", unexpected_dataset_creation)
    manifest_file = assets / "engram_multi_manifest.json"
    for key, changed in (
        ("max_token_len", model_config.max_token_len + 1),
        ("action_head_names", list(reversed(ROUTE_NAMES))),
        ("structured_prompt_probability", 0.75),
    ):
        manifest_file.write_text(json.dumps({**identity, key: changed}))
        with pytest.raises(ValueError, match="model or prompt mixture"):
            data_loader.create_torch_data_loader(
                data_config, model_config, 3, len(raw), num_workers=0, num_batches=1,
            )
    manifest_file.write_text(json.dumps(identity))


def test_multi_startup_audit_flags_propagate_from_recipe_to_loader(tmp_path, monkeypatch):
    token_model = Path("/tmp/engram-openpi-cache/big_vision/paligemma_tokenizer.model")
    if not token_model.is_file():
        pytest.skip("Cached PaliGemma sentencepiece model is unavailable")
    monkeypatch.setattr(tokenizer.download, "maybe_download", lambda *_a, **_kw: token_model)
    recipe = config.get_config("pi05_engram_multi")
    assert not recipe.data.verify_proxy_hash
    assert not recipe.data.audit_source_metadata
    for enabled in (False, True):
        factory = dataclasses.replace(recipe.data, verify_proxy_hash=enabled, audit_source_metadata=enabled)
        data_config = factory.create(tmp_path / "assets", recipe.model)
        assert data_config.engram_multi_verify_proxy_hash is enabled
        assert data_config.engram_multi_audit_source_metadata is enabled
        captured = {}

        def create_dataset(*args, _captured=captured, **kwargs):
            _captured.update(kwargs)
            return "dataset created"

        monkeypatch.setattr(multi, "EngramMultiEmbodimentDataset", create_dataset)
        assert data_loader.create_torch_dataset(data_config, 64, recipe.model) == "dataset created"
        assert captured["verify_proxy_hash"] is enabled

        def check_identity(*args, _captured=captured, **kwargs):
            _captured.update(kwargs)
            raise RuntimeError("identity check reached")

        monkeypatch.setattr(multi, "check_stats_identity", check_identity)
        with pytest.raises(RuntimeError, match="identity check reached"):
            data_loader.create_torch_data_loader(
                dataclasses.replace(data_config, norm_stats={}), recipe.model, 64, 256, num_workers=0
            )
        assert captured["audit_source_metadata"] is enabled
