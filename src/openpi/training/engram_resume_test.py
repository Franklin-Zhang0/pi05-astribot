import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openpi.shared import normalize
from openpi.training import checkpoints


def test_resume_requires_same_stats_and_manifest(tmp_path: Path, monkeypatch):
    stats = {"state": normalize.NormStats(mean=np.zeros(22), std=np.ones(22))}
    manifest = {"contract": "test", "threshold": 0.01}
    saved = tmp_path / "1/assets/engram/train"
    normalize.save(saved, stats)
    (saved / "engram_stats_manifest.json").write_text(json.dumps(manifest))
    current_manifest = tmp_path / "current.json"
    current_manifest.write_text(json.dumps(manifest))
    data = SimpleNamespace(
        engram_stats_manifest=str(current_manifest), asset_id="engram/train", norm_stats=stats,
        engram_video_mode="original", engram_proxy_subset_only=False, engram_video_identity=None,
    )
    loader = SimpleNamespace(data_config=lambda: data)
    manager = SimpleNamespace(
        directory=tmp_path,
        latest_step=lambda: 1,
        restore=lambda *args, **kwargs: {"train_state": "state", "params": {"params": "params"}},
    )
    monkeypatch.setattr(checkpoints, "_split_params", lambda state: (state, None))
    monkeypatch.setattr(checkpoints, "_merge_params", lambda state, params: (state, params))
    assert checkpoints.restore_state(manager, "initial", loader) == ("state", {"params": "params"})
    data.norm_stats = {"state": normalize.NormStats(mean=np.ones(22), std=np.ones(22))}
    with pytest.raises(ValueError, match="changed normalization"):
        checkpoints.restore_state(manager, "initial", loader)
    data.norm_stats = stats
    current_manifest.write_text(json.dumps({"contract": "changed"}))
    with pytest.raises(ValueError, match="changed normalization"):
        checkpoints.restore_state(manager, "initial", loader)
    current_manifest.write_text(json.dumps(manifest))
    data.engram_video_mode = "proxy_640"
    data.engram_video_identity = "a" * 64
    with pytest.raises(ValueError, match="Legacy checkpoint has no proxy identity"):
        checkpoints.restore_state(manager, "initial", loader)
    contract_path = saved / "engram_video_contract.json"
    contract_path.write_text(json.dumps(checkpoints._engram_video_contract(data)))  # noqa: SLF001
    assert checkpoints.restore_state(manager, "initial", loader) == ("state", {"params": "params"})
    data.engram_video_identity = "b" * 64
    with pytest.raises(ValueError, match="changed source-video/proxy identity"):
        checkpoints.restore_state(manager, "initial", loader)
