"""Video derivatives must preserve VFR alignment and short GOP boundaries."""

from pathlib import Path

import numpy as np
import pytest

from openpi.training import checkpoints
from openpi.training import config
from openpi.training import engram_video_proxy as proxy


def test_vfr_proxy_checks_every_timestamp_and_gop(monkeypatch):
    source = Path("source.mp4")
    target = Path("proxy.mp4")
    timestamps = np.array([0, 0.013744, 0.071693, 0.092065, 0.122151, 0.170468, 0.207057, 0.252241])
    keys = np.array([True, False, False, False, True, False, False, False])
    source_info = {"width": 3840, "height": 1080, "codec": "h264", "color_range": 2}
    target_info = {"width": 640, "height": 360, "codec": "h264", "color_range": 2}
    times = {source: timestamps.copy(), target: timestamps.copy()}
    monkeypatch.setattr(proxy, "packet_info", lambda path: (times[path], keys.copy(), source_info if path == source else target_info))
    result = proxy.verify_proxy(source, target, [(7, timestamps.copy(), 0.0)])
    assert result["frames"] == 8
    assert result["max_gop"] == 4
    assert result["max_pts_error_us"] == 0
    times[target][3] += 2e-6
    with pytest.raises(ValueError, match="presentation timestamps"):
        proxy.verify_proxy(source, target, [(7, timestamps, 0.0)])
    times[target][3] -= 2e-6
    with pytest.raises(ValueError, match="annotation"):
        proxy.verify_proxy(source, target, [(7, timestamps + 0.001, 0.0)])


def test_checkpoint_identity_changes_between_video_modes():
    original = config.DataConfig()
    converted = config.DataConfig(engram_video_mode="proxy_640", engram_video_identity="a" * 64)
    assert checkpoints._engram_video_contract(original) != checkpoints._engram_video_contract(converted)  # noqa: SLF001
