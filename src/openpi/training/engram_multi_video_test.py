"""Verify physical timestamp, geometry, color and derivative integrity contracts."""

from fractions import Fraction

import av
import numpy as np
import pytest

from openpi.policies.engram_multi_policy import camera_size
from openpi.training import engram_multi_video as video


def make_video(path, width, height, color_range):
    pts = np.array([0, 33333, 66666, 100001, 133333, 166667, 200000, 233333, 266667, 300000])
    with av.open(str(path), "w", options={"video_track_timescale": "1000000"}) as container:
        stream = container.add_stream("libx264", rate=30, options={"crf": "12", "preset": "veryfast"})
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuvj420p" if color_range == 2 else "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000000)
        stream.codec_context.color_range = color_range
        for index, value in enumerate(pts):
            pixels = np.zeros((height, width, 3), np.uint8)
            pixels[:, : width // 2, 0] = 100 + index
            pixels[:, width // 2 :, 2] = 200
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = int(value), Fraction(1, 1000000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return pts / 1000000


@pytest.mark.parametrize(
    ("robot", "role", "width", "height", "color_range"),
    [
        ("dex_w1", "head", 640, 180, 2),
        ("dex_w1", "left_wrist", 320, 240, 1),
        ("galbot", "head", 320, 240, 1),
        ("galbot", "right_wrist", 320, 180, 2),
    ],
)
def test_proxy_preserves_timing_and_geometry(tmp_path, robot, role, width, height, color_range):
    original = tmp_path / "original.mp4"
    timestamps = make_video(original, width, height, color_range)
    destination = tmp_path / "original_640.mp4"
    rows = [(0, timestamps, 0.0)]
    record = video.create_proxy(original, destination, tmp_path, robot, role, rows)
    assert record["timing"]["max_pts_error_us"] <= 1
    assert record["timing"]["max_gop"] <= 4
    size = camera_size(robot, role)
    assert (record["timing"]["proxy"]["width"], record["timing"]["proxy"]["height"]) == size
    assert record["timing"]["proxy"]["color_range"] == 2
    if role == "head" and robot == "dex_w1":
        with av.open(str(destination)) as container:
            image = next(container.decode(video=0)).to_ndarray(format="rgb24")
        assert np.abs(image[..., 0].mean() - 100) < 6
        assert image[..., 2].mean() < 6
    assert video.create_proxy(original, destination, tmp_path, robot, role, rows)["reused"]
    with destination.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="sidecar"):
        video.validate_proxy(tmp_path, original, destination, robot, role, 0, len(timestamps))


def test_dataset_cached_decoder_seeks_repeatedly_without_stale_frames(tmp_path):
    from collections import OrderedDict
    from types import SimpleNamespace

    from openpi.training.engram_multi_dataset import EngramMultiEmbodimentDataset

    original = tmp_path / "original.mp4"
    timestamps = make_video(original, 640, 180, 2)
    dataset = object.__new__(EngramMultiEmbodimentDataset)
    dataset.decode_images = True
    dataset.video_cache_size = 2
    dataset.video_tolerance_s = 0.0001
    dataset._video_cache = OrderedDict()  # noqa: SLF001
    episode = SimpleNamespace(embodiment_id=0, episode_index=0, videos={"head": (str(original), 0.0, True)})
    try:
        values = []
        for index in (0, 5, 1, 9, 0):
            image = dataset._read_image(episode, "head", float(timestamps[index]))  # noqa: SLF001
            values.append(float(image[..., 0].mean()))
            assert abs(values[-1] - (100 + index)) < 3
            assert image[..., 2].mean() < 6
        assert values[0] == values[-1]
        assert values[3] > values[2]
        assert len(dataset._video_cache) == 1  # noqa: SLF001
    finally:
        dataset.close()
