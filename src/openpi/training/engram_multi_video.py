"""Timestamp-preserving head and wrist proxies for mixed Engram robots."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import av
import numpy as np

from openpi.policies.engram_multi_policy import camera_size
from openpi.training.engram_video_proxy import packet_info

PROXY_VERSION = "engram_multi_camera_h264_gop4_v1"
LEGACY_HEAD_VERSION = "engram_left640_h264_gop4_v2"


def sha256(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def proxy_path(root: str | Path, original: str | Path, proxy_root: str | Path | None = None) -> Path:
    root, original = Path(root), Path(original)
    relative = original.relative_to(root)
    relative = relative.with_name(relative.stem + "_640.mp4")
    return original.with_name(relative.name) if proxy_root is None else Path(proxy_root) / root.parent.name / relative


def sidecar_path(proxy: str | Path) -> Path:
    path = Path(proxy)
    return path.with_name(path.name + ".json")


def verify_proxy(
    original: Path, proxy: Path, robot_type: str, role: str, episode_rows: list[tuple[int, np.ndarray, float]]
) -> dict:
    orig_pts, _, source = packet_info(original)
    proxy_pts, keys, result = packet_info(proxy)
    width, height = camera_size(robot_type, role)
    source_width = source["width"] // 2 if robot_type == "dex_w1" and role == "head" else source["width"]
    if abs(source_width / source["height"] - width / height) > 0.005:
        raise ValueError("Source camera aspect ratio differs from its declared role")
    if (result["width"], result["height"], result["codec"], result["color_range"]) != (width, height, "h264", 2):
        raise ValueError("Proxy dimensions, codec, or color range differ from the camera contract")
    if len(orig_pts) != len(proxy_pts):
        raise ValueError("Proxy changed frame count")
    error = float(np.max(np.abs(orig_pts - proxy_pts)))
    if error > 0.000001:
        raise ValueError("Proxy changed original presentation timestamps")
    key_indices = np.flatnonzero(keys)
    if (
        not len(key_indices)
        or key_indices[0] != 0
        or np.any(np.diff(key_indices) > 4)
        or len(proxy_pts) - key_indices[-1] > 4
    ):
        raise ValueError("Proxy keyframe gap exceeds four frames")
    checks = []
    for episode, timestamps, start in episode_rows:
        expected = np.asarray(timestamps) + start
        right = np.minimum(np.searchsorted(orig_pts, expected), len(orig_pts) - 1)
        left = np.maximum(right - 1, 0)
        nearest = np.minimum(np.abs(orig_pts[left] - expected), np.abs(orig_pts[right] - expected))
        maximum = float(nearest.max())
        if maximum > 0.0001:
            raise ValueError(f"Camera timing differs from episode {episode} annotation by {maximum}s")
        checks.append({"episode_index": int(episode), "rows": len(expected), "max_pts_error_us": maximum * 1e6})
    return {
        "source": source,
        "proxy": result,
        "frames": len(orig_pts),
        "max_pts_error_us": error * 1e6,
        "max_gop": int(np.max(np.diff(key_indices))) if len(key_indices) > 1 else len(proxy_pts),
        "episodes": checks,
    }


def validate_proxy(
    root: Path,
    original: Path,
    proxy: Path,
    robot_type: str,
    role: str,
    episode_id: int,
    row_count: int,
    *,
    verify_hash: bool = True,
) -> dict:
    record = json.loads(sidecar_path(proxy).read_text())
    width, height = camera_size(robot_type, role)
    legacy = robot_type == "dex_w1" and role == "head" and record.get("version") == LEGACY_HEAD_VERSION
    if not legacy and (
        record.get("version") != PROXY_VERSION
        or record.get("robot_type") != robot_type
        or record.get("camera_role") != role
    ):
        raise ValueError("Proxy recipe, robot, or camera role mismatch")
    timing = record["timing"]
    if (
        record["source_relative"] != str(original.relative_to(root))
        or record["source_bytes"] != original.stat().st_size
        or record["proxy_bytes"] != proxy.stat().st_size
        or timing["proxy"]["width"] != width
        or timing["proxy"]["height"] != height
        or timing["proxy"]["color_range"] != 2
        or timing["max_gop"] > 4
        or timing["max_pts_error_us"] > 1
        or not any(e["episode_index"] == episode_id and e["rows"] == row_count for e in timing["episodes"])
    ):
        raise ValueError("Proxy sidecar does not match source, geometry, timing, or annotation")
    if verify_hash and sha256(proxy) != record["proxy_sha256"]:
        raise ValueError("Proxy payload checksum differs from sidecar")
    return record


def _encode(original: Path, output: Path, robot_type: str, role: str, *, crf: int, decoder: str, cuda_device: str):
    if decoder not in ("pyav_cpu", "nvdec"):
        raise ValueError("Decoder must be pyav_cpu or nvdec")
    hw = None
    if decoder == "nvdec":
        from av.codec.hwaccel import HWAccel

        hw = HWAccel("cuda", device=cuda_device, allow_software_fallback=False)
    width, height = camera_size(robot_type, role)
    with (
        av.open(str(original), hwaccel=hw) as source,
        av.open(
            str(output), "w", format="mp4", options={"video_track_timescale": "1000000", "movflags": "+faststart"}
        ) as target,
    ):
        stream = source.streams.video[0]
        stream.codec_context.thread_count = 1
        color_range = int(stream.codec_context.color_range)
        if color_range == 0 and stream.codec_context.format.name in {"yuv420p", "nv12"}:
            # H.264 YUV without a full_range VUI uses studio/limited range.
            color_range = 1
        if color_range not in (1, 2):
            raise ValueError("Cannot establish source camera color range")
        out = target.add_stream(
            "libx264",
            rate=30,
            options={
                "crf": str(crf),
                "preset": "veryfast",
                "x264-params": "keyint=4:min-keyint=4:scenecut=0:bframes=0",
            },
        )
        out.width, out.height, out.pix_fmt = width, height, "yuvj420p"
        out.time_base = stream.time_base
        out.codec_context.time_base = stream.time_base
        out.codec_context.color_range = 2
        out.codec_context.thread_count = 1
        graph = None
        decoded = 0
        for frame in source.decode(stream):
            if decoder == "nvdec" and not stream.codec_context.is_hwaccel:
                raise RuntimeError("NVDEC was requested but is not active")
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Decoded frame has no physical timestamp")
            frame.color_range = color_range
            if graph is None:
                graph = av.filter.Graph()
                first = graph.add_buffer(template=frame)
                previous = first
                if robot_type == "dex_w1" and role == "head":
                    crop = graph.add("crop", "iw/2:ih:0:0")
                    previous.link_to(crop)
                    previous = crop
                source_range = "full" if color_range == 2 else "limited"
                scale = graph.add("scale", f"{width}:{height}:flags=area:in_range={source_range}:out_range=full")
                pixel = graph.add("format", "pix_fmts=yuvj420p")
                sink = graph.add("buffersink")
                previous.link_to(scale)
                scale.link_to(pixel)
                pixel.link_to(sink)
                graph.configure()
            graph.push(frame)
            converted = graph.pull()
            converted.pts, converted.time_base, converted.color_range = frame.pts, frame.time_base, 2
            for packet in out.encode(converted):
                target.mux(packet)
            decoded += 1
        if not decoded:
            raise ValueError("No source video frames decoded")
        for packet in out.encode():
            target.mux(packet)


def create_proxy(
    original: Path,
    destination: Path,
    root: Path,
    robot_type: str,
    role: str,
    episode_rows: list[tuple[int, np.ndarray, float]],
    *,
    crf: int = 18,
    decoder: str = "pyav_cpu",
    cuda_device: str = "0",
    scratch_root: Path | None = None,
    ffmpeg: str | None = None,
    verify_existing_source: bool = False,
) -> dict:
    """Publish a verified adjacent derivative without replacing originals."""
    original, destination, root = Path(original), Path(destination), Path(root)
    if not 0 <= crf <= 30:
        raise ValueError("CRF must be between 0 and 30")
    if destination.exists() and sidecar_path(destination).exists():
        record = None
        for episode, timestamps, _ in episode_rows:
            record = validate_proxy(root, original, destination, robot_type, role, episode, len(timestamps))
        if record is None or (verify_existing_source and sha256(original) != record["source_sha256"]):
            raise ValueError("Existing derivative source checksum changed")
        return {
            **record,
            "reused": True,
            "reused_source_validation": "sha256" if verify_existing_source else "size_and_saved_provenance",
        }
    if destination.exists() or sidecar_path(destination).exists():
        raise FileExistsError("Orphan proxy/sidecar must be inspected before rebuilding")
    scratch = Path(scratch_root or Path(tempfile.gettempdir()) / "engram-multi-encoding")
    scratch.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="engram-camera-", suffix=".mp4", dir=scratch)
    os.close(fd)
    temporary = Path(name)
    try:
        _encode(original, temporary, robot_type, role, crf=crf, decoder=decoder, cuda_device=cuda_device)
        timing = verify_proxy(original, temporary, robot_type, role, episode_rows)
        record = {
            "version": PROXY_VERSION,
            "robot_type": robot_type,
            "camera_role": role,
            "source_relative": str(original.relative_to(root)),
            "source_bytes": original.stat().st_size,
            "source_sha256": sha256(original),
            "proxy_bytes": temporary.stat().st_size,
            "proxy_sha256": sha256(temporary),
            "crf": crf,
            "decoder": decoder,
            "geometry": {
                "width": camera_size(robot_type, role)[0],
                "height": camera_size(robot_type, role)[1],
                "crop_left_stereo": robot_type == "dex_w1" and role == "head",
            },
            "timing": timing,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as out, temporary.open("rb") as inp:
            shutil.copyfileobj(inp, out, 1024 * 1024)
        sidecar = sidecar_path(destination)
        pending = sidecar.with_name(sidecar.name + ".partial")
        pending.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(pending, sidecar)
        return {**record, "reused": False}
    finally:
        temporary.unlink(missing_ok=True)
