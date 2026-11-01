"""Create and validate timestamp-preserving left-camera Engram video proxies."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq

from openpi.training import engram_dataset

# Version 2 requires explicit full-range planar output in the PyAV filter
# graph. Earlier NVDEC derivatives can have incorrect color despite matching
# all timing and metadata checks, so readers and builders must reject them.
PROXY_VERSION = "engram_left640_h264_gop4_v2"
PTS_TOLERANCE_S = 0.0001


def proxy_path(root: Path, original: Path, proxy_root: Path | None = None) -> Path:
    """Resolve an adjacent proxy, or the same relative path under a scratch root."""
    relative = original.relative_to(root)
    relative = relative.with_name(relative.stem + "_640.mp4")
    return original.with_name(relative.name) if proxy_root is None else proxy_root / root.parent.name / relative


def sidecar_path(proxy: Path) -> Path:
    return proxy.with_name(proxy.name + ".json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def packet_info(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Get presentation-ordered packet times, keyframes and video properties."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        meta = {
            "width": stream.width, "height": stream.height,
            "time_base_num": stream.time_base.numerator, "time_base_den": stream.time_base.denominator,
            "codec": stream.codec_context.name, "color_range": int(stream.codec_context.color_range),
            "declared_frames": stream.frames,
        }
        packets = [(float(packet.pts * packet.time_base), bool(packet.is_keyframe))
                   for packet in container.demux(stream) if packet.pts is not None]
    packets.sort(key=lambda item: item[0])
    if not packets or (meta["declared_frames"] and len(packets) != meta["declared_frames"]):
        raise ValueError(f"Missing or non-frame packets in {path}")
    timestamps = np.asarray([p[0] for p in packets], dtype=np.float64)
    keys = np.asarray([p[1] for p in packets], dtype=bool)
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"Invalid video presentation timestamps in {path}")
    return timestamps, keys, meta


def verify_proxy(original: Path, proxy: Path, episode_rows: list[tuple[int, np.ndarray, float]]) -> dict[str, Any]:
    """Verify every original/proxy timestamp, every annotation row and GOP."""
    orig_pts, _, orig = packet_info(original)
    proxy_pts, keyframes, dest = packet_info(proxy)
    if (orig["width"], orig["height"], orig["codec"], orig["color_range"]) != (3840, 1080, "h264", 2):
        raise ValueError(f"Unexpected source stereo video: {original}")
    if (dest["width"], dest["height"], dest["codec"], dest["color_range"]) != (640, 360, "h264", 2):
        raise ValueError(f"Proxy must be 640x360 full-range H.264: {proxy}")
    if len(orig_pts) != len(proxy_pts):
        raise ValueError("Source and proxy have different frame counts")
    error = float(np.max(np.abs(orig_pts - proxy_pts)))
    if error > 0.000001:
        raise ValueError(f"Proxy changed presentation timestamps by {error * 1e6:.2f} us")
    key_indices = np.flatnonzero(keyframes)
    if not len(key_indices) or key_indices[0] != 0 or np.any(np.diff(key_indices) > 4) or len(proxy_pts) - key_indices[-1] > 4:
        raise ValueError("Proxy keyframe spacing exceeds four frames")
    annotation_checks = []
    for episode_id, timestamps, video_start in episode_rows:
        expected = timestamps + video_start
        right = np.minimum(np.searchsorted(orig_pts, expected), len(orig_pts) - 1)
        left = np.maximum(right - 1, 0)
        nearest = np.minimum(np.abs(orig_pts[left] - expected), np.abs(orig_pts[right] - expected))
        max_error = float(np.max(nearest))
        if max_error > PTS_TOLERANCE_S:
            raise ValueError(f"Episode {episode_id}: source video and annotation differ by {max_error * 1e6:.2f} us")
        annotation_checks.append({"episode_index": episode_id, "rows": len(timestamps), "max_pts_error_us": max_error * 1e6})
    return {"source": orig, "proxy": dest, "frames": len(orig_pts), "max_pts_error_us": error * 1e6,
            "max_gop": int(np.max(np.diff(key_indices))) if len(key_indices) > 1 else len(proxy_pts),
            "episodes": annotation_checks}


def _encode_pyav(original: Path, output: Path, crf: int, decoder: str, cuda_device: str) -> None:
    """Decode with NVDEC (or CPU for qualification), then CPU-scale/encode.

    PyAV 14 copies CUDA frames to host memory before filters are invoked.
    Software fallback is forbidden in the NVDEC mode.
    """
    if decoder not in ("nvdec", "pyav_cpu"):
        raise ValueError(f"Unsupported PyAV decoder: {decoder}")
    hw = None
    if decoder == "nvdec":
        from av.codec.hwaccel import HWAccel
        hw = HWAccel("cuda", device=cuda_device, allow_software_fallback=False)
    with av.open(str(original), hwaccel=hw) as source, av.open(
        str(output), mode="w", format="mp4", options={"video_track_timescale": "1000000", "movflags": "+faststart"}
    ) as target:
        input_stream = source.streams.video[0]
        if input_stream.codec_context.color_range != 2:
            raise ValueError("Expected full-range source video")
        if input_stream.time_base.numerator != 1 or input_stream.time_base.denominator != 1_000_000:
            raise ValueError("NVDEC path currently requires the dataset's 1-microsecond input time base")
        if decoder == "pyav_cpu":
            input_stream.codec_context.thread_count = 1
        out = target.add_stream("libx264", rate=30, options={
            "crf": str(crf), "preset": "veryfast", "x264-params": "keyint=4:min-keyint=4:scenecut=0:bframes=0"
        })
        out.width, out.height, out.pix_fmt = 640, 360, "yuvj420p"
        out.time_base = input_stream.time_base
        out.codec_context.time_base = input_stream.time_base
        out.codec_context.color_range = 2
        out.codec_context.thread_count = 1
        graph = None
        decoded = 0
        for frame in source.decode(input_stream):
            if decoder == "nvdec" and not input_stream.codec_context.is_hwaccel:
                raise RuntimeError("NVDEC not active; refusing software fallback")
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Decoded frame is missing presentation time")
            # PyAV's CUDA transfer drops frame color metadata. The source is
            # verified full-range above; restore it before filtering.
            frame.color_range = 2
            if graph is None:
                graph = av.filter.Graph()
                graph_input = graph.add_buffer(template=frame)
                crop = graph.add("crop", "iw/2:ih:0:0")
                scale = graph.add("scale", "640:360:flags=area:in_range=full:out_range=full")
                # NVDEC returns NV12. Pin the planar full-range output inside
                # the graph rather than letting the encoder's implicit pixel
                # conversion reinterpret that frame as limited-range.
                pixel_format = graph.add("format", "pix_fmts=yuvj420p")
                sink = graph.add("buffersink")
                graph_input.link_to(crop)
                crop.link_to(scale)
                scale.link_to(pixel_format)
                pixel_format.link_to(sink)
                graph.configure()
            graph.push(frame)
            converted = graph.pull()
            if converted.format.name != "yuvj420p":
                raise ValueError(f"Expected planar full-range graph output, got {converted.format.name}")
            converted.color_range = 2
            converted.pts = frame.pts
            converted.time_base = frame.time_base
            for packet in out.encode(converted):
                target.mux(packet)
            decoded += 1
        if not decoded:
            raise ValueError("No video frames decoded")
        for packet in out.encode():
            target.mux(packet)


def create_proxy(original: Path, destination: Path, root: Path, episode_rows: list[tuple[int, np.ndarray, float]],
                 *, ffmpeg: str, crf: int = 18, scratch_root: Path | None = None,
                 decoder: str = "ffmpeg_cpu", cuda_device: str = "0") -> dict[str, Any]:
    """Encode and atomically publish a verified proxy; never overwrite a video."""
    original, destination, root = Path(original), Path(destination), Path(root)
    if destination.exists() and sidecar_path(destination).exists():
        for episode_id, timestamps, _ in episode_rows:
            validate_sidecar(root, original, destination, episode_id, len(timestamps))
        existing = json.loads(sidecar_path(destination).read_text())
        if (existing["crf"] != crf or existing.get("decoder", "ffmpeg_cpu") != decoder
                or existing["source_sha256"] != _sha256(original)
                or existing["proxy_sha256"] != _sha256(destination)):
            raise ValueError(f"Existing proxy differs from source or requested recipe: {destination}")
        return {**existing, "reused": True}
    if destination.exists() or sidecar_path(destination).exists():
        raise FileExistsError(f"Incomplete proxy or orphan sidecar requires inspection: {destination}")
    if not 0 <= crf <= 30:
        raise ValueError("CRF must be between 0 and 30")
    src_pts, _, source_meta = packet_info(original)
    if source_meta["time_base_num"] != 1 or source_meta["time_base_den"] > 2_000_000_000:
        raise ValueError("Cannot safely preserve this source video time base")
    scratch = Path(scratch_root or Path(tempfile.gettempdir()) / "engram-proxy-encoding")
    scratch.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=destination.stem + ".partial-", suffix=".mp4", dir=scratch)
    os.close(fd)
    temporary = Path(name)
    try:
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "1", "-copyts", "-i", str(original),
            "-map", "0:v:0", "-vf", "crop=iw/2:ih:0:0,scale=640:360:flags=area:in_range=full:out_range=full",
            "-fps_mode", "passthrough", "-enc_time_base", "demux", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", str(crf), "-g", "4", "-keyint_min", "4", "-sc_threshold", "0", "-bf", "0", "-threads", "1",
            "-pix_fmt", "yuv420p", "-color_range", "pc", "-video_track_timescale", str(source_meta["time_base_den"]),
            "-an", "-movflags", "+faststart", "-y", str(temporary),
        ]
        if decoder == "ffmpeg_cpu":
            subprocess.run(command, check=True)
        else:
            _encode_pyav(original, temporary, crf, decoder, cuda_device)
        check = verify_proxy(original, temporary, episode_rows)
        record = {"version": PROXY_VERSION, "source_relative": str(original.relative_to(root)),
                  "source_bytes": original.stat().st_size, "source_sha256": _sha256(original),
                  "proxy_bytes": temporary.stat().st_size, "proxy_sha256": _sha256(temporary),
                  "crf": crf, "decoder": decoder, "geometry": "left-half 16:9, area 640x360, no padding",
                  "timing": check, "source_frame_count": len(src_pts)}
        # The sidecar is the completion marker. A video without its sidecar is
        # rejected rather than consumed while another process is still writing.
        # Exclusive destination creation prevents accidental overwrites even
        # if two builders target the same name. An interrupted file lacks the
        # completion sidecar and is rejected by readers.
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as target, temporary.open("rb") as source:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        temp_sidecar = sidecar_path(destination).with_name(sidecar_path(destination).name + ".partial")
        temp_sidecar.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(temp_sidecar, sidecar_path(destination))
        return {**record, "reused": False}
    finally:
        temporary.unlink(missing_ok=True)


def validate_sidecar(root: Path, original: Path, proxy: Path, episode_id: int, row_count: int) -> dict[str, Any]:
    """Fail closed on missing, stale, incomplete or differently mapped proxies."""
    try:
        record = json.loads(sidecar_path(proxy).read_text())
        checks = record["timing"]
        valid_episode = any(item["episode_index"] == episode_id and item["rows"] == row_count
                            for item in checks["episodes"])
        if (record["version"] != PROXY_VERSION or record["source_relative"] != str(original.relative_to(root))
            or record["source_bytes"] != original.stat().st_size or record["proxy_bytes"] != proxy.stat().st_size
            or len(record["source_sha256"]) != 64 or len(record["proxy_sha256"]) != 64
            or checks["proxy"]["width"] != 640 or checks["proxy"]["height"] != 360
            or checks["proxy"]["color_range"] != 2 or checks["max_gop"] > 4 or checks["max_pts_error_us"] > 1
            or not valid_episode):
            raise ValueError("Proxy sidecar does not match source, format, timing or episode")
        if _sha256(proxy) != record["proxy_sha256"]:
            raise ValueError(f"Proxy content checksum differs from sidecar: {proxy}")
        return record
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Missing or invalid proxy sidecar for {proxy}") from exc


def episodes_for_video(root: Path, original: Path, *, info: dict | None = None, rows: list[dict] | None = None) -> list[tuple[int, np.ndarray, float]]:
    """Read all episode timestamps mapped to one source video."""
    info = info or json.loads((root / "meta/info.json").read_text())
    if rows is None:
        rows = [row for path in sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
                for row in pq.read_table(path).to_pylist()]
    prefix = f"videos/{engram_dataset.CAMERA_KEY}"
    result = []
    for row in rows:
        path = root / info["video_path"].format(video_key=engram_dataset.CAMERA_KEY,
                    chunk_index=int(row[f"{prefix}/chunk_index"]), file_index=int(row[f"{prefix}/file_index"]))
        if path != original:
            continue
        episode_id = int(row["episode_index"])
        shard = root / info["data_path"].format(chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"]))
        table = pq.read_table(shard, columns=["timestamp"], filters=[("episode_index", "=", episode_id)])
        timestamps = np.asarray(table.column("timestamp"), dtype=np.float64)
        if len(timestamps) != int(row["length"]) or not len(timestamps):
            raise ValueError(f"Episode {episode_id} timestamp rows do not match metadata")
        result.append((episode_id, timestamps, float(row[f"{prefix}/from_timestamp"])))
    if not result:
        raise ValueError(f"No episode metadata maps to {original}")
    return result
