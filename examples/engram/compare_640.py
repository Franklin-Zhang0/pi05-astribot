"""Compare model-input pixels from matching original, CPU and NVDEC proxies."""

import argparse
import json
import math

import av
import numpy as np
from openpi_client.image_tools import resize_with_pad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("original")
    parser.add_argument("cpu_proxy")
    parser.add_argument("nvdec_proxy")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    containers = [av.open(path) for path in (args.original, args.cpu_proxy, args.nvdec_proxy)]
    streams = [container.streams.video[0] for container in containers]
    frame_count = streams[0].frames
    if frame_count < args.samples or not all(stream.frames == frame_count for stream in streams):
        raise ValueError("Expected equal frame counts and enough frames")
    indices = set(np.linspace(0, frame_count - 1, args.samples, dtype=int))
    rows = []
    for i, frames in enumerate(zip(*(container.decode(stream) for container, stream in zip(containers, streams, strict=True)), strict=True)):
        pts = [float(frame.pts * frame.time_base) for frame in frames]
        if max(pts) - min(pts) > 0.000001:
            raise ValueError(f"PTS differs at frame {i}: {pts}")
        if i not in indices:
            continue
        images = [frame.to_ndarray(format="rgb24") for frame in frames]
        images[0] = images[0][:, : images[0].shape[1] // 2].copy()
        images = [np.asarray(resize_with_pad(image, 224, 224), dtype=np.float32) for image in images]
        entry = {"frame": i, "timestamp": pts[0]}
        for label, first, second in (("original_cpu", 0, 1), ("original_nvdec", 0, 2), ("cpu_nvdec", 1, 2)):
            diff = np.abs(images[first] - images[second])
            mse = float(np.mean(diff * diff))
            entry[label] = {"mae": float(diff.mean()), "p99": float(np.quantile(diff, 0.99)),
                            "max": float(diff.max()), "psnr_db": 99.0 if mse == 0 else 10 * math.log10(255**2 / mse)}
        rows.append(entry)
    if len(rows) != len(indices):
        raise ValueError("Missing sampled frames")
    for container in containers:
        container.close()
    print(json.dumps({"frames": frame_count, "samples": rows,
                      "mean_cpu_nvdec_mae": float(np.mean([r["cpu_nvdec"]["mae"] for r in rows])),
                      "max_cpu_nvdec_p99": float(max(r["cpu_nvdec"]["p99"] for r in rows))}, indent=2))


if __name__ == "__main__":
    main()
