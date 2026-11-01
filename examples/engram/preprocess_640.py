"""Build VFR-preserving 640x360 left-view proxies, never modifying source videos."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import json
from pathlib import Path
import shutil
import threading
import time

import pyarrow.parquet as pq

from openpi.training import engram_dataset
from openpi.training import engram_video_proxy


def _run_jobs(jobs, workers, run):
    """Cancel queued conversions on the first failure; let active ones finish."""
    stop = threading.Event()

    def guarded(job):
        if stop.is_set():
            return None
        try:
            return run(job)
        except BaseException:
            stop.set()
            raise

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(guarded, job) for job in jobs]
        try:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    print(json.dumps(result), flush=True)
        except BaseException:
            cancelled = sum(future.cancel() for future in futures)
            print(f"Preprocessing failed; cancelled {cancelled} queued conversions", flush=True)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    destinations = parser.add_mutually_exclusive_group(required=True)
    destinations.add_argument("--output-root", type=Path, help="Scratch root mirroring source-relative video paths")
    destinations.add_argument("--adjacent", action="store_true", help="Write *_640.mp4 next to originals")
    parser.add_argument("--split", choices=("train", "val", "all"), default="train")
    parser.add_argument("--limit-per-source", type=int, default=None)
    parser.add_argument("--shortest-first", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--ffmpeg", default=None)
    parser.add_argument("--scratch-root", type=Path, default=None, help="Local filesystem for FFmpeg's seekable output")
    parser.add_argument("--decoder", choices=("ffmpeg_cpu", "nvdec", "pyav_cpu"), default="ffmpeg_cpu")
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--cuda-devices", default=None, help="Comma-separated devices; jobs are assigned round-robin")
    parser.add_argument("--workers-per-gpu", type=int, default=1,
                        help="Maximum NVDEC conversions per GPU (default: 1, maximum: 8)")
    parser.add_argument("--source-contains", default=None, help="Only process source paths containing this string")
    parser.add_argument("--episode-index", type=int, default=None, help="Only process this episode index")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit_per_source is not None and args.limit_per_source < 1):
        parser.error("workers and limit-per-source must be positive")
    if not 1 <= args.workers_per_gpu <= 8:
        parser.error("workers-per-gpu must be between 1 and 8")
    devices = [item.strip() for item in (args.cuda_devices or args.cuda_device).split(",") if item.strip()]
    if not devices or len(set(devices)) != len(devices):
        parser.error("CUDA devices must be nonempty and unique")
    device_locks = {device: threading.BoundedSemaphore(args.workers_per_gpu) for device in devices}
    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    spec = engram_dataset._read_source_manifest(args.manifest)  # noqa: SLF001
    jobs = []
    for source in spec["sources"]:
        if args.source_contains is not None and args.source_contains not in source["root"]:
            continue
        root = Path(source["root"])
        info = json.loads((root / "meta/info.json").read_text())
        rows = [row for path in sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
                for row in pq.read_table(path).to_pylist()]
        plan = json.loads((root.parent / "export_plan.json").read_text())
        package = {item["annotation_id"]: item["data_package_id"] for item in plan}
        heldout = set(source["validation_packages"])
        selected = [row for row in rows if row.get("completeness") == "FULL" and row.get("annotation_result") == "SUCCESS"
                    and (args.split == "all" or ((package[row["annotation_id"]] in heldout) == (args.split == "val")))
                    and (args.episode_index is None or int(row["episode_index"]) == args.episode_index)]
        selected.sort(key=lambda r: (int(r["length"]), int(r["episode_index"])) if args.shortest_first else int(r["episode_index"]))
        if args.limit_per_source is not None:
            selected = selected[:args.limit_per_source]
        paths = set()
        prefix = f"videos/{engram_dataset.CAMERA_KEY}"
        for row in selected:
            original = root / info["video_path"].format(video_key=engram_dataset.CAMERA_KEY,
                chunk_index=int(row[f"{prefix}/chunk_index"]), file_index=int(row[f"{prefix}/file_index"]))
            if original in paths:
                continue
            paths.add(original)
            destination = engram_video_proxy.proxy_path(root, original, args.output_root)
            jobs.append((root, info, rows, original, destination, devices[len(jobs) % len(devices)]))
    if not jobs:
        raise ValueError("No videos selected")
    print(f"Selected {len(jobs)} {args.split} videos; workers={args.workers}; "
          f"workers_per_gpu={args.workers_per_gpu}; ffmpeg={ffmpeg}", flush=True)

    def run(job):
        root, info, rows, original, destination, device = job
        start = time.monotonic()
        episodes = engram_video_proxy.episodes_for_video(root, original, info=info, rows=rows)
        if args.decoder == "nvdec":
            with device_locks[device]:
                record = engram_video_proxy.create_proxy(original, destination, root, episodes, ffmpeg=ffmpeg,
                    crf=args.crf, scratch_root=args.scratch_root, decoder=args.decoder, cuda_device=device)
        else:
            record = engram_video_proxy.create_proxy(original, destination, root, episodes, ffmpeg=ffmpeg,
                crf=args.crf, scratch_root=args.scratch_root, decoder=args.decoder, cuda_device=device)
        return {"source": str(original), "proxy": str(destination), "frames": record["source_frame_count"],
                "seconds": time.monotonic() - start, "bytes": record["proxy_bytes"],
                "max_pts_error_us": record["timing"]["max_pts_error_us"], "max_gop": record["timing"]["max_gop"],
                "reused": record["reused"], "decoder": args.decoder, "cuda_device": device if args.decoder == "nvdec" else None}

    _run_jobs(jobs, args.workers, run)


if __name__ == "__main__":
    main()
