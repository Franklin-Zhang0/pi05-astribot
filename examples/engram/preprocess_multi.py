"""Build selected head/wrist derivatives for a pinned mixed Engram manifest."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import json
import multiprocessing
from pathlib import Path
import sys
import traceback

import numpy as np
import pyarrow.parquet as pq

from openpi.training import engram_multi_video as video
from openpi.training.engram_multi_dataset import CAMERAS
from openpi.training.engram_multi_dataset import read_manifest


def jobs_for_manifest(path: Path, *, split: str = "all", output_root: Path | None = None):
    spec = read_manifest(path)
    jobs = {}
    for source in spec["sources"]:
        selected = set(source["selected_episode_indices"])
        if not selected:
            continue
        root = Path(source["root"])
        info = json.loads((root / "meta/info.json").read_text())
        plan = json.loads((root.parent / "export_plan.json").read_text())
        packages = {r["annotation_id"]: r["data_package_id"] for r in plan}
        heldout = set(spec.get("heldout_packages", source.get("validation_packages", [])))
        rows = [
            row
            for p in sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
            for row in pq.read_table(p).to_pylist()
        ]
        for row in rows:
            episode = int(row["episode_index"])
            if episode not in selected:
                continue
            if row.get("completeness") != "FULL" or row.get("annotation_result") != "SUCCESS":
                raise ValueError("Preprocessing selection must contain full successful episodes")
            if split != "all" and (packages[row["annotation_id"]] in heldout) != (split == "val"):
                continue
            shard = root / info["data_path"].format(
                chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
            )
            timestamps = np.asarray(
                pq.read_table(shard, columns=["timestamp"], filters=[("episode_index", "=", episode)])["timestamp"],
                np.float64,
            )
            if len(timestamps) != int(row["length"]) or not len(timestamps):
                raise ValueError("Episode timestamp rows differ from metadata")
            for role, name in CAMERAS[source["robot_type"]].items():
                key = f"observation.images.{name}"
                if key not in info["features"]:
                    continue
                prefix = f"videos/{key}"
                original = root / info["video_path"].format(
                    video_key=key,
                    chunk_index=int(row[f"{prefix}/chunk_index"]),
                    file_index=int(row[f"{prefix}/file_index"]),
                )
                target = video.proxy_path(root, original, output_root)
                identity = str(target)
                if identity not in jobs:
                    jobs[identity] = {
                        "root": str(root),
                        "original": str(original),
                        "destination": str(target),
                        "robot_type": source["robot_type"],
                        "role": role,
                        "episodes": [],
                    }
                jobs[identity]["episodes"].append((episode, timestamps, float(row[f"{prefix}/from_timestamp"])))
    return list(jobs.values())


def process(job, *, decoder, device, scratch, crf, verify_existing_source):
    record = video.create_proxy(
        Path(job["original"]),
        Path(job["destination"]),
        Path(job["root"]),
        job["robot_type"],
        job["role"],
        job["episodes"],
        decoder=decoder,
        cuda_device=device,
        scratch_root=Path(scratch),
        crf=crf,
        verify_existing_source=verify_existing_source,
    )
    return {
        "destination": job["destination"],
        "robot_type": job["robot_type"],
        "role": job["role"],
        "reused": record["reused"],
        "frames": record["timing"]["frames"],
        "proxy_sha256": record["proxy_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--adjacent", action="store_true")
    destination.add_argument("--output-root", type=Path)
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--decoder", choices=("pyav_cpu", "nvdec"), default="pyav_cpu")
    parser.add_argument("--cuda-devices", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scratch-root", default="/tmp/engram-multi-encoding")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shortest-first", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--verify-existing-source", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    jobs = jobs_for_manifest(args.manifest, split=args.split, output_root=args.output_root)
    if args.shortest_first:
        jobs.sort(key=lambda job: max(len(e[1]) for e in job["episodes"]))
    if args.limit is not None:
        jobs = jobs[: args.limit]
    print(
        json.dumps(
            {
                "event": "PROXY_PLAN",
                "videos": len(jobs),
                "split": args.split,
                "camera_roles": dict(Counter(j["role"] for j in jobs)),
                "decoder": args.decoder,
            }
        ),
        flush=True,
    )
    if args.plan_only:
        return
    devices = [s.strip() for s in args.cuda_devices.split(",") if s.strip()]
    if not devices:
        parser.error("At least one CUDA device must be selected")
    receipt = args.receipt
    if receipt is not None:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        stream = receipt.open("x")
    else:
        stream = None
    try:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = []
            try:
                for i, job in enumerate(jobs):
                    futures.append(
                        pool.submit(
                            process,
                            job,
                            decoder=args.decoder,
                            device=devices[i % len(devices)],
                            scratch=args.scratch_root,
                            crf=args.crf,
                            verify_existing_source=args.verify_existing_source,
                        )
                    )
                for i, future in enumerate(as_completed(futures), 1):
                    result = future.result()
                    result.update(event="PROXY_COMPLETE", completed=i, total=len(jobs))
                    line = json.dumps(result)
                    print(line, flush=True)
                    if stream:
                        stream.write(line + "\n")
                        stream.flush()
            except BaseException:
                # Report before __exit__ waits for active encoders, and prevent
                # the remaining submission queue from continuing after failure.
                traceback.print_exc()
                sys.stderr.flush()
                for future in futures:
                    future.cancel()
                raise
    finally:
        if stream:
            stream.close()


if __name__ == "__main__":
    main()
