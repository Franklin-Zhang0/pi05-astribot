"""Run two isolated captures on an already allocated, idle CUDA GPU.

This tool never allocates resources or controls ACP lifecycle. It refuses to
share the selected GPU with existing compute processes and holds a local lock.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-seconds", type=int, default=3300)
    args = parser.parse_args()
    if args.gpu < 0 or args.max_seconds < 1:
        raise ValueError("GPU index must be nonnegative and time budget positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    script_dir = Path(__file__).resolve().parent
    package_src = script_dir.parent / "src"
    env = dict(os.environ)
    env.update(
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    env["PYTHONPATH"] = os.pathsep.join([str(package_src), str(script_dir), env.get("PYTHONPATH", "")])
    deadline = time.monotonic() + args.max_seconds
    with Path(f"/tmp/prism-engram-qualification-gpu-{args.gpu}.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for label, factory in (
            ("standalone", "prism_engram.bundle:load_bundle"),
            ("reference", "original_graph:load_bundle"),
        ):
            active = subprocess.check_output(
                ["nvidia-smi", "-i", str(args.gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                text=True,
            ).strip()
            if active:
                raise RuntimeError("Selected GPU has active compute processes; no capture launched")
            capture_env = dict(env)
            if label == "reference":
                capture_env.update(
                    PRISM_PARITY_REFERENCE_SOURCE=str(args.reference_source.resolve()),
                    PRISM_PARITY_REFERENCE_CHECKPOINT=str(args.reference_checkpoint.resolve()),
                )
            command = [
                sys.executable,
                str(script_dir / "capture.py"),
                "--bundle",
                str(args.bundle.resolve()),
                "--fixture",
                str(args.fixture.resolve()),
                "--output",
                str((args.output_dir / f"{label}.json").resolve()),
                "--factory",
                factory,
                "--device",
                f"cuda:{args.gpu}",
            ]
            with (args.output_dir / f"{label}.log").open("w") as stream:
                subprocess.run(
                    command,
                    env=capture_env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=max(1, deadline - time.monotonic()),
                    check=True,
                )
        from compare import compare_captures

        result = compare_captures(
            args.output_dir / "reference.json", args.output_dir / "standalone.json", require_gpu=True
        )
        (args.output_dir / "parity.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "qualified": result["qualified"],
                    "scope": result["qualification_scope"],
                    "receipt": str(args.output_dir / "parity.json"),
                }
            )
        )
        raise SystemExit(0 if result["qualified"] else 1)


if __name__ == "__main__":
    run()
