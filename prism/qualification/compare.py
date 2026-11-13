"""Compare isolated runtime captures without treating CPU checks as GPU proof.

Each capture is a JSON receipt and an adjacent NPZ containing numeric model
outputs.  The reference and standalone runtime must be captured in separate
processes because both expose the ``prism_vla`` Python namespace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare_captures(
    reference: Path, candidate: Path, *, atol: float = 1e-5, rtol: float = 1e-5, require_gpu: bool = False
) -> dict:
    """Require identical cases, weights and tensor inventories, then compare values."""
    if not (np.isfinite(atol) and np.isfinite(rtol)) or min(atol, rtol) < 0:
        raise ValueError("Comparison tolerances must be finite and nonnegative")
    receipts = [json.loads(path.read_text()) for path in (reference, candidate)]
    for record in receipts:
        if record.get("schema") != "prism_engram_runtime_capture_v1":
            raise ValueError("Unsupported capture schema")
        if not record.get("capture_complete"):
            raise ValueError("Capture did not complete")
    for key in ("fixture_sha256", "weights_sha256", "case_inventory", "flow_noise_seed"):
        if receipts[0].get(key) != receipts[1].get(key):
            raise ValueError(f"Capture identity mismatch: {key}")
        if receipts[0].get(key) is None:
            raise ValueError(f"Missing capture identity: {key}")
    gpu = all(
        str(record.get("device", "")).startswith("cuda") and record.get("cuda_device_name") for record in receipts
    )
    if require_gpu and not gpu:
        raise ValueError("GPU qualification requires two actual CUDA captures")
    if gpu and (
        not all(isinstance(case, dict) for case in receipts[0]["case_inventory"])
        or not any(case.get("observations", 0) >= 9 for case in receipts[0]["case_inventory"])
    ):
        raise ValueError("GPU qualification must cross the rolling eight-observation window")
    for record in receipts:
        proof = record.get("interleaved_session_isolation")
        if gpu and not isinstance(proof, dict):
            raise ValueError("GPU qualification requires interleaved session-isolation proof")
        if proof is not None and (
            not isinstance(proof, dict)
            or proof.get("passed") is not True
            or proof.get("distinct_identity") is not True
            or proof.get("sequence") != ["a0", "b0", "a1"]
        ):
            raise ValueError("Interleaved session-isolation proof failed or is incomplete")
    arrays = [np.load(path.with_suffix(".npz"), allow_pickle=False) for path in (reference, candidate)]
    try:
        if set(arrays[0].files) != set(arrays[1].files) or not arrays[0].files:
            raise ValueError("Captured tensor inventories differ or are empty")
        results = {}
        failures = []
        for name in sorted(arrays[0].files):
            left, right = arrays[0][name], arrays[1][name]
            if left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError(f"Captured tensor shape/dtype mismatch: {name}")
            if left.dtype.kind not in "biuf":
                raise ValueError(f"Captured tensor must be numeric: {name}")
            if not np.isfinite(left).all() or not np.isfinite(right).all():
                raise ValueError(f"Nonfinite captured tensor: {name}")
            if left.dtype.kind in "biu":
                equal = bool(np.array_equal(left, right))
            else:
                equal = bool(np.allclose(left, right, atol=atol, rtol=rtol))
            difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
            results[name] = {
                "shape": list(left.shape),
                "dtype": str(left.dtype),
                "maximum_absolute_difference": float(difference.max(initial=0)),
                "passed": equal,
            }
            if not equal:
                failures.append(name)
        reset_results = {}
        isolation_results = {}
        for record, archive in zip(receipts, arrays):
            for case in record["case_inventory"]:
                if not isinstance(case, dict):
                    continue
                name = case["name"]
                first = f"{name}/step000/"
                matching = [key for key in archive.files if key.startswith(first)]
                if not matching:
                    raise ValueError(f"No initial outputs for fixture case: {name}")
                for key in matching:
                    reset_key = f"{name}/after_reset/{key[len(first) :]}"
                    if reset_key not in archive:
                        raise ValueError(f"Missing reset replay output: {reset_key}")
                    passed = bool(np.allclose(archive[key], archive[reset_key], atol=atol, rtol=rtol))
                    check = f"{record.get('factory', 'runtime')}/{reset_key}"
                    reset_results[check] = passed
                    if not passed:
                        failures.append(check)
            proof = record.get("interleaved_session_isolation")
            if proof is not None:
                inventory = {case["name"] for case in record["case_inventory"] if isinstance(case, dict)}
                if (
                    proof.get("case_a") not in inventory
                    or proof.get("case_b") not in inventory
                    or proof["case_a"] == proof["case_b"]
                ):
                    raise ValueError("Session isolation references unknown or identical fixture cases")
                for label, tick, case_name in (
                    ("a", 0, proof["case_a"]),
                    ("a", 1, proof["case_a"]),
                    ("b", 0, proof["case_b"]),
                ):
                    baseline = f"{case_name}/step{tick:03d}/"
                    probe = f"interleaved/session_{label}/step{tick:03d}/"
                    matching = [key for key in archive.files if key.startswith(baseline)]
                    if not matching:
                        raise ValueError("Missing isolated outputs for interleaved comparison")
                    for key in matching:
                        probe_key = probe + key[len(baseline) :]
                        if probe_key not in archive:
                            raise ValueError(f"Missing interleaved session output: {probe_key}")
                        passed = bool(np.allclose(archive[key], archive[probe_key], atol=atol, rtol=rtol))
                        check = f"{record.get('factory', 'runtime')}/{probe_key}"
                        isolation_results[check] = passed
                        if not passed:
                            failures.append(check)
        return {
            "schema": "prism_engram_runtime_parity_v1",
            "receipt_closed": True,
            "qualified": not failures,
            "qualification_scope": "gpu_checkpoint_runtime_parity" if gpu else "cpu_capture_comparison",
            "reference": str(reference),
            "candidate": str(candidate),
            "fixture_sha256": receipts[0]["fixture_sha256"],
            "weights_sha256": receipts[0]["weights_sha256"],
            "atol": atol,
            "rtol": rtol,
            "failed_tensors": failures,
            "tensor_results": results,
            "reset_replay_results": reset_results,
            "interleaved_session_results": isolation_results,
            "policy_quality_evaluated": False,
        }
    finally:
        for archive in arrays:
            archive.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    result = compare_captures(
        args.reference, args.candidate, atol=args.atol, rtol=args.rtol, require_gpu=args.require_gpu
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("qualified", "qualification_scope", "failed_tensors")}))
    raise SystemExit(0 if result["qualified"] else 1)


if __name__ == "__main__":
    main()
