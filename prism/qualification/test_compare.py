"""Qualification gates must reject mismatches rather than relax their scope."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from compare import compare_captures


def capture(tmp_path: Path, name: str, *, values=None, device="cpu", identity="same"):
    path = tmp_path / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "schema": "prism_engram_runtime_capture_v1",
                "capture_complete": True,
                "fixture_sha256": identity,
                "weights_sha256": "weights",
                "case_inventory": ["history_then_reset"],
                "flow_noise_seed": 7,
                "device": device,
                "cuda_device_name": "H100" if device.startswith("cuda") else None,
            }
        )
    )
    np.savez(path.with_suffix(".npz"), prediction=np.asarray([1, 2] if values is None else values, dtype=np.float32))
    return path


def test_cpu_comparison_never_claims_gpu_qualification(tmp_path):
    a, b = capture(tmp_path, "reference"), capture(tmp_path, "candidate")
    result = compare_captures(a, b)
    assert result["qualified"]
    assert result["qualification_scope"] == "cpu_capture_comparison"
    with pytest.raises(ValueError, match="actual CUDA"):
        compare_captures(a, b, require_gpu=True)


def test_rejects_input_identity_mismatch(tmp_path):
    a = capture(tmp_path, "reference")
    b = capture(tmp_path, "candidate", identity="different")
    with pytest.raises(ValueError, match="fixture_sha256"):
        compare_captures(a, b)


def test_reports_numeric_mismatch(tmp_path):
    a = capture(tmp_path, "reference")
    b = capture(tmp_path, "candidate", values=[1, 2.5])
    result = compare_captures(a, b)
    assert not result["qualified"]
    assert result["failed_tensors"] == ["prediction"]
    assert result["tensor_results"]["prediction"]["maximum_absolute_difference"] == 0.5


def test_nonfinite_cannot_pass_comparison(tmp_path):
    a = capture(tmp_path, "reference")
    b = capture(tmp_path, "candidate", values=[1, np.nan])
    with pytest.raises(ValueError, match="Nonfinite"):
        compare_captures(a, b)


def test_empty_capture_is_not_qualification(tmp_path):
    a = capture(tmp_path, "reference")
    b = capture(tmp_path, "candidate")
    np.savez(a.with_suffix(".npz"))
    np.savez(b.with_suffix(".npz"))
    with pytest.raises(ValueError, match="inventories"):
        compare_captures(a, b)


def test_capture_must_replay_initial_outputs_after_reset(tmp_path):
    a, b = capture(tmp_path, "reference"), capture(tmp_path, "candidate")
    for path in (a, b):
        record = json.loads(path.read_text())
        record["case_inventory"] = [{"name": "history", "observations": 9}]
        path.write_text(json.dumps(record))
        np.savez(
            path.with_suffix(".npz"),
            **{
                "history/step000/normalized/native_joint": np.array([1], np.float32),
                "history/after_reset/normalized/native_joint": np.array([2], np.float32),
            },
        )
    # Comparing two equally broken implementations must still reject reset drift.
    result = compare_captures(a, b)
    assert not result["qualified"]
    assert all(not value for value in result["reset_replay_results"].values())


def test_gpu_scope_requires_recurrent_boundary_coverage(tmp_path):
    a, b = capture(tmp_path, "reference", device="cuda:0"), capture(tmp_path, "candidate", device="cuda:0")
    with pytest.raises(ValueError, match="eight-observation"):
        compare_captures(a, b, require_gpu=True)


def test_tolerances_cannot_disable_finite_comparison(tmp_path):
    a, b = capture(tmp_path, "reference"), capture(tmp_path, "candidate")
    with pytest.raises(ValueError, match="finite and nonnegative"):
        compare_captures(a, b, atol=float("inf"))


def test_gpu_capture_requires_interleaved_session_proof(tmp_path):
    a, b = capture(tmp_path, "reference", device="cuda:0"), capture(tmp_path, "candidate", device="cuda:0")
    for path in (a, b):
        record = json.loads(path.read_text())
        record["case_inventory"] = [{"name": "a", "observations": 9}, {"name": "b", "observations": 2}]
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="session-isolation proof"):
        compare_captures(a, b, require_gpu=True)


def isolation_capture(tmp_path, name, *, drift=0, passed=True):
    path = capture(tmp_path, name)
    record = json.loads(path.read_text())
    record["case_inventory"] = [{"name": "a", "observations": 9}, {"name": "b", "observations": 2}]
    record["interleaved_session_isolation"] = {
        "case_a": "a",
        "case_b": "b",
        "passed": passed,
        "distinct_identity": True,
        "sequence": ["a0", "b0", "a1"],
    }
    path.write_text(json.dumps(record))
    np.savez(
        path.with_suffix(".npz"),
        **{
            "a/step000/normalized/native_joint": np.array([1], np.float32),
            "a/step001/normalized/native_joint": np.array([2], np.float32),
            "a/after_reset/normalized/native_joint": np.array([1], np.float32),
            "b/step000/normalized/native_joint": np.array([3], np.float32),
            "b/after_reset/normalized/native_joint": np.array([3], np.float32),
            "interleaved/session_a/step000/normalized/native_joint": np.array([1], np.float32),
            "interleaved/session_a/step001/normalized/native_joint": np.array([2 + drift], np.float32),
            "interleaved/session_b/step000/normalized/native_joint": np.array([3], np.float32),
        },
    )
    return path


def test_interleaved_session_arrays_are_verified_independently_of_pass_flag(tmp_path):
    a, b = isolation_capture(tmp_path, "reference", drift=0.25), isolation_capture(tmp_path, "candidate", drift=0.25)
    result = compare_captures(a, b)
    assert not result["qualified"]
    assert any(not value for value in result["interleaved_session_results"].values())


def test_complete_cpu_isolation_fixture_stays_cpu_qualification(tmp_path):
    a, b = isolation_capture(tmp_path, "reference"), isolation_capture(tmp_path, "candidate")
    result = compare_captures(a, b)
    assert result["qualified"]
    assert result["interleaved_session_results"]
    assert result["qualification_scope"] == "cpu_capture_comparison"


def test_declared_interleaved_failure_rejects_capture(tmp_path):
    a, b = isolation_capture(tmp_path, "reference", passed=False), isolation_capture(tmp_path, "candidate")
    with pytest.raises(ValueError, match="failed or is incomplete"):
        compare_captures(a, b)
