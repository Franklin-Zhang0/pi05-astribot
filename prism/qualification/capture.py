"""Capture seeded checkpoint inference for comparison in an isolated process.

A fixture contains a JSON request schedule and an adjacent NPZ of RGB/state
arrays.  Array references are encoded as ``{"array": "npz_key"}``.  Supplying
``--factory module:function`` lets the private reference construct its original
model graph while using the identical serving adapter and explicit flow noise.
The default factory is the standalone bundle loader.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def resolve_arrays(value: Any, arrays) -> Any:
    if isinstance(value, dict):
        if set(value) == {"array"}:
            return np.array(arrays[value["array"]], copy=True)
        return {key: resolve_arrays(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_arrays(item, arrays) for item in value]
    return value


def numeric_outputs(value: Any, *, prefix: str, output: dict) -> None:
    """Record predicted arrays; omit nondeterministic server timing metadata."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key not in {"server_timing", "timing", "latency_ms", "infer_ms"}:
                numeric_outputs(item, prefix=f"{prefix}/{key}", output=output)
    elif isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError(f"Invalid predicted array: {prefix}")
        output[prefix] = np.array(value, copy=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--factory", default="prism_engram.bundle:load_bundle")
    parser.add_argument("--runtime", default="prism_engram.runtime:Runtime")
    parser.add_argument("--noise-seed", type=int, default=20261113)
    args = parser.parse_args()
    import torch

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Checkpoint qualification requires an available CUDA device")
    fixture = json.loads(args.fixture.read_text())
    if fixture.get("schema") != "prism_engram_parity_fixture_v1":
        raise ValueError("Unsupported qualification fixture")
    cases = fixture["cases"]
    if not cases or len({case["name"] for case in cases}) != len(cases):
        raise ValueError("Fixture case names must be unique and nonempty")
    if not any(len(case["requests"]) >= 9 for case in cases):
        raise ValueError("At least one case must cross the rolling eight-observation window")
    factory_module, factory_name = args.factory.split(":", 1)
    runtime_module, runtime_name = args.runtime.split(":", 1)
    factory = getattr(importlib.import_module(factory_module), factory_name)
    runtime_class = getattr(importlib.import_module(runtime_module), runtime_name)
    loaded = factory(args.bundle, device=args.device)
    runtime = runtime_class(loaded)
    predictions: dict[str, np.ndarray] = {}
    arrays_path = args.fixture.with_suffix(".npz")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        for case_index, case in enumerate(cases):
            session = runtime.new_session()
            session.reset(seed=int(case.get("seed", 0)))
            generator = torch.Generator(device="cpu").manual_seed(args.noise_seed + case_index)
            requests = [resolve_arrays(request, arrays) for request in case["requests"]]
            first_noise = None
            for index, request in enumerate(requests):
                noise = {
                    key: torch.randn((1, 160, 15), generator=generator).to(
                        device=args.device, dtype=runtime.model._model_dtype()
                    )
                    for key in ("native_joint", "robot_eef")
                }
                if first_noise is None:
                    first_noise = {key: value.clone() for key, value in noise.items()}
                result = session.infer(request, flow_noise=noise, return_normalized=True)
                numeric_outputs(result, prefix=f"{case['name']}/step{index:03d}", output=predictions)
            # Reset must remove all episode history. Reusing the same explicit
            # first-step noise isolates reset behavior from random sampling.
            session.reset(seed=int(case.get("seed", 0)))
            result = session.infer(requests[0], flow_noise=first_noise, return_normalized=True)
            numeric_outputs(result, prefix=f"{case['name']}/after_reset", output=predictions)

        # Independently cached sessions must reproduce isolated execution even
        # when a different robot/prompt activates the shared model in between.
        case_a_index = next(index for index, case in enumerate(cases) if len(case["requests"]) >= 2)
        case_a = cases[case_a_index]
        a_requests = [resolve_arrays(value, arrays) for value in case_a["requests"][:2]]
        a_prepared = runtime.adapter.prepare(a_requests[0])
        a_identity = (a_prepared.route_name, a_prepared.prompt_text, a_prepared.prompt_format)
        alternatives = []
        for index, case in enumerate(cases):
            request = resolve_arrays(case["requests"][0], arrays)
            prepared = runtime.adapter.prepare(request)
            identity = (prepared.route_name, prepared.prompt_text, prepared.prompt_format)
            if identity != a_identity:
                alternatives.append(
                    (
                        prepared.route_name == a_prepared.route_name,
                        prepared.prompt_text == a_prepared.prompt_text,
                        index,
                        request,
                    )
                )
        if not alternatives:
            raise ValueError("Session isolation needs a second case with a different robot or prompt")
        same_route, same_prompt, case_b_index, b_request = min(
            alternatives, key=lambda value: (value[0], value[1], value[2])
        )
        case_b = cases[case_b_index]
        session_a, session_b = runtime.new_session(), runtime.new_session()
        session_a.reset(seed=int(case_a.get("seed", 0)))
        session_b.reset(seed=int(case_b.get("seed", 0)))
        a_generator = torch.Generator(device="cpu").manual_seed(args.noise_seed + case_a_index)
        b_generator = torch.Generator(device="cpu").manual_seed(args.noise_seed + case_b_index)
        schedule = (
            ("a", 0, session_a, a_requests[0], a_generator),
            ("b", 0, session_b, b_request, b_generator),
            ("a", 1, session_a, a_requests[1], a_generator),
        )
        for label, tick, session, request, generator in schedule:
            noise = {
                key: torch.randn((1, 160, 15), generator=generator).to(
                    device=args.device, dtype=runtime.model._model_dtype()
                )
                for key in ("native_joint", "robot_eef")
            }
            result = session.infer(request, flow_noise=noise, return_normalized=True)
            numeric_outputs(result, prefix=f"interleaved/session_{label}/step{tick:03d}", output=predictions)
        isolation_failures = []
        isolation_checks = 0
        for label, tick, case_name in (("a", 0, case_a["name"]), ("a", 1, case_a["name"]), ("b", 0, case_b["name"])):
            baseline = f"{case_name}/step{tick:03d}/"
            probe = f"interleaved/session_{label}/step{tick:03d}/"
            keys = [key for key in predictions if key.startswith(baseline)]
            if not keys:
                raise RuntimeError("Missing isolated baseline outputs for session isolation")
            for key in keys:
                isolation_checks += 1
                probe_key = probe + key[len(baseline) :]
                if probe_key not in predictions or not np.allclose(
                    predictions[key], predictions[probe_key], atol=1e-5, rtol=1e-5
                ):
                    isolation_failures.append(probe_key)
        isolation = {
            "case_a": case_a["name"],
            "case_b": case_b["name"],
            "distinct_identity": True,
            "sequence": ["a0", "b0", "a1"],
            "distinct_route": not same_route,
            "distinct_formatted_prompt": not same_prompt,
            "atol": 1e-5,
            "rtol": 1e-5,
            "tensor_checks": isolation_checks,
            "passed": not isolation_failures,
            "failed_tensors": isolation_failures,
        }
    if not predictions or not any("normal" in key for key in predictions):
        raise RuntimeError("Runtime did not expose diagnostic normalized head outputs")
    weights = args.bundle / "pytorch_model.bin"
    if not weights.is_file():
        raise FileNotFoundError("Bundle weight identity requires pytorch_model.bin")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **predictions)
    receipt = {
        "schema": "prism_engram_runtime_capture_v1",
        "capture_complete": True,
        "factory": args.factory,
        "runtime": args.runtime,
        "device": args.device,
        "cuda_device_name": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__,
        "flow_noise_seed": args.noise_seed,
        "fixture_sha256": hashlib.sha256((sha256(args.fixture) + sha256(arrays_path)).encode()).hexdigest(),
        "weights_sha256": sha256(weights),
        "case_inventory": [{"name": case["name"], "observations": len(case["requests"])} for case in cases],
        "tensor_count": len(predictions),
        "interleaved_session_isolation": isolation,
        "scope": "original_graph_vs_standalone_graph_using_identical_serving_adapter",
        "training_dataset_required_for_runtime": False,
    }
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt, indent=2))
    if not isolation["passed"]:
        raise RuntimeError("Interleaved sessions differ from isolated execution; capture failed qualification")


if __name__ == "__main__":
    main()
