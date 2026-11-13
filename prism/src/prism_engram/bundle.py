"""Portable, weights-only PRISM bundles with verified local model assets.

Export runs beside the private checkpoint. Serving requires only its exported
directory and this package: neither training code nor training data is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "prism-engram-inference-bundle-v1"
SOURCE_REVISION = "de6415d53c031eff5c03ba68cb3fd6dec142be44"
QWEN_REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
CRADIO_REVISION = "c0457f5dc26ca145f954cd4fc5bb6114e5705ad8"
FAST_REVISION = "ec4d7aa71691cac0b8bed6942be45684db2110f4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _without_private_paths(value):
    if isinstance(value, dict):
        return {key: _without_private_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_without_private_paths(item) for item in value]
    if isinstance(value, str) and value.startswith(("/", "file://", "s3://")):
        return ""
    return value


def inference_config(source: dict) -> dict:
    """Retain graph/compute fields, omitting unrelated training source catalogs."""
    result = json.loads(json.dumps(source))
    # The active Engram heads use the separately exported Engram normalizers.
    # These inherited pretraining catalogs do not enter inference computation.
    for key in (
        "cotrain_contract",
        "action_head_normalization",
        "ego_action_normalization",
        "robot_action_normalization",
        "robot_proprio_normalization",
    ):
        result[key] = None
    result["stage1_checkpoint"] = ""
    result["qwen_model_name"] = "Qwen/Qwen3.5-2B-Base"
    result["instruction_encoder_model_name"] = "Qwen/Qwen3.5-2B-Base"
    result["fast_pretrained_path"] = "physical-intelligence/fast"
    result = _without_private_paths(result)
    extension = result.get("engram_native", {})
    if (
        extension.get("mode") != "joint_and_robot_eef"
        or extension.get("physical_horizon") != 32
        or extension.get("robot_eef_head") != "robot"
        or result.get("rotation_representation") != "rotation_6d"
        or result.get("visual_encoder_type") != "cradio"
        or result.get("visual_encoder_revision") != CRADIO_REVISION
        or result.get("instruction_encoder_revision") != QWEN_REVISION
    ):
        raise ValueError("Export requires the trained Engram hybrid32/Qwen2B/C-RADIO contract")
    return result


def inference_normalization(source: dict) -> dict:
    """Keep all numeric normalizers, masks and physical semantics, without datasets."""
    result = {key: value for key, value in source.items() if key not in {"source_manifest", "video_identity"}}
    native = dict(result["route_statistics"])
    native.pop("raw_cache_adoption", None)
    native.pop("hand_anchor_availability_audit", None)
    result["route_statistics"] = native
    result = _without_private_paths(result)
    if result.get("layout") != "engram_delta_joints_delta_hands_packed75_v2" or result.get("target_horizon") != 32:
        raise ValueError("Unsupported Engram normalization layout or action horizon")
    return result


def _copy_assets(source: Path, target: Path, *, radio: bool) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Architecture asset directory is missing: {source}")
    target.mkdir(parents=True)
    names = {
        "config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "LICENSE",
        "LICENSE.txt",
        "NOTICE",
        "NOTICE.txt",
    }
    for path in sorted(source.iterdir()):
        if not path.is_file() or not (path.name in names or (radio and path.suffix == ".py")):
            continue
        if path.name == "config.json":
            # Upstream model configs can contain training paths even though
            # those fields are unused by their model constructors.
            _write_json(target / path.name, _without_private_paths(json.loads(path.read_text())))
        else:
            shutil.copyfile(path, target / path.name)
    if not (target / "config.json").is_file():
        raise ValueError("Architecture assets require config.json")
    if radio and not (target / "hf_model.py").is_file():
        raise ValueError("C-RADIO assets require the pinned local implementation")


def export_bundle(checkpoint_dir, output_dir, *, qwen_assets=None, cradio_assets=None, fast_assets=None) -> Path:
    """Export a completed checkpoint without its optimizer or source catalogs.

    Existing output directories are never overwritten. Base model weights are
    unnecessary: the final checkpoint contains both Qwen networks and C-RADIO.
    Architecture/tokenizer files and C-RADIO Python definitions are copied.
    """
    source, output = Path(checkpoint_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    config_source = source / "config.json"
    config = json.loads(config_source.read_text())
    normalization_source = source / "engram_normalization.json"
    if not normalization_source.exists():
        normalization_source = source.parent / "engram_normalization.json"
    normalization = json.loads(normalization_source.read_text())
    state = source / "pytorch_model.bin"
    trainer_state = json.loads((source / "trainer_state.json").read_text())
    checkpoint_manifest = json.loads((source / "fsdp2_dmuon_manifest.json").read_text())
    step = trainer_state.get("global_step")
    if not isinstance(step, int) or step < 1 or checkpoint_manifest.get("step") != step:
        raise ValueError("Checkpoint completion step does not match its manifest")
    required = checkpoint_manifest.get("files", {}).values()
    if not required or any(not (source / name).is_file() or (source / name).stat().st_size == 0 for name in required):
        raise ValueError("Checkpoint manifest payloads are incomplete")
    if not state.is_file() or state.stat().st_size == 0:
        raise ValueError("Completed model weights are missing")
    qwen = Path(qwen_assets or config["qwen_model_name"]).resolve()
    if cradio_assets is None:
        from huggingface_hub import snapshot_download

        cradio_assets = snapshot_download("nvidia/C-RADIOv4-SO400M", revision=CRADIO_REVISION, local_files_only=True)
    radio = Path(cradio_assets).resolve()
    fast = Path(fast_assets or config["fast_pretrained_path"]).resolve()
    # Validate before copying the large checkpoint.
    clean_config = inference_config(config)
    clean_normalization = inference_normalization(normalization)
    output.mkdir(parents=True)
    try:
        _copy_assets(qwen, output / "assets/qwen", radio=False)
        _copy_assets(radio, output / "assets/cradio", radio=True)
        shutil.copyfile(Path(__file__).parent / "third_party/APACHE_2_0.txt", output / "assets/cradio/LICENSE.txt")
        # The frozen FAST head is retained for exact checkpoint topology. Its
        # tokenizer metadata defines its vocabulary even though Engram routes
        # never execute FAST action decoding.
        (output / "assets/fast").mkdir(parents=True)
        for name in ("tokenizer.json", "processor_config.json"):
            shutil.copyfile(fast / name, output / "assets/fast" / name)
        shutil.copyfile(Path(__file__).parent / "third_party/APACHE_2_0.txt", output / "assets/fast/LICENSE.txt")
        _write_json(output / "config.json", clean_config)
        _write_json(output / "normalization.json", clean_normalization)
        shutil.copyfile(state, output / "pytorch_model.bin")
        files = {
            str(path.relative_to(output)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(output.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema": SCHEMA,
            "source_revision": SOURCE_REVISION,
            "checkpoint_step": step,
            "source_config_sha256": sha256_file(config_source),
            "source_normalization_sha256": sha256_file(normalization_source),
            "dtype_policy": "checkpoint_storage_device_only_bf16_autocast",
            "qwen_revision": QWEN_REVISION,
            "cradio_revision": CRADIO_REVISION,
            "fast_revision": FAST_REVISION,
            "observation_hz": 10.0,
            "action_horizon": 32,
            "files": files,
            "complete": True,
        }
        _write_json(output / "manifest.json", manifest)
    except Exception:
        # Preserve an interrupted export for inspection, but never publish a
        # manifest that makes it look loadable.
        (output / "manifest.json").unlink(missing_ok=True)
        raise
    return output


def verify_bundle(directory) -> tuple[Path, dict, dict, dict]:
    """Verify every local asset before executing trusted model definitions."""
    path = Path(directory).resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("complete") is not True
        or manifest.get("source_revision") != SOURCE_REVISION
        or manifest.get("dtype_policy") != "checkpoint_storage_device_only_bf16_autocast"
    ):
        raise ValueError("Unsupported or incomplete PRISM bundle")
    files = manifest.get("files")
    required = {
        "config.json",
        "normalization.json",
        "pytorch_model.bin",
        "assets/qwen/config.json",
        "assets/qwen/tokenizer.json",
        "assets/cradio/config.json",
        "assets/cradio/hf_model.py",
        "assets/fast/tokenizer.json",
        "assets/fast/processor_config.json",
    }
    if not isinstance(files, dict) or not required <= files.keys():
        raise ValueError("Bundle manifest omits required runtime assets")
    actual = {
        str(item.relative_to(path)) for item in path.rglob("*") if item.is_file() and item != path / "manifest.json"
    }
    if actual != set(files):
        raise ValueError("Bundle contains unlisted or missing payloads")
    for name, record in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Bundle manifest contains an unsafe relative path")
        item = path / relative
        if item.is_symlink() or not item.is_file() or not item.resolve().is_relative_to(path):
            raise ValueError(f"Bundle payload is missing or escapes its directory: {name}")
        if item.stat().st_size != record["bytes"] or sha256_file(item) != record["sha256"]:
            raise ValueError(f"Bundle integrity mismatch: {name}")
    config = json.loads((path / "config.json").read_text())
    if inference_config(config) != config:
        raise ValueError("Bundle config is not portable or changes the inference contract")
    normalization = json.loads((path / "normalization.json").read_text())
    if inference_normalization(normalization) != normalization:
        raise ValueError("Bundle normalization contains unsupported metadata")
    return path, manifest, config, normalization


@dataclass
class LoadedBundle:
    path: Path
    model: Any
    tokenizer: Any
    config: dict
    normalization: dict
    manifest: dict


def restore_model_weights(model, state_path, *, device="cuda:0"):
    """Strictly restore names/shapes and exact checkpoint storage dtypes."""
    import torch

    state = torch.load(state_path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Model payload must be a tensor state dictionary")
    # Accelerate's empty-parameter registration can wrap a newly tied HF
    # parameter again. Reapply each model's declared construction policy after
    # leaving the empty context, before recording the original graph aliases.
    for module in tuple(model.modules()):
        tie_weights = getattr(module, "tie_weights", None)
        if callable(tie_weights):
            tie_weights()
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    alias_groups = [names for names in aliases.values() if len(names) > 1]
    for names in alias_groups:
        if any(name not in state for name in names):
            raise ValueError(f"Missing tied checkpoint tensor: {names}")
        reference = state[names[0]]
        for name in names[1:]:
            value = state[name]
            if value.dtype != reference.dtype or value.shape != reference.shape:
                raise ValueError(f"Tied checkpoint tensors have incompatible shapes or dtypes: {names}")
            same_view = (
                value.untyped_storage().data_ptr() == reference.untyped_storage().data_ptr()
                and value.storage_offset() == reference.storage_offset()
                and value.stride() == reference.stride()
            )
            if not same_view and not torch.equal(
                value.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8)
            ):
                raise ValueError(f"Tied checkpoint tensors have different values: {names}")

    def retie_parameters():
        for names in alias_groups:
            parameter = model.get_parameter(names[0])
            for name in names[1:]:
                parent, _, leaf = name.rpartition(".")
                setattr(model.get_submodule(parent) if parent else model, leaf, parameter)

    # assign=True is essential: copying into constructor BF16/FP32 tensors can
    # silently round the mixed BF16/FP32 parameters used by this training run.
    model.load_state_dict(state, strict=True, assign=True)
    for name, value in model.state_dict().items():
        if value.dtype != state[name].dtype:
            raise ValueError(f"Checkpoint dtype changed during restoration: {name}")
    # assign=True wraps state tensors in new Parameter objects, including names
    # originally tied by the constructor (Qwen embedding/output weights).
    # Restore the original object aliases before any device transfer.
    retie_parameters()
    del state
    for module_name, module in model.named_modules():
        for name, value in list(module.named_buffers(recurse=False)):
            if not value.is_meta:
                continue
            # Summary-token tuning keeps a nonpersistent pretraining reference
            # for a training-only auxiliary loss. Inference never consumes it.
            # Its constructor clone followed the meta parameter; retain a real
            # local reference after assignment so device transfer is defined.
            summary_parameter = getattr(module, "summary_token_parameter", None)
            if (
                name == "_initial_summary_tokens"
                and name in module._non_persistent_buffers_set
                and callable(summary_parameter)
            ):
                reference = summary_parameter()
                if reference.is_meta:
                    raise ValueError("The checkpoint did not restore C-RADIO summary tokens")
                module._buffers[name] = reference.detach().clone()
            elif name == "inv_freq" and type(module).__name__ == "TextDeepstackAdapter":
                # The source installer moves the adapter (including its
                # deterministic rotary clone) to the K-projection's device.
                # During empty construction that device is meta. Reproduce the
                # original clone/cast from the real backbone rotary buffer.
                backbone = getattr(getattr(model, "qwen", None), "model", None)
                rotary = getattr(backbone, "rotary_emb", None)
                reference = getattr(rotary, "inv_freq", None)
                if (
                    getattr(rotary, "rope_type", None) != "default"
                    or reference is None
                    or reference.is_meta
                    or reference.shape != value.shape
                ):
                    raise ValueError("Cannot restore the exact text-deepstack rotary frequencies")
                module._buffers[name] = reference.detach().to(device="cpu", dtype=value.dtype).clone()
            else:
                raise ValueError(f"Unmaterialized inference buffer after strict restoration: {module_name}.{name}")
    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError("The checkpoint did not materialize every model parameter")
    model.to(device=device)
    # Preserve the graph contract even on device backends whose conversion
    # replaces Parameter objects instead of updating their storage in place.
    retie_parameters()
    model._prism_parameter_alias_groups = tuple(tuple(names) for names in alias_groups)
    model.eval()
    model.requires_grad_(False)
    return model


def load_bundle(directory, device="cuda:0") -> LoadedBundle:
    """Load a verified standalone model without network or training dependencies."""
    path, manifest, config, normalization = verify_bundle(directory)
    from accelerate import init_empty_weights
    from prism_vla.modeling.config import PrismVLAConfig
    from prism_vla.modeling.model import PrismVLA
    from prism_vla.modeling.release_assets import architecture_assets
    from transformers import AutoTokenizer

    # Construct parameter metadata, not a second random 5B-parameter model.
    # Keep ordinary buffers on CPU: C-RADIO uses their actual scalar values in
    # construction, and rotary/sampling buffers must survive strict assignment.
    with architecture_assets(path / "assets"), init_empty_weights(include_buffers=False):
        model = PrismVLA(PrismVLAConfig.from_dict(config))
    restore_model_weights(model, path / "pytorch_model.bin", device=device)
    model.qwen.config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(str(path / "assets/qwen"), local_files_only=True)
    return LoadedBundle(path, model, tokenizer, config, normalization, manifest)


def main():
    parser = argparse.ArgumentParser(description="Export a portable, weights-only Engram PRISM checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qwen-assets")
    parser.add_argument("--cradio-assets")
    parser.add_argument("--fast-assets")
    args = parser.parse_args()
    result = export_bundle(
        args.checkpoint,
        args.output,
        qwen_assets=args.qwen_assets,
        cradio_assets=args.cradio_assets,
        fast_assets=args.fast_assets,
    )
    print(json.dumps({"exported": str(result), "manifest": str(result / "manifest.json")}))


if __name__ == "__main__":
    main()
