"""Portable export, integrity and exact mixed-dtype restoration contracts."""

import json

import pytest
from prism_engram.bundle import (
    CRADIO_REVISION,
    QWEN_REVISION,
    export_bundle,
    inference_config,
    restore_model_weights,
    verify_bundle,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def exported(tmp_path):
    source = tmp_path / "checkpoint-2"
    source.mkdir()
    config = {
        "engram_native": {"mode": "joint_and_robot_eef", "physical_horizon": 32, "robot_eef_head": "robot"},
        "rotation_representation": "rotation_6d",
        "visual_encoder_type": "cradio",
        "visual_encoder_revision": CRADIO_REVISION,
        "instruction_encoder_revision": QWEN_REVISION,
        "qwen_model_name": "/private/model/cache",
        "instruction_encoder_model_name": "/private/model/cache",
        "fast_pretrained_path": "/private/fast",
        "stage1_checkpoint": "/private/stage1",
        "cotrain_contract": {"secret_source": "/private/data"},
        "action_head_normalization": {"unused_pretraining_sources": "internal"},
    }
    write_json(source / "config.json", config)
    normalization = {
        "layout": "engram_delta_joints_delta_hands_packed75_v2",
        "target_horizon": 32,
        "source_manifest": {"path": "/private/data"},
        "route_statistics": {
            "routes": {"test": {"mean": [0.125], "std": [2.5]}},
            "raw_cache_adoption": {"path": "/private/cache"},
        },
    }
    write_json(source / "engram_normalization.json", normalization)
    write_json(source / "trainer_state.json", {"global_step": 2})
    write_json(source / "fsdp2_dmuon_manifest.json", {"step": 2, "files": {"model": "pytorch_model.bin"}})
    (source / "pytorch_model.bin").write_bytes(b"fixture-model-payload")
    qwen, radio, fast = (tmp_path / name for name in ("qwen", "radio", "fast"))
    for path in (qwen, radio):
        write_json(path / "config.json", {"architecture": "fixture", "unused_path": "/private/upstream"})
    write_json(qwen / "tokenizer.json", {})
    (radio / "hf_model.py").write_text("# Local trusted fixture; never executed by these tests.\n")
    write_json(fast / "tokenizer.json", {})
    write_json(fast / "processor_config.json", {"scale": 10, "min_token": -100})
    result = export_bundle(source, tmp_path / "release", qwen_assets=qwen, cradio_assets=radio, fast_assets=fast)
    return result, config, normalization


def test_export_is_weights_only_portable_and_keeps_numeric_stats(exported):
    path, source_config, source_norm = exported
    _, manifest, config, norm = verify_bundle(path)
    assert manifest["checkpoint_step"] == 2
    assert config == inference_config(source_config)
    assert config["cotrain_contract"] is None
    assert norm["route_statistics"]["routes"] == source_norm["route_statistics"]["routes"]
    assert "source_manifest" not in norm
    assert "raw_cache_adoption" not in norm["route_statistics"]
    assert not any("optimizer" in str(name) or "trainer_state" in str(name) for name in manifest["files"])
    for file in path.rglob("*.json"):
        assert "/private/" not in file.read_text()


def test_corrupted_weight_payload_is_rejected(exported):
    path, *_ = exported
    (path / "pytorch_model.bin").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_bundle(path)


def test_unlisted_python_payload_is_rejected_before_execution(exported):
    path, *_ = exported
    (path / "assets/cradio/injected.py").write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="unlisted"):
        verify_bundle(path)


def test_export_never_overwrites_existing_bundle(exported):
    path, *_ = exported
    with pytest.raises(FileExistsError):
        export_bundle(path.parent / "checkpoint-2", path)


def test_strict_restore_preserves_mixed_checkpoint_storage(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    saved = model.state_dict()
    saved["0.weight"] = torch.tensor([[1.0001, 0.00321], [-0.122345, 2.0301]], dtype=torch.float32)
    saved["1.weight"] = saved["1.weight"].bfloat16()
    path = tmp_path / "weights.bin"
    torch.save(saved, path)
    restored = restore_model_weights(model, path, device="cpu")
    for name, value in restored.state_dict().items():
        assert value.dtype == saved[name].dtype
        assert torch.equal(value, saved[name])
    assert not restored.training
    assert all(not parameter.requires_grad for parameter in restored.parameters())


def test_missing_model_parameter_fails_closed(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2)
    path = tmp_path / "weights.bin"
    torch.save({"weight": model.weight}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        restore_model_weights(model, path, device="cpu")


def test_local_cradio_loader_keeps_transitive_imports_and_bundle_immutable(tmp_path):
    from prism_vla.modeling.release_assets import architecture_assets, build_cradio_architecture

    path = tmp_path / "cradio"
    write_json(path / "config.json", {"fixture": 7})
    (path / "utils.py").write_text("VALUE = 5\n")
    (path / "hf_model.py").write_text(
        "from .utils import VALUE\n"
        "class RADIOConfig:\n"
        "    @staticmethod\n"
        "    def from_dict(value): return value\n"
        "class RADIOModel:\n"
        "    def __init__(self, config): self.answer = config['fixture'] + VALUE\n"
    )
    with architecture_assets(tmp_path):
        model = build_cradio_architecture()
    assert model.answer == 12
    assert not list(path.rglob("*.pyc"))


def test_meta_initialization_restores_mixed_parameters_and_keeps_real_buffers(tmp_path):
    torch = pytest.importorskip("torch")
    accelerate = pytest.importorskip("accelerate")

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(2, 2))
            self.bias = torch.nn.Parameter(torch.empty(2, dtype=torch.bfloat16))
            self.register_buffer("rotary_frequency", torch.arange(2, dtype=torch.float32), persistent=False)

    values = {
        "weight": torch.tensor([[1.0001, 2.12345], [-3.00321, 4.7777]], dtype=torch.float32),
        "bias": torch.tensor([0.5, -0.25], dtype=torch.bfloat16),
    }
    path = tmp_path / "weights.bin"
    torch.save(values, path)
    with accelerate.init_empty_weights(include_buffers=False):
        model = TinyModel()
    assert all(parameter.is_meta for parameter in model.parameters())
    assert not model.rotary_frequency.is_meta
    restore_model_weights(model, path, device="cpu")
    assert torch.equal(model.weight, values["weight"])
    assert model.weight.dtype == torch.float32
    assert torch.equal(model.bias, values["bias"])
    assert model.bias.dtype == torch.bfloat16
    assert torch.equal(model.rotary_frequency, torch.tensor([0.0, 1.0]))


def test_meta_summary_reference_materializes_only_after_checkpoint_restore(tmp_path):
    torch = pytest.importorskip("torch")
    accelerate = pytest.importorskip("accelerate")

    class SummaryModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tokens = torch.nn.Parameter(torch.empty(2, 3))
            self.register_buffer("_initial_summary_tokens", self.tokens.detach().clone(), persistent=False)

        def summary_token_parameter(self):
            return self.tokens

    path = tmp_path / "weights.bin"
    tokens = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch.save({"tokens": tokens}, path)
    with accelerate.init_empty_weights(include_buffers=False):
        model = SummaryModel()
    assert model._initial_summary_tokens.is_meta
    restore_model_weights(model, path, device="cpu")
    assert torch.equal(model._initial_summary_tokens, tokens)
    assert "_initial_summary_tokens" not in model.state_dict()


def test_meta_deepstack_rotary_clone_preserves_original_constructor_rounding(tmp_path):
    torch = pytest.importorskip("torch")
    from prism_vla.modeling.text_deepstack import TextDeepstackAdapter

    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.empty(1, device="meta")))
    model.qwen = torch.nn.Module()
    model.qwen.model = torch.nn.Module()
    model.qwen.model.rotary_emb = rotary = torch.nn.Module()
    rotary.rope_type = "default"
    rotary.attention_scaling = 1.0
    rotary.register_buffer("inv_freq", torch.tensor([1.0, 0.0123456], dtype=torch.float32), persistent=False)
    adapter = TextDeepstackAdapter(
        source_heads=1, source_head_dim=4, target_heads=1, target_head_dim=4, rotary_emb=rotary, source_layer=0
    )
    adapter.to(device="meta", dtype=torch.bfloat16)
    model.qwen.model.adapter = adapter
    path = tmp_path / "weights.bin"
    torch.save({"weight": torch.tensor([1.0001], dtype=torch.float32)}, path)
    restore_model_weights(model, path, device="cpu")
    assert not adapter.inv_freq.is_meta
    assert adapter.inv_freq.dtype == torch.bfloat16
    assert torch.equal(adapter.inv_freq, rotary.inv_freq.bfloat16())


def test_strict_meta_restore_preserves_tied_parameter_objects(tmp_path):
    torch = pytest.importorskip("torch")
    accelerate = pytest.importorskip("accelerate")

    class TiedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.output = torch.nn.Linear(4, 8, bias=False)
            self.tie_weights()

        def tie_weights(self):
            self.output.weight = self.embedding.weight

    weights = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 1001
    # Separate storage is legal if the checkpoint copies agree bit for bit.
    path = tmp_path / "tied.bin"
    torch.save({"embedding.weight": weights, "output.weight": weights.clone()}, path)
    with accelerate.init_empty_weights(include_buffers=False):
        model = TiedModel()
    restore_model_weights(model, path, device="cpu")
    assert model.embedding.weight is model.output.weight
    assert torch.equal(model.embedding.weight, weights)
    assert model.embedding.weight.dtype == torch.float32
    assert sum(parameter.numel() for parameter in model.parameters()) == 32
    assert model._prism_parameter_alias_groups == (("embedding.weight", "output.weight"),)
    model.to(dtype=torch.bfloat16)
    assert model.embedding.weight is model.output.weight


def test_conflicting_tied_checkpoint_values_are_rejected(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Module()
    model.a = torch.nn.Linear(2, 2, bias=False)
    model.b = torch.nn.Linear(2, 2, bias=False)
    model.b.weight = model.a.weight
    path = tmp_path / "bad-tie.bin"
    torch.save({"a.weight": torch.zeros(2, 2), "b.weight": torch.ones(2, 2)}, path)
    with pytest.raises(ValueError, match="different values"):
        restore_model_weights(model, path, device="cpu")
