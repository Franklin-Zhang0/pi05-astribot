"""Recurrent episode boundaries and camera provenance without GPU/model weights."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from prism_engram.runtime import Runtime, training_image
from test_adapter import normalization, request  # noqa: F401


class Encoder:
    patch_size = 16

    def preprocess_image_tensor(self, images, device):
        self.encoded_views = len(images)
        return images.permute(0, 3, 1, 2).to(device)

    def __call__(self, images):
        cls = torch.arange(1, len(images) + 1, dtype=torch.float32)[:, None].expand(-1, 4)
        return cls, cls[:, None].expand(-1, 4, -1)

    def forward_with_register_tokens(self, images):
        cls, patches = self(images)
        return cls, None, patches, cls[:, None].expand(-1, 7, -1)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            num_dino_cameras=3,
            dino_camera_fusion="concat",
            dino_cls_camera_fusion="mean",
            full_attention_window_policy="rolling",
            attention_window_timesteps=8,
            flow_sampling_steps=16,
            instruction_format="json",
            query_design="direct_query",
            direct_query_include_registers=True,
            visual_encoder_eval_image_hw=[32, 32],
        )
        self.dino_cls_encoder = Encoder()
        self.share_dino_cls_encoder_for_patches = True
        self.max_inst_len = 384
        self.calls = []

    def inference_reset(self, *args, **kwargs):
        return {"ticks": 0}

    def _model_dtype(self):
        return torch.float32

    def inference_step(self, cache, cls, patches, **kwargs):
        cache["ticks"] += 1
        self.calls.append(kwargs)
        output = {key: torch.zeros((1, 160, 15)) for key in ["native_joint", "robot_eef"]}
        return (output if kwargs["decode_actions"] else None), cache


class Tokenizer:
    def __call__(self, texts, **kwargs):
        self.texts = texts
        return {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}

    def pad(self, value, **kwargs):
        return {key: torch.tensor(array) for key, array in value.items()}


@pytest.fixture
def runtime(normalization):  # noqa: F811
    return Runtime(SimpleNamespace(model=Model(), tokenizer=Tokenizer(), normalization=normalization))


def test_dense_optional_cameras_are_scattered_and_cls_is_masked_mean(runtime):
    prepared = runtime.adapter.prepare(request(wrists=True))
    cls, patches, registers, grid = runtime.visual_features(prepared)
    assert runtime.encoder.encoded_views == 2
    assert cls.shape == (1, 4) and patches.shape == (1, 12, 4)
    torch.testing.assert_close(cls, torch.full((1, 4), 1.5))
    assert patches[:, 8:].count_nonzero() == 0  # missing right wrist has a fixed slot
    assert registers.shape == (1, 3, 7, 4) and registers[:, 2].count_nonzero() == 0
    assert grid == (2, 2)


def test_independent_sessions_reset_and_observe_do_not_decode(runtime):
    a, b = runtime.new_session(), runtime.new_session()
    sample = {**request(), "timestamp_s": 0.0}
    result = a.infer(sample, return_normalized=True)
    assert result["observation_step"] == 1 and a.cache["ticks"] == 1 and b.cache is None
    assert runtime.tokenizer.texts == ["pick up the cup"]  # plain stays plain in JSON checkpoint
    assert runtime.model.calls[-1]["flow_initial_noise"]["native_joint"].dtype == torch.float32
    observed = a.infer({**sample, "timestamp_s": 0.1}, decode=False)
    assert observed["observed"] and a.cache["ticks"] == 2
    assert runtime.model.calls[-1]["decode_actions"] is False
    assert runtime.model.calls[-1]["proprio_frame_ids"].tolist() == [9]
    assert runtime.model.calls[-1]["dino_camera_mask_new"].tolist() == [[True, False, False]]
    assert runtime.model.calls[-1]["dino_registers_new"].shape == (1, 3, 7, 4)
    assert runtime.model.calls[-1]["dino_patch_grid_hw_new"] == (2, 2)
    b.infer(sample)
    a.reset()
    assert b.cache["ticks"] == 1 and a.cache is None


def test_timestamp_prompt_change_and_invalid_input_do_not_advance_cache(runtime):
    session = runtime.new_session()
    sample = {**request(), "timestamp_s": 0.0}
    session.infer(sample)
    for bad in [{**sample, "timestamp_s": 0.4}, {**sample, "timestamp_s": 0.1, "prompt": "other task"}]:
        with pytest.raises(ValueError):
            session.infer(bad)
        assert session.cache["ticks"] == 1
    session.reset()
    session.infer({**sample, "timestamp_s": 1.0})
    assert session.cache["ticks"] == 1


def test_inference_failure_retires_partially_mutated_cache(runtime):
    def broken(cache, *args, **kwargs):
        cache["ticks"] += 1
        raise RuntimeError("simulated model failure")

    session = runtime.new_session()
    runtime.model.inference_step = broken
    with pytest.raises(RuntimeError):
        session.infer({**request(), "timestamp_s": 0.0})
    assert session.cache is None and session.last_timestamp is None


def test_output_conversion_failure_also_resets_cache(runtime):
    def malformed(cache, *args, **kwargs):
        cache["ticks"] += 1
        return {"native_joint": "invalid"}, cache

    session = runtime.new_session()
    runtime.model.inference_step = malformed
    with pytest.raises(AttributeError):
        session.infer({**request(), "timestamp_s": 0.0})
    assert session.cache is None and session.steps == 0


def test_cover_crop_matches_private_loader_pixel_formula():
    image = np.arange(5 * 9 * 3, dtype=np.uint8).reshape(5, 9, 3)
    expected = (
        torch.nn.functional.interpolate(
            torch.from_numpy(image).permute(2, 0, 1)[None].float(), size=(8, 15), mode="bilinear", align_corners=False
        )
        .round()
        .clamp(0, 255)
        .byte()
    )
    expected = expected[0, :, :, 3:11].permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(training_image(image, (8, 8), crop_area=1), expected)
