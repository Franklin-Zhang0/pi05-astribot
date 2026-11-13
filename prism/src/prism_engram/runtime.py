"""Batch-one recurrent inference, with checkpoint-owned cameras and action heads."""

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from numbers import Integral, Real
from threading import RLock
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from prism_engram.adapter import EngramAdapter

CAMERAS = ("image", "left_wrist_image", "right_wrist_image")


def training_image(image: np.ndarray, output_hw: tuple[int, int], crop_area: float = 0.9) -> np.ndarray:
    """Deterministic counterpart of the TRAIN cover crop and fixed-area augmentation.

    Cover resizing uses FP32 bilinear without antialiasing and rounds to uint8,
    as the training loader does. The 90% crop uses torchvision's antialiased
    bilinear operation. Random offsets and photometric augmentation are disabled.
    """
    if not 0 < crop_area <= 1:
        raise ValueError("crop_area must lie in (0,1]")
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected HWC uint8 RGB")
    h, w = image.shape[:2]
    out_h, out_w = output_hw
    if min(h, w, out_h, out_w) <= 0:
        raise ValueError("Empty images/resolutions are unsupported")
    scale = max(out_h / h, out_w / w)
    resized_hw = (max(out_h, math.ceil(h * scale)), max(out_w, math.ceil(w * scale)))
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].float()
    if (h, w) != resized_hw:
        tensor = F.interpolate(tensor, size=resized_hw, mode="bilinear", align_corners=False)
    tensor = tensor.round().clamp(0, 255).to(torch.uint8)
    top, left = (resized_hw[0] - out_h) // 2, (resized_hw[1] - out_w) // 2
    tensor = tensor[:, :, top : top + out_h, left : left + out_w]
    if crop_area < 1:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        crop_h = max(1, min(out_h, round(math.sqrt(crop_area) * out_h)))
        crop_w = max(1, min(out_w, round(math.sqrt(crop_area) * out_w)))
        tensor = TF.resized_crop(
            tensor,
            (out_h - crop_h) // 2,
            (out_w - crop_w) // 2,
            crop_h,
            crop_w,
            [out_h, out_w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
    return tensor[0].permute(1, 2, 0).contiguous().numpy()


class Runtime:
    """One model shared by serialized, independently cached client sessions."""

    def __init__(self, bundle: Any):
        self.bundle = bundle
        self.model = bundle.model
        self.tokenizer = bundle.tokenizer
        self.adapter = EngramAdapter(bundle.normalization)
        self.device = next(self.model.parameters()).device
        self.lock = RLock()
        config = self.model.config
        if (
            config.num_dino_cameras != 3
            or config.dino_camera_fusion != "concat"
            or config.dino_cls_camera_fusion != "mean"
            or config.full_attention_window_policy != "rolling"
            or config.attention_window_timesteps != 8
            or config.flow_sampling_steps != 16
            or config.instruction_format != "json"
            or config.query_design != "direct_query"
            or not config.direct_query_include_registers
        ):
            raise ValueError("This server supports the trained Engram three-camera/rolling8/flow16 contract")
        self.output_hw = tuple(config.visual_encoder_eval_image_hw)
        self.encoder = self.model.dino_cls_encoder
        if not self.model.share_dino_cls_encoder_for_patches or self.encoder is None:
            raise ValueError("Expected the checkpoint-owned shared C-RADIO encoder")

    @classmethod
    def from_bundle(cls, path: str, device: str = "cuda:0") -> Runtime:
        from prism_engram.bundle import load_bundle

        return cls(load_bundle(path, device=device))

    def new_session(self) -> PolicySession:
        return PolicySession(self)

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()

    def visual_features(self, prepared):
        """Encode real views and scatter fixed camera-major slots, as in TRAIN."""
        slots = [i for i, name in enumerate(CAMERAS) if prepared.camera_mask[i]]
        images = np.stack([training_image(prepared.frames[CAMERAS[i]], self.output_hw) for i in slots])
        inputs = self.encoder.preprocess_image_tensor(torch.from_numpy(images), device=self.device)
        cls, _aux, patches, registers = self.encoder.forward_with_register_tokens(inputs)
        if registers.shape[1] != 7:
            raise ValueError("The trained direct queries require seven C-RADIO registers per camera")
        indices = torch.tensor(slots, device=cls.device)
        cls_slots = cls.new_zeros(3, cls.shape[-1]).index_copy(0, indices, cls)
        patch_slots = patches.new_zeros(3, *patches.shape[1:]).index_copy(0, indices, patches)
        register_slots = registers.new_zeros(3, *registers.shape[1:]).index_copy(0, indices, registers)
        patch_size = self.encoder.patch_size
        ph, pw = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        grid = (inputs.shape[-2] // ph, inputs.shape[-1] // pw)
        # Only present CLS tokens contribute to the mean; absent dense slots stay zero.
        return (
            cls_slots.sum(0, keepdim=True) / len(slots),
            patch_slots.reshape(1, -1, patches.shape[-1]),
            register_slots[None],
            grid,
        )


class PolicySession:
    """An episode's causal state and RNG; reset never affects another client."""

    def __init__(self, runtime: Runtime):
        self.runtime = runtime
        self.reset()

    def reset(self, seed: int = 0):
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0,2**63)")
        self.episode_generation = getattr(self, "episode_generation", -1) + 1
        self.cache = None
        self.identity = None
        self.last_timestamp = None
        self.steps = 0
        self.generator = torch.Generator(device=self.runtime.device).manual_seed(int(seed))
        return {"reset": True, "observation_hz": 10.0, "seed": int(seed)}

    @contextmanager
    def _cache_transaction(self):
        try:
            yield
        except Exception:
            self.reset()
            raise

    @torch.inference_mode()
    def infer(self, request: dict, *, decode: bool = True, flow_noise=None, return_normalized: bool = False):
        runtime = self.runtime
        prepared = runtime.adapter.prepare(request)
        timestamp = request.get("timestamp_s")
        if isinstance(timestamp, bool) or not isinstance(timestamp, Real) or not math.isfinite(timestamp):
            raise ValueError("Supply finite timestamp_s for the 10Hz observation stream")
        if self.last_timestamp is not None and not 0.05 - 1e-6 <= timestamp - self.last_timestamp <= 0.15 + 1e-6:
            raise ValueError("Observations must advance at 10Hz (50–150ms tolerance); reset after gaps")
        identity = (prepared.route_name, prepared.prompt_text, prepared.prompt_format)
        if self.identity is not None and self.identity != identity:
            raise ValueError("Reset the episode before changing robot, end-effector or prompt")
        device = runtime.device
        with runtime.lock, self._cache_transaction(), runtime.autocast():
            cls, patches, registers, grid = runtime.visual_features(prepared)
            camera_mask = torch.as_tensor(prepared.camera_mask[None], device=device)
            if self.cache is None:
                from prism_vla.data.instruction_conditioning import InstructionOnlyText, tokenize_instructions

                prompt = (
                    InstructionOnlyText(prepared.prompt_text)
                    if prepared.prompt_format == "plain"
                    else prepared.prompt_text
                )
                tokens = tokenize_instructions(
                    runtime.tokenizer,
                    [prompt],
                    max_inst_len=runtime.model.max_inst_len,
                    instruction_format=runtime.model.config.instruction_format,
                )
                self.cache = runtime.model.inference_reset(
                    tokens["input_ids"].to(device),
                    tokens["attention_mask"].to(device),
                    cls,
                    patches,
                    dino_camera_mask_0=camera_mask,
                )
            if decode and flow_noise is None:
                # Action conditions are explicitly converted to model dtype by
                # the trained inference graph; encoder autocast dtype can differ.
                dtype = runtime.model._model_dtype()
                flow_noise = {
                    name: torch.randn((1, 160, 15), device=device, dtype=dtype, generator=self.generator)
                    for name in ("native_joint", "robot_eef")
                }
            try:
                output, self.cache = runtime.model.inference_step(
                    self.cache,
                    cls,
                    patches,
                    proprio_new=torch.as_tensor(prepared.normalized_state[None], device=device),
                    proprio_frame_ids=torch.full((1,), 9, device=device, dtype=torch.long),
                    end_effector_type_ids=torch.full(
                        (1, 2), int(prepared.route_id == 0), device=device, dtype=torch.long
                    ),
                    engram_route_ids=torch.tensor([prepared.route_id], device=device),
                    engram_state_mask=torch.as_tensor(prepared.state_mask[None], device=device),
                    engram_action_mask=torch.as_tensor(prepared.native_action_mask[None], device=device),
                    engram_robot_eef_mask=torch.as_tensor(prepared.robot_eef_mask[None], device=device),
                    dino_registers_new=registers,
                    dino_patch_grid_hw_new=grid,
                    dino_camera_mask_new=camera_mask,
                    flow_initial_noise=flow_noise,
                    cache_crop_interval=1,
                    decode_actions=decode,
                )
            except Exception:
                # A partially updated recurrent cache must not be reused.
                self.reset()
                raise
        with self._cache_transaction():
            self.identity, self.last_timestamp = identity, float(timestamp)
            self.steps += 1
            if not decode:
                return {"observed": True, "observation_step": self.steps, "timestamp_s": float(timestamp)}
            normalized = {name: value[0].float().cpu().numpy() for name, value in output.items()}
            if any(not np.isfinite(value).all() for value in normalized.values()):
                raise FloatingPointError("Nonfinite model output; session reset")
            result = runtime.adapter.decode(normalized, prepared)
            result.update(observation_step=self.steps, timestamp_s=float(timestamp))
            if return_normalized:
                result["normalized"] = normalized
            return result
