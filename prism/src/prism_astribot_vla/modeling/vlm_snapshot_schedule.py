"""Causal selection of sparse RGB snapshots for the frozen multimodal prefix."""
from __future__ import annotations

import math

import torch

from prism_astribot_vla.modeling.text_deepstack import TextDeepstackMemory, TextLayerKV
from prism_astribot_vla.modeling.token_reduction_backbone import FrameDeepstackMemory


def _join_memory(old, new):
    if old is None:
        return new
    if new is None:
        return old
    return TextDeepstackMemory(tuple(
        TextLayerKV(a.layer_index, torch.cat((a.key, b.key)), torch.cat((a.value, b.value)))
        for a, b in zip(old.layers, new.layers, strict=True)),
        torch.cat((old.attention_mask, new.attention_mask)), torch.cat((old.position_ids, new.position_ids)))


def _select_rows(memory, rows):
    if rows == list(range(len(rows))) and memory.attention_mask.shape[0] == len(rows):
        return memory
    return memory.index_select(torch.tensor(rows, device=memory.attention_mask.device))


def prepare_snapshot_memory(encoder, *, hz, images, seconds, capture_seconds, valid, present, ids, mask,
                            previous, reset, changed, compute_age=True):
    """Encode each due image once and route the latest causal prefix to each frame.

    Times are physical seconds per episode. A missing camera retains the most recent
    observed snapshot and a due refresh waits for a valid image. A text change can
    recompute the multimodal language context from the previously encoded image.
    """
    batch, steps = valid.shape
    if seconds is None or seconds.shape != (batch, steps) or not seconds.is_floating_point():
        raise ValueError("VLM image input requires observation_timestamps_s [B,T] in seconds")
    if present.shape != (batch, steps):
        raise ValueError("VLM image presence must be [B,T]")
    if images is not None and (images.ndim != 5 or images.shape[:3] != (batch, steps, 3)):
        raise ValueError("vlm_images must be [B,T,3,H,W] RGB")
    # Decide the complete schedule in one host pass. The frozen encoders process
    # only actual refreshes, including when several deadlines fall inside a chunk.
    times = seconds.detach().double().cpu().tolist()
    if capture_seconds is not None and capture_seconds.shape != seconds.shape:
        raise ValueError("vlm_capture_timestamps_s must match observation timestamps")
    captures = times if capture_seconds is None else capture_seconds.detach().double().cpu().tolist()
    valid_rows = valid.detach().bool().cpu().tolist()
    camera_rows = present.detach().bool().cpu().tolist()
    previous = previous or {}
    old_memory = previous.get("memory")
    old_features = previous.get("features")
    capture = list(previous.get("capture", [None] * batch))
    deadline = list(previous.get("deadline", [None] * batch))
    latest = list(previous.get("latest", [None] * batch))
    reset_rows = reset.detach().bool().cpu().tolist() if reset is not None else [False] * batch
    changed_rows = changed.detach().bool().cpu().tolist() if changed is not None else [False] * batch
    period = 1.0 / hz
    due_pixels, requests, routing, age_flags, final_images = [], [], [], [], []
    old_count = batch if old_memory is not None else 0
    for row in range(batch):
        if reset_rows[row]:
            capture[row] = deadline[row] = latest[row] = None
        image = row if capture[row] is not None else -1
        memory = row if old_count and not reset_rows[row] and not changed_rows[row] else -1
        frame_ids, ages = [], []
        for t, now in enumerate(times[row]):
            if valid_rows[row][t]:
                if not math.isfinite(now) or (latest[row] is not None and now <= latest[row]):
                    raise ValueError("Valid VLM observation timestamps must increase within each episode")
                latest[row] = now
                due = deadline[row]
                # Tolerate representational rounding at an exact deadline.
                if camera_rows[row][t] and (due is None or now + 1e-7 >= due):
                    image = batch + len(due_pixels)
                    due_pixels.append((row, t))
                    captured_at = captures[row][t]
                    if not math.isfinite(captured_at) or captured_at > now + 1e-7:
                        raise ValueError("VLM image capture must be finite and causal")
                    capture[row] = captured_at
                    deadline[row] = (now + period if due is None else
                                     due + (1 + math.floor((now + 1e-7 - due) / period)) * period)
                    memory = -1
            if memory < 0:
                memory = old_count + len(requests)
                requests.append((row, image))
            frame_ids.append(memory)
            if compute_age:
                has_image = valid_rows[row][t] and capture[row] is not None
                ages.append((math.log1p(max(0., now - capture[row])), 1.) if has_image else (0., 0.))
        routing.append(frame_ids)
        age_flags.append(ages)
        final_images.append(image)

    fresh = None
    if due_pixels:
        if images is None:
            raise ValueError("Missing main-camera RGB at scheduled VLM image refresh")
        row, tick = zip(*due_pixels)
        fresh = encoder.encode_snapshot_images(images[list(row), list(tick)])

    def select_images(indices):
        template = fresh if fresh is not None else old_features
        if template is None or all(index < 0 for index in indices):
            return None
        selected = []
        for index in indices:
            if index < 0:
                selected.append(template.new_zeros(template.shape[1:]))
            elif index < batch:
                selected.append(old_features[index])
            else:
                selected.append(fresh[index - batch])
        return torch.stack(selected)

    new_memory = None
    if requests:
        rows, image_indices = zip(*requests)
        new_memory = encoder.snapshot_deepstack_memory(
            ids[list(rows)], mask[list(rows)], select_images(image_indices),
            torch.tensor([idx >= 0 for idx in image_indices], device=ids.device))
    if old_count and all(all(idx >= old_count for idx in row) for row in routing):
        # A full-batch refresh can discard the previous chunk's bank.
        routing = [[idx - old_count for idx in row] for row in routing]
        old_memory = None
    bank = _join_memory(old_memory, new_memory)
    final_rows = [row[-1] for row in routing]
    memory = _select_rows(bank, final_rows)
    if old_features is not None and fresh is None and not any(reset_rows):
        features = old_features
    else:
        features = select_images(final_images)
    state = {"memory": memory, "features": features, "capture": capture,
             "latest": latest, "deadline": deadline}
    frame_memory = FrameDeepstackMemory(bank, torch.tensor(routing, device=ids.device))
    ages = torch.tensor(age_flags, device=ids.device, dtype=torch.float32) if compute_age else None
    return frame_memory, ages, state
