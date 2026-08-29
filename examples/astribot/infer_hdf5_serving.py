#!/usr/bin/env python3
"""Replay an Astribot schema-v4 HDF5 episode through a policy server."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import io
import json
from pathlib import Path
import time
import uuid

import h5py
import numpy as np
from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy
from PIL import Image
import zmq

from openpi.serving.astribot_policy_schema import ACTION_DIM as CANONICAL_ACTION_DIM
from openpi.serving.astribot_policy_schema import CANONICAL_ACTION_JOINTS
from openpi.serving.astribot_policy_schema import CANONICAL_BODY_JOINTS
from openpi.serving.astribot_policy_schema import HAND_CONTROL_JOINTS
from openpi.serving.astribot_policy_schema import HAND_STATE_JOINTS
from openpi.serving.astribot_policy_schema import POLICY_PROTOCOL_VERSION
from openpi.serving.astribot_policy_schema import POLICY_SCHEMA_HASH
from openpi.serving.astribot_policy_schema import validate_server_metadata
from openpi.serving.astribot_policy_schema import validate_single_action_response

BODY_PARTS: tuple[tuple[str, int], ...] = (
    ("astribot_torso", 4),
    ("astribot_arm_left", 7),
    ("astribot_arm_right", 7),
    ("astribot_head", 2),
)
CAMERAS: tuple[tuple[str, str], ...] = (
    ("head_stereo", "observation/head_rgbd"),
    ("left_wrist_rgbd", "observation/left_wrist_rgbd"),
    ("right_wrist_rgbd", "observation/right_wrist_rgbd"),
)
HAND_DRIVER_DIM = 16
ACTION_DIM = 52
ACTION_HORIZON = 64
IMAGE_SIZE = (224, 224)


@dataclasses.dataclass(frozen=True)
class CameraPayload:
    camera: str
    raw_indices: np.ndarray
    offsets: np.ndarray
    blob: np.ndarray
    rotation_degrees: int


def _decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values.tolist()]


def _extract_state_actions(file: h5py.File) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build the 52-D layout used by convert_astribot_v4_dexhand_to_lerobot.py."""

    state_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    names: list[str] = []
    sample_count: int | None = None

    for part, expected_dim in BODY_PARTS:
        root = f"streams/joints/{part}"
        state = np.asarray(file[f"{root}/state/position"][()], dtype=np.float32)
        actions = np.asarray(file[f"{root}/command/position"][()], dtype=np.float32)
        if state.ndim != 2 or state.shape[1] != expected_dim or actions.shape != state.shape:
            raise ValueError(f"Unexpected {part} shapes: state={state.shape}, actions={actions.shape}")
        if sample_count is None:
            sample_count = state.shape[0]
        if state.shape[0] != sample_count:
            raise ValueError(f"Mismatched sample count for {part}: {state.shape[0]} != {sample_count}")
        state_parts.append(state)
        action_parts.append(actions)
        names.extend(f"{part}_{index}" for index in range(expected_dim))

    for side in ("left", "right"):
        event_root = f"streams/joint_events/hands/{side}"
        state_names = _decode_strings(np.asarray(file[f"{event_root}/state/joint_names"][()]))
        driver_names = _decode_strings(np.asarray(file[f"{event_root}/command_driver_input/joint_names"][()]))
        if len(driver_names) != HAND_DRIVER_DIM:
            raise ValueError(f"{side} hand has {len(driver_names)} driver joints, expected {HAND_DRIVER_DIM}")
        state_index = {name: index for index, name in enumerate(state_names)}
        if len(state_index) != len(state_names) or len(set(driver_names)) != len(driver_names):
            raise ValueError(f"{side} hand joint names are not unique")
        try:
            selected = np.asarray([state_index[name] for name in driver_names], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"{side} hand driver joint is missing from state: {exc.args[0]}") from exc

        root = f"streams/joints/hand_{side}"
        hand_state = np.asarray(file[f"{root}/state/position"][()], dtype=np.float32)
        hand_actions = np.asarray(file[f"{root}/driver_input/position"][()], dtype=np.float32)
        if hand_state.shape[0] != sample_count or hand_actions.shape != hand_state.shape:
            raise ValueError(f"Unexpected {side} hand shapes: state={hand_state.shape}, actions={hand_actions.shape}")
        state_parts.append(hand_state[:, selected])
        action_parts.append(hand_actions[:, selected])
        names.extend(f"{side}_hand_{name}" for name in driver_names)

    state = np.concatenate(state_parts, axis=1)
    actions = np.concatenate(action_parts, axis=1)
    if state.shape != actions.shape or state.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected matching (*, {ACTION_DIM}) arrays, got {state.shape} and {actions.shape}")
    return state, actions, names


def _valid_rows(
    file: h5py.File,
    state: np.ndarray,
    actions: np.ndarray,
    *,
    trim_start_seconds: float,
    trim_end_seconds: float,
) -> np.ndarray:
    valid = np.isfinite(state).all(axis=1) & np.isfinite(actions).all(axis=1)
    timestamps = np.asarray(file["time"][()], dtype=np.float64)
    if timestamps.shape != (len(state),):
        raise ValueError(f"Unexpected time shape: {timestamps.shape}, expected {(len(state),)}")
    valid &= np.isfinite(timestamps)

    for camera, _ in CAMERAS:
        base = f"images_dict/{camera}"
        raw_indices = np.asarray(file[f"{base}/aligned/rgb_raw_index"][()])
        raw_sizes = np.asarray(file[f"{base}/rgb_size"][()])
        if raw_indices.shape != (len(state),):
            raise ValueError(f"Unexpected {camera} aligned index shape: {raw_indices.shape}")
        integer_indices = raw_indices.astype(np.int64, copy=False)
        in_range = (integer_indices >= 0) & (integer_indices < len(raw_sizes))
        valid &= in_range
        if len(raw_sizes) == 0:
            valid &= False
        else:
            safe_indices = np.clip(integer_indices, 0, len(raw_sizes) - 1)
            valid &= np.isfinite(raw_sizes[safe_indices]) & (raw_sizes[safe_indices] > 0)

    valid_rows = np.flatnonzero(valid)
    if len(valid_rows) == 0:
        raise ValueError("No complete state/action/image rows found")
    first_time = float(timestamps[valid_rows[0]])
    last_time = float(timestamps[valid_rows[-1]])
    if trim_start_seconds < 0 or trim_end_seconds < 0:
        raise ValueError("trim durations must be non-negative")
    selected = valid & (timestamps >= first_time + trim_start_seconds)
    selected &= timestamps <= last_time - trim_end_seconds
    rows = np.flatnonzero(selected)
    if len(rows) == 0:
        raise ValueError("No rows remain after time trimming")
    return rows


def _camera_payload(file: h5py.File, camera: str, rows: np.ndarray) -> CameraPayload:
    base = f"images_dict/{camera}"
    raw_indices = np.asarray(file[f"{base}/aligned/rgb_raw_index"][()], dtype=np.int64)[rows]
    sizes = np.asarray(file[f"{base}/rgb_size"][()], dtype=np.int64)
    offsets = np.empty(len(sizes) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(sizes, out=offsets[1:])
    blob = np.asarray(file[f"{base}/rgb"][()], dtype=np.uint8)
    if int(offsets[-1]) > len(blob):
        raise ValueError(f"{camera} JPEG offsets exceed blob size: {offsets[-1]} > {len(blob)}")
    return CameraPayload(
        camera=camera,
        raw_indices=raw_indices,
        offsets=offsets,
        blob=blob,
        rotation_degrees=int(file[f"{base}/rgb"].attrs.get("rotation_degrees", 0)),
    )


def _decode_image(payload: CameraPayload, position: int) -> np.ndarray:
    raw_index = int(payload.raw_indices[position])
    start = int(payload.offsets[raw_index])
    end = int(payload.offsets[raw_index + 1])
    with Image.open(io.BytesIO(bytes(payload.blob[start:end]))) as source:
        image = source.convert("RGB")
        if payload.camera == "head_stereo":
            width, height = image.size
            if width % 2:
                raise ValueError(f"Stereo image width must be even, got {image.size}")
            image = image.crop((0, 0, width // 2, height))
        if payload.rotation_degrees == 180:
            image = image.transpose(Image.Transpose.ROTATE_180)
        elif payload.rotation_degrees != 0:
            raise ValueError(f"Unsupported {payload.camera} rotation: {payload.rotation_degrees}")
        image = image.resize(IMAGE_SIZE, resample=Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8).copy()


def _decode_observations(
    payloads: list[CameraPayload],
    states: np.ndarray,
    rows: np.ndarray,
    prompt: str,
    workers: int,
) -> list[dict[str, object]]:
    if workers < 1:
        raise ValueError("decode workers must be at least 1")

    def decode_payload(payload: CameraPayload) -> tuple[str, list[np.ndarray]]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            images = list(executor.map(lambda position: _decode_image(payload, position), range(len(rows))))
        output_key = dict(CAMERAS)[payload.camera]
        return output_key, images

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(payloads)) as executor:
        decoded = dict(executor.map(decode_payload, payloads))

    observations: list[dict[str, object]] = []
    for position, row in enumerate(rows.tolist()):
        observations.append(
            {
                "observation/head_rgbd": decoded["observation/head_rgbd"][position],
                "observation/left_wrist_rgbd": decoded["observation/left_wrist_rgbd"][position],
                "observation/right_wrist_rgbd": decoded["observation/right_wrist_rgbd"][position],
                "observation/joint_position": states[row],
                "prompt": prompt,
            }
        )
    return observations


def _encode_png(image: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(stream, format="PNG")
    return stream.getvalue()


def _model_state_to_wire(state: np.ndarray) -> tuple[dict[str, object], np.ndarray]:
    """Convert the dataset's 52-D model layout into a canonical wire sample."""

    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape != (ACTION_DIM,):
        raise ValueError(f"Expected model state shape {(ACTION_DIM,)}, got {state.shape}")
    body = np.zeros(len(CANONICAL_BODY_JOINTS), dtype=np.float32)
    body[3:7] = state[:4]
    body[9:16] = state[4:11]
    body[16:23] = state[11:18]
    body[7:9] = state[18:20]

    def hand_state(values: np.ndarray) -> np.ndarray:
        result = np.zeros(len(HAND_STATE_JOINTS), dtype=np.float32)
        state_index = {name: index for index, name in enumerate(HAND_STATE_JOINTS)}
        for index, name in enumerate(HAND_CONTROL_JOINTS):
            result[state_index[name]] = values[index]
        return result

    left_hand = hand_state(state[20:36])
    right_hand = hand_state(state[36:52])
    observation = {
        "body": {
            "valid": True,
            "joint_names": list(CANONICAL_BODY_JOINTS),
            "position": body,
        },
        "hand_left": {
            "valid": True,
            "joint_names": list(HAND_STATE_JOINTS),
            "position": left_hand,
        },
        "hand_right": {
            "valid": True,
            "joint_names": list(HAND_STATE_JOINTS),
            "position": right_hand,
        },
    }
    return observation, body


def _model_action_to_canonical(action: np.ndarray, hold_state: np.ndarray) -> np.ndarray:
    """Convert a 52-D target to canonical action space, holding omitted joints."""

    wire_state, _ = _model_state_to_wire(action)
    measured_wire_state, _ = _model_state_to_wire(hold_state)
    result = np.zeros(CANONICAL_ACTION_DIM, dtype=np.float32)
    result[: len(CANONICAL_BODY_JOINTS)] = wire_state["body"]["position"]  # type: ignore[index]
    # The checkpoint has no chassis output. Match the server's mapping layer
    # by using the measured state as the expected no-motion target there.
    result[: len(CANONICAL_BODY_JOINTS[:3])] = measured_wire_state["body"]["position"][:3]  # type: ignore[index]
    left = np.asarray(wire_state["hand_left"]["position"], dtype=np.float32)  # type: ignore[index]
    right = np.asarray(wire_state["hand_right"]["position"], dtype=np.float32)  # type: ignore[index]
    control_indices = [HAND_STATE_JOINTS.index(name) for name in HAND_CONTROL_JOINTS]
    result[23:39] = left[control_indices]
    result[39:55] = right[control_indices]
    return result


class _ZmqDummyClient:
    """Small synchronous client for replaying samples through the ZMQ server."""

    def __init__(self, host: str, port: int, prompt: str) -> None:
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{host}:{port}")
        self._session_id = f"hdf5-dummy-{uuid.uuid4().hex}"
        self._prompt = prompt
        self._sequence = 0
        self._mask = np.ones(CANONICAL_ACTION_DIM, dtype=np.bool_)
        self._socket.send(
            msgpack_numpy.packb(
                {
                    "type": "hello",
                    "protocol_version": POLICY_PROTOCOL_VERSION,
                    "schema_hash": POLICY_SCHEMA_HASH,
                    "session_id": self._session_id,
                    "control_epoch": 0,
                }
            )
        )
        if not self._socket.poll(10_000):
            raise TimeoutError("timed out waiting for ZMQ metadata")
        metadata = msgpack_numpy.unpackb(self._socket.recv())
        if not isinstance(metadata, dict):
            raise ValueError("ZMQ metadata must be a mapping")
        validate_server_metadata(metadata)
        self._metadata = metadata
        self._socket.send(
            msgpack_numpy.packb(
                {
                    "type": "control",
                    "command": "arm",
                    "protocol_version": POLICY_PROTOCOL_VERSION,
                    "schema_hash": POLICY_SCHEMA_HASH,
                    "session_id": self._session_id,
                    "control_epoch": 0,
                }
            )
        )

    @property
    def metadata(self) -> dict[str, object]:
        return dict(self._metadata)

    def infer(self, observation: dict[str, object]) -> dict[str, object]:
        sequence = self._sequence
        self._sequence += 1
        state = np.asarray(observation["observation/joint_position"], dtype=np.float32)
        wire_observation, _ = _model_state_to_wire(state)
        head_image = np.asarray(observation["observation/head_rgbd"], dtype=np.uint8)
        left_wrist_image = Image.fromarray(
            np.asarray(observation["observation/left_wrist_rgbd"], dtype=np.uint8),
            mode="RGB",
        ).transpose(Image.Transpose.ROTATE_180)
        right_wrist_image = Image.fromarray(
            np.asarray(observation["observation/right_wrist_rgbd"], dtype=np.uint8),
            mode="RGB",
        ).transpose(Image.Transpose.ROTATE_180)
        images = {
            "head_stereo": {
                "valid": True,
                "encoding": "png",
                "payload": _encode_png(np.concatenate((head_image, head_image), axis=1)),
            },
            "left_wrist_rgbd": {
                "valid": True,
                "encoding": "png",
                "payload": _encode_png(np.asarray(left_wrist_image)),
            },
            "right_wrist_rgbd": {
                "valid": True,
                "encoding": "png",
                "payload": _encode_png(np.asarray(right_wrist_image)),
            },
        }
        request = {
            "type": "observation",
            "protocol_version": POLICY_PROTOCOL_VERSION,
            "session_id": self._session_id,
            "request_sequence": sequence,
            "control_epoch": 0,
            "schema_hash": POLICY_SCHEMA_HASH,
            "prompt": observation.get("prompt", self._prompt),
            "control_scope": "whole_body",
            "enabled_action_mask": self._mask,
            "action_joint_names": list(CANONICAL_ACTION_JOINTS),
            "observation": {**wire_observation, "images": images},
        }
        self._socket.send(msgpack_numpy.packb(request))
        if not self._socket.poll(30_000):
            raise TimeoutError(f"timed out waiting for action sequence {sequence}")
        response = msgpack_numpy.unpackb(self._socket.recv())
        if not isinstance(response, dict):
            raise ValueError("ZMQ action response must be a mapping")
        validate_single_action_response(
            response,
            session_id=self._session_id,
            expected_mask=self._mask,
        )
        return response

    def close(self) -> None:
        self._socket.close(0)
        self._context.term()


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _error_stats(abs_sum: float, square_sum: float, count: int, maximum: float) -> dict[str, float]:
    return {
        "mae": abs_sum / count,
        "rmse": float(np.sqrt(square_sum / count)),
        "max_abs": maximum,
        "values": count,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    hdf5_path = args.hdf5.resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(hdf5_path)

    with h5py.File(hdf5_path, "r") as file:
        if int(file.attrs.get("schema_version", -1)) != 4:
            raise ValueError(f"Expected schema_version=4, got {file.attrs.get('schema_version')!r}")
        if not bool(file.attrs.get("completed", False)) or bool(file.attrs.get("capture_incomplete", True)):
            raise ValueError("HDF5 capture is not marked complete")
        states, targets, joint_names = _extract_state_actions(file)
        rows = _valid_rows(
            file,
            states,
            targets,
            trim_start_seconds=args.trim_start_seconds,
            trim_end_seconds=args.trim_end_seconds,
        )
        if args.frame_stride < 1:
            raise ValueError("frame stride must be at least 1")
        rows = rows[:: args.frame_stride]
        if args.max_frames > 0:
            rows = rows[: args.max_frames]
        if len(rows) == 0:
            raise ValueError("No rows selected")
        payloads = [_camera_payload(file, camera, rows) for camera, _ in CAMERAS]

    decode_started = time.perf_counter()
    observations = _decode_observations(payloads, states, rows, args.prompt, args.decode_workers)
    decode_seconds = time.perf_counter() - decode_started
    target_series = targets[rows]

    if args.transport == "zmq":
        policy = _ZmqDummyClient(args.host, args.port, args.prompt)
        target_series_for_error = np.stack(
            [_model_action_to_canonical(action, states[row]) for action, row in zip(target_series, rows, strict=True)]
        )
    else:
        policy = websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
            api_key=args.api_key,
        )
        target_series_for_error = target_series
    try:
        for warmup_index in range(min(args.warmup, len(observations))):
            policy.infer(observations[warmup_index])

        latencies_ms: list[float] = []
        server_infer_ms: list[float] = []
        policy_infer_ms: list[float] = []
        first_abs_sum = 0.0
        first_square_sum = 0.0
        first_count = 0
        first_max = 0.0
        chunk_abs_sum = 0.0
        chunk_square_sum = 0.0
        chunk_count = 0
        chunk_max = 0.0

        for position, observation in enumerate(observations):
            started = time.perf_counter()
            output = policy.infer(observation)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if args.transport == "zmq":
                predicted = np.asarray(output["action"], dtype=np.float64)
                if predicted.shape != (CANONICAL_ACTION_DIM,):
                    raise ValueError(f"Expected ZMQ action shape {(CANONICAL_ACTION_DIM,)}, got {predicted.shape}")
                if not np.isfinite(predicted).all():
                    raise ValueError(f"Non-finite action returned at sample position {position}")
                current_error = predicted - target_series_for_error[position]
            else:
                predicted = np.asarray(output["actions"], dtype=np.float64)
                if predicted.shape != (ACTION_HORIZON, ACTION_DIM):
                    raise ValueError(f"Expected actions shape {(ACTION_HORIZON, ACTION_DIM)}, got {predicted.shape}")
                if not np.isfinite(predicted).all():
                    raise ValueError(f"Non-finite action returned at sample position {position}")
                current_error = predicted[0] - target_series_for_error[position]

            current_abs = np.abs(current_error)
            first_abs_sum += float(current_abs.sum())
            first_square_sum += float(np.square(current_error).sum())
            first_count += current_error.size
            first_max = max(first_max, float(current_abs.max()))

            if args.transport == "websocket":
                chunk_length = min(ACTION_HORIZON, len(target_series_for_error) - position)
                chunk_error = predicted[:chunk_length] - target_series_for_error[position : position + chunk_length]
                chunk_abs = np.abs(chunk_error)
                chunk_abs_sum += float(chunk_abs.sum())
                chunk_square_sum += float(np.square(chunk_error).sum())
                chunk_count += chunk_error.size
                chunk_max = max(chunk_max, float(chunk_abs.max()))

            latencies_ms.append(elapsed_ms)
            server_timing = output.get("server_timing", {})
            policy_timing = output.get("policy_timing", {})
            if isinstance(server_timing, dict) and "infer_ms" in server_timing:
                server_infer_ms.append(float(server_timing["infer_ms"]))
            if isinstance(policy_timing, dict) and "infer_ms" in policy_timing:
                policy_infer_ms.append(float(policy_timing["infer_ms"]))
    finally:
        if args.transport == "zmq":
            policy.close()

    result: dict[str, object] = {
        "status": "passed",
        "hdf5": str(hdf5_path),
        "host": args.host,
        "port": args.port,
        "transport": args.transport,
        "prompt": args.prompt,
        "sample_count": len(observations),
        "source_rows": [int(rows[0]), int(rows[-1])],
        "warmup_count": min(args.warmup, len(observations)),
        "joint_dim": len(joint_names),
        "action_shape": [CANONICAL_ACTION_DIM] if args.transport == "zmq" else [ACTION_HORIZON, ACTION_DIM],
        "decode_seconds": decode_seconds,
        "latency_ms": _stats(latencies_ms),
        # Keep transport and model measurements separate. A synchronous
        # dummy client measures round-trip throughput, while server timing
        # gives the model-compute ceiling and ZMQ metadata gives the
        # configured scheduler cap.
        "max_round_trip_frequency_hz": 1000.0 / min(latencies_ms),
        "configured_max_inference_frequency_hz": (
            float(policy.metadata["max_inference_hz"]) if args.transport == "zmq" else None
        ),
        "configured_action_horizon": (
            policy.metadata.get("action_horizon") if args.transport == "zmq" else None
        ),
        "configured_ensemble_decay": (
            float(policy.metadata["ensemble_decay"]) if args.transport == "zmq" else None
        ),
        "max_inference_frequency_hz": (
            min(
                float(policy.metadata.get("max_inference_hz", float("inf"))),
                1000.0 / min(server_infer_ms),
            )
            if args.transport == "zmq" and server_infer_ms
            else 1000.0 / min(latencies_ms)
        ),
        "server_inference_frequency_hz": 1000.0 / min(server_infer_ms) if server_infer_ms else None,
        "server_infer_ms": _stats(server_infer_ms) if server_infer_ms else None,
        "policy_infer_ms": _stats(policy_infer_ms) if policy_infer_ms else None,
        "action_error": {
            "units": "recorded joint-position units",
            "first_step": _error_stats(first_abs_sum, first_square_sum, first_count, first_max),
            "chunk": (
                _error_stats(chunk_abs_sum, chunk_square_sum, chunk_count, chunk_max)
                if args.transport == "websocket"
                else None
            ),
            "first_step_comparison": (
                "single canonical action versus recorded command at the observation row"
                if args.transport == "zmq"
                else "actions[0] versus recorded command at the observation row"
            ),
            "chunk_comparison": (
                "actions[0:64] versus recorded commands from the observation row onward"
                if args.transport == "websocket"
                else None
            ),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--transport", choices=("websocket", "zmq"), default="websocket")
    parser.add_argument("--api-key")
    parser.add_argument("--prompt", default="Light the candle")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all selected rows")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--trim-start-seconds", type=float, default=0.0)
    parser.add_argument("--trim-end-seconds", type=float, default=5.0)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()

    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload, flush=True)
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.result_json.with_suffix(args.result_json.suffix + ".tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(args.result_json)


if __name__ == "__main__":
    main()
