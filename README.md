# Astribot mobile PRISM inference

Serve the full Astribot mobile fine-tune over OpenPI WebSocket. Use
`checkpoint-20000` from
`prism-vla-astribot-mobile-stereoleft-chassis-textdrop05-cos20k-fsdp-4h100-b16-20261125-r2`.
The server selects the `robot` flow head and its fitted Astribot training
statistics. This is the head that includes chassis targets.

## Start the server

Use Linux x86-64, Python 3.12.12, `uv`, and an NVIDIA GPU with a CUDA
12.8-compatible driver. The native convolution extension may require the CUDA
toolkit to build. Have the full checkpoint directory (`config.json` and
`model.pt`). The branch includes the matching token-reduction model code.

```bash
git clone --single-branch --branch codex/prism-astribot-mobile-inference \
  https://github.com/Franklin-Zhang0/pi05-astribot.git
cd pi05-astribot/prism
uv sync --locked --extra native-conv
uv run --no-sync prism-astribot-serve --checkpoint /path/to/checkpoint-20000
```

The command uses `cuda:0` and listens on `127.0.0.1:8000` by default. Set
`--host`, `--port` or `--device` as needed. The first launch resolves pinned
Qwen configuration and tokenizer files, FAST tokenizer files, and the frozen
NVIDIA C-RADIOv4-SO400M encoder from Hugging Face. The checkpoint supplies
both Qwen text towers' weights. For an offline host, point `HF_HUB_CACHE`
at a cache containing these files and set `HF_HUB_OFFLINE=1`. You can also
provide local metadata directories with `--qwen-assets` and `--fast-assets`.

## Stream observations

Send one aligned observation every **100 ms** in capture order. Use a common
monotonic clock in seconds for the camera and measured state. Training
formed a 20 Hz grid from the latest available samples at or before each
tick, then took every other tick for observations. Replay should follow the
same causal alignment and treat samples older than 100 ms as unavailable.
The client owns sensor alignment. The server accepts observation intervals
of 50–150 ms and advances its recurrent history by one nominal 10 Hz step
per request.

Use `observe()` to advance state on ticks where you do not need a new plan,
including while executing a previous chunk. Call either `observe()` or
`infer()` for each capture. `infer()` also returns a **32-step plan at
20 Hz**. Call `reset()` once at the start of each episode, after reconnecting,
or when restarting the stream after a gap.

```python
from prism_engram.client import PrismClient

with PrismClient() as client:
    assert client.metadata["model"] == "astribot_mobile_prism"
    assert client.metadata["action_head"] == "robot"
    client.reset(seed=0)
    # episode_stream yields aligned samples and your replanning decisions.
    for sample, request_chunk in episode_stream:
        observation = {
            "timestamp_s": sample.timestamp_s,
            "stereo_left": sample.left_eye_rgb,
            "head_rgb": sample.head_rgb,
            "left_wrist": sample.left_wrist_rgb,
            "right_wrist": sample.right_wrist_rgb,
            "eef_pose": sample.eef_pose,
            "gripper_percent": sample.gripper_percent,
            "chassis_xyyaw": sample.chassis_xyyaw,
            "previous_chassis_xyyaw": sample.previous_chassis_xyyaw,
            "prompt": sample.task or "",
            "subtask": sample.subtask or "",
        }
        if request_chunk:
            plan = client.infer(observation)
        else:
            client.observe(observation)
```

### Client input contract

| Field | What to send |
|---|---|
| `stereo_left`, `head_rgb` | One main-camera image as an HWC `uint8` RGB array. Use the left half when decoding packed side-by-side stereo. The server prefers `stereo_left` and uses `head_rgb` when it is absent or `None`. |
| `left_wrist`, `right_wrist` | HWC `uint8` RGB in robot-side order. Omit unavailable wrists or send `None` so the server masks those views. |
| `eef_pose` | Measured left then right SDK tool poses in the chassis-relative controller basis, shape `[2, 7]`. XYZ is in metres, followed by a unit XYZW quaternion. |
| `gripper_percent` | Measured values in the SDK's native 0–100 convention, left then right. The server handles the conversion to `[0, 1]` internally. |
| `chassis_xyyaw`, `previous_chassis_xyyaw` | Measured SDK local-axis accumulator values `[x_m, y_m, yaw_rad]` at the current and previous **20 Hz** state ticks. For a capture at 1.00 s, use the state at 0.95 s as the predecessor. Keep the SDK's accumulated yaw convention. The predecessor may be omitted on the first request, which makes the chassis delta zero. Omit both keys for arm-only inference. |
| `prompt`, `subtask` | Optional task and current subtask strings. In recorded replay, use the subtask active at the capture timestamp and use an empty string for gaps. The server builds the training JSON profile for a dual-gripper robot in the real world. When the text changes it refreshes the prefix and retains visual history. |

Training selected one primary camera stream per recording, in the order
`stereo_left`, left eye of `head_stereo`, then `head`. Use that same stream
throughout a recorded replay. Send camera frames in the orientation used
by the recordings. When replaying
HDF5, apply each RGB stream's `rotation_degrees` metadata once, after
selecting the left eye for packed stereo. The wire format has no orientation
metadata. Convert BGR streams to RGB and send the original pixel dimensions.
The server handles aspect-preserving resize, center crop to 256 × 256 and
C-RADIO preprocessing.
Training used resolution buckets and random image augmentation. Serving
uses a fixed, deterministic crop.

Use measured SDK poses and accumulator values, including while a command is
in flight. If deriving tool poses from joints, match the training tool links
and chassis origin. The training FK applied a +0.097 m Z correction when
starting from the no-wheel URDF's torso base. SDK-native tool poses already
use the requested origin.

### Plan coordinates and timing

| Reply field | Shape | Meaning |
|---|---|---|
| `eef_pose` | `[32, 2, 7]` | Absolute XYZ in metres and XYZW quaternions in the input controller basis, left then right |
| `gripper_percent` | `[32, 2]` | Absolute gripper targets, left then right |
| `chassis_xyyaw` | `[32, 3]` or `None` | Absolute targets in the SDK local accumulator, using the current observed chassis anchor |
| `timestamp_s` | scalar | Capture time of the observation used as the chunk anchor |

Row `i` corresponds to `timestamp_s + i / 20`. Row 0 is the command for the
capture tick. All rows are absolute targets. The server denormalizes with
the robot head's Astribot statistics, adds XYZ and chassis offsets to the
measured chunk-start state, and composes arm rotations as
`R_target = R_offset @ R_observed`. The client can pass these absolute
targets through without applying its own normalization, cumulative sum,
or chunk-start transform. Gripper commands retain the SDK's 0–100 scale.

### Compare against training

For open-loop tests, replay the recorded observations in order with a fixed
reset seed and the same `observe()`/`infer()` schedule. Compare the returned
absolute robot-head plan with recorded **commands** at the same 20 Hz ticks.
Use the label validity mask, including the absent chassis labels in legacy
recordings. The trainer's main sampled-action metrics use the checkpoint's
`eef_local` primary head. A same-head comparison should select `robot`
on both sides and use the same crop, flow noise and scoring ticks.
The clean-wheel test verified strict loading and finite CPU replay over
WebSocket. CUDA behavior and physical robot performance remain unverified.
