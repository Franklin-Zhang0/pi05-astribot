# Astribot mobile PRISM inference

Serve the full Astribot mobile fine-tune (`checkpoint-20000`, run ending in
`20261125-r2`) over OpenPI WebSocket. The server uses the checkpoint's
robot-frame flow head and Astribot TRAIN statistics for both arms, grippers
and the mobile base.

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

Send one capture at **10 Hz**, including while a previous chunk is in use.
`observe()` advances recurrent state without sampling. `infer()` advances
the same state and returns a **32-step plan at 20 Hz** (1.6 seconds). Start
with `reset()` at each episode boundary, after reconnecting, or after a
timestamp gap. Current task and subtask strings become the checkpoint's JSON
prefix. A changed prefix updates in place while visual history is retained.

```python
from prism_engram.client import PrismClient

with PrismClient() as client:
    client.reset(seed=0)
    # sample contains the current capture and measured robot state.
    observation = {
        "timestamp_s": sample.timestamp_s,
        "stereo_left": sample.left_eye_rgb,        # HWC uint8 RGB
        "head_rgb": sample.head_rgb,               # fallback if left stereo is absent
        "left_wrist": sample.left_wrist_rgb,       # optional
        "right_wrist": sample.right_wrist_rgb,     # optional
        "eef_pose": sample.eef_pose,               # [left, right] × [x,y,z,qx,qy,qz,qw]
        "gripper_percent": sample.gripper_percent, # [left, right], 0–100
        "chassis_xyyaw": sample.chassis_xyyaw,     # SDK local accumulator [m,m,rad]
        "previous_chassis_xyyaw": sample.previous_chassis_xyyaw,
        "subtask": "Put the cup in the box.",
    }
    if request_chunk:
        plan = client.infer(observation)
    else:
        client.observe(observation)
```

Use the same chassis-relative controller basis as the recordings for EEF
poses. Supply the **left eye** as a single, correctly oriented RGB image.
Leave a missing camera absent or `None`. `previous_chassis_xyyaw` comes
from the previous **20 Hz** state tick (50 ms earlier), and may be omitted
on the first observation. Omit both chassis keys for arm-only inference.
Use `prompt` for an optional task-level instruction and `subtask` for
the current subtask. An empty subtask matches training's empty subtask field.

| Reply field | Shape | Meaning |
|---|---|---|
| `eef_pose` | `[32, 2, 7]` | Absolute XYZ in metres and XYZW quaternions in the input controller basis, left then right |
| `gripper_percent` | `[32, 2]` | Absolute gripper targets, left then right |
| `chassis_xyyaw` | `[32, 3]` or `None` | Absolute targets in the SDK local accumulator, using the current observed chassis anchor |
| `timestamp_s` | scalar | Capture time of the observation used as the chunk anchor |

The server only returns predictions. Chassis quality was not established
by the fine-tune's held-out metrics. Inspect replayed output before
connecting the planner to a robot controller.
