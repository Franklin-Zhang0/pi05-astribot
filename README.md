# Shanghai Coffee PRISM inference

Serve the Shanghai Coffee `step-020000.pt` checkpoint and stream observations
to it over WebSocket. Each prediction returns 32 action rows at 30 Hz.

## Prerequisites

- A Linux x86-64 machine with Python 3.12.12, `uv`, and an NVIDIA GPU with a
  CUDA 12.8-compatible driver. Building the native convolution extension may
  require the CUDA toolkit.
- The checkpoint-compatible `UVA_Finetune` checkout and the Coffee
  `step-020000.pt` adapter.
- The original PRISM `checkpoint-50000` directory with `config.json`,
  `model.pt`, `dmuon_master.pt`, and `fsdp2_dmuon_manifest.json`.
- The local Qwen3.5-2B-Base snapshot and a Hugging Face cache containing the
  pinned C-RADIOv4-SO400M snapshot used for training.

## Start the server

```bash
git clone --branch codex/prism-coffee-inference --single-branch \
  https://github.com/Franklin-Zhang0/pi05-astribot.git
cd pi05-astribot/prism
uv sync --locked --extra native-conv

HF_HUB_OFFLINE=1 HF_HUB_CACHE=/path/to/huggingface/hub \
  uv run --no-sync prism-coffee-serve \
    --fine-tune-dir /path/to/UVA_Finetune \
    --base-checkpoint /path/to/checkpoint-50000 \
    --checkpoint /path/to/step-020000.pt \
    --qwen-assets /path/to/Qwen3.5-2B-Base \
    --device cuda:0 --port 8000
```

The server listens on `127.0.0.1` by default. Use `--host` to set a different
bind address.

## Stream observations and get actions

Build each request from the latest measured state, three HWC `uint8` RGB images,
and the Coffee task text. Keep the task text fixed within an episode. Use the
sensor capture time in seconds for `timestamp_s`.

```python
from prism_engram.client import PrismClient

observation = {
    "timestamp_s": capture_time_s,
    "prompt": task_text,
    "observation.images.kfc_compressed": main_rgb,
    "observation.images.camera_l": left_wrist_rgb,
    "observation.images.camera_r": right_wrist_rgb,
    "observation.state": measured_state,  # 32 values in Coffee training order
}

with PrismClient(host="127.0.0.1", port=8000) as client:
    client.reset(seed=0)
    actions = client.infer(observation)["actions"]  # (32 time steps, 32 channels)
```

In deployment, keep that connection open and send one fresh observation every
**100 ms**. Call `infer()` on ticks where you need a new plan and `observe()` on
the other ticks, including while the current plan is executing. A full action
chunk lasts about **1.07 seconds**. Waiting until it ends to send another
observation exceeds the server's 50–150 ms interval. The client calls are
synchronous, so keep robot command playback independent of the request loop.

The server pairs the previous camera frames with the current frames and builds
the model's positions for that two-frame window. After `reset()`, it uses the
first frames for both slots. Reset the inference session at episode start,
after a gap or reconnect, or when the task changes. This reset clears the
server's observation history.

`actions[i]` is the 32-channel command for time step `i`. The server
denormalizes it into the training data's original channel order and units.
Use the Coffee feature metadata to map those channels to the controller.
