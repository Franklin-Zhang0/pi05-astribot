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

Use `PrismClient.observe(observation)` to update camera history without producing
actions. It sends `"op": "observe"` and returns an acknowledgment with
`"observed": True`. Use `PrismClient.infer(observation)` to get an action chunk.
The `decode=False` keyword belongs to the server's internal
`CoffeeSession.infer()` method. The server sets it when it receives the
`observe` operation.

Feed the loop one fresh sample every **100 ms**. In this example,
`sensor_stream` yields the capture timestamp in seconds, three RGB images in main,
left-wrist, right-wrist order, the measured state, and whether to request a
new chunk on that tick. Use HWC `uint8` images, an unnormalized 32-channel
state in Coffee training order, and fixed task text for the episode.

```python
from prism_engram.client import PrismClient

with PrismClient(host="127.0.0.1", port=8000) as client:
    client.reset(seed=0)

    for timestamp_s, (main_rgb, left_wrist_rgb, right_wrist_rgb), state, request_chunk in sensor_stream:
        observation = {
            "timestamp_s": timestamp_s,
            "prompt": task_text,
            "observation.images.kfc_compressed": main_rgb,
            "observation.images.camera_l": left_wrist_rgb,
            "observation.images.camera_r": right_wrist_rgb,
            "observation.state": state,
        }
        if request_chunk:
            actions = client.infer(observation)["actions"]  # (32 time steps, 32 channels)
            submit_plan(actions)  # Hand off to the controller without waiting for playback.
        else:
            client.observe(observation)  # Update history while the current plan executes.
```

A full action chunk lasts about **1.07 seconds**. Keep sending observations
during playback to stay within the server's 50–150 ms interval between capture
timestamps. The client calls are synchronous, so keep robot command playback
independent of the request loop.

The server pairs the previous camera frames with the current frames and builds
the model's positions for that two-frame window. After `reset()`, it uses the
first frames for both slots. Reset the inference session at episode start,
after a gap or reconnect, or when the task changes. This reset clears the
server's observation history.

`actions[i]` is the 32-channel command for time step `i`. The server
denormalizes it into the training data's original channel order and units.
Use the Coffee feature metadata to map those channels to the controller.
