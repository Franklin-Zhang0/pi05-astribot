# Shanghai Coffee PRISM inference

This server supports the 32-channel Shanghai Coffee LeRobot adapters. Use
`step-020000.pt` for the final weights or `step-010000.pt` for the
intermediate weights.

## Start the server

Use the checkpoint-compatible `UVA_Finetune` training checkout and the original
PRISM `checkpoint-50000` directory. The base directory needs its original
`config.json`, `model.pt`, `dmuon_master.pt`, and
`fsdp2_dmuon_manifest.json`. Use a local Qwen3.5-2B-Base snapshot and a
Hugging Face cache containing the pinned C-RADIO snapshot. The LeRobot data is
only needed to replay recorded observations.

On an allocated CUDA node, run from the `prism` directory of this branch.
Replace the paths with your local model assets and Coffee training checkout.

```bash
uv sync --locked --extra native-conv

HF_HUB_OFFLINE=1 HF_HUB_CACHE=/path/to/huggingface/hub \
uv run --no-sync prism-coffee-serve \
  --fine-tune-dir /path/to/UVA_Finetune \
  --base-checkpoint /path/to/checkpoint-50000 \
  --checkpoint /path/to/step-020000.pt \
  --qwen-assets /path/to/Qwen3.5-2B-Base --device cuda:0 --port 8000
```

The server calls the Coffee training policy and its original image resize and
prompt formatter. It reads the state and action mean/std from the adapter.
The CLI uses the package's OpenPI-compatible WebSocket transport.

## Send observations

Supply the three RGB images as HWC `uint8` arrays and the unnormalized
32-channel state in the same ordering and units as the Coffee training data.
Use the original task text. Send a new observation at 10 Hz. The first
request after reset repeats its images to fill the two-observation window.
Call `observe()` to update that window without sampling actions, or `infer()`
to update it and sample. Reset after a gap, reconnect, or task change.

```python
from prism_engram.client import PrismClient

with PrismClient(port=8000) as client:
    client.reset(seed=0)
    result = client.infer({
        "timestamp_s": 0.0,
        "prompt": task_text,
        "observation.images.kfc_compressed": main_rgb,
        "observation.images.camera_l": left_wrist_rgb,
        "observation.images.camera_r": right_wrist_rgb,
        "observation.state": measured_state,
    })
    actions = result["actions"]  # (32, 32), consecutive frames at 30 Hz
```

Actions are denormalized using the Coffee training statistics. Their channels
retain the Coffee dataset's action order and units. Use the Coffee dataset
metadata when mapping them to a robot controller. The server only returns
predictions.
