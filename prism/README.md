# Engram PRISM inference

This package serves the Engram hybrid PRISM policy without the private training
repository, training datasets, W&B, ACP tools, or optimizer state. The included
runtime is selected from the exact trained implementation; its source hashes
and release adaptations are recorded in [RUNTIME_SOURCES.md](RUNTIME_SOURCES.md).
The π0.5 code and environment in the parent repository remain independent.

[Validation status](qualification/RESULTS.md): 64 CPU tests and a complete
offline checkpoint restore pass. Full-model GPU parity remains pending; this
does not establish closed-loop robot performance.

## Start a server

Use Linux x86-64, Python 3.12.12, CUDA 12.8-compatible NVIDIA drivers and a GPU
with sufficient memory for the complete model. The first release retains all
checkpoint tensors, including dormant heads. Full-model GPU memory and latency
qualification are recorded separately from CPU adapter tests.

```bash
git clone --branch codex/prism-engram-inference https://github.com/Franklin-Zhang0/pi05-astribot.git
cd pi05-astribot/prism
uv sync --extra native-conv
uv run prism-serve --checkpoint /path/to/engram-prism-bundle --port 8000
```

`prism/uv.lock` pins the separate Torch/Transformers/KDA environment. Run these
commands from `prism`, rather than syncing the parent π0.5 environment. The
`native-conv` extra selects the causal convolution implementation used during
training; source builds may require a CUDA toolkit. Its build runs against the
project's pinned Torch using uv's [two-phase installation](https://docs.astral.sh/uv/concepts/projects/config/#disabling-build-isolation),
instead of resolving another Torch version in an isolated build. The graph can use the
upstream Torch fallback when that extra is omitted, but that is a different
kernel path and is not automatically covered by native-kernel GPU parity.

The bundle contains the weights, model/processor/tokenizer definitions,
normalization, physical layouts and a SHA-256 file manifest. Serving verifies
the bundle and reads its local assets; no Hugging Face login, private model
cache, training mount or network download is needed. Relocating a bundle does
not change its predictions. Default listening address is `127.0.0.1`.

## Request a prediction

The wire format matches `openpi_client.WebsocketClientPolicy`. Its stock
`reset()` is a no-op, so reset PRISM explicitly through an inference message:

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy

client = WebsocketClientPolicy(host="127.0.0.1", port=8000)
client.infer({"op": "reset", "seed": 0})
prediction = client.infer({
    "timestamp_s": 0.0,
    "robot_type": "dex_w1",
    "end_effector_type": "linker_l6",
    "prompt": "pick up the cup",
    "observation/head_rgbd": left_head_rgb,
    "observation/left_wrist_rgbd": left_wrist_rgb,   # optional
    "observation/right_wrist_rgbd": right_wrist_rgb, # optional
    "observation/joint_position": measured_joints_rad, # (20,)
    "observation/hand_position": measured_hands_percent, # optional (12,)
    "observation/ee_poses": measured_ee_poses, # (2,7): XYZ,QW,QX,QY,QZ
})
eef_targets = prediction["eef_actions"]           # (32,2,9): absolute XYZ + rotation6D
body_head_targets = prediction["body_head_actions"] # (32,6), radians
hand_targets = prediction.get("end_effector_actions") # (32,12), percent
```

The package also supplies `prism_engram.client.PrismClient`, whose `reset()`,
`infer()` and `observe()` send the corresponding messages. Use a separate
connection for each robot/episode stream. Every connection has its own
recurrent state and random generator; model execution is serialized.

Send observations at **10 Hz**, with the next `timestamp_s` equal to `0.1`,
then `0.2`, and so on. These are sensor timestamps, not the server's completion
times. The server admits 50–150 ms intervals to allow capture jitter. It rejects
gaps, repeated/reordered observations and prompt/embodiment changes until reset.
For a tick that should update history without decoding a new action chunk,
send the same observation fields with `"op": "observe"`. Do not resend the
same observation once per executed action frame. Reset on a new task, episode,
reconnect or observation gap. Frame zero is consumed once, in the first step.
Rolling attention retains seven preceding observations while KDA state carries
the complete episode history, matching the eight-observation training window.

RGB images must be HWC `uint8`. W1 head images use the left eye; pass
`"head_camera_layout": "stereo"` for a side-by-side stereo image. Galbot head
images use its left eye. Inputs are brought to the trained proxy geometry,
then receive the training loader's cover resize, deterministic center crop and
the center counterpart of the 90% crop augmentation. Color jitter is disabled.
Encoding uses checkpoint-owned C-RADIO and the saved evaluation resolution
256×256. Present wrists contribute dense patches and register tokens; absent
wrists keep their fixed camera slots and are explicitly masked. Live RGB cannot
reproduce lossy training-video compression bit-for-bit.

## Prompts and routes

Plain text is used exactly as supplied. Structured input uses the canonical
pretraining JSON schema, for example:

```python
"prompt": {
    "task instruction": "pick up the cup",
    "end effector type": "hand",
    "end effector number": "dual",
    "embodiment": "robot",
    "setting": "engram_dex_w1_linker_l6",
}
```

The server fills the route's fixed setup fields when omitted and rejects
conflicts. It also accepts the π0.5 convenience form containing `instruction`,
`robot_type`, and `end_effector_type`, then serializes the canonical PRISM
fields. Route fields are required even with a plain prompt. Do not randomly
switch prompt styles within an episode: the 50/50 mix is a training augmentation,
while inference keeps the selected instruction prefix stable. Complete prompts
must fit the saved 384-token budget; they are never silently truncated.

| Robot | End-effector | Measured joints | Optional measured hand state | Output |
|---|---|---:|---:|---|
| `dex_w1` | `gripper` | 20 | 2 percentages | EEF + body/head + available grippers |
| `dex_w1` | `linker_l6` | 20 | 12 percentages | EEF + body/head + available fingers |
| `dex_w1` | `linker_l20` | 20 | 32 signed percentages | EEF + body/head + available fingers |
| `galbot` | `inspire_rh56dfx_hand` | 14 | Unavailable | 14 arm joints |

Omit unavailable hand feedback. The server masks it and omits hand targets;
placeholder zeros must not be presented as measured feedback. Galbot finger
outputs are unavailable because its demonstrations have no finger labels.
W1 requires the current measured EEF poses of both arms. It does not turn EEF
predictions into arm joint targets; that requires a robot-specific controller
or IK outside this policy. There is no silent joint-only W1 fallback.

## Action coordinates, normalization and timing

All predictions cover **32 native frames**, nominally 30 Hz. The output never
exposes the model's padded 160×15 storage as executable actions.

- W1 input joint order: `ANKLE, KNEE, BUTTOCK, WAIST, NECK1, NECK2,
  LEFT_J1…LEFT_J7, RIGHT_J1…RIGHT_J7`, in radians. `body_head_actions[:,0:6]`
  returns the first six absolute targets. Galbot order is right-arm joints
  1–7, followed by left-arm joints 1–7.
- Hand outputs are left hand followed by right hand. Gripper/L6 units are
  percentages; L20 units are signed percentages. No 3D hand keypoints or
  fabricated finger radians are used.
- Native state/action normalization is per route using saved 1st/99th-percentile
  ranges, with at least 0.1 rad for joints and 5 percentage points for hands.
  Tails remain unclipped. Native outputs are deltas from the current measured
  anchor, which the server adds back exactly once.
- EEF normalization uses saved TRAIN mean/std statistics. It predicts
  `dp = p_future - p_current` and `dR = R_future @ R_current.T`. The server returns
  `p_current + dp` and `dR @ R_current`. Rotation6D is the **first two matrix
  rows flattened**, matching training. `eef_position`, `eef_rotation_matrix`
  and `eef_rotation_6d` are also returned. EEF order is left then right.
- W1 EEF coordinates are the recorded, inferred torso/arm-base-attached frame,
  in metres. Supply feedback in that same frame; no guessed camera/world/base
  calibration is applied. These are future feedback trajectories, not recorded
  EEF command labels. Body motion is separately supervised by native targets.
- Joint/hand command rows correspond to native ticks **0…31** from the anchor;
  EEF feedback rows correspond to ticks **1…32**. The response includes both
  tick-offset arrays. Controllers must preserve this distinction.

Outputs are raw model predictions. The server returns targets and does not
execute commands, solve IK or clip predictions into hardware limits.

## Export a checkpoint (model owner)

Run the exporter on a machine that has a complete training checkpoint and its
pinned cached tokenizer/architecture assets:

```bash
uv run prism-export --checkpoint /path/to/checkpoint-10000 \
    --output /path/to/engram-prism-bundle \
    --qwen-assets /path/to/pinned-qwen-snapshot \
    --cradio-assets /path/to/pinned-cradio-snapshot \
    --fast-assets /path/to/fast-tokenizer
```

The exporter checks checkpoint completion metadata, copies only final model
weights plus inference assets, removes dataset catalogs/private mount paths,
and publishes its manifest last. It refuses to overwrite an existing directory.
Do not distribute the full training checkpoint or its optimizer files. The
bundle loader restores every state tensor strictly, preserves the checkpoint's
mixed BF16/FP32 parameter storage, and uses BF16 CUDA autocast without globally
rounding FP32 parameters. It preserves the trained embedding/output weight
sharing and constructs parameter metadata before memory-mapped restoration.
Base Qwen/C-RADIO weights are already in the final
checkpoint and are not downloaded or loaded a second time.

## Verification

```bash
uv run pytest tests qualification/test_compare.py
```

CPU tests cover normalization and exact anchors, all four routes, canonical
prompts, missing cameras/hands, rotation composition, bundle integrity, strict
mixed-dtype loading, session isolation, resets and OpenPI wire compatibility.
Full checkpoint comparison uses [qualification/capture.py](qualification/capture.py)
and [qualification/compare.py](qualification/compare.py) in separate processes
for original and standalone graphs, with identical inputs and explicit noise.
It covers history across the rolling window and reset replay. GPU parity is a
separate qualification from these CPU tests and does not establish robot policy
quality. See `qualification/README.md` for the recorded scope and commands.
