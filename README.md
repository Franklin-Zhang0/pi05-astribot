# Engram π0.5 and PRISM inference

## Standalone PRISM inference

This branch also contains an isolated [PRISM inference package](prism/README.md)
for the Engram hybrid EEF/joint/hand checkpoints. It uses OpenPI's WebSocket
wire format, with explicit episode resets and one recurrent cache per client.
It requires an exported PRISM bundle and an NVIDIA GPU; it does not require
the training repository or dataset mounts. Its environment is separate from
the π0.5 environment below.

```bash
cd prism
uv sync --extra native-conv
uv run prism-serve --checkpoint /path/to/engram-prism-bundle --port 8000
```

PRISM's 32-frame W1 output contains absolute EEF trajectories, body/head joint
targets and available hand targets. Galbot returns 14 arm joint targets.
See the [exact inputs, action split and reset protocol](prism/README.md) before
connecting a robot controller. Existing π0.5 commands and outputs remain below.

## Engram multi-embodiment π0.5 fine-tuning and inference

The `pi05_engram_multi` recipe trains Dex W1 gripper, Linker L6 and Linker L20 demonstrations alongside Galbot demonstrations. It starts from `pi05_base`, uses a shared diffusion transformer with separate input and output projections for each robot/end-effector route, and trains with the head camera plus wrist cameras wherever recorded. The earlier 22-channel recipe and server remain available for existing checkpoints; their instructions are below.

## Multi-embodiment recipe, 2026-11-12

The pinned [dataset manifest](examples/engram/kitchen_multi_20261112.json) selects complete successful episodes from `kitchen_new`, excludes partial and failed demonstrations, checks duplicate annotation windows, and holds out recording packages across every source. It preserves the earlier validation packages and extends the split to approximately 5% of packages. The audit selects:

| Robot / end effector | Train episodes | Validation episodes | Total |
| --- | ---: | ---: | ---: |
| Dex W1 / gripper | 670 | 35 | 705 |
| Dex W1 / Linker L6 | 712 | 38 | 750 |
| Dex W1 / Linker L20 | 36 | 2 | 38 |
| Galbot / Inspire RH56DFX hand | 179 | 9 | 188 |
| **Total** | **1,597** | **84** | **1,681** |

These selected episodes contain approximately **25.49 hours** of demonstrations before sample-level motion filtering.

Some W1 exports lack end-effector commands and feedback; the Galbot export lacks all finger labels. They still contribute joint supervision. Within labeled exports, a missing hand measurement or command/mode anywhere in a sample's future horizon disables hand supervision for that sample while retaining valid joint supervision. Missing hand state and loss channels are masked; no hand commands are inferred as labels. Galbot inference therefore returns arm targets only. The old 113-episode export with incomplete provenance stays excluded.

### Inputs, prompts and actions

Each sample contains RGB images, current measured joints and available hand feedback, an explicit numeric embodiment route, an availability mask, and a task instruction. Each of the four routes has independent projections into and out of the shared diffusion transformer. The route is supplied numerically even for plain prompts. Half of training samples use the original instruction text; half use canonical JSON, for example:

```json
{"end_effector_state_available":true,"end_effector_type":"linker_l6","instruction":"pick up the cup","robot_type":"dex_w1"}
```

JSON keys and instruction text are preserved during tokenization. Plain prompts with missing end-effector state receive `End-effector state unavailable. ` before the instruction. JSON prompts mark the same condition with `false`. The 352-token budget includes the discretized 64-channel state; overlong inputs raise an error rather than silently truncating.

The model uses a **64-step, 64-channel packed action**. Only the physical channels for the selected route receive loss. All joint targets are deltas from the **current measured joint state**, shared by the whole 64-step horizon. Hand targets are absolute positions; they are never joint deltas.

| Route | Joint channels (radians) | Hand channels | Unused channels |
| --- | --- | --- | --- |
| W1 / gripper | `0:20` | `20:22`, left then right, percent `[0,100]` | `22:64` |
| W1 / Linker L6 | `0:20` | `20:32`, 6 left then 6 right, percent `[0,100]` | `32:64` |
| W1 / Linker L20 | `0:20` | `20:52`, 16 left then 16 right, signed percent `[-100,100]` | `52:64` |
| Galbot / Inspire | `0:14` | `20:32` reserved, **unlabeled and masked** | `14:20`, `32:64` |

W1 joint order is `ANKLE, KNEE, BUTTOCK, WAIST, NECK1, NECK2, LEFT_J1–LEFT_J7, RIGHT_J1–RIGHT_J7`. Galbot order is `right_arm_joint1–7, left_arm_joint1–7`. Hand channel order follows the recorded command arrays within each side. Padded and unavailable channels are excluded from loss. Route-specific statistics use training anchors only, with at least a 0.1-radian joint quantile span and a 5-percentage-point hand span. Samples are dropped when **all** future joint movements are below 0.01 rad and there is no recorded hand movement. Available hand-only movements remain eligible.

Training uses the W1 stereo camera's **left eye**, resized to 640×360; Galbot uses its left head camera at 480×360, preserving its 4:3 aspect ratio. W1 wrists stay 640×480 and Galbot wrists stay 640×360. The final model transform letterboxes every camera to 224×224. Absent wrist cameras are zero-filled with attention masks. Only these needed camera streams are preprocessed. Proxies are adjacent `*_640.mp4` files with a maximum four-frame GOP; preprocessing verifies original timestamps, frame counts, geometry and source/proxy hashes and saves sidecars. Original videos remain unchanged.

Training starts without repeating video checksum scans or source provenance audits. It uses the prepared sidecars and normalization assets, retaining lightweight schema and training-contract checks. This fast startup is the default for future runs; the approximately 5% validation split is unchanged. For an explicit dataset audit, add `--data.verify-proxy-hash --data.audit-source-metadata` to the training command.

The multi-embodiment recipe caches its prepared annotations in `/tmp/openpi-engram-prepared-index`. The first launch on a node builds the index once; subsequent launches reuse it. Spawned workers open the same read-only arrays instead of receiving full dataset copies. The cache preserves the original frame timelines, action chunks and masks. A changed source manifest, loader implementation or sampling setting selects a new cache. Use `--data.prepared-index-dir /path/on/local/disk` to choose another local directory, or `--data.rebuild-prepared-index` after changing annotations in place. Explicit integrity audits also rebuild the cache. Shared AOSS, FUSE and network mounts cannot host the memory-mapped index.

```bash
# From the repository root, after uv sync:
uv run examples/engram/preprocess_multi.py examples/engram/kitchen_multi_20261112.json \
  --adjacent --split all --decoder nvdec --cuda-devices 0,1,2,3 --workers 8 \
  --receipt /path/to/run/preprocess.json
export ENGRAM_ASSETS_DIR=/path/to/run/assets
bash examples/engram/finetune_multi_pi05.sh stats --batch-size 256 --num-workers 0
bash examples/engram/finetune_multi_pi05.sh train my-engram-multi \
  --batch-size 128 --num-train-steps 30000 --save-interval 10000 --keep-period 10000 \
  --fsdp-devices 4 --gradient-accumulation-steps 2 --checkpoint-completed-steps \
  --checkpoint-dir-override /hand_keypoint_data/engram_data/ckpt/pi05/DATE_EXPNAME
```

The default global batch is 128: two accumulated global microbatches of 64 samples, 16 samples per GPU on four GPUs. Statistics require the full retained training split; sampled diagnostic statistics are rejected by production training and serving. Validation packages never contribute statistics or training samples. The checkpoint includes its exact statistics, source provenance, route/camera/prompt contract and original training instruction list (`assets/engram/multi_embodiment_train/prompt.json`). Resume rejects changed contracts.

### Multi-embodiment inference

Use the new server for `pi05_engram_multi` checkpoints. It reads checkpoint-owned assets and does not need the training dataset mounted. Set `OPENPI_DATA_HOME` to a cache containing `big_vision/paligemma_tokenizer.model`.

```bash
uv run examples/engram/serve_multi_pi05.py \
  --checkpoint-dir /hand_keypoint_data/engram_data/ckpt/pi05/DATE_EXPNAME/30000 \
  --host 127.0.0.1 --port 8000 --head-layout left
```

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy

client = WebsocketClientPolicy(host="127.0.0.1", port=8000)
prediction = client.infer({
    "observation/head_rgbd": left_head_rgb,        # HWC RGB uint8
    "observation/left_wrist_rgbd": left_wrist_rgb, # optional
    "observation/right_wrist_rgbd": right_wrist_rgb, # optional
    "observation/joint_position": measured_joints_rad, # shape (20,) for W1
    "observation/hand_position": measured_hand_percent, # optional; (12,) for L6
    "prompt": {
        "instruction": "pick up the cup",
        "robot_type": "dex_w1",
        "end_effector_type": "linker_l6",
        "end_effector_state_available": True,
    },
})
joint_targets = prediction["joint_actions"]
hand_targets = prediction.get("end_effector_actions")
```

A plain `"prompt": "pick up the cup"` is also supported; provide `robot_type` and `end_effector_type` as top-level fields. Canonical end-effector names are `gripper`, `linker_l6`, `linker_l20`, and `inspire_rh56dfx_hand`; robot names are `dex_w1` and `galbot`. Set `head_camera_layout="stereo"` only for a W1 side-by-side image. Galbot's head input is its left eye. Camera geometry is checked against training preprocessing.

Omit hand feedback when unavailable, or explicitly set `end_effector_state_available=false`. Do not pass placeholder hand values as measured feedback. Hand feedback can alternatively be appended to the joint array. The new interface derives its mask from these fields and rejects caller-supplied `action_mask`. Galbot accepts 14 arm joints and no hand feedback because its hand channels have no training labels.

The server unnormalizes route-specific predictions and adds the current measured joint anchor exactly once, returning **absolute targets**. `joint_actions` is `(64,20)` for W1 or `(64,14)` for Galbot. With measured and supervised hand feedback, `end_effector_actions` is `(64,2)`, `(64,12)` or `(64,32)` for gripper, L6 or L20. `actions` concatenates physical joints and available hands, giving `(64,22)`, `(64,32)`, `(64,52)`, or joint-only shapes. It never includes packed padding. Outputs are raw, unclipped predictions at nominal 30 Hz; the server does not execute robot commands.

## Earlier 22-channel gripper recipe

The following sections describe the 2026-11-01 checkpoint and its legacy `serve_pi05.py` server. That recipe used 820 complete episodes, one head camera and a padded 32-channel model head. It is separate from the four-route recipe above.

## Setup

This branch builds on [OpenPI](https://github.com/Physical-Intelligence/openpi) and uses `uv` with its committed lockfile. From a machine with access to the repository:

```bash
git clone --branch engram-finetuning git@github.com:Franklin-Zhang0/pi05-astribot.git engram-pi05
cd engram-pi05
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

Training requires the π0.5 base checkpoint, the selected Engram data, and suitable GPU resources. The current recipe was configured for two H100 GPUs. The default base checkpoint is `gs://openpi-assets/checkpoints/pi05_base/params`; an existing OpenPI cache can also be used. Serving requires the PaliGemma tokenizer in the OpenPI cache and a completed Engram checkpoint. Video preprocessing requires FFmpeg and its probing tools. CPU tests use small diagnostic models and do not establish trained-policy quality.

## Action split and horizon

The physical action uses the same zero-based ordering as the measured input state:

| Indices | Dimensions | Body part | Channels |
| --- | ---: | --- | --- |
| 0–3 | 4 | Body | `ANKLE`, `KNEE`, `BUTTOCK`, `WAIST` |
| 4–5 | 2 | Neck | `NECK1`, `NECK2` |
| 6–12 | 7 | Left arm | `LEFT_J1`–`LEFT_J7` |
| 13–19 | 7 | Right arm | `RIGHT_J1`–`RIGHT_J7` |
| 20 | 1 | Left gripper | Absolute position in percent, when available |
| 21 | 1 | Right gripper | Absolute position in percent, when available |

The first 20 channels are joint positions in radians. Before noise augmentation and normalization, the training target at each step `h` is `joint_command[t+h] - measured_joint_position[t]`; all 64 steps use the **same current observation** as their reference. Gripper targets are absolute positions in percent and are supervised only when their labels are present. Joint noise augmentation compensates the joint delta so that it preserves the original absolute target.

OpenPI pads the 22 physical slots with 10 zero-valued dimensions at indices 22–31 for the model's 32-dimensional action head. These padding dimensions are not robot actions and are removed from the served result. When gripper feedback is unavailable, the two gripper slots are masked during training and omitted from the server response; the hand-only exports do not provide finger-action labels.

The server unnormalizes the predictions and adds the current measured joint positions back to the joint deltas. Thus, `prediction["actions"]` contains **absolute** targets over a 64-step horizon at nominal 30 Hz:

- Shape `(64, 22)` when both grippers are available: `[:, :20]` contains joint targets, `[:, 20]` the left gripper, and `[:, 21]` the right gripper.
- Shape `(64, 20)` when gripper feedback is unavailable: `[:, :20]` contains the joint targets and there are no gripper columns.

The server returns raw predictions without clipping; the 20 joint outputs are radians and gripper outputs are percentages. It does not execute the targets.

## Serve a completed checkpoint

The Engram server uses OpenPI's existing WebSocket policy protocol. It reads a completed checkpoint's parameters, normalization statistics and provenance, and checks the saved statistics hash and strict 640-pixel training contract. A completed checkpoint may be copied to a directory with a nonnumeric name. It does not open the training dataset or require its original mount paths. The server imposes the active run's single-camera configuration; the checkpoint's saved video contract does not itself record that flag. The PaliGemma tokenizer must be available at `$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model` (or `~/.cache/openpi/big_vision/paligemma_tokenizer.model` if the variable is unset).

From the repository root, after the 10,000-step checkpoint has completed:

```bash
export OPENPI_DATA_HOME=/path/to/openpi_cache  # contains big_vision/paligemma_tokenizer.model
uv run examples/engram/serve_pi05.py \
  --checkpoint-dir /hand_keypoint_data/engram_data/ckpt/pi05/20261101_engram_pi05_strict_640g4_b256_2h100/10000 \
  --host 127.0.0.1 --port 8000 --camera-layout left
```

The `--camera-layout` applies to every request: `left` accepts an already-cropped left-eye RGB image; `stereo` selects the left half of a side-by-side stereo RGB image. The selected eye must be 16:9. Images must be HWC `uint8` RGB or floating RGB in `[0, 1]`; the server area-resizes the eye to 640×360, then the unchanged OpenPI pipeline letterboxes it to 224×224. Existing 640×360 left frames are not cropped or rescaled again. Training proxies also underwent YUV conversion and lossy H.264 encoding, so live RGB preprocessing is not bit-identical to those videos.

The state is ordered as **ANKLE, KNEE, BUTTOCK, WAIST, NECK1, NECK2, LEFT_J1–LEFT_J7, RIGHT_J1–RIGHT_J7**, in radians. Supply 20 measured joint positions, or append measured left and right gripper positions in percent `[0, 100]`. A 20D state marks end-effector feedback unavailable. For 22D state, an omitted `action_mask` means both grippers are measured; to use the 22D placeholder layout from the joint-only dataset, supply `action_mask=[True]*20+[False]*2`. Both end-effector mask entries must agree. The server supplies the unavailable marker and masked normalized state; provide the original task prompt without adding that marker yourself. It rejects requests if OpenPI's exact prompt-plus-state token sequence would exceed the model's 200-token limit.

```python
import numpy as np
from PIL import Image
from openpi_client.websocket_client_policy import WebsocketClientPolicy

client = WebsocketClientPolicy(host="127.0.0.1", port=8000)
observation = {
    "observation/head_rgbd": np.asarray(Image.open("left_rgb.png").convert("RGB")),
    "observation/joint_position": np.load("measured_joints_rad.npy"),  # shape (20,)
    "prompt": "dual gripperopen bread bag and take out bread slice",
}
prediction = client.infer(observation)
print(prediction["actions"].shape, prediction["gripper_available"])  # (64, 20), False
```

The example prompt is a task string from training; a caller can provide another task instruction. The returned `actions` are **predicted absolute targets** at the nominal 30 Hz, for a 64-step horizon. They contain 20 joints when end-effector feedback is unavailable, or 20 joints plus two absolute gripper percentages when it is available. These are raw, unbounded predictions: values are not clipped, so predicted gripper percentages can fall outside `[0, 100]`. The policy does not infer finger commands from the unlabeled hand videos; do not represent a dexterous hand as two gripper values. This server returns predictions and does not execute robot commands.

## Mixed-end-effector snapshot, 2026-11-01

The pinned [multi-source manifest](examples/engram/kitchen_new_20261101.json) combines the original 450 fully labeled gripper episodes with 110 `st026` and 260 `st031` full, successful episodes. `st026` uses grippers and `st031` shows dexterous hands; neither new export records any end-effector feedback or commands. The previous 113-episode incomplete source and partial fragments stay excluded. The validation split preserves the original 23 held-out episodes and adds 19 from the new sources, for 778 training and 42 validation episodes. Package IDs are held out across all sources, and duplicate annotations, ambiguous mappings, missing packages, unexpected schema changes and episode counts fail closed.

All sources supervise the same 20 joint deltas. For the 370 episodes without end-effector annotations, a prompt marker explicitly says `End-effector state unavailable`, neutral normalized state tokens occupy the two unavailable gripper slots, and their flow inputs are pure noise and their output loss is masked. No finger labels are invented. For these joint-only episodes, the training filter drops chunks when all joint target movements across the horizon are below 0.01 rad. Their end-effector activity is unobserved, so this can also drop gripper-only or finger-only activity. Labeled gripper episodes keep the original joint-or-gripper motion pruning rule. Gripper normalization statistics use only labeled training anchors; joint statistics use all retained training anchors. This filter changes the statistics contract: recompute full training statistics before training. The full stats pass includes the final partial batch and saves sample counts, the source snapshot, and a hash in the statistics manifest.

The commands below use the current run’s single-camera 640×360 video mode and require completed adjacent proxies for the corresponding training and validation videos. See [Optional 640-pixel video derivatives](#optional-640-pixel-video-derivatives).

For an effective global batch of 256 with two-way FSDP, `--gradient-accumulation-steps 32` performs 32 global microbatches of eight (four samples per device) per optimizer update. This reduces activation memory but has not by itself established that a full-size model fits or performs well on a specific H100 allocation. Set `--checkpoint-completed-steps` to label checkpoint directories by completed optimizer updates; the default keeps legacy OpenPI naming.

```bash
export ENGRAM_DATA_ROOT="$PWD/examples/engram/kitchen_new_20261101.json"
export ENGRAM_ASSETS_DIR=/path/to/run/assets
examples/engram/finetune_pi05.sh stats --batch-size 256 --num-workers 0
examples/engram/finetune_pi05.sh train my-mixed-engram-run \
  --batch-size 256 --num-train-steps 30000 --save-interval 10000 --keep-period 10000 \
  --fsdp-devices 2 --gradient-accumulation-steps 32 --checkpoint-completed-steps \
  --data.single-camera --data.video-mode proxy_640 \
  --checkpoint-dir-override /path/to/exact/run/checkpoint/directory
uv run examples/engram/evaluate_pi05.py --data-root "$ENGRAM_DATA_ROOT" \
  --single-camera --video-mode proxy_640 \
  --checkpoint-dir /path/to/exact/run/checkpoint/directory/30000
```

The held-out evaluator reports joint errors across all chosen samples and gripper errors only for samples with actual gripper labels. The mixed policy does not acquire direct finger-control supervision from the hand videos.

To match a single-camera 640-pixel proxy training run, prepare timestamp-validated proxies and sidecars for the **validation** videos as well, then pass `--single-camera --video-mode proxy_640` to the evaluator. By default it reads adjacent `*_640.mp4` files; add `--proxy-root /path/to/proxies` for derivatives stored separately. Missing or incompatible validation proxies fail closed. Without these options the evaluator retains its original-video input and three image slots with absent wrist cameras masked. The reported result records video mode, proxy identity, camera mode and checkpoint normalization hashes; checkpoint training statistics are used without training-time observation augmentation.

### Optional 640-pixel video derivatives

`examples/engram/preprocess_640.py` creates opt-in `*_640.mp4` videos from the left eye of each selected full-success training recording. The 1920×1080 eye is area-resized to 640×360 without padding. Full-range H.264 CRF 18 with a maximum four-frame GOP is lossy: model inputs change slightly. Every derivative is validated against **every original presentation timestamp**, the per-episode annotated timestamps (100 µs tolerance), frame count, full-range metadata, and GOP length before publication. Each has a JSON sidecar containing source and proxy SHA-256. Encoding and MP4 faststart occur in local scratch; publication is exclusive and the sidecar is written last. Completed outputs are reused only after hash verification. An interrupted/orphan output is never overwritten automatically.

Default encoding uses CPU FFmpeg/libx264. `--decoder nvdec --cuda-device 0` explicitly requires CUDA decoding with no software fallback; PyAV transfers decoded frames to host memory, so crop/scale and libx264 encoding remain CPU operations. Inspect CPU-vs-GPU RGB/color-range results with `probe_video_decode.py` before using NVDEC. The source streams must be full range, and the decoder restores that metadata after CUDA transfer.

For a bounded concurrency trial, pass `--cuda-devices 0,1 --workers 4 --workers-per-gpu 2`, or use `--workers 8 --workers-per-gpu 4`. Videos are assigned round-robin, and the per-device limit defaults to one and cannot exceed eight. Each process still encodes on CPU, so benchmark throughput and CPU pressure before scaling up.

```bash
# Test one complete train video from each source without changing source data.
uv run examples/engram/preprocess_640.py examples/engram/kitchen_new_20261101.json \
  --output-root /tmp/engram-proxies --scratch-root /tmp/engram-encoding \
  --split train --limit-per-source 1 --shortest-first --workers 2

# After qualification, --adjacent replaces --output-root and writes *_640.mp4
# beside the original recordings. Omit --limit-per-source for all 778 train videos.

# Use the exact same completed-proxy episodes/anchors to compare loader/trainer speed:
examples/engram/finetune_pi05.sh train proxy-smoke --data.video-mode proxy_640 \
  --data.proxy-root /tmp/engram-proxies --data.proxy-subset-only
examples/engram/finetune_pi05.sh train original-smoke --data.video-mode original \
  --data.proxy-root /tmp/engram-proxies --data.proxy-subset-only
```

For production proxies beside originals, use `--data.video-mode proxy_640` without `--data.proxy-root` or `--data.proxy-subset-only`. The reader verifies sidecar format, source size, proxy SHA-256 and episode mapping, and never silently falls back to original video. It reads the left-view derivatives directly, without cropping them a second time. The source SHA-256 is verified on builder reuse; the reader checks source size and trusted sidecar identity, rather than rehashing every 4K source at each startup. Checkpoint assets pin the selected proxy identities and prevent resume with a different video mode or profiling subset. Validation videos are not built by the default train-only pass; use the default original-video mode for evaluation, or explicitly preprocess the validation split.

`examples/engram/benchmark_training.py` performs real pretrained π0.5 updates for the subset with completed proxies, using batch 256 on two H100s and saving no checkpoints. Invoke it twice with `--mode original` and `--mode proxy_640`, respectively, using the same seed and other options. Omit `--proxy-root` to compare adjacent proxies on their actual storage. After all training proxies have been created, add `--full-corpus` to sample across the entire training split; the benchmark prints actual per-source episode and anchor counts. The reported loader wait follows each completed GPU step and allows worker prefetch during that step; explicit synchronization makes this diagnostic different from a fully asynchronous end-to-end timing.

`--data.single-camera` is an Engram-only opt-in that removes the two permanently masked wrist-image streams from π0.5's vision and transformer inputs. It does not change the physical state/action data or the statistics. This mode rejects samples that actually provide wrist images; omit the flag to keep the existing three-slot policy interface. The CPU smoke compares losses and gradients against the ordinary masked-wrist path.

## Original single-export recipe

This recipe starts from `pi05_base`, uses a 64-step action horizon, and trains on the full successful episodes of the gripper-equipped export. The source dataset is read-only; the 113-episode export without gripper commands and partial fragments are not included. It uses the OpenPI-pinned LeRobot v2 environment with a local v3 reader rather than attempting to open v3 metadata with the v2 loader.

For implementation background, see the [upstream OpenPI training configuration](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py) and [LeRobot v3 format documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3). This OpenPI repository is distributed under its [Apache-2.0 license](LICENSE).

The default split holds out entire source recording packages. The train split has 427 episodes, and the validation split has 23 episodes (5.1% of 450). The selected package IDs, shared with the UVA recipe, are `5af748f3-6e02-4506-9938-90bb165c0953` and `07bc4327-6a40-42f0-8a30-35168b9d73b2`; they were selected deterministically by SHA-256 ranking within the two main task types. The loader checks the parent `export_plan.json` and fails on missing or ambiguous package mappings. `ENGRAM_SPLIT=all` is available for an explicitly chosen all-data run; it has a separate normalization namespace.

Each state/action is ordered as `[20 body/head/arm joints, left gripper, right gripper]`. The joint values are radians, grippers are percentages. The measured state uses joint feedback and gripper feedback. Targets use joint position commands and gripper position commands; non-position gripper modes and invalid action values are rejected. Joint targets are converted to deltas from the current joint state; gripper targets remain absolute. OpenPI pads the 22 physical dimensions to the pretrained model's native action dimension of 32. Proprioceptive noise uses `std=0.001`; compensation is exact, so noisy state plus noisy delta reconstructs the original joint target. Gripper channels are unaffected. The inverse restores absolute joint targets and leaves grippers absolute.

For this original recipe, the observation is the full left 1920×1080 half of the side-by-side `kfc_compressed` video, letterboxed by OpenPI to 224×224. The other camera slots are masked. Video is decoded on demand at the anchor's actual timestamp using bounded keyframe seeking and must match the row timestamp within 100 μs. For each anchor, the 64 actions are selected nearest to times `t + k/30 s`, using source timestamps. Episodes with gaps greater than 250 ms are rejected, and anchors without a complete future horizon are excluded because the model loss does not consume a padding mask. A missing or invalid gripper feedback value excludes that observation anchor but does not remove its otherwise valid action from neighboring horizons.

For labeled gripper episodes, the training split excludes an anchor only when **all** 20 joint targets across its entire future horizon differ from the current joint state by less than 0.01 rad **and** all gripper targets equal their current measured gripper positions. This keeps gripper-only chunks as well as any chunk with later meaningful joint motion; retained samples supervise both grippers. Validation does not apply the static filter. Set `ENGRAM_PRUNE_THRESHOLD_RAD=0` for an explicitly chosen unfiltered training run, or set `ENGRAM_GRIPPER_THRESHOLD_PERCENT` to a positive deadband if desired; recompute statistics after changing either setting.

Fit quantile statistics on the selected, filtered training split before training. Fitting skips video decoding but uses the same timing, delta conversion and noise transform as training. Minimum full quantile spans are 0.1 rad for each joint and 5 percentage points for each gripper, avoiding severe amplification of nearly constant axes; values are not clipped. Saved adjusted quantiles make the normal OpenPI inverse consistent. The stats manifest binds the stats to the source metadata, split, horizon and pruning threshold. It is not a checksum of every video or data file.

```bash
# Optional: use durable local storage for training artifacts.
export ENGRAM_ASSETS_DIR=/path/to/assets
export ENGRAM_CHECKPOINTS_DIR=/path/to/checkpoints

# Default: complete TRAIN split, no source-data writes.
examples/engram/finetune_pi05.sh stats
examples/engram/finetune_pi05.sh train my-engram-run --fsdp-devices 8

# Offline held-out action error; uses the training stats saved in the checkpoint.
uv run examples/engram/evaluate_pi05.py --checkpoint-dir /path/to/checkpoint/20000 --max-samples 256
```

Use a new experiment name to avoid overwriting a run. Training requires the π0.5 base checkpoint at `gs://openpi-assets/checkpoints/pi05_base/params` (or an existing OpenPI cache), and suitable GPU resources. Set `ENGRAM_DATA_ROOT` to relocate the same export; its parent directory must contain `export_plan.json` for the default package split. The split and both thresholds must match between stats and training. For a bounded diagnostic stats pass, append `--max-frames 1024 --batch-size 8 --num-workers 0` to the `stats` command and use a separate assets directory: sampled stats are a smoke test, not the full training statistics. Evaluation reports offline action RMSE in radians and gripper percentage points; it is not a robot rollout or success-rate evaluation.

The default recipe is 20,001 steps, batch size 64, 10,000-step checkpoint retention, and one FSDP device; choose a matching `--fsdp-devices` for the actual single-node allocation. This repository's normal `scripts/train.py` and `scripts/compute_norm_stats.py` entrypoints also support the config directly. CPU tests exercise the data and transformation paths with a small diagnostic model; they do not establish full-checkpoint GPU training or policy quality.

For a reproducible CPU optimizer smoke, first compute sampled stats in an isolated directory, then run the diagnostic random-weight model:

```bash
ENGRAM_ASSETS_DIR=/tmp/engram-pi-smoke examples/engram/finetune_pi05.sh stats --max-frames 8 --batch-size 4 --num-workers 0
JAX_PLATFORM_NAME=cpu uv run examples/engram/smoke_cpu.py --assets-base-dir /tmp/engram-pi-smoke
```

Sampled statistics are rejected for ordinary training unless `--data.allow-sampled-stats` is explicitly selected; the smoke script makes that diagnostic-only choice itself.
