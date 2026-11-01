# π0.5 fine-tuning: Engram Dex W1 sandwich

This recipe starts from `pi05_base`, follows the Astribot 64-step fine-tuning recipe, and trains on the full successful episodes of the gripper-equipped export. The source dataset is read-only; the 113-episode export without gripper commands and partial fragments are not included. It uses the OpenPI-pinned LeRobot v2 environment with a local v3 reader rather than attempting to open v3 metadata with the v2 loader.

The implementation follows this repository's Astribot checkout. For background, see the [upstream OpenPI training configuration](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py) and [LeRobot v3 format documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3). This OpenPI repository is distributed under its [Apache-2.0 license](../../LICENSE).

The default split holds out entire source recording packages. The train split has 427 episodes, and the validation split has 23 episodes (5.1% of 450). The selected package IDs, shared with the UVA recipe, are `5af748f3-6e02-4506-9938-90bb165c0953` and `07bc4327-6a40-42f0-8a30-35168b9d73b2`; they were selected deterministically by SHA-256 ranking within the two main task types. The loader checks the parent `export_plan.json` and fails on missing or ambiguous package mappings. `ENGRAM_SPLIT=all` is available for an explicitly chosen all-data run; it has a separate normalization namespace.

Each state/action is ordered as `[20 body/head/arm joints, left gripper, right gripper]`. The joint values are radians, grippers are percentages. The measured state uses joint feedback and gripper feedback. Targets use joint position commands and gripper position commands; non-position gripper modes and invalid action values are rejected. Joint targets are converted to deltas from the current joint state; gripper targets remain absolute. OpenPI pads the 22 physical dimensions to the pretrained model's native action dimension of 32. Like the Astribot recipe, proprioceptive noise uses `std=0.001`; here compensation is exact, so noisy state plus noisy delta reconstructs the original joint target. Gripper channels are unaffected. The inverse restores absolute joint targets and leaves grippers absolute.

The current observation is the full left 1920×1080 half of the side-by-side `kfc_compressed` video, letterboxed by OpenPI to 224×224. The other camera slots are masked. Video is decoded on demand at the anchor's actual timestamp using bounded keyframe seeking and must match the row timestamp within 100 μs. For each anchor, the 64 actions are selected nearest to times `t + k/30 s`, using source timestamps. Episodes with gaps greater than 250 ms are rejected, and anchors without a complete future horizon are excluded because the model loss does not consume a padding mask. A missing or invalid gripper feedback value excludes that observation anchor but does not remove its otherwise valid action from neighboring horizons.

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

## Mixed-end-effector snapshot, 2026-11-01

The pinned [multi-source manifest](kitchen_new_20261101.json) combines the original 450 fully labeled gripper episodes with 110 `st026` and 260 `st031` full, successful episodes. `st026` uses grippers and `st031` shows dexterous hands; neither new export records any end-effector feedback or commands. The previous 113-episode incomplete source and partial fragments stay excluded. The validation split preserves the original 23 held-out episodes and adds 19 from the new sources, for 778 training and 42 validation episodes. Package IDs are held out across all sources, and duplicate annotations, ambiguous mappings, missing packages, unexpected schema changes and episode counts fail closed.

All sources supervise the same 20 joint deltas. For the 370 episodes without end-effector annotations, a prompt marker explicitly says `End-effector state unavailable`, neutral normalized state tokens occupy the two unavailable gripper slots, and their flow inputs are pure noise and their output loss is masked. No finger labels are invented. For these joint-only episodes, the training filter drops chunks when all joint target movements across the horizon are below 0.01 rad. Their end-effector activity is unobserved, so this can also drop gripper-only or finger-only activity. Labeled gripper episodes keep the original joint-or-gripper motion pruning rule. Gripper normalization statistics use only labeled training anchors; joint statistics use all retained training anchors. This filter changes the statistics contract: recompute full training statistics before training. The full stats pass includes the final partial batch and saves sample counts, the source snapshot, and a hash in the statistics manifest.

For an effective global batch of 256 with two-way FSDP, `--gradient-accumulation-steps 32` performs 32 global microbatches of eight (four samples per device) per optimizer update. This reduces activation memory but has not by itself established that a full-size model fits or performs well on a specific H100 allocation. Set `--checkpoint-completed-steps` to label checkpoint directories by completed optimizer updates; the default keeps legacy OpenPI naming.

```bash
export ENGRAM_DATA_ROOT="$PWD/examples/engram/kitchen_new_20261101.json"
export ENGRAM_ASSETS_DIR=/path/to/run/assets
examples/engram/finetune_pi05.sh stats --batch-size 256 --num-workers 0
examples/engram/finetune_pi05.sh train my-mixed-engram-run \
  --batch-size 256 --num-train-steps 30000 --save-interval 10000 --keep-period 10000 \
  --fsdp-devices 2 --gradient-accumulation-steps 32 --checkpoint-completed-steps \
  --checkpoint-dir-override /path/to/exact/run/checkpoint/directory
uv run examples/engram/evaluate_pi05.py --data-root "$ENGRAM_DATA_ROOT" \
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
