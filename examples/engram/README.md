# π0.5 fine-tuning: Engram Dex W1 sandwich

This recipe starts from `pi05_base`, follows the Astribot 64-step fine-tuning recipe, and trains on the full successful episodes of the gripper-equipped export. The source dataset is read-only; the 113-episode export without gripper commands and partial fragments are not included. It uses the OpenPI-pinned LeRobot v2 environment with a local v3 reader rather than attempting to open v3 metadata with the v2 loader.

The implementation follows this repository's Astribot checkout. For background, see the [upstream OpenPI training configuration](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py) and [LeRobot v3 format documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3). This OpenPI repository is distributed under its [Apache-2.0 license](../../LICENSE).

The default split holds out entire source recording packages. The train split has 427 episodes, and the validation split has 23 episodes (5.1% of 450). The selected package IDs, shared with the UVA recipe, are `5af748f3-6e02-4506-9938-90bb165c0953` and `07bc4327-6a40-42f0-8a30-35168b9d73b2`; they were selected deterministically by SHA-256 ranking within the two main task types. The loader checks the parent `export_plan.json` and fails on missing or ambiguous package mappings. `ENGRAM_SPLIT=all` is available for an explicitly chosen all-data run; it has a separate normalization namespace.

Each state/action is ordered as `[20 body/head/arm joints, left gripper, right gripper]`. The joint values are radians, grippers are percentages. The measured state uses joint feedback and gripper feedback. Targets use joint position commands and gripper position commands; non-position gripper modes and invalid action values are rejected. Joint targets are converted to deltas from the current joint state; gripper targets remain absolute. OpenPI pads the 22 physical dimensions to the pretrained model's native action dimension of 32. Like the Astribot recipe, proprioceptive noise uses `std=0.001`; here compensation is exact, so noisy state plus noisy delta reconstructs the original joint target. Gripper channels are unaffected. The inverse restores absolute joint targets and leaves grippers absolute.

The current observation is the full left 1920×1080 half of the side-by-side `kfc_compressed` video, letterboxed by OpenPI to 224×224. The other camera slots are masked. Video is decoded on demand at the anchor's actual timestamp using bounded keyframe seeking and must match the row timestamp within 100 μs. For each anchor, the 64 actions are selected nearest to times `t + k/30 s`, using source timestamps. Episodes with gaps greater than 250 ms are rejected, and anchors without a complete future horizon are excluded because the model loss does not consume a padding mask. A missing or invalid gripper feedback value excludes that observation anchor but does not remove its otherwise valid action from neighboring horizons.

The training split excludes an anchor only when **all** 20 joint targets across its entire future horizon differ from the current joint state by less than 0.01 rad **and** all gripper targets equal their current measured gripper positions. This keeps gripper-only chunks as well as any chunk with later meaningful joint motion; retained samples supervise both grippers. Validation does not apply the static filter. Set `ENGRAM_PRUNE_THRESHOLD_RAD=0` for an explicitly chosen unfiltered training run, or set `ENGRAM_GRIPPER_THRESHOLD_PERCENT` to a positive deadband if desired; recompute statistics after changing either setting.

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
