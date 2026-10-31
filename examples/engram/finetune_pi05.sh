#!/usr/bin/env bash
# Run from any directory. Source data are read-only; assets and checkpoints
# are written only to the locations selected below.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

engram_root="${ENGRAM_DATA_ROOT:-/hand_keypoint_data/engram_data/kitchen_new/req0148_st028_1793266078098_lerobot/trainable}"
engram_assets="${ENGRAM_ASSETS_DIR:-./assets}"
engram_checkpoints="${ENGRAM_CHECKPOINTS_DIR:-./checkpoints}"
engram_split="${ENGRAM_SPLIT:-train}"
engram_threshold="${ENGRAM_PRUNE_THRESHOLD_RAD:-0.01}"
engram_gripper_threshold="${ENGRAM_GRIPPER_THRESHOLD_PERCENT:-0.0}"

if [[ $# -lt 1 ]]; then
  printf 'Usage: %s stats [stats options...] | train EXPERIMENT_NAME [training options...]\n' "$0" >&2
  exit 2
fi
stage="$1"
shift
case "$stage" in
  stats)
    uv run scripts/compute_norm_stats.py --config-name pi05_engram_w1_sandwich \
      --engram-data-root "$engram_root" --engram-split "$engram_split" \
      --engram-prune-threshold-rad "$engram_threshold" --engram-gripper-threshold-percent "$engram_gripper_threshold" \
      --assets-base-dir "$engram_assets" "$@"
    ;;
  train)
    if [[ $# -lt 1 ]]; then printf 'A unique experiment name is required.\n' >&2; exit 2; fi
    experiment="$1"
    shift
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
      uv run scripts/train.py pi05_engram_w1_sandwich --exp-name "$experiment" \
        --data.data-root "$engram_root" --data.split "$engram_split" \
        --data.prune-threshold-rad "$engram_threshold" --data.gripper-threshold-percent "$engram_gripper_threshold" \
        --assets-base-dir "$engram_assets" \
        --checkpoint-base-dir "$engram_checkpoints" "$@"
    ;;
  *)
    printf 'Unknown stage: %s\n' "$stage" >&2
    exit 2
    ;;
esac
