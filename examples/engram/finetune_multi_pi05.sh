#!/usr/bin/env bash
# Full statistics and training for the pinned multi-embodiment kitchen manifest.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
engram_root="${ENGRAM_DATA_ROOT:-$PWD/examples/engram/kitchen_multi_20261112.json}"
engram_assets="${ENGRAM_ASSETS_DIR:-$PWD/assets}"
if [[ $# -lt 1 ]]; then
  printf 'Usage: %s stats [options...] | train EXPERIMENT_NAME [options...]\n' "$0" >&2
  exit 2
fi
stage="$1"
shift
case "$stage" in
  stats)
    uv run scripts/compute_norm_stats.py --config-name pi05_engram_multi \
      --engram-data-root "$engram_root" --engram-split train \
      --assets-base-dir "$engram_assets" "$@"
    ;;
  train)
    if [[ $# -lt 1 ]]; then printf 'A unique experiment name is required.\n' >&2; exit 2; fi
    experiment="$1"
    shift
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
      uv run scripts/train.py pi05_engram_multi --exp-name "$experiment" \
        --data.data-root "$engram_root" --data.split train \
        --assets-base-dir "$engram_assets" "$@"
    ;;
  *) printf 'Unknown stage: %s\n' "$stage" >&2; exit 2 ;;
esac
