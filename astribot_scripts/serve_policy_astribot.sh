#!/usr/bin/env bash
set -euo pipefail

# Serve the pi05 Astribot light-candle checkpoint without mutating its assets.

cd "$(dirname "$0")/.."  # -> openpi/

DEFAULT_CKPT=/data2/202608/astribot/checkpoints/pi05_astribot_light_candle/astribot-light-candle-20260827-235217/20000
CKPT_DIR="${1:-${DEFAULT_CKPT}}"
PORT="${2:-8000}"
POLICY_CONFIG="${3:-pi05_astribot_light_candle}"
DEFAULT_PROMPT="${4:-Light the candle}"
ASSET_ID=astribot/astribot_light_candle_lerobot
NORM_STATS="${CKPT_DIR}/assets/${ASSET_ID}/norm_stats.json"

if [[ ! -d "${CKPT_DIR}/params" ]]; then
  echo "Checkpoint params not found: ${CKPT_DIR}/params" >&2
  exit 2
fi
if [[ ! -f "${NORM_STATS}" ]]; then
  echo "Checkpoint-owned norm stats not found: ${NORM_STATS}" >&2
  exit 2
fi

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/data2/202608/astribot/training_artifacts/openpi_cache}"

echo "Serving checkpoint: ${CKPT_DIR}"
echo "Config: ${POLICY_CONFIG}; prompt: ${DEFAULT_PROMPT}; port: ${PORT}"

exec .venv/bin/python scripts/serve_policy.py \
  --port="${PORT}" \
  --default-prompt="${DEFAULT_PROMPT}" \
  policy:checkpoint \
  --policy.config="${POLICY_CONFIG}" \
  --policy.dir="${CKPT_DIR}"
