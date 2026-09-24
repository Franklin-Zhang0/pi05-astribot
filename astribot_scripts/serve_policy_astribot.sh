#!/usr/bin/env bash
set -euo pipefail

# Serve the pi05 Astribot light-candle checkpoint without mutating its assets.

cd "$(dirname "$0")/.."  # -> openpi/

END_EFFECTOR="${END_EFFECTOR:-apexhand}"
DEFAULT_CKPT="${DEFAULT_CKPT:-/data_disk/checkpoints/astribot/policy}"
CKPT_DIR="${1:-${DEFAULT_CKPT}}"
PORT="${2:-8000}"
if [[ "${END_EFFECTOR}" == "gripper" ]]; then
  DEFAULT_CONFIG=pi05_astribot_full_finetune
  DEFAULT_PROMPT_VALUE="${GRIPPER_PROMPT:-Execute the trained task}"
else
  DEFAULT_CONFIG=pi05_astribot_light_candle
  DEFAULT_PROMPT_VALUE="${APEXHAND_PROMPT:-Light the candle}"
fi
POLICY_CONFIG="${3:-${POLICY_CONFIG:-${DEFAULT_CONFIG}}}"
DEFAULT_PROMPT="${4:-${DEFAULT_PROMPT_VALUE}}"
TRANSPORT="${TRANSPORT:-websocket}"
MAX_INFERENCE_HZ="${MAX_INFERENCE_HZ:-10.0}"
ACTION_HORIZON="${ACTION_HORIZON:-64}"
ENSEMBLE_DECAY="${ENSEMBLE_DECAY:-0.01}"
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
echo "End effector: ${END_EFFECTOR}; config: ${POLICY_CONFIG}; prompt: ${DEFAULT_PROMPT}; transport: ${TRANSPORT}; port: ${PORT}; action horizon: ${ACTION_HORIZON}; ensemble decay: ${ENSEMBLE_DECAY}"

exec .venv/bin/python scripts/serve_policy.py \
  --port="${PORT}" \
  --transport="${TRANSPORT}" \
  --max-inference-hz="${MAX_INFERENCE_HZ}" \
  --action-horizon="${ACTION_HORIZON}" \
  --ensemble-decay="${ENSEMBLE_DECAY}" \
  --end-effector="${END_EFFECTOR}" \
  --default-prompt="${DEFAULT_PROMPT}" \
  policy:checkpoint \
  --policy.config="${POLICY_CONFIG}" \
  --policy.dir="${CKPT_DIR}"
