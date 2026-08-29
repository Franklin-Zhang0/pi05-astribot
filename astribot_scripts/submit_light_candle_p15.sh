#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---dry-run}"
if [[ "${MODE}" != "--dry-run" && "${MODE}" != "--submit" ]]; then
  echo "Usage: $0 [--dry-run|--submit]" >&2
  exit 2
fi

STAMP=$(date -u +%Y%m%d-%H%M%S)
RUN_PREFIX="${RUN_PREFIX:-astribot-light-candle}"
RUN_TAG="${RUN_PREFIX}-${STAMP}"
JOB_NAME="${RUN_TAG}"
TRAIN_CONFIG="${TRAIN_CONFIG:-pi05_astribot_light_candle}"
DATASET_ROOT="${DATASET_ROOT:-/data2/202608/astribot/astribot_light_candle_lerobot}"
DATASET_REPO_ID="${DATASET_REPO_ID:-astribot/astribot_light_candle_lerobot}"
TRIM_START_SECONDS="${TRIM_START_SECONDS:-0}"
TRIM_END_SECONDS="${TRIM_END_SECONDS:-5}"
REMOTE_ENTRY=/data/250010203/franklin/project/pi_astribot_finetune/openpi/astribot_scripts/train_light_candle_acp.sh
COMMAND="export RUN_TAG=${RUN_TAG} TRAIN_CONFIG=${TRAIN_CONFIG} DATASET_ROOT=${DATASET_ROOT} DATASET_REPO_ID=${DATASET_REPO_ID} TRIM_START_SECONDS=${TRIM_START_SECONDS} TRIM_END_SECONDS=${TRIM_END_SECONDS}; bash ${REMOTE_ENTRY}"

UV_CACHE_DIR=/tmp/uv-cache uv run --no-project python \
  /data/250010203/franklin/project/pi_astribot_finetune/openpi/astribot_scripts/acp_create_light_candle.py \
  --workspace-name p15-dexterous-eye-hand-coordina \
  --aec2-name share-cluster \
  --zone cn-sh-01g \
  --job-name "${JOB_NAME}" \
  --worker-spec n6ls.iu.i40.8 \
  --worker-nodes 1 \
  --quota-type reserved \
  --with-default-afs \
  --aoss-bucket astribot_serve_mm_bean \
  --aoss-mount-path /astribot \
  --command "${COMMAND}" \
  "${MODE}"
