#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/data/250010203/franklin/project/pi_astribot_finetune/openpi
CHECKPOINT_DIR=/data2/202608/astribot/checkpoints/pi05_astribot_light_candle/astribot-light-candle-20260827-235217/20000
RUN_TAG="${RUN_TAG:-astribot-light-candle-serving-test}"
RESULT_DIR="/data2/202608/astribot/training_artifacts/serving_tests/${RUN_TAG}"
PORT="${PORT:-8000}"

mkdir -p /data2
if [[ ! -e /data2/202608 ]]; then
  ln -s /mnt/afs /data2/202608
fi
test "$(readlink -f /data2/202608)" = /mnt/afs

mkdir -p "${RESULT_DIR}"
exec > >(tee -a "${RESULT_DIR}/runtime.log") 2>&1

echo "SERVING_TEST_BEGIN utc=$(date -u +%FT%TZ) run_tag=${RUN_TAG}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
test "$(nvidia-smi -L | wc -l)" -eq 1
test -d "${CHECKPOINT_DIR}/params"
test -f "${CHECKPOINT_DIR}/assets/astribot/astribot_light_candle_lerobot/norm_stats.json"

cd "${PROJECT_DIR}"
export HF_LEROBOT_HOME=/data2/202608
export HF_HOME=/data2/202608/astribot/training_artifacts/hf_cache
export OPENPI_DATA_HOME=/data2/202608/astribot/training_artifacts/openpi_cache
export LEROBOT_VIDEO_BACKEND=pyav
export JAX_COMPILATION_CACHE_DIR=/data2/202608/astribot/training_artifacts/jax_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

.venv/bin/python -m pytest -q src/openpi/transforms_test.py -k joint_position_noise
bash -n astribot_scripts/serve_policy_astribot.sh

bash astribot_scripts/serve_policy_astribot.sh "${CHECKPOINT_DIR}" "${PORT}" \
  >"${RESULT_DIR}/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

timeout 45m .venv/bin/python examples/astribot/infer_checkpoint_sanity.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --policy-host 127.0.0.1 \
  --policy-port "${PORT}" \
  --result-json "${RESULT_DIR}/result.json"

test "$(jq -r .status "${RESULT_DIR}/result.json")" = passed
test "$(jq -c .action_shape "${RESULT_DIR}/result.json")" = '[64,52]'
test "$(jq -r .actions_finite "${RESULT_DIR}/result.json")" = true
test "$(jq -r .transport "${RESULT_DIR}/result.json")" = websocket
echo "SERVING_TEST_PASSED utc=$(date -u +%FT%TZ) result=${RESULT_DIR}/result.json"
sync
