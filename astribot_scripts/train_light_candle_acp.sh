#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/250010203/franklin/project/pi_astribot_finetune/openpi}"
SOURCE_ROOT="${SOURCE_ROOT:-/astribot/astribot_light_candle}"
DATASET_ROOT="${DATASET_ROOT:-/data2/202608/astribot/astribot_light_candle_lerobot}"
RUN_TAG="${RUN_TAG:-astribot-light-candle-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ROOT="/data2/202608/astribot/training_artifacts/runs/${RUN_TAG}"
LOG_FILE="${RUN_ROOT}/acp_entry.log"

mkdir -p /data2
if [[ ! -e /data2/202608 ]]; then
  ln -s /mnt/afs /data2/202608
fi
test "$(readlink -f /data2/202608)" = /mnt/afs
mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

on_exit() {
  status=$?
  echo "RUN_EXIT status=${status} utc=$(date -u +%FT%TZ)"
  sync
  exit "${status}"
}
trap on_exit EXIT

echo "RUN_BEGIN utc=$(date -u +%FT%TZ)"
echo "RUN_TAG=${RUN_TAG}"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "SOURCE_ROOT=${SOURCE_ROOT}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "TRAIN_CONTRACT config=pi05_astribot_light_candle steps=20001 save_interval=10000 keep_period=10000 fsdp_devices=8 proprio_noise_std=0.001"

test -d "${PROJECT_DIR}"
test -d "${SOURCE_ROOT}"
test -d /data2/202608/astribot
grep -E ' /(astribot|data|mnt/afs) ' /proc/mounts || true

GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "GPU_COUNT=${GPU_COUNT}"
test "${GPU_COUNT}" -eq 8
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd "${PROJECT_DIR}"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
test -x "${PYTHON}"

export HF_LEROBOT_HOME=/data2/202608
export HF_HOME=/data2/202608/astribot/training_artifacts/hf_cache
export OPENPI_DATA_HOME=/data2/202608/astribot/training_artifacts/openpi_cache
export WANDB_CACHE_DIR=/data2/202608/astribot/training_artifacts/wandb_cache
export WANDB_DIR="${RUN_ROOT}/wandb"
export JAX_COMPILATION_CACHE_DIR=/data2/202608/astribot/training_artifacts/jax_cache
# Leave explicit device-memory headroom for NCCL collectives. At 0.95, XLA's
# arena left only ~4 GiB per H100 and the first replicated step aborted in
# ncclGroupEnd with CUDA OOM before step 0 could be logged.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/astribot_scripts/python_startup${PYTHONPATH:+:${PYTHONPATH}}"
export LEROBOT_VIDEO_BACKEND=pyav
mkdir -p "${HF_HOME}" "${OPENPI_DATA_HOME}" "${WANDB_CACHE_DIR}" "${WANDB_DIR}" "${JAX_COMPILATION_CACHE_DIR}"

"${PYTHON}" -c 'import sys, typing; assert typing.__file__ and "/cv2/typing/" not in typing.__file__; print("PYTHON_STARTUP_TYPING_OK", typing.__file__, "cached_at_startup", "sitecustomize" in sys.modules)'
"${PYTHON}" -c 'import os; assert os.environ["LEROBOT_VIDEO_BACKEND"]=="pyav"; print("LEROBOT_VIDEO_BACKEND_OK", os.environ["LEROBOT_VIDEO_BACKEND"])'
echo "XLA_MEMORY_CONTRACT fraction=${XLA_PYTHON_CLIENT_MEM_FRACTION} nccl_debug=${NCCL_DEBUG}"

if [[ -z "${WANDB_API_KEY:-}" && -f /data/250010203/franklin/.netrc ]]; then
  WANDB_API_KEY=$("${PYTHON}" -c 'import netrc; print(netrc.netrc("/data/250010203/franklin/.netrc").authenticators("api.wandb.ai")[2])')
  export WANDB_API_KEY
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  export WANDB_MODE=disabled
fi
echo "WANDB_CONTRACT mode=${WANDB_MODE:-online} dir=${WANDB_DIR} api_key_present=$([[ -n "${WANDB_API_KEY:-}" ]] && echo yes || echo no)"

git -c safe.directory="${PROJECT_DIR}" rev-parse --abbrev-ref HEAD
git -c safe.directory="${PROJECT_DIR}" rev-parse HEAD
git -c safe.directory="${PROJECT_DIR}" status --short

DATASET_READY=0
CONVERSION_APPEND_RESUME=0
if [[ -f "${DATASET_ROOT}/conversion_complete.json" ]]; then
  if DATASET_ROOT="${DATASET_ROOT}" "${PYTHON}" -c 'import json, os, pathlib; m=json.loads((pathlib.Path(os.environ["DATASET_ROOT"])/"conversion_complete.json").read_text()); assert m.get("status")=="complete"; assert m.get("head_camera")=={"source":"head_stereo","view":"left","crop":"left_half_before_resize"}; assert m.get("time_trim_seconds")=={"start":0.0,"end":5.0,"basis":"recorded_time"}'; then
    DATASET_READY=1
    echo "CONVERSION_REUSE marker=${DATASET_ROOT}/conversion_complete.json"
  fi
fi

if [[ "${DATASET_READY}" -eq 0 && -e "${DATASET_ROOT}" ]]; then
  if [[ ! -e "${DATASET_ROOT}/conversion_complete.json" && -f "${DATASET_ROOT}/meta/info.json" ]]; then
    CONVERSION_APPEND_RESUME=1
    echo "CONVERSION_APPEND_RESUME dataset=${DATASET_ROOT}"
  else
    SUPERSEDED_ROOT=/data2/202608/astribot/superseded
    SUPERSEDED_DATASET="${SUPERSEDED_ROOT}/astribot_light_candle_lerobot_contract_mismatch_${RUN_TAG}"
    mkdir -p "${SUPERSEDED_ROOT}"
    mv "${DATASET_ROOT}" "${SUPERSEDED_DATASET}"
    echo "CONVERSION_SUPERSEDED old=${SUPERSEDED_DATASET}"
  fi
fi

if [[ "${DATASET_READY}" -eq 0 ]]; then
  CONVERT_ARGS=(
    --data-dir "${SOURCE_ROOT}"
    --output-dir "${DATASET_ROOT}"
    --repo-id astribot/astribot_light_candle_lerobot
    --default-task "Light the candle"
    --trim-start-seconds 0
    --trim-end-seconds 5
    --decode-workers 48
    --image-writer-processes 16
    --image-writer-threads 4
    --video-encode-workers 3
  )
  if [[ "${CONVERSION_APPEND_RESUME}" -eq 1 ]]; then
    CONVERT_ARGS+=(--resume)
  fi
  "${PYTHON}" examples/astribot/convert_astribot_v4_dexhand_to_lerobot.py "${CONVERT_ARGS[@]}"
fi

"${PYTHON}" -c 'import json, pathlib; p=pathlib.Path("/data2/202608/astribot/astribot_light_candle_lerobot/conversion_complete.json"); m=json.loads(p.read_text()); assert m["status"]=="complete"; assert m["state_dim"]==52 and m["action_dim"]==52; assert m["converted_episodes"]>0 and m["converted_frames"]>0; assert m["head_camera"]=={"source":"head_stereo","view":"left","crop":"left_half_before_resize"}; assert m["time_trim_seconds"]=={"start":0.0,"end":5.0,"basis":"recorded_time"}; print({k:m[k] for k in ("status","converted_episodes","converted_frames","state_dim","action_dim","head_camera","time_trim_seconds")})'

NORM_STATS_FILE=/data2/202608/astribot/training_artifacts/assets/pi05_astribot_light_candle/astribot/astribot_light_candle_lerobot/norm_stats.json
validate_norm_stats() {
  NORM_STATS_FILE="${NORM_STATS_FILE}" "${PYTHON}" -c 'import json, math, os, pathlib; p=pathlib.Path(os.environ["NORM_STATS_FILE"]); s=json.loads(p.read_text())["norm_stats"]; assert set(s)=={"state","actions"}; fields=("mean","std","q01","q99"); assert all(len(s[key][field])==52 for key in s for field in fields); assert all(math.isfinite(value) for key in s for field in fields for value in s[key][field]); print("NORM_STATS_CONTRACT_OK", p)'
}

if [[ -f "${NORM_STATS_FILE}" ]] && validate_norm_stats; then
  echo "NORM_STATS_REUSE file=${NORM_STATS_FILE}"
else
  if [[ -f "${NORM_STATS_FILE}" ]]; then
    INVALID_NORM_STATS="${RUN_ROOT}/superseded_invalid_norm_stats.json"
    mv "${NORM_STATS_FILE}" "${INVALID_NORM_STATS}"
    echo "NORM_STATS_SUPERSEDED old=${INVALID_NORM_STATS}"
  fi
  "${PYTHON}" scripts/compute_norm_stats.py --config-name pi05_astribot_light_candle
  validate_norm_stats
fi

"${PYTHON}" -c 'from openpi.training import config as c; from openpi import transforms as t; x=c.get_config("pi05_astribot_light_candle"); d=x.data.create(x.assets_dirs,x.model); assert x.num_train_steps==20001; assert x.save_interval==10000; assert x.keep_period==10000; assert x.fsdp_devices==8; aug=[v for v in d.data_transforms.inputs if isinstance(v,t.JointPositionNoiseCompensateActions)]; assert len(aug)==1 and aug[0].std==0.001; print("CONFIG_CONTRACT_OK",x)'

"${PYTHON}" scripts/train.py pi05_astribot_light_candle \
  --exp-name "${RUN_TAG}" \
  --overwrite

CHECKPOINT_DIR="/data2/202608/astribot/checkpoints/pi05_astribot_light_candle/${RUN_TAG}"
test -d "${CHECKPOINT_DIR}/10000"
test -d "${CHECKPOINT_DIR}/20000"
FOUND_STEPS=$(find "${CHECKPOINT_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -n | tr '\n' ' ')
echo "CHECKPOINT_STEPS=${FOUND_STEPS}"
test "${FOUND_STEPS}" = "10000 20000 "
echo "TRAINING_COMPLETE checkpoint_dir=${CHECKPOINT_DIR}"
