#!/usr/bin/env bash
set -euo pipefail

# Serve a fine-tuned Astribot policy checkpoint via websocket.
#
# Starts [`openpi/scripts/serve_policy.py`](openpi/scripts/serve_policy.py:1) using the Astribot training config
# [`pi05_astribot`](openpi/src/openpi/training/config.py:565).

cd "$(dirname "$0")/.."  # -> openpi/

CKPT_DIR="${1:-}"
if [[ -z "${CKPT_DIR}" ]]; then
  echo "Usage: $0 <checkpoint_dir> [port]"
  echo "Example: $0 checkpoints/pi05_astribot/astribot_serve_small_mm_beans/20000 8000"
  exit 2
fi

POLICY_CONFIG="${2}"

PORT="${3:-8000}"

if [ -e "$CKPT_DIR/assets/astribot/norm_stats.json" ]; then
  echo skip cp normstat
else
  mkdir -p $CKPT_DIR/assets/astribot
  cp $CKPT_DIR/assets/local/*/norm_stats.json $CKPT_DIR/assets/astribot
fi

echo "Serving checkpoint: ${CKPT_DIR} on port ${PORT} using config ${POLICY_CONFIG}"

(
echo Waiting for compiling...
sleep 45
python /data/250010203/franklin/project/pi_astribot_finetune/astribot_pi/src/astribot_pi/traj_replay_offline.py \
  --policy-host 127.0.0.1 --policy-port $PORT --prompt "Serve small MM beans" --hdf5 /data/250010203/franklin/project/pi_astribot_finetune/astribot_data_example/Serve_small_MM_episode_abs_0.hdf5 > /dev/null
echo Still waiting for compiling...

sleep 30
python /data/250010203/franklin/project/pi_astribot_finetune/astribot_pi/src/astribot_pi/traj_replay_offline.py \
  --policy-host 127.0.0.1 --policy-port $PORT --prompt "Serve small MM beans" --hdf5 /data/250010203/franklin/project/pi_astribot_finetune/astribot_data_example/Serve_small_MM_episode_abs_0.hdf5 > /dev/null
echo Still waiting for compiling...

sleep 30
python /data/250010203/franklin/project/pi_astribot_finetune/astribot_pi/src/astribot_pi/traj_replay_offline.py \
  --policy-host 127.0.0.1 --policy-port $PORT --prompt "Serve small MM beans" --hdf5 /data/250010203/franklin/project/pi_astribot_finetune/astribot_data_example/Serve_small_MM_episode_abs_0.hdf5 > /dev/null
echo Still waiting for compiling...

python /data/250010203/franklin/project/pi_astribot_finetune/astribot_pi/src/astribot_pi/traj_replay_offline.py \
  --policy-host 127.0.0.1 --policy-port $PORT --prompt "Serve small MM beans" --hdf5 /data/250010203/franklin/project/pi_astribot_finetune/astribot_data_example/Serve_small_MM_episode_abs_0.hdf5 > /dev/null

)&

/data/250010203/franklin/project/pi_astribot_finetune/openpi/.venv/bin/python \
  scripts/serve_policy.py --port="${PORT}" policy:checkpoint \
  --policy.config=$POLICY_CONFIG \
  --policy.dir="${CKPT_DIR}"

# run one to start compile
