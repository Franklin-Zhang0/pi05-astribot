# Astribot + openpi (pi05_base) fine-tuning quickstart

This repo contains a minimal pipeline to fine-tune [`pi05_base`](openpi/src/openpi/training/config.py:1) for Astribot tasks using raw Astribot recordings under `/astribot_data`.

You will:
1) convert raw Astribot `*.hdf5` to a LeRobot dataset,
2) compute normalization stats,
3) fine-tune `pi05_base`,
4) serve the checkpoint via websocket,
5) replay either recorded trajectories or policy actions on the robot.

---

## 0) Prerequisites

- You need the Astribot SDK available for robot-side replay and live cameras (see [`astribot_sdk/README.md`](astribot_sdk/README.md:1)).
- You need the `openpi` python env (this repo uses [`openpi/.venv`](openpi/.python-version:1) in examples).

---

## 1) Convert `/astribot_data` → LeRobot dataset

Use the converter [`openpi/examples/astribot/convert_astribot_hdf5_to_lerobot.py`](openpi/examples/astribot/convert_astribot_hdf5_to_lerobot.py:1).

Example (full conversion):

```bash
/data/250010203/franklin/project/pi_astribot_finetune/openpi/.venv/bin/python \
  openpi/examples/astribot/convert_astribot_hdf5_to_lerobot.py \
  --data-dir /astribot_data/serve_small_mm_beans \
  --repo-id local/astribot_serve_small_mm_beans \
  --default-task "Serve small MM beans" \
  --decode-backend thread \
  --decode-workers 8
```

Smoke-test conversion (faster):

```bash
/data/250010203/franklin/project/pi_astribot_finetune/openpi/.venv/bin/python \
  openpi/examples/astribot/convert_astribot_hdf5_to_lerobot.py \
  --data-dir /astribot_data/serve_small_mm_beans \
  --repo-id local/astribot_serve_small_mm_beans_smoke \
  --default-task "Serve small MM beans" \
  --frame-stride 2 \
  --max-frames-per-segment 50 \
  --decode-backend thread \
  --decode-workers 8 \
  --max-episodes 2
```

---

## 2) Configure training to use your dataset

The training config is [`pi05_astribot`](openpi/src/openpi/training/config.py:565).

You can either:
- edit the config’s `repo_id` (recommended for a stable setup), or
- override it on the command line with `--data.repo-id ...`.

---

## 3) Compute norm stats

Norm stats are required for sensible training. Run:

```bash
cd openpi
/data/250010203/franklin/project/pi_astribot_finetune/openpi/.venv/bin/python scripts/compute_norm_stats.py --config-name pi05_astribot
cd ..
```

This writes stats under:
- [`openpi/assets/pi05_astribot/astribot`](openpi/assets:1)

---

## 4) Fine-tune `pi05_base`

### Option A: use the provided finetune script

Script: [`openpi/astribot_scripts/finetune_pi05_astribot.sh`](openpi/astribot_scripts/finetune_pi05_astribot.sh:1)

```bash
bash openpi/astribot_scripts/finetune_pi05_astribot.sh astribot_serve_small_mm_beans
```

### Option B: run train directly (recommended for experimentation)

```bash
cd openpi

# Disable wandb with the boolean flag form: --no-wandb-enabled
/data/250010203/franklin/project/pi_astribot_finetune/openpi/.venv/bin/python scripts/train.py pi05_astribot_full_finetune \
  --exp-name astribot_serve_small_mm_beans_full_tune_bs32_delta \
  --overwrite  \
  --data.repo-id local/astribot_serve_small_mm_beans_300_with_nops_downsampled \
  --data.assets.asset-id local/astribot_serve_small_mm_beans_300_with_nops_downsampled  \
  --fsdp_devices 4 \
  --assets-base-dir assets
```

100-step smoke test:

```bash
cd openpi
/data/250010203/franklin/project/pi_astribot_finetune/openpi/.venv/bin/python openpi/scripts/train.py pi05_astribot \
  --exp-name smoke_lora \
  --num-train-steps 100 \
  --save-interval 50 \
  --log-interval 10 \
  --batch-size 4 \
  --no-wandb-enabled \
  --overwrite \
  --assets-base-dir openpi/assets \
  --data.assets.asset-id local/astribot_serve_small_mm_beans_example
```

Checkpoints are written under:
- [`openpi/checkpoints/pi05_astribot/<exp_name>/<step>/`](openpi/checkpoints:1)

---

## 5) Serve the dexterous-hand light-candle checkpoint

Script: [`openpi/astribot_scripts/serve_policy_astribot.sh`](openpi/astribot_scripts/serve_policy_astribot.sh:1)

```bash
bash openpi/astribot_scripts/serve_policy_astribot.sh \
  /data2/202608/astribot/checkpoints/pi05_astribot_light_candle/astribot-light-candle-20260827-235217/20000 \
  8000
```

The wrapper uses config `pi05_astribot_light_candle`, prompt `Light the candle`, and the
normalization statistics embedded in the checkpoint. It does not copy or modify checkpoint
assets. The response contract is `(64, 52)` actions ordered as torso (4), left arm (7), right
arm (7), head (2), left-hand driver joints (16), and right-hand driver joints (16).

For the current Orin streaming client, use the ZMQ transport. The server returns one
canonical 55-element action per frame, keeps only the newest observation, and limits policy
calls to the configured rate. It applies ACT-style temporal ensembling to the configured
number of action rows. `ENSEMBLE_DECAY` is the exponential coefficient: each newer
surviving prediction is weighted by `exp(-ENSEMBLE_DECAY * i)` relative to the oldest.

```bash
env OPENPI_DATA_HOME=/data_disk/astribot_runtime/openpi_cache \
  JAX_COMPILATION_CACHE_DIR=/data_disk/astribot_runtime/jax_cache \
  TRANSPORT=zmq MAX_INFERENCE_HZ=10 ACTION_HORIZON=64 ENSEMBLE_DECAY=0.01 \
  bash astribot_scripts/serve_policy_astribot.sh \
  /data_disk/checkpoints/astribot/astribot-light-candle-20260827-235217/10000 \
  5555
```

Set the Orin policy client to `transport=zmq` and
`server_uri=zmq://<policy-server-ip>:5555`. Its camera selection may provide either
`head_rgbd` or `head_stereo`; the adapter maps both to the trained head input and crops
stereo before the model resize.

Replay the HDF5 dummy client against a running ZMQ server and write metrics:

```bash
.venv/bin/python examples/astribot/infer_hdf5_serving.py \
  --transport zmq \
  --host 127.0.0.1 \
  --port 5555 \
  --hdf5 /data_disk/datasets/task_candle_0826/task_candle_0826_episode_146.hdf5 \
  --result-json /data_disk/astribot_runtime/hdf5_zmq_full.json
```

Run a checkpoint-backed smoke test before connecting a robot:

```bash
export HF_LEROBOT_HOME=/data2/202608
export LEROBOT_VIDEO_BACKEND=pyav
export OPENPI_DATA_HOME=/data2/202608/astribot/training_artifacts/openpi_cache

openpi/.venv/bin/python openpi/examples/astribot/infer_checkpoint_sanity.py \
  --checkpoint-dir /data2/202608/astribot/checkpoints/pi05_astribot_light_candle/astribot-light-candle-20260827-235217/20000
```

Pass `--policy-host 127.0.0.1 --policy-port 8000` to validate a running websocket server
instead of loading the checkpoint in the smoke-test process.

Live observations must contain a 52-D joint state in the same order. Use `head_rgbd` or the
left half of `head_stereo` (crop before resizing), plus both wrist cameras. Proprioceptive
noise remains a training-only augmentation and is not injected by the serving policy.

---

## 6) Replay

Replay script: [`astribot_pi/traj_replay.py`](astribot_pi/traj_replay.py:1)

### (A) Replay recorded commands from HDF5

```bash
python astribot_pi/traj_replay.py \
  --hdf5 /astribot_data/serve_small_mm_beans/episode_0.hdf5 \
  --seg 0 \
  --freq 100
```

### (B) Policy-driven replay (robot live cameras → websocket inference → joint commands)

Start the server first (see section 5), then on the robot:

```bash
python astribot_pi/traj_replay.py \
  --policy-host <policy-server-ip> \
  --policy-port 8000 \
  --prompt "Serve small MM beans" \
  --control-hz 30
```
