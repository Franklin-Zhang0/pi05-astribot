#!/usr/bin/env bash
set -euo pipefail

# Read-only readiness check for the inference desktop. It never starts a
# server, downloads weights, connects to Orin, or sends a policy action.
cd "$(dirname "$0")/.."

VENV="${OPENPI_VENV:-/data_disk/astribot_runtime/openpi-policy-venv}"
PORT="${POLICY_PORT:-5555}"
if [[ ! -d /data_disk ]]; then
  echo "ERROR: /data_disk is not mounted" >&2
  exit 2
fi
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "ERROR: Python environment not found at ${VENV}" >&2
  exit 2
fi

echo "repo=$(pwd)"
echo "branch=$(git branch --show-current)"
echo "commit=$(git rev-parse --short HEAD)"
echo "python=$(${VENV}/bin/python --version 2>&1)"
echo "data_disk=$(df -h /data_disk | tail -1)"
"${VENV}/bin/python" - <<'PY'
import importlib

for name in ("numpy", "zmq", "PIL", "openpi", "torch", "jax"):
    module = importlib.import_module(name)
    print(f"{name}={getattr(module, '__version__', 'ok')}")

import torch
print(f"torch_cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch_gpu={torch.cuda.get_device_name(0)}")
import jax
print("jax_devices=" + ",".join(str(device) for device in jax.devices()))
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
  echo "WARNING: nvidia-smi not found" >&2
fi

if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :${PORT}" | tail -n +2 | grep -q .; then
  echo "WARNING: TCP port ${PORT} is already listening" >&2
else
  echo "port_${PORT}=free"
fi

echo "policy_host=ready"
