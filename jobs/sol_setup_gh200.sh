#!/bin/bash
# One-time Grace Hopper (aarch64) environment setup for eigenflow.
# MUST run on an arm / GH200 node — x86_64 envs will not work here.
#
# Usage (from repo root on Sol login, then):
#   interactive -c 8 -G 1 -t 60 -p arm -q public --mem=64G
#   bash jobs/sol_setup_gh200.sh
#
# Hardware: NVIDIA Grace CPU (72× aarch64) + 1× GH200 480GB
# Docs: https://docs.rc.asu.edu/supercomputer-hardware/
#       https://docs.rc.asu.edu/requesting-resources/  (Grace Hopper/GH200)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

ARCH="$(uname -m)"
if [[ "${ARCH}" != "aarch64" ]]; then
  echo "ERROR: this setup is for Grace Hopper (aarch64). Got arch=${ARCH}."
  echo "Request an arm node first, e.g.:"
  echo "  interactive -c 8 -G 1 -t 60 -p arm -q public --mem=64G"
  exit 1
fi

module load mamba/latest
module load cuda/12.2 2>/dev/null || module load cuda 2>/dev/null || true

ENV_NAME="${EIGENFLOW_ENV:-eigenflow-gh200}"

echo "host=$(hostname) arch=${ARCH} env=${ENV_NAME}"
nvidia-smi -L || true

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "Creating mamba env: ${ENV_NAME} (python 3.11, aarch64)"
  mamba create -y -n "${ENV_NAME}" -c conda-forge python=3.11 pip
fi

# shellcheck disable=SC1091
source activate "${ENV_NAME}"

python -m pip install -U pip
python -m pip install -e ".[dev]"

# CUDA JAX for Hopper on aarch64 (install on the GH200 node itself)
python -m pip install --upgrade "jax[cuda12]" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo ""
echo "=== Smoke checks ==="
python -c "import platform; print('python', platform.python_version(), platform.machine())"
python -c "import jax; print('jax devices', jax.devices())"
python experiments/run_benchmark.py --quick --models mlp,spline,fourier --tasks I.12.1 \
  --platform gpu --out /tmp/ef_gh200_smoke.csv

echo ""
echo "Env ready: ${ENV_NAME}"
echo "Submit fair suite:  GPU=gh200 bash jobs/submit_overnight.sh"
