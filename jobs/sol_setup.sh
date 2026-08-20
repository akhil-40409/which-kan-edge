#!/bin/bash
# One-time Sol environment setup for eigenflow (run inside an interactive job).
# Usage:
#   interactive -c 4 -t 60 -p general -q public
#   bash jobs/sol_setup.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

module load mamba/latest
# CUDA module helps GPU JAX builds when present
module load cuda/12.2 2>/dev/null || module load cuda 2>/dev/null || true

ENV_NAME="${EIGENFLOW_ENV:-eigenflow}"

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "Creating mamba env: ${ENV_NAME}"
  mamba create -y -n "${ENV_NAME}" -c conda-forge python=3.11 pip
fi

# shellcheck disable=SC1091
source activate "${ENV_NAME}"

python -m pip install -U pip
python -m pip install -e ".[dev]"

# CPU-only JAX is default from pip; for GPU nodes install CUDA jaxlib separately:
#   python -m pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
echo ""
echo "Env ready: ${ENV_NAME}"
echo "Test: python -c 'import jax; print(jax.devices())'"
echo "Smoke: python experiments/run_benchmark.py --quick --models mlp --tasks I.12.1"
