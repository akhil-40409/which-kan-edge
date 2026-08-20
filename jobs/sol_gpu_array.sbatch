#!/bin/bash
#SBATCH -J ef-gpu
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -G a100:1
#SBATCH -t 0-08:00:00
#SBATCH -p general
#SBATCH -q public
#SBATCH -o logs/gpu_%A_%a.out
#SBATCH -e logs/gpu_%A_%a.err
#SBATCH --export=NONE
#
# Fair/paper GPU array for JAX models (spline, fourier, qkan, mlp).
# Submit via jobs/submit_overnight.sh.

set -euo pipefail

module load mamba/latest
module load cuda/12.2 2>/dev/null || module load cuda 2>/dev/null || true
# shellcheck disable=SC1091
source activate "${EIGENFLOW_ENV:-eigenflow}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p logs results/shards

N_SHARDS="${N_SHARDS:-32}"
IDX="${SLURM_ARRAY_TASK_ID:?Set via sbatch --array=}"
SUITE="${SUITE:-fair}"
MODELS="${MODELS:-spline,fourier,qkan,mlp}"

export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "host=$(hostname) array=${IDX}/${N_SHARDS} suite=${SUITE} models=${MODELS}"
nvidia-smi -L || true
python -c "import jax; print('jax devices', jax.devices())"

python experiments/run_benchmark.py \
  --suite "${SUITE}" \
  --models "${MODELS}" \
  --platform gpu \
  --shard "${IDX}/${N_SHARDS}" \
  --out "results/shards/${SUITE}_gpu_${IDX}.csv"
