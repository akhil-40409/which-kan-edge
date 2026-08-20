#!/bin/bash
#SBATCH -J ef-gh200
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -G 1
#SBATCH -t 0-08:00:00
#SBATCH -p arm
#SBATCH -q public
#SBATCH -o logs/gh200_%A_%a.out
#SBATCH -e logs/gh200_%A_%a.err
#SBATCH --export=NONE
#
# Fair/paper array on Sol Grace Hopper (GH200) nodes.
# Partition ``arm`` = aarch64 Grace CPU + 1× GH200 480GB.
# Requires aarch64 env from jobs/sol_setup_gh200.sh (default: eigenflow-gh200).
#
#   sbatch --array=0-31 --export=NONE,N_SHARDS=32,SUITE=fair,EIGENFLOW_ENV=eigenflow-gh200 \
#     jobs/sol_gh200_array.sbatch
#
# Or:  GPU=gh200 bash jobs/submit_overnight.sh

set -euo pipefail

module load mamba/latest
module load cuda/12.2 2>/dev/null || module load cuda 2>/dev/null || true
# shellcheck disable=SC1091
source activate "${EIGENFLOW_ENV:-eigenflow-gh200}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p logs results/shards

N_SHARDS="${N_SHARDS:-32}"
IDX="${SLURM_ARRAY_TASK_ID:?Set via sbatch --array=}"
SUITE="${SUITE:-fair}"
MODELS="${MODELS:-spline,fourier,qkan,mlp}"

export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "host=$(hostname) arch=$(uname -m) array=${IDX}/${N_SHARDS} suite=${SUITE}"
nvidia-smi -L || true
python -c "import platform, jax; print(platform.machine(), jax.devices())"

python experiments/run_benchmark.py \
  --suite "${SUITE}" \
  --models "${MODELS}" \
  --platform gpu \
  --shard "${IDX}/${N_SHARDS}" \
  --out "results/shards/${SUITE}_gh200_${IDX}.csv"
