#!/bin/bash
#SBATCH -J ef-cpu
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0-08:00:00
#SBATCH -p general
#SBATCH -q public
#SBATCH -o logs/cpu_%A_%a.out
#SBATCH -e logs/cpu_%A_%a.err
#SBATCH --export=NONE
#
# Fair/paper CPU array. Submit via jobs/submit_overnight.sh (exports SUITE, MODELS, N_SHARDS).

set -euo pipefail

module load mamba/latest
# shellcheck disable=SC1091
source activate "${EIGENFLOW_ENV:-eigenflow}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p logs results/shards

N_SHARDS="${N_SHARDS:-64}"
IDX="${SLURM_ARRAY_TASK_ID:?Set via sbatch --array=}"
SUITE="${SUITE:-fair}"
MODELS="${MODELS:-spline,fourier,qkan,mlp}"

export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=${OMP_NUM_THREADS}"

echo "host=$(hostname) array=${IDX}/${N_SHARDS} suite=${SUITE} models=${MODELS}"
python experiments/run_benchmark.py --suite "${SUITE}" --models "${MODELS}" --list-jobs

python experiments/run_benchmark.py \
  --suite "${SUITE}" \
  --models "${MODELS}" \
  --platform cpu \
  --shard "${IDX}/${N_SHARDS}" \
  --out "results/shards/${SUITE}_cpu_${IDX}.csv"
