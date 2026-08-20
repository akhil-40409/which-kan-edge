#!/bin/bash
# Submit overnight fair/paper arrays on ASU Sol.
# Run from the repo root on a Sol login node (after setup).
#
#   bash jobs/submit_overnight.sh                 # CPU + A100 (default)
#   GPU=gh200 bash jobs/submit_overnight.sh       # CPU + Grace Hopper only
#   GPU=both  bash jobs/submit_overnight.sh       # CPU + A100 + GH200
#   GPU=none  bash jobs/submit_overnight.sh       # CPU only
#   SUITE=paper GPU=gh200 bash jobs/submit_overnight.sh
#
# Grace Hopper needs a separate aarch64 env first:
#   interactive -c 8 -G 1 -t 60 -p arm -q public --mem=64G
#   bash jobs/sol_setup_gh200.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs results/shards

module load mamba/latest
# shellcheck disable=SC1091
# List-jobs can use any working env (x86 is fine on login/CPU).
source activate "${EIGENFLOW_ENV:-eigenflow}"

SUITE="${SUITE:-fair}"
MODELS="${MODELS:-spline,fourier,qkan,mlp}"
GPU="${GPU:-a100}"   # a100 | gh200 | both | none

echo "=== Suite=${SUITE} models=${MODELS} GPU=${GPU} ==="
python experiments/run_benchmark.py --suite "${SUITE}" --models "${MODELS}" --list-jobs

CPU_SHARDS="${CPU_SHARDS:-64}"
GPU_SHARDS="${GPU_SHARDS:-32}"
GH200_SHARDS="${GH200_SHARDS:-32}"
GH200_ENV="${GH200_ENV:-eigenflow-gh200}"

JOB_IDS=()

echo ""
echo "Submitting CPU array 0-$((CPU_SHARDS - 1)) ..."
CPU_JOB=$(sbatch --parsable --array="0-$((CPU_SHARDS - 1))" \
  --export=NONE,N_SHARDS="${CPU_SHARDS}",EIGENFLOW_ENV="${EIGENFLOW_ENV:-eigenflow}",SUITE="${SUITE}",MODELS="${MODELS}" \
  jobs/sol_cpu_array.sbatch)
echo "CPU array job id: ${CPU_JOB}"
JOB_IDS+=("${CPU_JOB}")

if [[ "${GPU}" == "a100" || "${GPU}" == "both" ]]; then
  echo "Submitting A100 GPU array 0-$((GPU_SHARDS - 1)) ..."
  A100_JOB=$(sbatch --parsable --array="0-$((GPU_SHARDS - 1))" \
    --export=NONE,N_SHARDS="${GPU_SHARDS}",EIGENFLOW_ENV="${EIGENFLOW_ENV:-eigenflow}",SUITE="${SUITE}",MODELS="${MODELS}" \
    jobs/sol_gpu_array.sbatch)
  echo "A100 array job id: ${A100_JOB}"
  JOB_IDS+=("${A100_JOB}")
fi

if [[ "${GPU}" == "gh200" || "${GPU}" == "both" ]]; then
  echo "Submitting Grace Hopper (arm/GH200) array 0-$((GH200_SHARDS - 1)) ..."
  echo "  using env: ${GH200_ENV} (must be aarch64 — see jobs/sol_setup_gh200.sh)"
  GH_JOB=$(sbatch --parsable --array="0-$((GH200_SHARDS - 1))" \
    --export=NONE,N_SHARDS="${GH200_SHARDS}",EIGENFLOW_ENV="${GH200_ENV}",SUITE="${SUITE}",MODELS="${MODELS}" \
    jobs/sol_gh200_array.sbatch)
  echo "GH200 array job id: ${GH_JOB}"
  JOB_IDS+=("${GH_JOB}")
fi

if [[ "${GPU}" == "none" ]]; then
  echo "Skipping GPU arrays (GPU=none)."
fi

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Cancel:   scancel ${JOB_IDS[*]}"
echo "When done:"
echo "  python experiments/merge_results.py --glob 'results/shards/*.csv' --out results/${SUITE}_benchmark.csv"
