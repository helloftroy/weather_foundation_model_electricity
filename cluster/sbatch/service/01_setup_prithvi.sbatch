#!/bin/bash
#SBATCH --job-name=prithvi_setup
#SBATCH --partition=service
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/prithvi_setup_%j.out
#SBATCH --error=logs/prithvi_setup_%j.err

set -euo pipefail

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"

# TODO: confirm this matches wherever the project actually lives on the
# cluster once created (see PROJECT_NOTES.md step 0).
export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/morrill/users/hmp278/weather_electricity_foundation}"
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_ROOT}/pip_cache}"
export TMPDIR="${TMPDIR:-${SCRATCH_ROOT}/tmp}"
mkdir -p "${SCRATCH_ROOT}" "${HF_HOME}" "${PIP_CACHE_DIR}" "${TMPDIR}" logs

echo "SCRATCH_ROOT=${SCRATCH_ROOT}"
echo "HF_HOME=${HF_HOME}"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

bash "${REPO_ROOT}/src/prithvi/setup_prithvi_env.sh"

source "$(conda info --base)/etc/profile.d/conda.sh"
PRITHVI_ENV="$(cat "${SCRATCH_ROOT}/prithvi_env_path.txt")"
conda activate "${PRITHVI_ENV}"
echo "Activated env for weight download: ${PRITHVI_ENV}"

python "${REPO_ROOT}/src/prithvi/download_prithvi_weights.py" \
  --out-dir "${SCRATCH_ROOT}/weights/prithvi.wxc.2300m.v1"

echo "Job finished: $(date)"
