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

# $0 is unreliable under sbatch (SLURM runs a spooled copy of the script, not
# the original path), so locate the repo via SLURM_SUBMIT_DIR -- the
# directory `sbatch` was invoked from -- instead.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -d "${REPO_ROOT}/src" ]; then
  echo "REPO_ROOT=${REPO_ROOT} doesn't look like the repo (no src/ dir). Run sbatch from the repo root." >&2
  exit 1
fi

# Data/weights/caches are gitignored subdirs co-located with the repo by
# default, so there's nothing to keep in sync with a separately-guessed path.
export SCRATCH_ROOT="${SCRATCH_ROOT:-${REPO_ROOT}}"
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_ROOT}/pip_cache}"
export TMPDIR="${TMPDIR:-${SCRATCH_ROOT}/tmp}"
mkdir -p "${SCRATCH_ROOT}" "${HF_HOME}" "${PIP_CACHE_DIR}" "${TMPDIR}" logs

echo "REPO_ROOT=${REPO_ROOT}"
echo "SCRATCH_ROOT=${SCRATCH_ROOT}"
echo "HF_HOME=${HF_HOME}"

bash "${REPO_ROOT}/src/prithvi/setup_prithvi_env.sh"

source "$(conda info --base)/etc/profile.d/conda.sh"
PRITHVI_ENV="$(cat "${SCRATCH_ROOT}/prithvi_env_path.txt")"
conda activate "${PRITHVI_ENV}"
echo "Activated env for weight download: ${PRITHVI_ENV}"

python "${REPO_ROOT}/src/prithvi/download_prithvi_weights.py" \
  --out-dir "${SCRATCH_ROOT}/weights/prithvi.wxc.2300m.v1"

echo "Job finished: $(date)"
