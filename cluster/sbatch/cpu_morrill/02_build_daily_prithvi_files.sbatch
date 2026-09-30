#!/bin/bash
#SBATCH --job-name=build_daily_prithvi_files
#SBATCH --partition=morrill
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/build_daily_prithvi_files_%j.out
#SBATCH --error=logs/build_daily_prithvi_files_%j.err

set -euo pipefail
echo "Job started: $(date)"
echo "Host: $(hostname)"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -d "${REPO_ROOT}/src" ]; then
  echo "REPO_ROOT=${REPO_ROOT} doesn't look like the repo (no src/ dir). Run sbatch from the repo root." >&2
  exit 1
fi
export SCRATCH_ROOT="${SCRATCH_ROOT:-${REPO_ROOT}}"
# Several libraries (matplotlib, misc XDG-aware tools) default to $HOME
# for cache dirs regardless of HF_HOME/PIP_CACHE_DIR -- redirect those too
# (a full $HOME caused a hard failure here before; see
# NEW_PROJECT_MORRILL_CONTEXT.md).
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/xdg_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRATCH_ROOT}/mpl_cache}"
mkdir -p logs "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

source "${CONDA_SH:-/scratch/morrill/users/hmp278/miniforge3/etc/profile.d/conda.sh}"
conda activate "$(cat "${SCRATCH_ROOT}/prithvi_env_path.txt")"

DATA_DIR="${DATA_DIR:-${SCRATCH_ROOT}/data/merra2_2024}"
VARIABLES_CONFIG="${VARIABLES_CONFIG:-${REPO_ROOT}/configs/merra2_variables.yaml}"
OUT_DIR_SURFACE="${OUT_DIR_SURFACE:-${SCRATCH_ROOT}/data/merra2_2024_daily/surface}"
OUT_DIR_VERTICAL="${OUT_DIR_VERTICAL:-${SCRATCH_ROOT}/data/merra2_2024_daily/vertical}"
START="${START:-2024-01-01}"
END="${END:-2024-12-31}"

python "${REPO_ROOT}/src/merra2/build_daily_prithvi_files.py" \
  --data-dir "${DATA_DIR}" \
  --variables-config "${VARIABLES_CONFIG}" \
  --out-dir-surface "${OUT_DIR_SURFACE}" \
  --out-dir-vertical "${OUT_DIR_VERTICAL}" \
  --start "${START}" --end "${END}"

echo "Job finished: $(date)"
