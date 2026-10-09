#!/bin/bash
#SBATCH --job-name=isone_lmp_2024
#SBATCH --partition=service
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=logs/isone_lmp_%j.out
#SBATCH --error=logs/isone_lmp_%j.err

set -euo pipefail

# Year to process. Override at submit time: sbatch --export=ALL,YEAR=2023 ...
YEAR="${YEAR:-2024}"
echo "Year: ${YEAR}"
echo "Job started: $(date)"
echo "Host: $(hostname)"

# Requires ISONE_WS_USERNAME / ISONE_WS_PASSWORD to be set (ISO Express
# account, see src/isone/download_isone_lmp.py docstring). Export them in
# the shell before `sbatch`-ing this, or source a gitignored credentials.sh.

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
mkdir -p logs "${SCRATCH_ROOT}/data/isone_lmp_${YEAR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

source "${CONDA_SH:-/scratch/morrill/users/hmp278/miniforge3/etc/profile.d/conda.sh}"
conda activate "$(cat "${SCRATCH_ROOT}/prithvi_env_path.txt")"

python "${REPO_ROOT}/src/isone/download_isone_lmp.py" \
  --start "${START:-${YEAR}-01-01}" \
  --end "${END:-${YEAR}-12-31}" \
  --out-dir "${SCRATCH_ROOT}/data/isone_lmp_${YEAR}"

echo "Job finished: $(date)"
