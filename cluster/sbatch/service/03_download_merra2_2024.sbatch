#!/bin/bash
#SBATCH --job-name=merra2_2024
#SBATCH --partition=service
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/merra2_%j.out
#SBATCH --error=logs/merra2_%j.err

set -euo pipefail

# Year to process. Override at submit time: sbatch --export=ALL,YEAR=2023 ...
YEAR="${YEAR:-2024}"
echo "Year: ${YEAR}"
echo "Job started: $(date)"
echo "Host: $(hostname)"

# Only run this after 02_download_merra2_sample.sbatch has been verified to
# work end-to-end with the real Prithvi preprocessing pipeline.

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
mkdir -p logs "${SCRATCH_ROOT}/data/merra2_${YEAR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

source "${CONDA_SH:-/scratch/morrill/users/hmp278/miniforge3/etc/profile.d/conda.sh}"
conda activate "$(cat "${SCRATCH_ROOT}/prithvi_env_path.txt")"

START="${START:-${YEAR}-01-01}"
END="${END:-${YEAR}-12-31}"
LAT_MIN="${LAT_MIN:-30.0}"
LAT_MAX="${LAT_MAX:-59.5}"
LON_MIN="${LON_MIN:--90.0}"
LON_MAX="${LON_MAX:--50.625}"
VARIABLES_CONFIG="${VARIABLES_CONFIG:-${REPO_ROOT}/configs/merra2_variables.yaml}"

python "${REPO_ROOT}/src/merra2/download_merra2.py" \
  --variables-config "${VARIABLES_CONFIG}" \
  --start "${START}" --end "${END}" \
  --lat-min "${LAT_MIN}" --lat-max "${LAT_MAX}" \
  --lon-min "${LON_MIN}" --lon-max "${LON_MAX}" \
  --out-dir "${SCRATCH_ROOT}/data/merra2_${YEAR}"

echo "Job finished: $(date)"
