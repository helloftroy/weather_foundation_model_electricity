#!/bin/bash
#SBATCH --job-name=crop_climatology
#SBATCH --partition=morrill
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/crop_climatology_%j.out
#SBATCH --error=logs/crop_climatology_%j.err

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

CLIMATOLOGY_DIR="${CLIMATOLOGY_DIR:-${SCRATCH_ROOT}/weights/prithvi.wxc.2300m.v1/climatology}"
OUT_DIR="${OUT_DIR:-${SCRATCH_ROOT}/data/climatology_regional}"
LAT_MIN="${LAT_MIN:-30.0}"
LAT_MAX="${LAT_MAX:-59.5}"
LON_MIN="${LON_MIN:--90.0}"
LON_MAX="${LON_MAX:--50.625}"

python "${REPO_ROOT}/src/merra2/crop_climatology.py" \
  --climatology-dir "${CLIMATOLOGY_DIR}" \
  --out-dir "${OUT_DIR}" \
  --lat-min "${LAT_MIN}" --lat-max "${LAT_MAX}" \
  --lon-min "${LON_MIN}" --lon-max "${LON_MAX}"

echo "Job finished: $(date)"
