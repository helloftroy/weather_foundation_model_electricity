#!/bin/bash
#SBATCH --job-name=extract_conventional_weather
#SBATCH --partition=morrill
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/extract_conventional_weather_%j.out
#SBATCH --error=logs/extract_conventional_weather_%j.err

set -euo pipefail

# Show Python output in the log as it happens instead of in delayed chunks.
export PYTHONUNBUFFERED=1

# Year to process. Override at submit time: sbatch --export=ALL,YEAR=2023 ...
YEAR="${YEAR:-2024}"
echo "Year: ${YEAR}"
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

WEIGHTS_DIR="${WEIGHTS_DIR:-${SCRATCH_ROOT}/weights/prithvi.wxc.2300m.v1}"
# GRID=global (default) runs on the full MERRA-2 grid, which is the only
# setup in which the model performs properly (see sample_validation.py).
# GRID=regional is the old 30-60N, 90-50W crop, kept for comparison.
GRID="${GRID:-global}"
if [ "${GRID}" = "global" ]; then
  DAILY_DIR="${SCRATCH_ROOT}/data/merra2_global_${YEAR}_daily"
  # 2020 ships ready-made with the model weights; use it unless we built our own.
  if [ ! -d "${DAILY_DIR}" ] && [ "${YEAR}" = "2020" ]; then DAILY_DIR="${WEIGHTS_DIR}/merra-2"; fi
  DEFAULT_CLIMATOLOGY="${WEIGHTS_DIR}/climatology"
else
  DAILY_DIR="${SCRATCH_ROOT}/data/merra2_${YEAR}_daily"
  DEFAULT_CLIMATOLOGY="${SCRATCH_ROOT}/data/climatology_regional"
fi
echo "Grid: ${GRID}; daily MERRA-2 files: ${DAILY_DIR}"
MERRA2_SURFACE_DIR="${MERRA2_SURFACE_DIR:-${DAILY_DIR}}"
OUT_PARQUET="${OUT_PARQUET:-${SCRATCH_ROOT}/data/compact/conventional_weather_${YEAR}.parquet}"

python "${REPO_ROOT}/src/merra2/extract_conventional_weather.py" \
  --merra2-surface-dir "${MERRA2_SURFACE_DIR}" \
  --year "${YEAR}" \
  --out-parquet "${OUT_PARQUET}"

echo "Job finished: $(date)"
