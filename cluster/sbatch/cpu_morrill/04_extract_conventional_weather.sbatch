#!/bin/bash
#SBATCH --job-name=extract_conventional_weather
#SBATCH --partition=morrill
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/extract_conventional_weather_%j.out
#SBATCH --error=logs/extract_conventional_weather_%j.err

set -euo pipefail

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

MERRA2_SURFACE_DIR="${MERRA2_SURFACE_DIR:-${SCRATCH_ROOT}/data/merra2_${YEAR}_daily}"
OUT_PARQUET="${OUT_PARQUET:-${SCRATCH_ROOT}/data/compact/conventional_weather_${YEAR}.parquet}"

python "${REPO_ROOT}/src/merra2/extract_conventional_weather.py" \
  --merra2-surface-dir "${MERRA2_SURFACE_DIR}" \
  --out-parquet "${OUT_PARQUET}"

echo "Job finished: $(date)"
