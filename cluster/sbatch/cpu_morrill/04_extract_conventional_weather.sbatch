#!/bin/bash
#SBATCH --job-name=extract_conventional_weather
#SBATCH --partition=morrill
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/extract_conventional_weather_%j.out
#SBATCH --error=logs/extract_conventional_weather_%j.err

set -euo pipefail
echo "Job started: $(date)"
echo "Host: $(hostname)"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -d "${REPO_ROOT}/src" ]; then
  echo "REPO_ROOT=${REPO_ROOT} doesn't look like the repo (no src/ dir). Run sbatch from the repo root." >&2
  exit 1
fi
export SCRATCH_ROOT="${SCRATCH_ROOT:-${REPO_ROOT}}"
mkdir -p logs

source "${CONDA_SH:-/scratch/morrill/users/hmp278/miniforge3/etc/profile.d/conda.sh}"
conda activate "$(cat "${SCRATCH_ROOT}/prithvi_env_path.txt")"

MERRA2_SURFACE_DIR="${MERRA2_SURFACE_DIR:-${SCRATCH_ROOT}/data/merra2_2024_daily/surface}"
OUT_PARQUET="${OUT_PARQUET:-${SCRATCH_ROOT}/data/compact/conventional_weather_2024.parquet}"

python "${REPO_ROOT}/src/merra2/extract_conventional_weather.py" \
  --merra2-surface-dir "${MERRA2_SURFACE_DIR}" \
  --out-parquet "${OUT_PARQUET}"

echo "Job finished: $(date)"
