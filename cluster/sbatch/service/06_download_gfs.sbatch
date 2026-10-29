#!/bin/bash
#SBATCH --job-name=gfs_download
#SBATCH --partition=service
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/gfs_download_%j.out
#SBATCH --error=logs/gfs_download_%j.err

set -euo pipefail

# Show Python output in the log as it happens instead of in delayed chunks.
export PYTHONUNBUFFERED=1

# Archived GFS 24 h and 27 h forecasts over New England, for benchmarking the
# Prithvi decoder (src/gfs/download_gfs_forecasts.py). Public NOAA archive on
# AWS, no login; available from 2021. Output: data/gfs/gfs_ne_YEAR.nc (~100 MB).
# Override at submit time: sbatch --export=ALL,YEAR=2023 ...
YEAR="${YEAR:-2021}"
WORKERS="${WORKERS:-8}"
# Optional wider crop for regional features: BOX="32 50 -90 -64" (writes gfs_wide_YEAR.nc).
BOX="${BOX:-}"
echo "Year: ${YEAR}"
echo "Job started: $(date)"
echo "Host: $(hostname)"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -d "${REPO_ROOT}/src" ]; then
  echo "REPO_ROOT=${REPO_ROOT} doesn't look like the repo (no src/ dir). Run sbatch from the repo root." >&2
  exit 1
fi
export SCRATCH_ROOT="${SCRATCH_ROOT:-${REPO_ROOT}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/xdg_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRATCH_ROOT}/mpl_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_ROOT}/pip_cache}"
mkdir -p logs "${SCRATCH_ROOT}/data/gfs" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

source "${CONDA_SH:-/scratch/morrill/users/hmp278/miniforge3/etc/profile.d/conda.sh}"
conda activate "$(cat "${SCRATCH_ROOT}/prithvi_env_path.txt")"

# GRIB reader (pip wheels include the ecCodes library). Installed once.
python -c "import eccodes, requests" 2>/dev/null || pip install --quiet eccodes eccodeslib requests

python "${REPO_ROOT}/src/gfs/download_gfs_forecasts.py" \
  --year "${YEAR}" \
  --out-dir "${SCRATCH_ROOT}/data/gfs" \
  --workers "${WORKERS}" \
  ${BOX:+--box ${BOX} --name ${NAME:-wide}} \
  ${SOURCE:+--source ${SOURCE}} \
  ${ALSO_NE:+--also-ne}

echo "Job finished: $(date)"
