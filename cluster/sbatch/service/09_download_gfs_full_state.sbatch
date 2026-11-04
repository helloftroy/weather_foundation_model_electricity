#!/bin/bash
#SBATCH --job-name=gfs_full_state
#SBATCH --partition=service
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/gfs_full_state_%j.out
#SBATCH --error=logs/gfs_full_state_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

# Full GFS state (150 fields: 9 variables x 14 pressure levels + 25 single-level fields), 0.5 deg, 24/27 h forecasts,
# cropped to the Northeast (src/gfs/download_gfs_full_state.py). Public NOAA archive on AWS, no login.
# One year per job: sbatch --account=... --export=ALL,YEAR=2023 cluster/sbatch/service/09_download_gfs_full_state.sbatch
# Output in data/gfs_full/: gfs_full_YEAR_meta.npz, gfs_full_YEAR_ne_grid.npy (~1.8 GB), parts/ (~1.6 GB, deletable after).
YEAR="${YEAR:?Set YEAR}"
REGION="${REGION:-ne}"   # ne or texas (ERCOT replication: gfs_full_texas_YEAR_*)
WORKERS="${WORKERS:-16}"
echo "Year: ${YEAR}  started: $(date)  host: $(hostname)"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -d "${REPO_ROOT}/src" ] || { echo "Run sbatch from the repo root." >&2; exit 1; }
export SCRATCH_ROOT="${SCRATCH_ROOT:-${REPO_ROOT}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/xdg_cache}"
mkdir -p logs "${SCRATCH_ROOT}/data/gfs_full" "${XDG_CACHE_HOME}"

source "${CONDA_SH:-/scratch/morrill/users/hmp278/miniforge3/etc/profile.d/conda.sh}"
conda activate "$(cat "${SCRATCH_ROOT}/prithvi_env_path.txt")"
python -c "import eccodes, requests" 2>/dev/null || pip install --quiet eccodes eccodeslib requests

cd "${REPO_ROOT}/src/gfs"
python download_gfs_full_state.py --year "${YEAR}" --out-dir "${SCRATCH_ROOT}/data/gfs_full" --workers "${WORKERS}" --region "${REGION}"
echo "Job finished: $(date)"
