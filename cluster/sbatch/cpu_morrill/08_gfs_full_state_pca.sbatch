#!/bin/bash
#SBATCH --job-name=gfs_full_pca
#SBATCH --partition=morrill
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/gfs_full_pca_%j.out
#SBATCH --error=logs/gfs_full_pca_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

# Gridded PCA-50 of the full GFS state over New England, one basis per year-ahead fold (src/gfs/gfs_full_state_pca.py).
# Needs all four years from 09_download_gfs_full_state.sbatch. Output: data/gfs_full/gfs_full_state_features.npz
# (~40 MB) -> copy to the local data/gfs/.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -d "${REPO_ROOT}/src" ] || { echo "Run sbatch from the repo root." >&2; exit 1; }
export SCRATCH_ROOT="${SCRATCH_ROOT:-${REPO_ROOT}}"
source "${CONDA_SH:-/scratch/morrill/users/hmp278/miniforge3/etc/profile.d/conda.sh}"
conda activate "$(cat "${SCRATCH_ROOT}/prithvi_env_path.txt")"
echo "started: $(date)  host: $(hostname)"
python "${REPO_ROOT}/src/gfs/gfs_full_state_pca.py" --dir "${SCRATCH_ROOT}/data/gfs_full"
echo "Job finished: $(date)"
