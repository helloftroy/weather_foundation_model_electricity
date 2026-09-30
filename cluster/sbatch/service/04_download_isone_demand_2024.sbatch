#!/bin/bash
#SBATCH --job-name=isone_demand_2024
#SBATCH --partition=service
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=logs/isone_demand_2024_%j.out
#SBATCH --error=logs/isone_demand_2024_%j.err

set -euo pipefail
echo "Job started: $(date)"
echo "Host: $(hostname)"

# Requires ISONE_WS_USERNAME / ISONE_WS_PASSWORD to be set (ISO Express
# account, see src/isone/download_isone_demand.py docstring). Export them in
# the shell before `sbatch`-ing this, or source a gitignored credentials.sh.

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -d "${REPO_ROOT}/src" ]; then
  echo "REPO_ROOT=${REPO_ROOT} doesn't look like the repo (no src/ dir). Run sbatch from the repo root." >&2
  exit 1
fi
export SCRATCH_ROOT="${SCRATCH_ROOT:-${REPO_ROOT}}"
mkdir -p logs "${SCRATCH_ROOT}/data/isone_2024"

source "${CONDA_SH:-/scratch/morrill/users/hmp278/miniforge3/etc/profile.d/conda.sh}"
conda activate "$(cat "${SCRATCH_ROOT}/prithvi_env_path.txt")"

python "${REPO_ROOT}/src/isone/download_isone_demand.py" \
  --start "${START:-2024-01-01}" \
  --end "${END:-2024-12-31}" \
  --out-dir "${SCRATCH_ROOT}/data/isone_2024"

echo "Job finished: $(date)"
