#!/bin/bash
#SBATCH --job-name=qc_merra2_isone
#SBATCH --partition=morrill
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/qc_merra2_isone_%j.out
#SBATCH --error=logs/qc_merra2_isone_%j.err

# NOTE: "morrill" is used here as instructed (the CPU node reserved for
# non-internet work). Confirm this is the correct --partition value with
# `sinfo` on the cluster -- if morrill is actually a specific node within a
# different partition, swap this for e.g. `--partition=<name> --nodelist=morrill`.

set -euo pipefail
echo "Job started: $(date)"
echo "Host: $(hostname)"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -d "${REPO_ROOT}/src" ]; then
  echo "REPO_ROOT=${REPO_ROOT} doesn't look like the repo (no src/ dir). Run sbatch from the repo root." >&2
  exit 1
fi
export SCRATCH_ROOT="${SCRATCH_ROOT:-${REPO_ROOT}}"
mkdir -p logs "${SCRATCH_ROOT}/qc"

source "${CONDA_SH:-/scratch/morrill/users/hmp278/miniforge3/etc/profile.d/conda.sh}"
conda activate "$(cat "${SCRATCH_ROOT}/prithvi_env_path.txt")"

python "${REPO_ROOT}/src/qc/qc_merra2_isone.py" \
  --merra2-dir "${SCRATCH_ROOT}/data/merra2_2024" \
  --isone-dir "${SCRATCH_ROOT}/data/isone_2024" \
  --out-json "${SCRATCH_ROOT}/qc/qc_report_2024.json"

echo "Job finished: $(date)"
