#!/bin/bash
#SBATCH --job-name=merra2_2024
#SBATCH --partition=service
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/merra2_2024_%j.out
#SBATCH --error=logs/merra2_2024_%j.err

set -euo pipefail
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
mkdir -p logs "${SCRATCH_ROOT}/data/merra2_2024"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$(cat "${SCRATCH_ROOT}/prithvi_env_path.txt")"

START="${START:-2024-01-01}"
END="${END:-2024-12-31}"
LAT_MIN="${LAT_MIN:-39.0}"
LAT_MAX="${LAT_MAX:-48.0}"
LON_MIN="${LON_MIN:--80.0}"
LON_MAX="${LON_MAX:--65.0}"
VARIABLES_CONFIG="${VARIABLES_CONFIG:-${REPO_ROOT}/configs/merra2_variables.yaml}"

python "${REPO_ROOT}/src/merra2/download_merra2.py" \
  --variables-config "${VARIABLES_CONFIG}" \
  --start "${START}" --end "${END}" \
  --lat-min "${LAT_MIN}" --lat-max "${LAT_MAX}" \
  --lon-min "${LON_MIN}" --lon-max "${LON_MAX}" \
  --out-dir "${SCRATCH_ROOT}/data/merra2_2024"

echo "Job finished: $(date)"
