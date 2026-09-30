#!/bin/bash
#SBATCH --job-name=merra2_sample
#SBATCH --partition=service
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/merra2_sample_%j.out
#SBATCH --error=logs/merra2_sample_%j.err

set -euo pipefail
echo "Job started: $(date)"
echo "Host: $(hostname)"

export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/morrill/users/hmp278/weather_electricity_foundation}"
mkdir -p logs "${SCRATCH_ROOT}/data/merra2_sample"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$(cat "${SCRATCH_ROOT}/prithvi_env_path.txt")"

# 2 days only, per Phase 1 step 3: validate the pipeline before the full year.
START="${START:-2024-06-01}"
END="${END:-2024-06-02}"
LAT_MIN="${LAT_MIN:-39.0}"
LAT_MAX="${LAT_MAX:-48.0}"
LON_MIN="${LON_MIN:--80.0}"
LON_MAX="${LON_MAX:--65.0}"
VARIABLES_CONFIG="${VARIABLES_CONFIG:-${REPO_ROOT}/configs/merra2_variables.yaml}"

if [ ! -f "${VARIABLES_CONFIG}" ]; then
  echo "Missing ${VARIABLES_CONFIG}." >&2
  echo "Copy configs/merra2_variables.yaml.template and fill it in from the" >&2
  echo "Prithvi-WxC repo's own dataloader/config first (Phase 1 step 2)." >&2
  exit 1
fi

python "${REPO_ROOT}/src/merra2/download_merra2.py" \
  --variables-config "${VARIABLES_CONFIG}" \
  --start "${START}" --end "${END}" \
  --lat-min "${LAT_MIN}" --lat-max "${LAT_MAX}" \
  --lon-min "${LON_MIN}" --lon-max "${LON_MAX}" \
  --out-dir "${SCRATCH_ROOT}/data/merra2_sample"

echo "Job finished: $(date)"
