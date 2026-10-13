#!/bin/bash
#SBATCH --job-name=merra2_2024
#SBATCH --partition=service
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/merra2_%j.out
#SBATCH --error=logs/merra2_%j.err

set -euo pipefail

# Show Python output in the log as it happens instead of in delayed chunks.
export PYTHONUNBUFFERED=1

# Year to process. Override at submit time: sbatch --export=ALL,YEAR=2023 ...
YEAR="${YEAR:-2024}"
echo "Year: ${YEAR}"
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
# Several libraries (matplotlib, misc XDG-aware tools) default to $HOME
# for cache dirs regardless of HF_HOME/PIP_CACHE_DIR -- redirect those too
# (a full $HOME caused a hard failure here before; see
# NEW_PROJECT_MORRILL_CONTEXT.md).
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/xdg_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRATCH_ROOT}/mpl_cache}"
mkdir -p logs "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

source "${CONDA_SH:-/scratch/morrill/users/hmp278/miniforge3/etc/profile.d/conda.sh}"
conda activate "$(cat "${SCRATCH_ROOT}/prithvi_env_path.txt")"

# GRID=global (default) runs on the full MERRA-2 grid, which is the only
# setup in which the model performs properly (see sample_validation.py).
# GRID=regional is the old 30-60N, 90-50W crop, kept for comparison.
GRID="${GRID:-global}"
if [ "${GRID}" = "global" ]; then
  OUT_DIR="${OUT_DIR:-${SCRATCH_ROOT}/data/merra2_global_${YEAR}}"
  LAT_MIN="${LAT_MIN:--90.0}"; LAT_MAX="${LAT_MAX:-90.0}"; LON_MIN="${LON_MIN:--180.0}"; LON_MAX="${LON_MAX:-180.0}"
  # Whole files are downloaded and cut down locally: reading a full global
  # field through remote range requests was far too slow.
  EXTRA_ARGS="--whole-file --tmp-dir ${SCRATCH_ROOT}/tmp/merra_raw_${SLURM_JOB_ID:-$$}"
else
  OUT_DIR="${OUT_DIR:-${SCRATCH_ROOT}/data/merra2_${YEAR}}"
  EXTRA_ARGS=""
fi
mkdir -p "${OUT_DIR}"
START="${START:-${YEAR}-01-01}"
END="${END:-${YEAR}-12-31}"
LAT_MIN="${LAT_MIN:-30.0}"
LAT_MAX="${LAT_MAX:-59.5}"
LON_MIN="${LON_MIN:--90.0}"
LON_MAX="${LON_MAX:--50.625}"
VARIABLES_CONFIG="${VARIABLES_CONFIG:-${REPO_ROOT}/configs/merra2_variables.yaml}"

python "${REPO_ROOT}/src/merra2/download_merra2.py" \
  --variables-config "${VARIABLES_CONFIG}" \
  --start "${START}" --end "${END}" \
  --lat-min "${LAT_MIN}" --lat-max "${LAT_MAX}" \
  --lon-min "${LON_MIN}" --lon-max "${LON_MAX}" \
  --out-dir "${OUT_DIR}" ${EXTRA_ARGS}

echo "Job finished: $(date)"
