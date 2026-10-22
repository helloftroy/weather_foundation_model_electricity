#!/bin/bash
#SBATCH --job-name=parse_isone_demand
#SBATCH --partition=morrill
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/parse_isone_demand_%j.out
#SBATCH --error=logs/parse_isone_demand_%j.err

set -euo pipefail

# Show Python output in the log as it happens instead of in delayed chunks.
export PYTHONUNBUFFERED=1

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

# Raw ISO-NE demand JSON -> tables. The same two steps run inside
# 05_assemble_compact_dataset.sbatch; this job does them on their own, for
# years where the embeddings (which assembly waits for) are not run yet.
for SERIES in realtime dayahead; do
  python "${REPO_ROOT}/src/isone/parse_isone_demand.py" \
    --isone-dir "${SCRATCH_ROOT}/data/isone_${YEAR}" \
    --series "${SERIES}_hourly_demand" \
    --out-parquet "${SCRATCH_ROOT}/data/compact/isone_${SERIES}_demand_${YEAR}.parquet"
done

echo "Job finished: $(date)"
