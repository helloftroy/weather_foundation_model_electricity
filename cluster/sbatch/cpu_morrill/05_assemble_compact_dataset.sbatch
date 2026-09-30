#!/bin/bash
#SBATCH --job-name=assemble_compact_dataset
#SBATCH --partition=morrill
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/assemble_compact_dataset_%j.out
#SBATCH --error=logs/assemble_compact_dataset_%j.err

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

# Step 1: parse raw ISO-NE JSON into a clean table (both series, kept
# separate -- real-time is the primary "actual load" series used downstream).
python "${REPO_ROOT}/src/isone/parse_isone_demand.py" \
  --isone-dir "${SCRATCH_ROOT}/data/isone_2024" \
  --series realtime_hourly_demand \
  --out-parquet "${SCRATCH_ROOT}/data/compact/isone_realtime_demand_2024.parquet"

python "${REPO_ROOT}/src/isone/parse_isone_demand.py" \
  --isone-dir "${SCRATCH_ROOT}/data/isone_2024" \
  --series dayahead_hourly_demand \
  --out-parquet "${SCRATCH_ROOT}/data/compact/isone_dayahead_demand_2024.parquet"

# Step 2: join embeddings + conventional weather + real-time demand.
python "${REPO_ROOT}/src/assemble_compact_dataset.py" \
  --embeddings-parquet "${SCRATCH_ROOT}/data/compact/prithvi_embeddings_2024.parquet" \
  --conventional-weather-parquet "${SCRATCH_ROOT}/data/compact/conventional_weather_2024.parquet" \
  --isone-demand-parquet "${SCRATCH_ROOT}/data/compact/isone_realtime_demand_2024.parquet" \
  --out-parquet "${SCRATCH_ROOT}/data/compact/modeling_input_2024.parquet"

echo "Job finished: $(date)"
echo "Transfer ${SCRATCH_ROOT}/data/compact/ to your local machine, e.g.:"
echo "  scp -r hpc2:${SCRATCH_ROOT}/data/compact/ ./"
