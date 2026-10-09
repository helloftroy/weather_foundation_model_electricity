#!/bin/bash
# Submit the whole pipeline for one or more years, with each step waiting on
# the ones it needs.
#
#   export ACCOUNT=...                       # Slurm account
#   source credentials.sh                    # EARTHDATA_USERNAME/PASSWORD, ISONE_WS_USERNAME/PASSWORD
#   cluster/submit_year.sh 2022 2023
#
# SKIP_EMBEDDINGS=1 submits only the downloads, daily files, weather table
# and price parsing (no GPU job, no final assembly).
#
# Downloads run one year at a time (ISO-NE rate-limits, and the service node
# is slow); everything else for a year starts as soon as its inputs exist.
# A year whose download fails stops there; later years still run.
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: cluster/submit_year.sh YEAR [YEAR ...]" >&2
  exit 1
fi
: "${ACCOUNT:?Set ACCOUNT to your Slurm account}"
for v in EARTHDATA_USERNAME EARTHDATA_PASSWORD ISONE_WS_USERNAME ISONE_WS_PASSWORD; do
  [ -n "${!v:-}" ] || { echo "Set ${v} first (e.g. source credentials.sh)." >&2; exit 1; }
done
[ -d cluster/sbatch ] || { echo "Run from the repo root." >&2; exit 1; }

S=cluster/sbatch
prev_merra=""
prev_isone=""

submit() {  # submit NAME DEPENDENCY SCRIPT -> prints job id
  local name="$1" dep="$2" script="$3"
  local args=(--parsable --account="${ACCOUNT}" --job-name="${name}_${YEAR}" --export=ALL,YEAR="${YEAR}")
  [ -n "${dep}" ] && args+=(--dependency="${dep}")
  sbatch "${args[@]}" "${script}"
}

for YEAR in "$@"; do
  [[ "${YEAR}" =~ ^(19|20)[0-9]{2}$ ]] || { echo "Not a year: ${YEAR}" >&2; exit 1; }
  export YEAR

  merra=$(submit merra2_dl "${prev_merra:+afterany:${prev_merra}}" $S/service/03_download_merra2_2024.sbatch)
  demand=$(submit isone_demand_dl "${prev_isone:+afterany:${prev_isone}}" $S/service/04_download_isone_demand_2024.sbatch)
  lmp=$(submit isone_lmp_dl "afterany:${demand}" $S/service/05_download_isone_lmp_2024.sbatch)
  prev_merra="${merra}"
  prev_isone="${lmp}"

  daily=$(submit build_daily "afterok:${merra}" $S/cpu_morrill/02_build_daily_prithvi_files.sbatch)
  weather=$(submit conv_weather "afterok:${daily}" $S/cpu_morrill/04_extract_conventional_weather.sbatch)
  parse_lmp=$(submit parse_lmp "afterok:${lmp}" $S/cpu_morrill/06_parse_isone_lmp.sbatch)
  line="${YEAR}: merra2 ${merra}, demand ${demand}, lmp ${lmp}, daily ${daily}, weather ${weather}, parse_lmp ${parse_lmp}"

  if [ "${SKIP_EMBEDDINGS:-0}" != "1" ]; then
    emb=$(submit embeddings "afterok:${daily}" $S/gpu/02_extract_embeddings.sbatch)
    assemble=$(submit assemble "afterok:${emb}:${weather}:${demand}" $S/cpu_morrill/05_assemble_compact_dataset.sbatch)
    line="${line}, embeddings ${emb}, assemble ${assemble}"
  fi
  echo "${line}"
done
echo "Watch with: squeue -u \$USER    Outputs land in data/compact/*_<year>.parquet"
