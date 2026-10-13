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
# SKIP_MERRA=1 skips the MERRA-2 download and daily-file build, for a year
# whose daily files already exist (2020 ships with the model weights).
# GRID=global (the default in every job) uses the full MERRA-2 grid.
#
# Downloads run one year at a time (ISO-NE rate-limits, and the service node
# is slow), with each year's MERRA-2 fetched as four quarters in parallel.
# Everything else for a year starts as soon as its inputs exist. Safe to
# re-run for a year that was interrupted: finished files are skipped.
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

submit() {  # submit NAME DEPENDENCY SCRIPT [EXTRA_EXPORTS] -> prints job id
  local name="$1" dep="$2" script="$3" extra="${4:-}"
  local args=(--parsable --account="${ACCOUNT}" --job-name="${name}_${YEAR}" --export=ALL,YEAR="${YEAR}${extra:+,${extra}}")
  [ -n "${dep}" ] && args+=(--dependency="${dep}")
  sbatch "${args[@]}" "${script}"
}

YEARS=("$@")
for YEAR in "${YEARS[@]}"; do
  [[ "${YEAR}" =~ ^(19|20)[0-9]{2}$ ]] || { echo "Not a year: ${YEAR}" >&2; exit 1; }
  export YEAR

  demand=$(submit isone_demand_dl "${prev_isone:+afterany:${prev_isone}}" $S/service/04_download_isone_demand_2024.sbatch)
  lmp=$(submit isone_lmp_dl "afterany:${demand}" $S/service/05_download_isone_lmp_2024.sbatch)
  prev_isone="${lmp}"
  parse_lmp=$(submit parse_lmp "afterok:${lmp}" $S/cpu_morrill/06_parse_isone_lmp.sbatch)

  if [ "${SKIP_MERRA:-0}" = "1" ]; then
    after_daily=""
    line="${YEAR}: demand ${demand}, lmp ${lmp}, parse_lmp ${parse_lmp} (MERRA-2 steps skipped)"
  else
    # The MERRA-2 download is the slow step, so each year is fetched as four
    # quarters in parallel. Files are written under a temporary name and
    # renamed, so the quarters cannot corrupt each other.
    merra=""
    for q in "01-01 03-31" "04-01 06-30" "07-01 09-30" "10-01 12-31"; do
      set -- $q
      id=$(submit "merra2_dl_${1%%-*}" "${prev_merra:+afterany:${prev_merra}}" $S/service/03_download_merra2_2024.sbatch "START=${YEAR}-$1,END=${YEAR}-$2")
      merra="${merra:+${merra}:}${id}"
    done
    prev_merra="${merra}"
    daily=$(submit build_daily "afterok:${merra}" $S/cpu_morrill/02_build_daily_prithvi_files.sbatch)
    after_daily="afterok:${daily}"
    line="${YEAR}: merra2 ${merra}, daily ${daily}, demand ${demand}, lmp ${lmp}, parse_lmp ${parse_lmp}"
  fi
  weather=$(submit conv_weather "${after_daily}" $S/cpu_morrill/04_extract_conventional_weather.sbatch)
  line="${line}, weather ${weather}"

  if [ "${SKIP_EMBEDDINGS:-0}" != "1" ]; then
    emb=$(submit embeddings "${after_daily}" $S/gpu/02_extract_embeddings.sbatch)
    assemble=$(submit assemble "afterok:${emb}:${weather}:${demand}" $S/cpu_morrill/05_assemble_compact_dataset.sbatch)
    line="${line}, embeddings ${emb}, assemble ${assemble}"
  fi
  echo "${line}"
done
echo "Watch with: squeue -u \$USER    Outputs land in data/compact/*_<year>.parquet"
