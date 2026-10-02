# Project Notes: Weather Foundation Model → Electricity Demand

## Background

Testing whether representations from a frozen, pretrained weather foundation
model (**Prithvi WxC**, NASA-IMPACT/IBM, 2.3B params, pretrained on global
MERRA-2) encode information useful for predicting **electricity demand**, and
later prices, better than conventional summarized weather variables.

Pipeline: `MERRA-2 (regional Northeast cube) → frozen Prithvi WxC → weather
embedding → CatBoost → electricity demand`. Prithvi is not retrained. First
target: 2024 hourly electricity demand across all 8 ISO-NE load zones
(ME, NH, VT, CT, RI, NEMA, SEMA, WCMA). Later stages (not in scope yet):
MA zonal prices (NEMA/Boston, SEMA, WCMA), demand→price chaining,
wind/solar generation, ERCOT as a contrast market.

Downstream models will also get explicit calendar/context features (hour,
day-of-week, day-of-year, holiday, year/trend, zone) alongside the Prithvi
embedding.

## Repo

GitHub: https://github.com/helloftroy/weather_foundation_model_electricity
Cloned onto the cluster; large data/weights/caches live under
`/scratch/morrill/users/hmp278/weather_foundation_model_electricity/`, never `$HOME`.

## Cluster logistics

- **service** partition: internet access, used for cloning repos, downloading
  weights/climatology/MERRA-2/ISO-NE data. Slow for compute -- no heavy CPU
  work here.
- **gpu-a100** partition: Prithvi inference / embedding extraction.
- **morrill** partition: CPU-only work with no internet need (QC,
  preprocessing, later CatBoost). *(Confirm this is the correct `--partition`
  value with `sinfo` once on the cluster -- see the note in
  `cluster/sbatch/cpu_morrill/01_qc_merra2_isone.sbatch`.)*
- Every job is submitted with `#SBATCH --time=48:00:00` regardless of
  expected runtime -- no queueing penalty for requesting the full 48h, and it
  avoids "job got killed for running long" failures (slow downloads,
  rate-limit backoff, etc.) that otherwise cost a full debug-resubmit cycle.
- Environment: reusing the existing **faire-agent** conda env
  (`/scratch/morrill/users/hmp278/conda_envs/faire-agent`, from
  `FAIRe_Ocean_Agent`) -- confirmed python 3.11.15, torch 2.13.0+cu130, which
  is compatible, so Prithvi-WxC installs into it rather than a new env
  (`cuda.is_available()` reads False on the login node since there's no GPU
  there; that's expected, not a disqualifier). Conda itself isn't on `PATH`
  on login/compute nodes -- sourced from
  `/scratch/morrill/users/hmp278/miniforge3/etc/profile.d/conda.sh`, same as
  `FAIRe_Ocean_Agent/cluster/env_activate.sh` does. `HF_HOME`, pip cache, and
  tmp are all redirected into scratch. See `src/prithvi/setup_prithvi_env.sh`.

## Directory layout on the cluster

```
/scratch/morrill/users/hmp278/weather_foundation_model_electricity/
├── Prithvi-WxC/          # cloned NASA-IMPACT repo
├── envs/prithvi_wxc/     # dedicated conda env, only if FAIRe wasn't reusable
├── weights/              # HF snapshot of prithvi.wxc.2300m.v1
├── hf_cache/              # HF_HOME
├── pip_cache/
├── tmp/
├── data/
│   ├── merra2_sample/    # 1-2 day validation pull
│   ├── merra2_2024/      # full year, organized by MERRA-2 collection
│   └── isone_2024/       # raw per-day JSON + index CSVs, by series
└── qc/                   # QC report(s)
```

Designed so expanding from 2024 to ~10 years later is just new subdirectories
under `data/`, not a redesign.

## Commands, in order

All commands assume this repo is cloned at, e.g.,
`~/weather_foundation_model_electricity` or directly under scratch, and are
run via `sbatch` from the repo root (relative `logs/` output paths assume
this). Steps 0-1 need `service`; step 2 needs `gpu-a100`; steps 3-4 need
`service`; step 5 needs `morrill`.

### 0. Reconnaissance (done once, manually, no job needed)

```bash
# Inspect multimodal_granite's .sbatch conventions and FAIRe env
ls ~/multimodal_granite
cat ~/multimodal_granite/fish/run_extract_granite_fish_embeddings.sbatch
conda env list
conda activate FAIRe && python -c "import torch; print(torch.__version__)"

# Check scratch space
df -h /scratch/morrill/users/hmp278/
mkdir -p /scratch/morrill/users/hmp278/weather_foundation_model_electricity
```

### 1. Set up Prithvi WxC (service)

```bash (DONE ~~)
sbatch --account=191001-364393 cluster/sbatch/service/01_setup_prithvi.sbatch
```
Clones NASA-IMPACT/Prithvi-WxC, resolves FAIRe-vs-new-env, `pip install
-e .[examples]`, downloads `Prithvi-WxC/prithvi.wxc.2300m.v1` weights +
climatology into scratch via `huggingface_hub`.

### 2. Sample inference validation (gpu-a100)

```bash (DONE ~~)
sbatch --account=191001-364393 cluster/sbatch/gpu/01_prithvi_sample_inference.sbatch
```
Runs `src/prithvi/sample_validation.py` -- the same official
`Merra2Dataset`/`PrithviWxC`/`preproc` code the notebook uses, but pointed at
the sample MERRA-2/climatology data already downloaded in step 1, instead of
running `examples/PrithviWxC_inference.ipynb` directly. That notebook's own
download cells point at `ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M`, whose
`merra-2/`/`climatology/` directories are empty upstream (confirmed via
direct HTTP checks, 2026-09-30) -- likely deprecated in favor of
`Prithvi-WxC/prithvi.wxc.2300m.v1`, which we already fully download in step
1. This also directly exercises the encoder-hook embedding extraction from
`docs/prithvi_embedding_candidates.md` against real weights, printing the
observed `global_shape_mu`/embed_dim for a sanity check before the full
Phase 2 run.

### 3. MERRA-2: sample, then full year (service)

Variable list and region bounds are already confirmed and filled in
(`configs/merra2_variables.yaml`, `docs/merra2_region_and_variables.md`) --
sourced directly from the pretrained model's own `config.yaml` and official
inference notebook, not guessed. Re-run `src/merra2/define_region.py` only
if the region needs to change.

```bash (DONE~~)
sbatch --account=191001-364393 cluster/sbatch/service/02_download_merra2_sample.sbatch   # 2 days, validate pipeline

sbatch --account=191001-364393 \
  --export=ALL,EARTHDATA_USERNAME='parkmhelen',EARTHDATA_PASSWORD='!U$xPzmDS)%&%!8' \
  cluster/sbatch/service/02_download_merra2_sample.sbatch

## DONE~~
sbatch --account=191001-364393 \
  --export=ALL,EARTHDATA_USERNAME='parkmhelen',EARTHDATA_PASSWORD='!U$xPzmDS)%&%!8' \
  cluster/sbatch/service/03_download_merra2_2024.sbatch     # full year, only after 02 checks out
```

### 4. ISO-NE electricity demand, 2024 (service)

Requires a free ISO Express account (see `docs/isone_demand_sources.md`).

```bash (DONE~~)
export ISONE_WS_USERNAME="parkmhelen@gmail.com"
export ISONE_WS_PASSWORD="jSJ9!NkcCex4yjF"
sbatch --account=191001-364393 \
  --export=ALL,ISONE_WS_USERNAME='parkmhelen@gmail.com',ISONE_WS_PASSWORD='jSJ9!NkcCex4yjF' \
  cluster/sbatch/service/04_download_isone_demand_2024.sbatch
```

### 5. QC (morrill, CPU, no internet)

```bash
sbatch cluster/sbatch/cpu_morrill/01_qc_merra2_isone.sbatch
```
Produces `qc/qc_report_2024.json`: MERRA-2 timestamp coverage/variables and
ISO-NE row counts/zone/date coverage. Does not interpolate or merge the two
datasets.

## Phase 1 stopping point (from the source instructions)

Stop once we have: a validated Prithvi WxC install + weights; a successful
sample inference run; one full year of regional MERRA-2; one full year of
8-zone ISO-NE demand; basic timestamp/data QC; notes on candidate embedding
tensors; reproducible scripts/sbatch files. **Do not start CatBoost or
large-scale embedding extraction yet** -- that's Phase 2.

---

## Phase 2: extract compact representations, transfer locally

Goal: run the frozen Prithvi encoder over 2024, save compact per-timestamp
feature tables (not the raw MERRA-2 cube) to scratch, then transfer only
those small files locally. See `docs/prithvi_embedding_candidates.md` for
the full architecture derivation (extraction point, why regional cropping is
valid, exact tensor shapes) done ahead of running anything on the cluster.

Depends on Phase 1's MERRA-2 download (`data/merra2_2024/`, organized by GES
DISC collection) and the Prithvi weights/climatology
(`weights/prithvi.wxc.2300m.v1/`) already being in place.

### 1. Build Prithvi-format daily MERRA-2 files (morrill)

`Merra2Dataset` expects one merged `MERRA2_sfc_YYYYMMDD.nc` /
`MERRA_pres_YYYYMMDD.nc` file per day (its "global 361x576" shape properties
are dead code, never actually enforced -- confirmed by reading the source,
see `docs/prithvi_embedding_candidates.md`), so this merges our
per-GES-DISC-collection downloads into that format, resampling the
half-hour-offset tavg1 surface products onto the exact 3-hourly synoptic
grid the vertical data uses.

```bash
sbatch cluster/sbatch/cpu_morrill/02_build_daily_prithvi_files.sbatch
```

### 2. Crop the global climatology to our region (morrill)

`residual="climate"` is mandatory for the pretrained weights (not swappable
-- see docs), so climatology is needed even though we discard the decoder
output. The global climatology already downloaded in Phase 1 just needs
cropping to our bbox:

```bash
sbatch cluster/sbatch/cpu_morrill/03_crop_climatology.sbatch
```

### 3. Extract embeddings (gpu-a100)

```bash
sbatch cluster/sbatch/gpu/02_extract_embeddings.sbatch
```
Runs the frozen encoder (`mask_ratio_inputs=0.0`, full token coverage) over
every 2024 timestamp, saving both a compact global-pooled embedding
(`global_emb_*`, 2560 cols) and a modestly spatially-pooled one
(`region{0-3}_emb_*`, 4 regions x 2560 cols) to
`data/compact/prithvi_embeddings_2024.parquet`. Watch the printed
`global_shape_mu`/`embed_dim` at startup match
`docs/prithvi_embedding_candidates.md`'s expected values before letting it
run the full year.

### 4. Conventional weather baseline (morrill)

```bash
sbatch cluster/sbatch/cpu_morrill/04_extract_conventional_weather.sbatch
```
T2M + QV2M/U10M/V10M/SWGNT, regional mean and per-zone (nearest MERRA-2
gridcell to each zone's approximate centroid -- see script docstring for
caveats).

### 5. Parse ISO-NE demand + assemble the compact modeling table (morrill)

```bash
sbatch cluster/sbatch/cpu_morrill/05_assemble_compact_dataset.sbatch
```
Parses the raw ISO-NE JSON (Phase 1 only saved raw responses + an index --
the cleaned `timestamp | zone | demand_MW` table happens here), then joins
embeddings + conventional weather + real-time demand on nearest timestamp
(90 min tolerance) into `data/compact/modeling_input_2024.parquet`.

**Important, unverified until run for real:** the exact JSON field casing
from ISO-NE's XML-to-JSON conversion (`src/isone/parse_isone_demand.py`
tries the documented field names and fails loudly with the raw keys if they
don't match -- fix is a quick look at one real file, not a redesign), and
whether `BeginDate` is UTC or Eastern local time (do not assume either until
checked against ISO-NE's own dashboard for a known hour).

### 6. Transfer to local machine

```bash
scp -r hpc2:/scratch/morrill/users/hmp278/weather_foundation_model_electricity/data/compact/ ./data/
```
Only `data/compact/` (a handful of parquet files) leaves the cluster. Raw
MERRA-2, daily merged files, climatology, and model weights all stay on
scratch.

## Phase 3: CatBoost locally

Runs entirely on the local machine -- no cluster access needed from here.
`pip install -r local/requirements.txt` once.

### 1. Add calendar features

```bash
python local/prepare_modeling_table.py \
  --compact-parquet data/compact/modeling_input_2024.parquet \
  --out-parquet data/modeling_table_2024.parquet
```
Adds hour/day-of-week/day-of-year/month/year/is_weekend/is_holiday, and a
per-row "own-zone weather" view (`weather_T2M` etc., looked up from each
row's own zone). **Check the timezone caveat above first** -- these
calendar features are only meaningful once the timestamp's timezone is
confirmed.

### 2. Train the comparison models

```bash
python local/train_catboost_comparisons.py \
  --modeling-table data/modeling_table_2024.parquet \
  --train-end 2024-10-31 \
  --out-dir results/catboost_2024_comparison/
```
Trains 4 pooled CatBoost models (zone as a categorical feature, start
simple): calendar-only, calendar+weather, calendar+prithvi,
calendar+weather+prithvi. Time-respecting split (train through `--train-end`,
test after). Outputs: `comparison_summary.csv` (overall R2/RMSE/MAE per
model), `comparison_by_zone.csv`, `comparison_extreme_weather.csv`
(hot/cold-decile breakdown), `feature_importance_*.csv` for the
Prithvi-containing models, `observed_vs_predicted.png`, and the saved
`.cbm` model files.

## Phase 3 stopping point (from the source instructions)

Stop after a clean first comparison of calendar vs. simple weather vs.
Prithvi representation for 2024 ISO-NE zonal demand, with test metrics,
observed-vs-predicted plots, per-zone performance, a summary table, and
notes on failure modes. Do not move on to electricity-price prediction yet.

---

## Findings

*(Empty until Phase 1/2/3 execution completes on the cluster and locally.)*
