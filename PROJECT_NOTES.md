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
`/scratch/morrill/users/hmp278/weather_electricity_foundation/`, never `$HOME`.

## Cluster logistics

- **service** partition: internet access, used for cloning repos, downloading
  weights/climatology/MERRA-2/ISO-NE data. Slow for compute -- no heavy CPU
  work here.
- **gpu-a100** partition: Prithvi inference / embedding extraction.
- **morrill** partition: CPU-only work with no internet need (QC,
  preprocessing, later CatBoost). *(Confirm this is the correct `--partition`
  value with `sinfo` once on the cluster -- see the note in
  `cluster/sbatch/cpu_morrill/01_qc_merra2_isone.sbatch`.)*
- Environment: reuse the existing **FAIRe** conda env if it already has a
  compatible Python (>=3.10) and torch install; otherwise create a dedicated
  `prithvi_wxc` env under scratch. `HF_HOME`, pip cache, and tmp are all
  redirected into scratch. See `src/prithvi/setup_prithvi_env.sh`.

## Directory layout on the cluster

```
/scratch/morrill/users/hmp278/weather_electricity_foundation/
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
mkdir -p /scratch/morrill/users/hmp278/weather_electricity_foundation
```

### 1. Set up Prithvi WxC (service)

```bash
sbatch cluster/sbatch/service/01_setup_prithvi.sbatch
```
Clones NASA-IMPACT/Prithvi-WxC, resolves FAIRe-vs-new-env, `pip install
-e .[examples]`, downloads `Prithvi-WxC/prithvi.wxc.2300m.v1` weights +
climatology into scratch via `huggingface_hub`.

### 2. Sample inference validation + architecture inspection (gpu-a100)

```bash
sbatch cluster/sbatch/gpu/01_prithvi_sample_inference.sbatch
```
Executes the repo's own `examples/PrithviWxC_inference.ipynb` as the official
small/sample validation test. Then, interactively or via a follow-up script,
use `src/prithvi/inspect_architecture.py` (fill in the model-loading call
first, per its TODO) to identify candidate embedding tensors and record
findings in `docs/prithvi_embedding_candidates.md`.

### 3. MERRA-2: sample, then full year (service)

First confirm the exact variable list from the cloned repo's own MERRA-2
dataloader/config, and fill in `configs/merra2_variables.yaml` from
`configs/merra2_variables.yaml.template` (do not guess this list). Then:

```bash
python src/merra2/define_region.py --patch-size <PRITHVI_PATCH_SIZE>
# adjust docs/merra2_region_and_variables.md with the confirmed bounds, then:

sbatch cluster/sbatch/service/02_download_merra2_sample.sbatch   # 2 days, validate pipeline
sbatch cluster/sbatch/service/03_download_merra2_2024.sbatch     # full year, only after 02 checks out
```

### 4. ISO-NE electricity demand, 2024 (service)

Requires a free ISO Express account (see `docs/isone_demand_sources.md`).

```bash
export ISONE_WS_USERNAME="..."
export ISONE_WS_PASSWORD="..."
sbatch cluster/sbatch/service/04_download_isone_demand_2024.sbatch
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

## Findings

*(Empty until Phase 1 execution completes on the cluster.)*
