# Weather Foundation Model → Electricity

Testing whether representations from a frozen pretrained weather foundation model
(Prithvi WxC, trained on global MERRA-2) contain useful information about
electricity-system behavior, starting with New England electricity demand.

See [PROJECT_NOTES.md](PROJECT_NOTES.md) for the full methods summary, the exact
commands used at each step, and (once we get there) findings.

## Pipeline

```
MERRA-2 (regional Northeast cube) -> frozen Prithvi WxC -> weather embedding -> CatBoost -> electricity demand
```

## Repo layout

```
cluster/sbatch/service/     sbatch scripts that need internet (clone repos, download weights/data)
cluster/sbatch/gpu/         sbatch scripts for GPU work (Prithvi inference / embedding extraction)
cluster/sbatch/cpu_morrill/ sbatch scripts for CPU-only work on morrill (QC, preprocessing, CatBoost)
src/prithvi/                Prithvi WxC setup + architecture inspection
src/merra2/                 MERRA-2 region definition + download
src/isone/                  ISO-NE electricity demand download
src/qc/                     Timestamp / coverage QC between MERRA-2 and ISO-NE
docs/                       Reference notes filled in as we go (embedding candidates, variable lists, data source decisions)
```

## Cluster

Runs on the `hpc2` (Mississippi State HPC2 / Orion) cluster. Large data, weights,
caches, and environments live under `/scratch/morrill/users/hmp278/weather_foundation_model_electricity/`,
never under `$HOME`. See [PROJECT_NOTES.md](PROJECT_NOTES.md) for the exact
directory layout and setup commands.

This repo (code only) is cloned onto the cluster and run from there; raw/large
data products are git-ignored and stay in scratch.
