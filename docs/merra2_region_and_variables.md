# MERRA-2 region and variable selection

## Grid

MERRA-2 native grid: 0.5° latitude x 0.625° longitude, global
(lat -90:0.5:90, lon -180:0.625:179.375). `src/merra2/define_region.py`
snaps a rough requested box to this grid (analytically, or from a real
sample file's coordinate arrays once one is downloaded).

## Rough target box (to be confirmed/adjusted after running define_region.py)

lat [39.0, 48.0], lon [-80.0, -65.0] -- intended to cover all six New England
states, NY, adjacent Quebec/Maritime Canada, and western Atlantic margin.

**Status: not yet run against Prithvi's actual patch-size requirement.**
Once the repo is cloned, check its patch size (likely referenced in the model
config / dataloader) and re-run `define_region.py --patch-size <N>` to catch
any non-divisible grid dimensions before downloading.

## Variables

**Status: not yet filled in.** Prithvi-WxC was pretrained on ~160 MERRA-2
variables (per the model card). The authoritative list must come from the
repo's own dataloader/config, not be guessed --
`configs/merra2_variables.yaml.template` has the format to fill in
(`configs/merra2_variables.yaml`, gitignored-by-convention since it may end
up large, or kept if it's small text -- check before committing).

## Recorded findings (fill in once confirmed on cluster)

- Final grid-snapped lat/lon bounds:
- Resulting grid dimensions (n_lat x n_lon):
- Number of variables/channels:
- Temporal resolution:
- Number of expected weather states (2024):
- Downloaded size:
- Missing/corrupt timestamps:
