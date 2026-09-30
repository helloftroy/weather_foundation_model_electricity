# MERRA-2 region and variable selection

**Status: confirmed** (from the pretrained model's own `config.yaml` and the
official inference notebook -- see `docs/prithvi_embedding_candidates.md`
for the full derivation and sourcing).

## Grid and region

MERRA-2 native grid: 0.5 deg latitude x 0.625 deg longitude, global.
Prithvi's pretrained `mask_unit_size_px = [30, 32]` (lat, lon pixels) means a
regional crop's pixel counts MUST be exact multiples of 30 (lat) and 32
(lon) -- i.e. exact multiples of 15 deg lat x 20 deg lon -- or the model
constructor's own assert fails.

**Chosen box** (2x2 mask units, giving non-degenerate global attention
across 4 regions instead of a degenerate single global position):

```
lat: [30.0, 59.5]   -> 60 pixels (exactly 2 x 30)
lon: [-90.0, -50.625] -> 64 pixels (exactly 2 x 32)
```

Covers all six New England states, NY, the Maritimes/southern Quebec, and a
wide western-Atlantic margin. Both bounds are exact MERRA-2 grid points
(verified: `-90 + n*0.5 = 30` at `n=240`; `-180 + n*0.625 = -90` at `n=144`,
both integers) -- see `src/merra2/define_region.py` for the derivation and
divisibility check.

## Variables

Confirmed exact list and order from
`NASA-IMPACT/Prithvi-WxC/examples/PrithviWxC_inference.ipynb` (`main`
branch, the repo's default -- re-checked against both `main` and `develop`
after they turned out to have diverged; this specific variable/level list is
identical on both) -- **do not** add/remove/reorder these; the pretrained
weights' normalization scalers and channel ordering depend on it exactly.

- **Surface (20)**: EFLUX, GWETROOT, HFLUX, LAI, LWGAB, LWGEM, LWTUP, PS,
  QV2M, SLP, SWGNT, SWTNT, T2M, TQI, TQL, TQV, TS, U10M, V10M, Z0M
- **Static surface (4)**: FRACI, FRLAND, FROCEAN, PHIS
- **Vertical (10)**: CLOUD, H, OMEGA, PL, QI, QL, QV, T, U, V
- **Levels (14, native MERRA-2 model-level indices out of 72)**: 34, 39, 41,
  43, 44, 45, 48, 51, 53, 56, 63, 68, 71, 72

Total: 20 + 10*14 = 160 channels, matching the model card's "160 variables"
claim exactly.

GES DISC collection mapping (standard NASA schema, best-effort -- see
`configs/merra2_variables.yaml` for per-variable assignment and confidence
notes): M2T1NXFLX, M2T1NXLND, M2T1NXRAD, M2T1NXSLV (hourly surface),
M2C0NXASM (static constants), M2I3NVASM (3-hourly instantaneous native
vertical levels).

## Known data-alignment issue (see `src/merra2/build_daily_prithvi_files.py`)

The tavg1 surface collections (FLX/LND/RAD/SLV) are hourly
INTERVAL-AVERAGED products timestamped ~30 min off the synoptic hour;
M2I3NVASM is true instantaneous at 00,03,...,21 UTC. `Merra2Dataset` requires
an exact timestamp match, so our merge step explicitly resamples surface
data onto the clean 3-hourly synoptic grid (nearest-neighbor, 90 min
tolerance) before writing Prithvi-format daily files, rather than fighting
GES DISC's raw timestamp conventions bit-for-bit.

## Recorded findings (fill in once the full year is downloaded)

- Actual downloaded size:
- Missing/corrupt timestamps:
- Number of expected weather states (2024): ~2920 (365 days x 8 synoptic
  times/day), pending confirmation once the download completes.
