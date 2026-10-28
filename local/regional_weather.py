"""Coarse regional weather outside New England, from GFS, the Prithvi
decoder and MERRA-2 (actual), all on the same five areas and variables, for
each 3-hourly target time T (24 h ahead).

Areas (lat / lon boxes, degrees; plain area means, land and sea alike):
    NY-PA          40-45 N, 80-75 W   New York, Pennsylvania
    MidAtlantic    36-40 N, 80-74 W   NJ, DE, MD, DC, northern VA
    GreatLakes     38-46 N, 90-80 W   Great Lakes and Ohio Valley
    Southeast      32-36 N, 90-75 W
    Quebec-Mar     45-50 N, 80-64 W   southern Quebec, New Brunswick, Nova Scotia
Variables: T2M, QV2M, U10M, V10M, SWGNT and wind speed (as NAMES), so each
block is [T, 5 areas x 6] = 30 features.

    GFS regional       GFS 24-27 h forecast (gfs_wide_YEAR.nc, 0.25 deg)
    Prithvi regional   Prithvi decoder 24 h forecast (the wide_recon5 field of
                       the sample issued at T - 24 h; 0.5 x 0.625 deg)
    true regional      MERRA-2 at T (the wide_inputs5 field of the sample
                       issued at T): perfect knowledge, the ceiling
"""
import json

import numpy as np
import pandas as pd

from analyze_global_year import WEATHER_VARS
from gfs_demand_benchmark import GFS_VARS, NAMES, gfs_for_targets, prithvi_physical

REGIONS = {"NY-PA": (40, 45, -80, -75), "MidAtlantic": (36, 40, -80, -74), "GreatLakes": (38, 46, -90, -80),
           "Southeast": (32, 36, -90, -75), "Quebec-Mar": (45, 50, -80, -64)}
REGIONAL_NAMES = [f"{r}_{n}" for r in REGIONS for n in NAMES]


def _masks(lat, lon):
    return [((lat >= a) & (lat <= b))[:, None] & ((lon >= c) & (lon <= d))[None, :] for a, b, c, d in REGIONS.values()]


def gfs_regional(path, targets):
    ds, pick = gfs_for_targets(path, targets)
    masks = _masks(ds.lat.values, ds.lon.values)
    out = np.full((len(targets), len(REGIONS), len(NAMES)), np.nan, np.float32)
    ok = pick >= 0
    for m, v in enumerate(WEATHER_VARS):
        f = ds[GFS_VARS[v]].values[pick[ok]]
        for r, mk in enumerate(masks):
            out[ok, r, m] = f[:, mk].mean(axis=1)
    out[..., 5] = np.hypot(out[..., 2], out[..., 3])   # wind speed of the area-mean wind, as for the zone areas
    return out.reshape(len(targets), -1)


def prithvi_regional(data_dir, y, targets, weather):
    meta = json.loads((data_dir / f"prithvi_tokens_ne_{y}_lead24.json").read_text())
    wide = meta["wide"]
    assert wide["channels"] == WEATHER_VARS, wide["channels"]
    issue = pd.DatetimeIndex(meta["timestamps"])
    inp_ne = np.load(data_dir / meta["fields"]["inputs"]["file"], mmap_mode="r")
    physical = prithvi_physical(meta, inp_ne, weather, y)
    masks = _masks(np.array(wide["pixel_lats"]), np.array(wide["pixel_lons"]))
    out = {}
    for name, file, shift in [("Prithvi regional", wide["recon_file"], 24), ("true regional", wide["inputs_file"], 0)]:
        a = np.asarray(np.load(data_dir / file, mmap_mode="r"), np.float32)          # [S, 5, lat, lon], normalised
        reg = np.stack([a[:, :, mk].mean(axis=2) for mk in masks], axis=1)           # [S, area, 5]
        reg = physical(reg)                                                           # linear, so area mean commutes
        pos = issue.get_indexer(targets - pd.Timedelta(hours=shift))
        o = np.full((len(targets), len(REGIONS), len(NAMES)), np.nan, np.float32)
        o[pos >= 0] = reg[pos[pos >= 0]]
        out[name] = o.reshape(len(targets), -1)
    return out
