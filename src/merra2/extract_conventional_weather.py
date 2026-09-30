"""Small conventional-weather baseline table: T2M + a few simple predictors,
per ISO-NE zone and regionally, for comparison against the Prithvi embedding.

Reads directly from the daily merged surface files built by
build_daily_prithvi_files.py (same files Merra2Dataset consumes), so no
extra download is needed. Deliberately minimal per the source instructions:
temperature at minimum, plus humidity/wind/solar if easy -- not a large
handcrafted feature set.

Zone coordinates below are APPROXIMATE representative points (state/
sub-region geographic centers), not official ISO-NE zone centroids or
population-weighted load centers -- refine later with real zone geometry if
the nearest-MERRA2-gridcell approximation turns out to matter. At MERRA-2's
~50km resolution, the three Massachusetts sub-zones (NEMA/SEMA/WCMA) may
map to the same or adjacent grid cells; that's expected, not a bug.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ZONE_CENTROIDS = {
    "ME": (45.0, -69.0),
    "NH": (43.7, -71.5),
    "VT": (44.3, -72.6),
    "CT": (41.6, -72.7),
    "RI": (41.7, -71.5),
    "NEMA": (42.36, -71.06),
    "SEMA": (41.9, -70.9),
    "WCMA": (42.3, -72.6),
}

BASELINE_VARS = ["T2M", "QV2M", "U10M", "V10M", "SWGNT"]


def nearest_index(coord_array: np.ndarray, value: float) -> int:
    return int(np.abs(coord_array - value).argmin())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merra2-surface-dir", type=Path, required=True)
    parser.add_argument("--out-parquet", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.merra2_surface_dir.glob("MERRA2_sfc_*.nc"))
    if not files:
        raise SystemExit(f"No MERRA2_sfc_*.nc files found under {args.merra2_surface_dir}")
    print(f"Reading {len(files)} daily surface files")

    # time is minutes-since-midnight-of-that-file's-day; reconstruct real
    # timestamps from the begin_date attr per file instead of relying on a
    # single global decode (each file has its own begin_date).
    all_frames = []
    for f in files:
        day_ds = xr.open_dataset(f, decode_times=False)
        begin_date = str(day_ds["time"].attrs["begin_date"])
        day = pd.Timestamp(begin_date)
        timestamps = [day + pd.Timedelta(minutes=int(m)) for m in day_ds["time"].values]

        lat = day_ds["lat"].values
        lon = day_ds["lon"].values

        zone_idx = {z: (nearest_index(lat, la), nearest_index(lon, lo)) for z, (la, lo) in ZONE_CENTROIDS.items()}

        rows = []
        for ti, ts in enumerate(timestamps):
            row = {"timestamp": ts}
            for var in BASELINE_VARS:
                if var not in day_ds:
                    continue
                arr = day_ds[var].values[ti]  # [lat, lon]
                row[f"regional_mean_{var}"] = float(np.nanmean(arr))
                for zone, (yi, xi) in zone_idx.items():
                    row[f"{zone}_{var}"] = float(arr[yi, xi])
            rows.append(row)
        all_frames.append(pd.DataFrame(rows))
        day_ds.close()

    df = pd.concat(all_frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet, index=False)
    print(f"Wrote {len(df)} rows x {len(df.columns)} cols to {args.out_parquet}")


if __name__ == "__main__":
    main()
