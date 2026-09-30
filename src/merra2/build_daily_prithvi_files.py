"""Merge our per-collection regional MERRA-2 downloads into Merra2Dataset's
expected daily-file format.

PrithviWxC/dataloaders/merra2.py's Merra2Dataset reads two files per day via
h5py, using ONLY the file's own lat/lon dataset lengths to size arrays (its
`upper_shape`/`surface_shape` properties that hardcode 361x576 are dead code,
never referenced by the actual read path) -- so it works unmodified on a
regional crop as long as we hand it files in exactly its expected format:

    MERRA2_sfc_YYYYMMDD.nc  -- one per day, containing:
        'lat', 'lon': 1D coordinate arrays
        'time': 1D array of MINUTES SINCE MIDNIGHT that day (int), with
            attrs 'begin_date' (YYYYMMDD int) and 'begin_time' (HHMMSS int,
            0 for midnight)
        one (time, lat, lon) dataset per surface_vars short name
        one (lat, lon) dataset per static_surface_vars short name (no time
            dimension -- these are read directly by key, undated)
    MERRA_pres_YYYYMMDD.nc -- one per day, containing:
        'lat', 'lon', 'lev' (native MERRA-2 model level index, 1..72),
            'time' as above
        one (time, lev, lat, lon) dataset per vertical_vars short name

Important MERRA-2 gotcha this script works around: the tavg1 surface
collections (M2T1NXFLX/LND/RAD/SLV) are hourly INTERVAL-AVERAGED products
timestamped at half-past-the-hour (e.g. 00:30, 01:30, ...), not the exact
synoptic instants (00:00, 03:00, ...) that the M2I3NVASM vertical collection
uses. Merra2Dataset looks up an EXACT timestamp match (no tolerance), so
mixing the two naively would raise "not in list" errors. Instead of matching
raw GES DISC timestamps bit-for-bit, this script explicitly resamples/
relabels the surface data onto the clean 3-hourly synoptic grid (nearest
available hour, default 90 min tolerance) before writing -- introducing the
tavg1 series' inherent ~30 min lead/lag, which is an accepted simplification
here, not silently glossed over.

VERIFY ON CLUSTER before trusting a full year of output:
  - that the downloaded GES DISC files actually use the variable short names
    assumed here (should match, since Prithvi's names ARE the standard
    MERRA-2 short names, but confirm against one real downloaded file);
  - that the nearest-neighbor reindex tolerance (90 min) doesn't silently
    drop timestamps if a collection has gaps.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

STATIC_COLLECTION = "M2C0NXASM"
VERTICAL_COLLECTION = "M2I3NVASM"
SYNOPTIC_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]


def load_variable_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def open_collection_day(data_dir: Path, collection: str, day: pd.Timestamp) -> xr.Dataset | None:
    coll_dir = data_dir / collection
    if not coll_dir.is_dir():
        return None
    # download_merra2.py writes one file per granule; a granule is typically
    # one UTC day. Glob broadly and let xarray's time selection narrow it.
    files = sorted(coll_dir.glob("*.nc"))
    if not files:
        return None
    ds = xr.open_mfdataset(files, combine="by_coords", decode_times=True)
    day_start = day
    day_end = day + pd.Timedelta(hours=23, minutes=59)
    if "time" in ds.dims:
        ds = ds.sel(time=slice(day_start, day_end))
        if ds.sizes.get("time", 0) == 0:
            return None
    return ds


def resample_to_synoptic(ds: xr.Dataset, day: pd.Timestamp, tolerance_minutes: int = 90) -> xr.Dataset:
    target_times = [day + pd.Timedelta(hours=h) for h in SYNOPTIC_HOURS]
    return ds.reindex(time=target_times, method="nearest", tolerance=pd.Timedelta(minutes=tolerance_minutes))


def write_surface_file(
    out_path: Path,
    day: pd.Timestamp,
    lat: np.ndarray,
    lon: np.ndarray,
    surface_arrays: dict,
    static_arrays: dict,
) -> None:
    minutes = np.array([h * 60 for h in SYNOPTIC_HOURS], dtype=np.int64)

    data_vars = {}
    for name, arr in surface_arrays.items():
        data_vars[name] = (("time", "lat", "lon"), arr.astype(np.float32))
    for name, arr in static_arrays.items():
        data_vars[name] = (("lat", "lon"), arr.astype(np.float32))

    out_ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": ("time", minutes, {"begin_date": int(day.strftime("%Y%m%d")), "begin_time": 0}),
            "lat": ("lat", lat.astype(np.float32)),
            "lon": ("lon", lon.astype(np.float32)),
        },
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(out_path, engine="h5netcdf")


def write_vertical_file(
    out_path: Path,
    day: pd.Timestamp,
    lat: np.ndarray,
    lon: np.ndarray,
    lev: np.ndarray,
    vertical_arrays: dict,
) -> None:
    minutes = np.array([h * 60 for h in SYNOPTIC_HOURS], dtype=np.int64)

    data_vars = {
        name: (("time", "lev", "lat", "lon"), arr.astype(np.float32))
        for name, arr in vertical_arrays.items()
    }
    out_ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": ("time", minutes, {"begin_date": int(day.strftime("%Y%m%d")), "begin_time": 0}),
            "lev": ("lev", lev.astype(np.float32)),
            "lat": ("lat", lat.astype(np.float32)),
            "lon": ("lon", lon.astype(np.float32)),
        },
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(out_path, engine="h5netcdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Root with per-collection subdirs from download_merra2.py")
    parser.add_argument("--variables-config", type=Path, required=True)
    parser.add_argument("--out-dir-surface", type=Path, required=True)
    parser.add_argument("--out-dir-vertical", type=Path, required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    config = load_variable_config(args.variables_config)
    surface_collections = {
        k: v for k, v in config.items() if k not in (STATIC_COLLECTION, VERTICAL_COLLECTION)
    }

    # Static vars: read once, reused for every day.
    static_ds = open_collection_day(args.data_dir, STATIC_COLLECTION, pd.Timestamp(args.start))
    if static_ds is None:
        raise SystemExit(
            f"No data found for static collection {STATIC_COLLECTION} under {args.data_dir}. "
            "The const_2d_asm_Nx collection has no time dimension to match against a day -- "
            "if this fails, re-download it without a --start/--end temporal filter."
        )
    static_arrays = {}
    for var in config.get(STATIC_COLLECTION, []):
        arr = static_ds[var].values
        static_arrays[var] = arr[0] if arr.ndim == 3 else arr  # drop leading time dim of length 1, if present

    days = pd.date_range(args.start, args.end, freq="D")
    n_written = 0
    for day in days:
        surface_arrays = {}
        lat = lon = None
        missing_collections = []
        for collection, variables in surface_collections.items():
            ds = open_collection_day(args.data_dir, collection, day)
            if ds is None:
                missing_collections.append(collection)
                continue
            ds = resample_to_synoptic(ds, day)
            if lat is None:
                lat = ds["lat"].values
                lon = ds["lon"].values
            for var in variables:
                if var not in ds:
                    raise SystemExit(f"{day.date()}: variable '{var}' not found in collection {collection}")
                surface_arrays[var] = ds[var].values

        if missing_collections:
            print(f"{day.date()}: skipping, missing surface collections {missing_collections}")
            continue

        vert_ds = open_collection_day(args.data_dir, VERTICAL_COLLECTION, day)
        if vert_ds is None:
            print(f"{day.date()}: skipping, missing vertical collection {VERTICAL_COLLECTION}")
            continue
        vert_ds = resample_to_synoptic(vert_ds, day)
        lev = vert_ds["lev"].values
        vertical_arrays = {}
        for var in config.get(VERTICAL_COLLECTION, []):
            if var not in vert_ds:
                raise SystemExit(f"{day.date()}: variable '{var}' not found in {VERTICAL_COLLECTION}")
            vertical_arrays[var] = vert_ds[var].values

        write_surface_file(
            args.out_dir_surface / f"MERRA2_sfc_{day.strftime('%Y%m%d')}.nc",
            day, lat, lon, surface_arrays, static_arrays,
        )
        write_vertical_file(
            args.out_dir_vertical / f"MERRA_pres_{day.strftime('%Y%m%d')}.nc",
            day, lat, lon, lev, vertical_arrays,
        )
        n_written += 1
        if n_written % 30 == 0:
            print(f"...{n_written} days written (latest {day.date()})")

    print(f"Done. Wrote {n_written}/{len(days)} days to {args.out_dir_surface} and {args.out_dir_vertical}")


if __name__ == "__main__":
    main()
