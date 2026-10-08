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
            dimension -- Merra2Dataset's _read_static_data reads these
            directly by key with no time indexing at all, so each must be a
            bare 2D array here). Sourced from M2C0NXCTM (const_2d_ctm_Nx),
            NOT M2C0NXASM (const_2d_asm_Nx) -- confirmed from GES DISC's own
            MERRA-2 file specification: M2C0NXASM's constants table has no
            ice-fraction variable at all, while M2C0NXCTM has a variable
            named exactly "FRACI" alongside FRLAND/FROCEAN/PHIS, so all four
            of Prithvi's static_surface_vars come from this one collection.
            It's a single granule but NOT a true single instant like
            M2C0NXASM -- "time-invariant but duplicated for each month" (a
            repeating monthly climatology, time=12) -- so the time index
            used here is (day.month - 1), not always 0.
    MERRA_pres_YYYYMMDD.nc -- one per day, containing:
        'lat', 'lon', 'lev' (native MERRA-2 model level index, 1..72),
            'time' as above
        one (time, lev, lat, lon) dataset per vertical_vars short name

Processing follows the repo's own input-preparation code
(PrithviWxC/download.py, extract_prithvi_wxc_input_data):

  - Single-level state variables (PS, T2M, U10M, ...) come from the
    INSTANTANEOUS collection M2I1NXASM (inst1_2d_asm_Nx), on the hour.
  - The hourly time-averaged collections (M2T1NXFLX/LND/RAD) are stamped at
    half past the hour. Each 3-hourly value is the mean of the two averages
    either side of it (HH-1:30 and HH:30), which centres it on the hour. For
    00:00 that needs the previous day's 23:30; where that isn't available
    (the first day downloaded) the 00:30 value is used alone.
  - GWETROOT and LAI are undefined over ocean in MERRA-2. They are filled
    with the official NAN_VALS (1.0 and 0.0). The loader and model have no
    missing-value handling, so a single NaN makes every embedding NaN.

Every output file is checked for non-finite values before it is written.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

STATIC_COLLECTION = "M2C0NXCTM"
VERTICAL_COLLECTION = "M2I3NVASM"
SYNOPTIC_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]
# From PrithviWxC/definitions.py: fill values for variables that are
# undefined (NaN) away from land.
NAN_VALS = {"GWETROOT": 1.0, "LAI": 0.0}


def load_variable_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def open_static_collection(data_dir: Path, collection: str) -> xr.Dataset | None:
    """Open a constants collection with NO day-based time filtering -- its
    time dimension (e.g. M2C0NXCTM's 12 monthly slices) doesn't represent
    real per-day instants, so open_collection_day's .sel(time=slice(day,
    ...)) would incorrectly try to match it against actual calendar dates."""
    coll_dir = data_dir / collection
    if not coll_dir.is_dir():
        return None
    files = sorted(coll_dir.glob("*.nc"))
    if not files:
        return None
    return xr.open_mfdataset(files, combine="by_coords", decode_times=True)


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
    day_start = day - pd.Timedelta(hours=1)  # previous day's 23:30, for centring 00:00
    day_end = day + pd.Timedelta(hours=23, minutes=59)
    if "time" in ds.dims:
        ds = ds.sel(time=slice(day_start, day_end))
        if ds.sizes.get("time", 0) == 0:
            return None
    return ds


def resample_to_synoptic(ds: xr.Dataset, day: pd.Timestamp) -> xr.Dataset:
    """Put a collection on the 3-hourly synoptic grid for `day`.

    Instantaneous collections (stamped on the hour) are selected exactly.
    Time-averaged collections (stamped at half past) are centred on the hour
    by averaging the two neighbouring half-hour values.
    """
    targets = pd.DatetimeIndex([day + pd.Timedelta(hours=h) for h in SYNOPTIC_HOURS])
    minutes = pd.DatetimeIndex(ds["time"].values).minute
    if (minutes == 0).all():
        return ds.reindex(time=targets)
    if not (minutes == 30).all():
        raise SystemExit(f"{day.date()}: unexpected time stamps (minutes {sorted(set(minutes))}); expected :00 or :30.")
    half = pd.Timedelta(minutes=30)
    before = ds.reindex(time=targets - half).assign_coords(time=targets)
    after = ds.reindex(time=targets + half).assign_coords(time=targets)
    # Where the earlier half-hour is missing altogether (first day), use the
    # later one alone rather than leaving a gap.
    has_before = xr.DataArray(
        np.isin((targets - half).values, ds["time"].values), coords={"time": targets}, dims="time"
    )
    return xr.where(has_before, (before + after) / 2, after)


def fill_and_check(arrays: dict, label: str) -> dict:
    """Apply the official NaN fills, then refuse to continue if anything
    non-finite is left."""
    out = {}
    bad = {}
    for name, arr in arrays.items():
        arr = np.asarray(arr)
        if name in NAN_VALS:
            arr = np.nan_to_num(arr, nan=NAN_VALS[name])
        n_bad = int((~np.isfinite(arr)).sum())
        if n_bad:
            bad[name] = f"{n_bad}/{arr.size}"
        out[name] = arr
    if bad:
        raise SystemExit(f"{label}: non-finite values remain after filling: {bad}")
    return out


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

    # Opened once (one small file covering all 12 "monthly" slices) and
    # re-sliced per day below -- NOT reused unchanged across all days, since
    # M2C0NXCTM is a monthly climatology (time=12), not a true single
    # constant like M2C0NXASM.
    static_ds = open_static_collection(args.data_dir, STATIC_COLLECTION)
    if static_ds is None:
        raise SystemExit(
            f"No data found for static collection {STATIC_COLLECTION} under {args.data_dir}. "
            "This constants collection has no time dimension to match against a specific day -- "
            "if this fails, re-download it without a --start/--end temporal filter."
        )
    if "time" not in static_ds.dims or static_ds.sizes["time"] != 12:
        raise SystemExit(
            f"Expected {STATIC_COLLECTION} to have a time dimension of size 12 (one per "
            f"calendar month), got dims={dict(static_ds.sizes)}. The month-index lookup "
            "below assumes this; update it if the collection's actual layout differs."
        )
    # Don't assume Jan..Dec storage order -- if the time coordinate decoded
    # as real dates, map each calendar month to its actual slice index;
    # otherwise fall back to the natural assumption (index 0 = January).
    try:
        actual_months = pd.DatetimeIndex(static_ds["time"].values).month
        month_to_time_idx = {int(m): i for i, m in enumerate(actual_months)}
        assert set(month_to_time_idx) == set(range(1, 13))
    except Exception:
        month_to_time_idx = {m: m - 1 for m in range(1, 13)}

    days = pd.date_range(args.start, args.end, freq="D")
    n_written = 0
    for day in days:
        static_arrays = fill_and_check({
            # .isel (not raw .values[idx]) to index the "time" dim by label
            # regardless of its actual position among the var's axes.
            var: static_ds[var].isel(time=month_to_time_idx[day.month]).values
            for var in config.get(STATIC_COLLECTION, [])
        }, f"{day.date()} static")

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

        surface_arrays = fill_and_check(surface_arrays, f"{day.date()} surface")
        vertical_arrays = fill_and_check(vertical_arrays, f"{day.date()} vertical")

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
