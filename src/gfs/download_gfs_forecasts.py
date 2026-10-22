"""Download archived GFS forecasts over New England, for benchmarking the
Prithvi decoder's forecasts against a real operational weather model.

Source: NOAA's GFS archive on AWS (s3://noaa-gfs-bdp-pds, public, no login),
0.25 degree grid, available from January 2021. Each forecast file has a
small text index (.idx) giving the byte range of every field, so only the
fields needed are fetched (about 1 MB each), not the 500 MB file.

What is fetched, for every run (00, 06, 12, 18 UTC) and forecast hours 24
and 27:
    TMP   2 m temperature, K                      (MERRA-2 T2M)
    SPFH  2 m specific humidity, kg/kg            (QV2M)
    UGRD  10 m eastward wind, m/s                 (U10M)
    VGRD  10 m northward wind, m/s                (V10M)
    DSWRF, USWRF  surface down/up shortwave, W/m2; net = DSWRF - USWRF
                                                  (SWGNT). GFS gives these
                                                  as averages over the
                                                  preceding 6 h (f024: hours
                                                  18-24) or 3 h (f027: 24-27),
                                                  not hourly values.

Why hours 24 and 27: the lead-24 Prithvi samples are 3-hourly. For a target
time T, the comparable GFS forecast is the latest run at or before T - 24 h,
which is T - 24 h or T - 27 h, i.e. forecast hour 24 or 27. Like the Prithvi
setup, this ignores the 3-4 h GFS takes to be published.

Each field is cropped to the New England box (LAT_RANGE, LON_RANGE) and
saved; at the end everything is combined into one file per year:
    gfs_ne_{year}.nc   dims (sample, lat, lon); coords run_time, lead_hours,
                       valid_time per sample
Resumable: one small .npz per forecast file, skipped if present.
"""
import argparse
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import eccodes
import numpy as np
import pandas as pd
import xarray as xr

BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
FIELDS = {  # output name: text that identifies the field in the .idx file
    "TMP_2m": ":TMP:2 m above ground:",
    "SPFH_2m": ":SPFH:2 m above ground:",
    "UGRD_10m": ":UGRD:10 m above ground:",
    "VGRD_10m": ":VGRD:10 m above ground:",
    "DSWRF_sfc": ":DSWRF:surface:",
    "USWRF_sfc": ":USWRF:surface:",
}
LAT_RANGE = (38.0, 50.0)
LON_RANGE = (282.0, 296.0)  # degrees east, = 78W to 64W
FHOURS = (24, 27)
RUNS = (0, 6, 12, 18)


def get(url, byte_range=None, tries=6):
    for attempt in range(1, tries + 1):
        req = urllib.request.Request(url, headers={"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except Exception as e:  # noqa: BLE001 - network errors of every kind are retried
            err = e
        wait = min(10 * 2 ** (attempt - 1), 300)
        print(f"  retry {attempt}/{tries} in {wait}s: {url.rsplit('/', 3)[-3:]} {byte_range or ''}: {str(err)[:120]}", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"{url}: {err}")


def file_url(run, fhour):
    """Files moved into an atmos/ subfolder in March 2021; try both."""
    day, hh = run.strftime("%Y%m%d"), run.strftime("%H")
    name = f"gfs.t{hh}z.pgrb2.0p25.f{fhour:03d}"
    for sub in ("atmos/", ""):
        url = f"{BUCKET}/gfs.{day}/{hh}/{sub}{name}"
        idx = get(url + ".idx")
        if idx is not None:
            return url, idx.decode()
    return None, None


def byte_ranges(idx_text):
    lines = [ln.split(":") for ln in idx_text.strip().splitlines()]
    out = {}
    for name, key in FIELDS.items():
        for i, parts in enumerate(lines):
            if key in ":" + ":".join(parts[3:6]) + ":":
                start = int(parts[1])
                end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else ""
                out[name] = (start, end, ":".join(parts[3:7]))
                break
    return out


def decode(msg_bytes):
    h = eccodes.codes_new_from_message(msg_bytes)
    try:
        nj, ni = eccodes.codes_get(h, "Nj"), eccodes.codes_get(h, "Ni")
        lat0, lat1 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees"), eccodes.codes_get(h, "latitudeOfLastGridPointInDegrees")
        lon0, lon1 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees"), eccodes.codes_get(h, "longitudeOfLastGridPointInDegrees")
        vals = eccodes.codes_get_values(h).reshape(nj, ni)
    finally:
        eccodes.codes_release(h)
    return vals, np.linspace(lat0, lat1, nj), np.linspace(lon0, lon1, ni)


def fetch_one(run, fhour, out_dir):
    out = out_dir / f"gfs_{run:%Y%m%d%H}_f{fhour:03d}.npz"
    if out.exists():
        return out, "kept"
    url, idx = file_url(run, fhour)
    if url is None:
        return None, "missing"
    ranges = byte_ranges(idx)
    lacking = set(FIELDS) - set(ranges)
    if lacking:
        raise RuntimeError(f"{url}: fields not in the index: {sorted(lacking)}")
    arrays, desc = {}, {}
    for name, (start, end, text) in ranges.items():
        vals, lats, lons = decode(get(url, (start, end)))
        r = (lats >= LAT_RANGE[0]) & (lats <= LAT_RANGE[1])
        c = (lons >= LON_RANGE[0]) & (lons <= LON_RANGE[1])
        arrays[name] = vals[np.ix_(r, c)].astype(np.float32)
        desc[name] = text
        lat, lon = lats[r], lons[c]
    tmp = out.with_name(out.stem + ".part.npz")
    np.savez(tmp, lat=lat, lon=lon, fields=np.array(list(desc.items())), **arrays)
    tmp.replace(out)
    return out, "fetched"


def combine(out_dir, year, out_nc):
    files = sorted(out_dir.glob("gfs_??????????_f0??.npz"))  # all runs, incl. Dec 30-31 of the previous year (they cover Jan 1)
    if not files:
        raise SystemExit("Nothing to combine.")
    first = np.load(files[0])
    data = {k: [] for k in FIELDS}
    runs, leads = [], []
    for f in files:
        z = np.load(f)
        runs.append(pd.Timestamp(f.stem.split("_")[1][:10][:8] + " " + f.stem.split("_")[1][8:10] + ":00"))
        leads.append(int(f.stem.split("_f")[1]))
        for k in FIELDS:
            data[k].append(z[k])
    run_time = pd.DatetimeIndex(runs)
    lead = np.array(leads)
    ds = xr.Dataset({k: (("sample", "lat", "lon"), np.stack(v)) for k, v in data.items()},
                    coords={"lat": first["lat"], "lon": first["lon"] - 360.0, "run_time": ("sample", run_time),
                            "lead_hours": ("sample", lead), "valid_time": ("sample", run_time + pd.to_timedelta(lead, unit="h"))})
    ds["SW_net_sfc"] = ds["DSWRF_sfc"] - ds["USWRF_sfc"]
    ds.attrs["source"] = f"NOAA GFS 0.25 deg, {BUCKET}; fields: {dict(first['fields'].tolist())}"
    ds.to_netcdf(out_nc)
    print(f"Wrote {out_nc}: {ds.sizes['sample']} forecasts, lat {float(ds.lat.min())}..{float(ds.lat.max())}, lon {float(ds.lon.min())}..{float(ds.lon.max())}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--start", default=None, help="First run date (default: Dec 30 of the previous year, so the year's first targets are covered)")
    parser.add_argument("--end", default=None, help="Last run date (default: Dec 31 of the year)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    start = pd.Timestamp(args.start or f"{args.year - 1}-12-30")
    end = pd.Timestamp(args.end or f"{args.year}-12-31") + pd.Timedelta(hours=18)
    runs = [r for r in pd.date_range(start, end, freq="6h") if r.hour in RUNS]
    raw_dir = args.out_dir / f"gfs_ne_{args.year}_parts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(r, f) for r in runs for f in FHOURS]
    print(f"{len(runs)} runs x {len(FHOURS)} forecast hours = {len(jobs)} files, {args.workers} at a time", flush=True)
    counts = {"kept": 0, "fetched": 0, "missing": 0}
    missing = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(fetch_one, r, f, raw_dir): (r, f) for r, f in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            _, status = fut.result()
            counts[status] += 1
            if status == "missing":
                missing.append(futures[fut])
            if i % 200 == 0 or i == len(jobs):
                print(f"...{i}/{len(jobs)} ({counts})", flush=True)
    if missing:
        print(f"{len(missing)} forecast files not in the archive, e.g. {[f'{r:%Y-%m-%d %H}z f{f:03d}' for r, f in missing[:5]]}")
    combine(raw_dir, args.year, args.out_dir / f"gfs_ne_{args.year}.nc")


if __name__ == "__main__":
    main()
