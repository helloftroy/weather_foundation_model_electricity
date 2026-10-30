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

--source ncar (for 2020 and earlier, which the AWS archive lacks): NCAR GDEX
dataset d084001, "NCEP GFS 0.25 Degree Global Forecast Grids Historical
Archive" (2015 onward; https://data.gdex.ucar.edu/d084001/YYYY/YYYYMMDD/
gfs.0p25.YYYYMMDDHH.fFFF.grib2; public, anonymous byte-range access). It has
no .idx files, so the index is built by reading each GRIB message's first
512 bytes (section 0 gives the message length, section 4 the parameter and
level) and jumping to the next, stopping once all six fields are found.
The same fields as from AWS are identified by GRIB2 codes (discipline 0):
TMP 0/0/0 and SPFH 0/1/0 at 2 m above ground; UGRD 0/2/2 and VGRD 0/2/3 at
10 m; DSWRF 0/4/192 and USWRF 0/4/193 (NCEP local numbers) at the surface,
time-averaged (product template 4.8). The fields sit in the last 30-40% of
each file, so the scan starts from a message boundary found near 60% of the
file (falling back to a scan from the start if a field is missed): about
100 header reads per file instead of about 600. Checked against the AWS files on a 2021 run before use.

--also-ne writes gfs_ne_YEAR.nc as well, cropped from the --box output
(the New England box lies inside the wide box; for 2021 the crop
reproduces gfs_ne_2021.nc exactly), so each file is scanned only once.
--also-crop NAME LAT_MIN LAT_MAX LON_MIN LON_MAX (repeatable) writes further
crops of the --box output the same way, e.g. "wide" and "texas" from one
union box: each field is fetched whole (global) anyway, so extra crops cost
no extra downloading.

--fhours: other forecast hours, for longer leads, e.g. 72 75 (3 days), 120
123 (5 days), 168 171 (7 days); GFS 0.25 deg has 3-hourly output to 384 h.
Outputs then carry the first hour as a suffix (gfs_ne_2023_f072.nc), and the
default first run date moves back so the year's first targets are covered.
"""
import argparse
import struct
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import eccodes
import numpy as np
import pandas as pd
import requests
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


def fetch_message(url, start, end, tries=4):
    """One field's bytes, checked to be a whole GRIB message (starts 'GRIB',
    ends '7777'). A truncated or corrupted transfer is fetched again; if it
    stays bad, returns None and the forecast hour is skipped."""
    for attempt in range(1, tries + 1):
        msg = get(url, (start, end))
        if msg and msg[:4] == b"GRIB" and msg.rstrip(b"\n")[-4:] == b"7777":
            return msg
        print(f"  incomplete GRIB message ({len(msg or b'')} bytes) for {url.rsplit('/', 1)[-1]} bytes {start}-{end}, try {attempt}/{tries}", flush=True)
        time.sleep(5 * attempt)
    return None


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


NCAR = "https://data.gdex.ucar.edu/d084001"
GRIB_CODES = {  # output name: (discipline, category, number, first surface type, its scaled value, need a time average)
    "TMP_2m": (0, 0, 0, 103, 2, False),
    "SPFH_2m": (0, 1, 0, 103, 2, False),
    "UGRD_10m": (0, 2, 2, 103, 10, False),
    "VGRD_10m": (0, 2, 3, 103, 10, False),
    "DSWRF_sfc": (0, 4, 192, 1, None, True),   # NCEP local parameter numbers (wgrib2 names them DSWRF / USWRF)
    "USWRF_sfc": (0, 4, 193, 1, None, True),
}


_local = threading.local()


def ncar_get(url, byte_range=None, tries=6):
    """GET with a per-thread kept-alive connection (a new TLS connection per header read is ~2x slower).
    Returns b"" for a range past the end of the file (HTTP 416), None for a missing file."""
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    for attempt in range(1, tries + 1):
        try:
            r = _local.session.get(url, headers={"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {}, timeout=120)
            if r.status_code == 404:
                return None
            if r.status_code == 416:
                return b""
            if r.status_code in (200, 206):
                return r.content
            err = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001 - network errors of every kind are retried
            err = e
            _local.session = requests.Session()
        wait = min(10 * 2 ** (attempt - 1), 300)
        print(f"  retry {attempt}/{tries} in {wait}s: {url.rsplit('/', 1)[-1]} {byte_range or ''}: {str(err)[:120]}", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"{url}: {err}")


def ncar_size(url):
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    r = _local.session.head(url, timeout=60)
    return int(r.headers["Content-Length"]) if r.status_code == 200 else None


def find_boundary(url, offset, chunk=2_000_000):
    """Offset of the first GRIB2 message starting at or after `offset` (checked: edition 2, and the message ends
    in '7777' where its length says it does). None if none found in the chunk."""
    data = ncar_get(url, (offset, offset + chunk - 1)) or b""
    p = data.find(b"GRIB")
    while p >= 0:
        if p + 16 <= len(data) and data[p + 7] == 2:
            total = struct.unpack(">Q", data[p + 8:p + 16])[0]
            if 100 < total < 200_000_000:
                end = data[p + total - 4:p + total] if p + total <= len(data) else ncar_get(url, (offset + p + total - 4, offset + p + total - 1))
                if end == b"7777":
                    return offset + p
        p = data.find(b"GRIB", p + 1)
    return None


def ncar_url(run, fhour):
    return f"{NCAR}/{run:%Y}/{run:%Y%m%d}/gfs.0p25.{run:%Y%m%d%H}.f{fhour:03d}.grib2"


def describe_header(head):
    """(discipline, category, number, template, surface type, scaled surface value) from a message's first bytes."""
    discipline = head[6]
    pos = 16
    while pos + 5 <= len(head):
        sec_len, sec_num = struct.unpack(">IB", head[pos:pos + 5])
        if sec_num == 4:
            sec = head[pos:pos + sec_len]
            if len(sec) < 28:
                return None
            template = struct.unpack(">H", sec[7:9])[0]
            return discipline, sec[9], sec[10], template, sec[22], struct.unpack(">I", sec[24:28])[0]
        if sec_len == 0:
            return None
        pos += sec_len
    return None


# Where the fields sit in a GFS 0.25 deg file, as a fraction of its size (measured 2020-2024: the 2 m/10 m fields at
# 0.69-0.72, the averaged shortwave at 0.79-0.80). Walking headers from just before each group needs ~25 small
# requests per file instead of ~130 from 0.6, which NCAR's server throttles (connection resets) at 12 workers.
GROUPS = ((0.66, ("TMP_2m", "SPFH_2m", "UGRD_10m", "VGRD_10m")), (0.765, ("DSWRF_sfc", "USWRF_sfc")))


def scan_ranges(url, max_messages=2000):
    """Build {field: (start, end, text)} for FIELDS by walking the GRIB message headers with byte ranges, from a
    message boundary just before each group of fields; from the start of the file if that misses a field."""
    size = ncar_size(url)
    if size is None:
        return None
    out = {}
    for fraction, names in GROUPS:
        start = find_boundary(url, int(size * fraction))
        if start is not None:
            out.update(walk(url, start, 60, wanted=names))
    if len(out) < len(GRIB_CODES):  # other layouts (e.g. later GFS versions): walk from 60% of the file, then from the start
        start = find_boundary(url, int(size * 0.6))
        out = walk(url, start, max_messages) if start is not None else {}
    if len(out) < len(GRIB_CODES):
        out = walk(url, 0, max_messages)
    return out


def walk(url, offset, max_messages, wanted=None):
    wanted = set(wanted or GRIB_CODES)
    out = {}
    for _ in range(max_messages):
        head = ncar_get(url, (offset, offset + 511))
        if head is None:
            return {}
        if len(head) < 16 or head[:4] != b"GRIB":
            break
        total = struct.unpack(">Q", head[8:16])[0]
        d = describe_header(head)
        if d:
            for name, (disc, cat, num, surf, val, avg) in GRIB_CODES.items():
                if name in wanted and name not in out and d[:3] == (disc, cat, num) and d[4] == surf \
                        and (val is None or d[5] == val) and (not avg or d[3] == 8) and (avg or d[3] == 0):
                    out[name] = (offset, offset + total - 1, f"GRIB {disc}/{cat}/{num} surface {surf}:{d[5]} template 4.{d[3]}")
        if len(out) == len(wanted):
            break
        offset += total
    return out


def fetch_one(run, fhour, out_dir, lat_range=LAT_RANGE, lon_range=LON_RANGE, source="aws"):
    try:
        return _fetch_one(run, fhour, out_dir, lat_range, lon_range, source)
    except RuntimeError as e:
        print(f"  giving up on {run:%Y-%m-%d %H}z f{fhour:03d} for now: {str(e)[:160]}", flush=True)
        return None, "failed"


def _fetch_one(run, fhour, out_dir, lat_range=LAT_RANGE, lon_range=LON_RANGE, source="aws"):
    out = out_dir / f"gfs_{run:%Y%m%d%H}_f{fhour:03d}.npz"
    if out.exists():
        return out, "kept"
    if source == "ncar":
        url = ncar_url(run, fhour)
        ranges = scan_ranges(url)
        if ranges is None:
            return None, "missing"
    else:
        url, idx = file_url(run, fhour)
        if url is None:
            return None, "missing"
        ranges = byte_ranges(idx)
    lacking = set(FIELDS) - set(ranges)
    if lacking:
        raise RuntimeError(f"{url}: fields not in the index: {sorted(lacking)}")
    arrays, desc = {}, {}
    for name, (start, end, text) in ranges.items():
        msg = fetch_message(url, start, end)
        if msg is None:
            return None, "bad"
        try:
            vals, lats, lons = decode(msg)
        except Exception as e:  # noqa: BLE001 - an undecodable field skips this forecast hour, not the job
            print(f"  could not decode {name} in {url.rsplit('/', 1)[-1]}: {e}", flush=True)
            return None, "bad"
        r = (lats >= lat_range[0]) & (lats <= lat_range[1])
        c = (lons >= lon_range[0]) & (lons <= lon_range[1])
        arrays[name] = vals[np.ix_(r, c)].astype(np.float32)
        desc[name] = text
        lat, lon = lats[r], lons[c]
    tmp = out.with_name(out.stem + ".part.npz")
    np.savez(tmp, lat=lat, lon=lon, fields=np.array(list(desc.items())), **arrays)
    tmp.replace(out)
    return out, "fetched"


def combine(out_dir, year, out_nc, source_note=BUCKET, also_ne=None, crops=()):
    files = sorted(out_dir.glob("gfs_??????????_f???.npz"))  # all runs, incl. Dec 30-31 of the previous year (they cover Jan 1)
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
    ds.attrs["source"] = f"NOAA GFS 0.25 deg, {source_note}; fields: {dict(first['fields'].tolist())}"
    ds.to_netcdf(out_nc)
    print(f"Wrote {out_nc}: {ds.sizes['sample']} forecasts, lat {float(ds.lat.min())}..{float(ds.lat.max())}, lon {float(ds.lon.min())}..{float(ds.lon.max())}")
    if also_ne is not None:
        ne = ds.sel(lat=slice(LAT_RANGE[1], LAT_RANGE[0]), lon=slice(LON_RANGE[0] - 360.0, LON_RANGE[1] - 360.0))
        ne.to_netcdf(also_ne)
        print(f"Wrote {also_ne}: New England crop, {ne.sizes['lat']} x {ne.sizes['lon']}")
    for path, (lat0, lat1, lon0, lon1) in crops:
        c = ds.sel(lat=slice(lat1, lat0), lon=slice(lon0, lon1))
        c.to_netcdf(path)
        print(f"Wrote {path}: crop {c.sizes['lat']} x {c.sizes['lon']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--start", default=None, help="First run date (default: Dec 30 of the previous year, so the year's first targets are covered)")
    parser.add_argument("--end", default=None, help="Last run date (default: Dec 31 of the year)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--box", type=float, nargs=4, default=None, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                        help="Crop box in degrees (longitudes -180..180); default is the New England box. "
                             "Output then goes to gfs_<name>_YEAR.nc with --name")
    parser.add_argument("--name", default="ne", help="Label for the output files (default 'ne')")
    parser.add_argument("--source", choices=["aws", "ncar"], default="aws", help="aws (2021 onward) or ncar (GDEX d084001, 2015 onward)")
    parser.add_argument("--also-ne", action="store_true", help="Also write gfs_ne_YEAR.nc, cropped from the --box output")
    parser.add_argument("--also-crop", nargs=5, action="append", default=[], metavar=("NAME", "LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                        help="Also write gfs_NAME_YEAR.nc cropped from the --box output (repeatable; longitudes -180..180)")
    parser.add_argument("--fhours", type=int, nargs="+", default=list(FHOURS),
                        help="Forecast hours to fetch (default 24 27); others get a _fHHH suffix on the output names")
    args = parser.parse_args()
    fhours = tuple(args.fhours)
    suffix = "" if fhours == FHOURS else f"_f{min(fhours):03d}"
    back = max(fhours) // 24 + 2   # days before Jan 1 whose runs still reach the year
    start = pd.Timestamp(args.start or (pd.Timestamp(f"{args.year}-01-01") - pd.Timedelta(days=back)).strftime("%Y-%m-%d"))
    end = pd.Timestamp(args.end or f"{args.year}-12-31") + pd.Timedelta(hours=18)
    runs = [r for r in pd.date_range(start, end, freq="6h") if r.hour in RUNS]
    lat_range, lon_range = LAT_RANGE, LON_RANGE
    if args.box:
        lat_range, lon_range = (args.box[0], args.box[1]), (args.box[2] % 360, args.box[3] % 360)
    raw_dir = args.out_dir / f"gfs_{args.name}_{args.year}{suffix}_parts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(r, f) for r in runs for f in fhours]
    print(f"{len(runs)} runs x {len(fhours)} forecast hours = {len(jobs)} files, {args.workers} at a time", flush=True)
    counts = {"kept": 0, "fetched": 0, "missing": 0, "bad": 0, "failed": 0}
    missing = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(fetch_one, r, f, raw_dir, lat_range, lon_range, args.source): (r, f) for r, f in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            _, status = fut.result()
            counts[status] += 1
            if status in ("missing", "bad"):
                missing.append(futures[fut])
            if i % 200 == 0 or i == len(jobs):
                print(f"...{i}/{len(jobs)} ({counts})", flush=True)
    if counts["failed"]:
        raise SystemExit(f"{counts['failed']} files failed after retries (server resets?); resubmit the same job to fetch "
                         "just those (finished files are kept), then it combines")
    if missing:
        print(f"{len(missing)} forecast files missing from the archive or unreadable, e.g. {[f'{r:%Y-%m-%d %H}z f{f:03d}' for r, f in missing[:5]]}")
    crops = [(args.out_dir / f"gfs_{c[0]}_{args.year}{suffix}.nc", tuple(float(v) for v in c[1:])) for c in args.also_crop]
    combine(raw_dir, args.year, args.out_dir / f"gfs_{args.name}_{args.year}{suffix}.nc", NCAR if args.source == "ncar" else BUCKET,
            args.out_dir / f"gfs_ne_{args.year}{suffix}.nc" if args.also_ne else None, crops)


if __name__ == "__main__":
    main()
