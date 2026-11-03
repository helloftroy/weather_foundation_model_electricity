"""Download the full GFS atmospheric state over the Northeast, as a control for the Prithvi tokens: does a broad
description of the forecast atmosphere (all variables and levels, from the operational model the market uses) do as
well as the tokens?

Source: NOAA's GFS archive on AWS (s3://noaa-gfs-bdp-pds), 0.5 degree files (pgrb2.0p50; the grid of Prithvi's own
MERRA-2 inputs is 0.5 x 0.625 degree), runs 00/06/12/18 UTC, forecast hours 24 and 27 (as for the existing GFS
features: for a target time T, the latest run at or before T - 24 h). Only the wanted fields are fetched, by byte range
from the .idx index (adjacent fields in one request).

Fields (150, chosen to mirror Prithvi's 160 MERRA-2 input channels and present in both GFS v15 (to 2021-03-22) and v16):
  9 variables x 14 pressure levels (1000, 975, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa; Prithvi's
  14 model levels span about 985-48 hPa): HGT, TMP, RH (v15 has no SPFH on levels), VVEL (not at 50 hPa: missing in v15),
  UGRD, VGRD, cloud water (named CLWMR or CLMR, depending on the date), ICMR, TCDC -> 125;
  25 single-level fields: PRMSL, surface PRES and TMP, 2 m TMP / SPFH / RH, 10 m UGRD / VGRD, PWAT, CWAT, total cloud
  (TCDC entire atmosphere), surface LHTFL / SHTFL / DSWRF / USWRF / DLWRF / ULWRF, top-of-atmosphere ULWRF / USWRF,
  top-layer soil moisture and temperature, HPBL, GUST, surface CAPE, tropopause height. Where GFS gives both an
  instantaneous and a time-averaged value (total cloud in v16), the average is used, since v15 has only that.

Each field is cropped to the Northeast box 38-50 N, 80-62 W (25 x 37 points at 0.5 degree); one .npz per forecast file
(resumable). At the end, per year:
  gfs_full_{year}_meta.npz   run_time, lead_hours, valid_time per sample; field names; box means [sample, field] over
                             "ne" (40-48 N, 75-66.25 W, the extent the tokens and MERRA-2 fields are pooled over) and
                             "northeast" (the whole crop)
  gfs_full_{year}_ne_grid.npy  [sample, field, 17, 18] float32, the "ne" box at full resolution (for gridded PCA)
"""
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import download_gfs_forecasts as gf

BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
LEVELS = (1000, 975, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
LEVEL_VARS = ("HGT", "TMP", "RH", "VVEL", "UGRD", "VGRD", "CLWMR", "ICMR", "TCDC")
ALIASES = {"CLWMR": ("CLWMR", "CLMR")}
SINGLE = (("PRMSL", "mean sea level"), ("PRES", "surface"), ("TMP", "surface"), ("TMP", "2 m above ground"),
          ("SPFH", "2 m above ground"), ("RH", "2 m above ground"), ("UGRD", "10 m above ground"), ("VGRD", "10 m above ground"),
          ("PWAT", "entire atmosphere (considered as a single layer)"), ("CWAT", "entire atmosphere (considered as a single layer)"),
          ("TCDC", "entire atmosphere"), ("LHTFL", "surface"), ("SHTFL", "surface"), ("DSWRF", "surface"), ("USWRF", "surface"),
          ("DLWRF", "surface"), ("ULWRF", "surface"), ("ULWRF", "top of atmosphere"), ("USWRF", "top of atmosphere"),
          ("SOILW", "0-0.1 m below ground"), ("TSOIL", "0-0.1 m below ground"), ("HPBL", "surface"), ("GUST", "surface"),
          ("CAPE", "surface"), ("HGT", "tropopause"))
FIELDS = [(v, f"{lv} mb") for v in LEVEL_VARS for lv in LEVELS if not (v == "VVEL" and lv == 50)] + list(SINGLE)
NAMES = [f"{v}_{lv.split()[0]}" if lv.endswith(" mb") else f"{v}_{lv.split(' (')[0].replace(' ', '_')}" for v, lv in FIELDS]
CROP = (38.0, 50.0, -80.0, -62.0)
NE = (40.0, 48.0, -75.0, -66.25)
RUNS, FHOURS = (0, 6, 12, 18), (24, 27)

_local = threading.local()


def http(url, rng=None, tries=6):
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    err = None
    for attempt in range(tries):
        try:
            r = _local.s.get(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}, timeout=120)
            if r.status_code == 404:
                return None
            if r.status_code in (200, 206):
                return r.content
            err = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001 - retried
            err, _local.s = e, requests.Session()
        gf.time.sleep(min(10 * 2 ** attempt, 300))
    raise RuntimeError(f"{url}: {err}")


def locate(run, fhour):
    day, hh = f"{run:%Y%m%d}", f"{run:%H}"
    for sub in ("atmos/", ""):
        url = f"{BUCKET}/gfs.{day}/{hh}/{sub}gfs.t{hh}z.pgrb2.0p50.f{fhour:03d}"
        idx = http(url + ".idx")
        if idx is not None:
            return url, idx.decode()
    return None, None


def ranges(idx):
    """[(field index, start, end)] for FIELDS; end is '' for the file's last message. Prefers a time-averaged entry."""
    lines = [ln.split(":") for ln in idx.strip().splitlines()]
    found = {}
    for i, p in enumerate(lines):
        key, ave = (p[3], p[4]), "ave" in p[5]
        end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else ""
        for k, (v, lv) in enumerate(FIELDS):
            if p[3] in ALIASES.get(v, (v,)) and p[4] == lv and (k not in found or (ave and not found[k][2])):
                found[k] = (int(p[1]), end, ave)
    return {k: (s, e) for k, (s, e, _) in found.items()}


def fetch_one(run, fhour, out_dir):
    out = out_dir / f"gfs_{run:%Y%m%d%H}_f{fhour:03d}.npz"
    if out.exists():
        return "kept"
    try:
        url, idx = locate(run, fhour)
        if url is None:
            return "missing"
        rg = ranges(idx)
        if len(rg) < len(FIELDS):
            print(f"  {url.rsplit('/', 3)[-3:]}: lacks {[NAMES[k] for k in range(len(FIELDS)) if k not in rg]}", flush=True)
            return "bad"
        # merge runs of adjacent messages into one request
        order = sorted(rg, key=lambda k: rg[k][0])
        groups, cur = [], [order[0]]
        for k in order[1:]:
            prev_end = rg[cur[-1]][1]
            if prev_end != "" and rg[k][0] == prev_end + 1:
                cur.append(k)
            else:
                groups.append(cur)
                cur = [k]
        groups.append(cur)
        arr = np.full((len(FIELDS), 25, 37), np.nan, np.float32)
        for g in groups:
            start, end = rg[g[0]][0], rg[g[-1]][1]
            blob = http(url, (start, end))
            for k in g:
                s, e = rg[k]
                msg = blob[s - start:(len(blob) if e == "" else e - start + 1)]
                if msg[:4] != b"GRIB" or msg.rstrip(b"\n")[-4:] != b"7777":
                    print(f"  bad message {NAMES[k]} in {url.rsplit('/', 1)[-1]}", flush=True)
                    return "bad"
                vals, lats, lons = gf.decode(msg)
                lons = (lons + 180) % 360 - 180
                r = (lats >= CROP[0] - 1e-6) & (lats <= CROP[1] + 1e-6)
                c = (lons >= CROP[2] - 1e-6) & (lons <= CROP[3] + 1e-6)
                sub = vals[np.ix_(r, c)]
                sub = sub[np.argsort(-lats[r])][:, np.argsort(lons[c])]   # north to south, west to east
                arr[k] = sub
    except RuntimeError as e:
        print(f"  giving up on {run:%Y-%m-%d %H}z f{fhour:03d} for now: {str(e)[:160]}", flush=True)
        return "failed"
    tmp = out.with_name(out.stem + ".part.npz")
    np.savez(tmp, fields=arr)
    tmp.replace(out)
    return "fetched"


def combine(parts, year, out_dir):
    files = sorted(parts.glob("gfs_??????????_f???.npz"))
    run = pd.DatetimeIndex([pd.Timestamp(f.stem[4:12] + " " + f.stem[12:14] + ":00") for f in files])
    lead = np.array([int(f.stem.split("_f")[1]) for f in files])
    valid = run + pd.to_timedelta(lead, unit="h")
    keep = valid.year == year
    files, run, lead, valid = [f for f, k in zip(files, keep) if k], run[keep], lead[keep], valid[keep]
    lat = np.arange(CROP[1], CROP[0] - 0.01, -0.5)
    lon = np.arange(CROP[2], CROP[3] + 0.01, 0.5)
    rn = (lat >= NE[0]) & (lat <= NE[1])
    cn = (lon >= NE[2]) & (lon <= NE[3])
    grid = np.lib.format.open_memmap(out_dir / f"gfs_full_{year}_ne_grid.npy", mode="w+", dtype=np.float32,
                                     shape=(len(files), len(FIELDS), int(rn.sum()), int(cn.sum())))
    ne_mean = np.empty((len(files), len(FIELDS)), np.float32)
    nea_mean = np.empty_like(ne_mean)
    for i, f in enumerate(files):
        a = np.load(f)["fields"]
        grid[i] = a[:, rn][:, :, cn]
        ne_mean[i] = grid[i].mean(axis=(1, 2))
        nea_mean[i] = a.mean(axis=(1, 2))
    grid.flush()
    np.savez(out_dir / f"gfs_full_{year}_meta.npz", run_time=run.values, lead_hours=lead, valid_time=valid.values,
             names=np.array(NAMES), ne_mean=ne_mean, northeast_mean=nea_mean, ne_lat=lat[rn], ne_lon=lon[cn])
    print(f"Wrote gfs_full_{year}_meta.npz and _ne_grid.npy: {len(files)} forecasts x {len(FIELDS)} fields, "
          f"NaN in {np.isnan(ne_mean).any(axis=1).sum()} samples", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    runs = [r for r in pd.date_range(f"{args.year - 1}-12-30", f"{args.year}-12-31 18:00", freq="6h") if r.hour in RUNS]
    parts = args.out_dir / f"gfs_full_{args.year}_parts"
    parts.mkdir(parents=True, exist_ok=True)
    jobs = [(r, f) for r in runs for f in FHOURS]
    print(f"{len(jobs)} forecast files x {len(FIELDS)} fields, {args.workers} at a time", flush=True)
    counts = {"kept": 0, "fetched": 0, "missing": 0, "bad": 0, "failed": 0}
    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(fetch_one, r, f, parts) for r, f in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            counts[fut.result()] += 1
            if i % 200 == 0 or i == len(jobs):
                print(f"...{i}/{len(jobs)} {counts}", flush=True)
    if counts["failed"]:
        raise SystemExit(f"{counts['failed']} files failed after retries; resubmit the same job (finished files are kept)")
    combine(parts, args.year, args.out_dir)


if __name__ == "__main__":
    main()
