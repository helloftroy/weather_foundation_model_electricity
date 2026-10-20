"""Download a regional, variable-subset MERRA-2 cube via NASA earthaccess.

Streams each granule via earthaccess.open() (cloud-native HTTPS/S3 access,
now that MERRA-2 is on AWS us-west-2) and immediately subsets to the
grid-snapped Northeast bbox and the requested variables with xarray before
writing to disk -- this avoids pulling full global granules just to crop
them locally, per the "don't download global MERRA-2 unnecessarily" goal.

The exact collections/variables to request MUST come from the config file,
not from guesses baked into this script -- fill in
docs/merra2_region_and_variables.md and a matching --variables-config
(collection short name -> variable list) after inspecting Prithvi-WxC's own
MERRA-2 dataloader/config (Phase 1 step 2), then point this script at it.
Use src/merra2/define_region.py first to get the grid-snapped lat/lon bounds
to pass in here.

Usage (small sample, for pipeline validation before the full year):
    python download_merra2.py \
      --variables-config configs/merra2_variables.yaml \
      --start 2024-01-01 --end 2024-01-02 \
      --lat-min 39.0 --lat-max 48.0 --lon-min -80.0 --lon-max -65.0 \
      --out-dir /scratch/morrill/users/hmp278/weather_electricity_foundation/data/merra2_sample

Full year:
    python download_merra2.py \
      --variables-config configs/merra2_variables.yaml \
      --start 2024-01-01 --end 2024-12-31 \
      --lat-min 39.0 --lat-max 48.0 --lon-min -80.0 --lon-max -65.0 \
      --out-dir /scratch/morrill/users/hmp278/weather_electricity_foundation/data/merra2_2024

Requires Earthdata Login credentials in ~/.netrc (machine urs.earthdata.nasa.gov)
or EARTHDATA_USERNAME / EARTHDATA_PASSWORD env vars, and network access
(run on the service node).
"""
import argparse
import signal
import time
from datetime import datetime
from pathlib import Path

import earthaccess
import xarray as xr
import yaml

# M2C0NXCTM (const_2d_ctm_Nx) is MERRA-2's constants-for-CTM-usage
# collection -- a single global granule with no real date, not one per day
# (its 12 "monthly" slices are all in that one granule). Applying a
# --start/--end temporal filter to it (as to every other collection)
# excludes its one granule whenever the requested range doesn't happen to
# cover its actual (arbitrary, pre-2024) timestamp, silently yielding zero
# results. See docs/merra2_region_and_variables.md.
STATIC_COLLECTIONS = {"M2C0NXCTM"}


# The 14 model levels Prithvi uses. The vertical collection has 72; fetching
# only these cuts download time and file size about fivefold.
KEEP_LEVELS = [34.0, 39.0, 41.0, 43.0, 44.0, 45.0, 48.0, 51.0, 53.0, 56.0, 63.0, 68.0, 71.0, 72.0]
GRANULE_TIMEOUT_S = 20 * 60


class GranuleTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise GranuleTimeout(f"no result after {GRANULE_TIMEOUT_S // 60} minutes")


def is_complete(path: Path) -> bool:
    """True if `path` is a readable NetCDF file. A job killed mid-write
    leaves a truncated file behind, which must not count as done."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        # No engine given: earlier runs wrote with xarray's default engine,
        # which may not have been HDF5. xarray picks by the file's signature.
        with xr.open_dataset(path):
            return True
    except Exception:
        return False


def search_with_retry(max_tries=12, **kwargs):
    """earthaccess.search_data, retried, and an empty answer treated as a
    failure. During NASA search-service (CMR) outages it either raises
    (HTTP 500, "index closed") or, worse, answers "0 granules" for ranges
    that certainly have data; the second used to let a job finish
    "successfully" with nothing downloaded. Waits up to about 6 hours."""
    for attempt in range(1, max_tries + 1):
        try:
            results = earthaccess.search_data(**kwargs)
            if results:
                return results
            problem = "0 granules found (search service outage?)"
        except Exception as e:
            problem = str(e)[:200]
        if attempt == max_tries:
            raise SystemExit(f"Search for {kwargs.get('short_name')} still failing after {max_tries} tries: {problem}")
        wait = min(60 * 2 ** (attempt - 1), 3600)
        print(f"  search failed (attempt {attempt}/{max_tries}: {problem}); retrying in {wait // 60} min...", flush=True)
        time.sleep(wait)


def fetch_crop_write(granule, variables, lat_min, lat_max, lon_min, lon_max, out_path, max_retries=5, tmp_dir=None):
    """Fetch one granule, crop to the region (and to the levels Prithvi
    uses), and write it out.

    Retries on transient network/service errors, and on a stall: a remote
    read that never returns would otherwise hold the job until its time
    limit. The file is written under a temporary name and renamed, so a
    killed job never leaves a partial file under the final name.
    """
    tmp_path = out_path.with_name(out_path.name + ".part")
    for attempt in range(1, max_retries + 1):
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(GRANULE_TIMEOUT_S * (3 if tmp_dir else 1))
        raw_path = None
        try:
            if tmp_dir is not None:
                # Whole-file mode: download the granule over plain HTTPS, cut
                # it down locally, delete the original. Much faster than
                # remote range reads when most of the file is wanted.
                tmp_dir.mkdir(parents=True, exist_ok=True)
                raw_path = Path(earthaccess.download([granule], local_path=str(tmp_dir), threads=1)[0])
                fileset = [raw_path]
            else:
                fileset = earthaccess.open([granule])
            # earthaccess.open() returns fsspec file-like objects with no
            # filename/extension to sniff, so xarray's engine auto-guessing
            # fails even though the underlying format is fine -- MERRA-2 is
            # HDF5-based netCDF4, and h5netcdf is already installed (a
            # Prithvi-WxC dependency), so just say so explicitly.
            ds = xr.open_dataset(fileset[0], engine="h5netcdf")
            missing = [v for v in variables if v not in ds.variables]
            if missing:
                raise SystemExit(f"variables not found in granule: {missing}")

            cropped = ds[variables].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
            if "lev" in cropped.dims:
                cropped = cropped.sel(lev=KEEP_LEVELS)
            cropped.load().to_netcdf(tmp_path, engine="h5netcdf")
            ds.close()
            tmp_path.replace(out_path)
            if raw_path is not None:
                raw_path.unlink(missing_ok=True)
            return cropped.nbytes
        except SystemExit:
            raise
        except Exception as e:
            if attempt == max_retries:
                raise
            wait = min(30 * attempt, 300)
            print(f"  attempt {attempt}/{max_retries} failed ({e}); retrying in {wait}s...", flush=True)
            signal.alarm(0)
            time.sleep(wait)
        finally:
            signal.alarm(0)
            if raw_path is not None and raw_path.exists() and not out_path.exists():
                raw_path.unlink(missing_ok=True)  # failed attempt: drop the partial download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variables-config", required=True, type=Path,
                         help="YAML: {collection_short_name: [var1, var2, ...]}")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--lat-min", type=float, required=True)
    parser.add_argument("--lat-max", type=float, required=True)
    parser.add_argument("--lon-min", type=float, required=True)
    parser.add_argument("--lon-max", type=float, required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--whole-file", action="store_true",
                        help="Download each granule in full and subset it locally (use for the global grid)")
    parser.add_argument("--tmp-dir", type=Path, default=None, help="Where whole-file downloads are held before subsetting")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.variables_config.read_text())

    # earthaccess.login(strategy="netrc") RAISES LoginStrategyUnavailable
    # (rather than returning an unauthenticated result) when ~/.netrc exists
    # but lacks a urs.earthdata.nasa.gov entry -- catch it and fall through
    # to the environment strategy instead of letting it crash the script.
    auth = None
    try:
        auth = earthaccess.login(strategy="netrc")
    except Exception as e:
        print(f"netrc login strategy unavailable ({e}); trying environment variables instead.")
    if auth is None or not auth.authenticated:
        try:
            auth = earthaccess.login(strategy="environment")
        except Exception as e:
            print(f"environment login strategy failed: {e}")
            auth = None
    if auth is None or not auth.authenticated:
        raise SystemExit(
            "Earthdata login failed. Set up ~/.netrc for urs.earthdata.nasa.gov "
            "or EARTHDATA_USERNAME/EARTHDATA_PASSWORD."
        )

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    bbox = (args.lon_min, args.lat_min, args.lon_max, args.lat_max)  # earthaccess uses (W,S,E,N)

    manifest_rows = []
    for collection, variables in config.items():
        coll_dir = args.out_dir / collection
        coll_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Collection: {collection} | variables: {variables} ===")

        search_kwargs = dict(short_name=collection, bounding_box=bbox)
        if collection not in STATIC_COLLECTIONS:
            search_kwargs["temporal"] = (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        else:
            print(f"{collection} is a static/time-invariant collection -- searching with no temporal filter.")
        results = search_with_retry(**search_kwargs)
        print(f"Found {len(results)} granules for {collection}", flush=True)
        n_skipped = 0

        for granule in results:
            granule_id = granule.get("meta", {}).get("native-id") or granule["umm"]["GranuleUR"]
            out_path = coll_dir / f"{granule_id}.nc"

            # Resumable: a long full-year download can get interrupted by a
            # transient error even with retries (or a job timeout, a manual
            # Ctrl-C, etc.) -- skip granules already written by a previous
            # run instead of redoing completed work.
            if is_complete(out_path):
                manifest_rows.append({"collection": collection, "path": str(out_path)})
                n_skipped += 1
                continue
            if out_path.exists():
                print(f"  {out_path.name} exists but is unreadable (interrupted write?); fetching again.", flush=True)

            nbytes = fetch_crop_write(
                granule, variables,
                args.lat_min, args.lat_max, args.lon_min, args.lon_max,
                out_path, tmp_dir=(args.tmp_dir or args.out_dir / "_raw") if args.whole_file else None,
            )
            manifest_rows.append({"collection": collection, "path": str(out_path)})
            print(f"  wrote {out_path} ({nbytes / 1e6:.1f} MB)", flush=True)
        if n_skipped:
            print(f"  {n_skipped} granules were already on disk and readable; skipped.", flush=True)

    manifest_path = args.out_dir / "download_manifest.csv"
    import csv
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["collection", "path"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nWrote manifest: {manifest_path} ({len(manifest_rows)} files)")


if __name__ == "__main__":
    main()
