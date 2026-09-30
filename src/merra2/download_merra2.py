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
from datetime import datetime
from pathlib import Path

import earthaccess
import xarray as xr
import yaml


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

        results = earthaccess.search_data(
            short_name=collection,
            temporal=(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")),
            bounding_box=bbox,
        )
        print(f"Found {len(results)} granules for {collection}")

        for granule in results:
            # One granule = one file, so this is always a single-item list
            # -- open_dataset (not open_mfdataset) is the right tool; the
            # "mf" (multi-file, dask-backed) variant isn't needed here and
            # requires dask, which isn't otherwise a dependency of anything
            # in this project.
            fileset = earthaccess.open([granule])
            # earthaccess.open() returns fsspec file-like objects with no
            # filename/extension to sniff, so xarray's engine auto-guessing
            # fails even though the underlying format is fine -- MERRA-2 is
            # HDF5-based netCDF4, and h5netcdf is already installed (a
            # Prithvi-WxC dependency), so just say so explicitly.
            ds = xr.open_dataset(fileset[0], engine="h5netcdf")
            missing = [v for v in variables if v not in ds.variables]
            if missing:
                raise SystemExit(f"{collection}: variables not found in granule: {missing}")

            cropped = ds[variables].sel(
                lat=slice(args.lat_min, args.lat_max),
                lon=slice(args.lon_min, args.lon_max),
            )
            granule_id = granule.get("meta", {}).get("native-id") or granule["umm"]["GranuleUR"]
            out_path = coll_dir / f"{granule_id}.nc"
            cropped.load().to_netcdf(out_path)
            ds.close()
            manifest_rows.append({"collection": collection, "path": str(out_path)})
            print(f"  wrote {out_path} ({cropped.nbytes / 1e6:.1f} MB)")

    manifest_path = args.out_dir / "download_manifest.csv"
    import csv
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["collection", "path"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nWrote manifest: {manifest_path} ({len(manifest_rows)} files)")


if __name__ == "__main__":
    main()
