"""Snap a rough Northeast-US bounding box onto the native MERRA-2 grid.

MERRA-2 is on a fixed 0.5 deg (lat) x 0.625 deg (lon) lat/lon grid, so "on the
native grid" just means picking existing grid-cell edges rather than
arbitrary decimal bounds -- it does not require reading a sample file. This
script still accepts an optional sample file/OPeNDAP URL so the coordinate
arrays can be read directly and the bbox can be checked against the *exact*
values MERRA-2 ships (protects against off-by-one drift in the assumed 0.5/0.625
spacing), and optionally checked for even divisibility against a Prithvi patch
size once that is confirmed from the cloned repo.

Rough target coverage (fill in / adjust after looking at the result):
  all six New England states, NY, adjacent Quebec/Maritime Canada,
  and enough western Atlantic to avoid an artificially land-locked crop.
"""
import argparse

import numpy as np
import xarray as xr

# Conservative rough box requested: New England + NY + adjacent Canada + western Atlantic.
# These are NOT snapped yet -- run this script to get the actual grid-aligned bounds.
DEFAULT_LAT_MIN, DEFAULT_LAT_MAX = 39.0, 48.0
DEFAULT_LON_MIN, DEFAULT_LON_MAX = -80.0, -65.0

MERRA2_LAT_STEP = 0.5
MERRA2_LON_STEP = 0.625


def snap_from_grid(lat_min, lat_max, lon_min, lon_max, lat_coord, lon_coord):
    lat_idx = np.where((lat_coord >= lat_min) & (lat_coord <= lat_max))[0]
    lon_idx = np.where((lon_coord >= lon_min) & (lon_coord <= lon_max))[0]
    return {
        "lat_slice": (int(lat_idx.min()), int(lat_idx.max())),
        "lon_slice": (int(lon_idx.min()), int(lon_idx.max())),
        "lat_bounds": (float(lat_coord[lat_idx.min()]), float(lat_coord[lat_idx.max()])),
        "lon_bounds": (float(lon_coord[lon_idx.min()]), float(lon_coord[lon_idx.max()])),
        "n_lat": len(lat_idx),
        "n_lon": len(lon_idx),
    }


def snap_analytic(lat_min, lat_max, lon_min, lon_max):
    """Fallback: MERRA-2's grid is documented as lat=-90:0.5:90, lon=-180:0.625:179.375."""
    lat_coord = np.round(np.arange(-90.0, 90.0 + MERRA2_LAT_STEP, MERRA2_LAT_STEP), 3)
    lon_coord = np.round(np.arange(-180.0, 180.0, MERRA2_LON_STEP), 3)
    return snap_from_grid(lat_min, lat_max, lon_min, lon_max, lat_coord, lon_coord)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX)
    parser.add_argument("--lon-min", type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon-max", type=float, default=DEFAULT_LON_MAX)
    parser.add_argument(
        "--sample-file",
        default=None,
        help="Path or OPeNDAP URL to one MERRA-2 granule; if given, use its "
        "actual lat/lon coordinate arrays instead of the analytic grid.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="If given, warn when n_lat/n_lon are not evenly divisible by this.",
    )
    args = parser.parse_args()

    if args.sample_file:
        ds = xr.open_dataset(args.sample_file)
        result = snap_from_grid(
            args.lat_min, args.lat_max, args.lon_min, args.lon_max,
            ds["lat"].values, ds["lon"].values,
        )
    else:
        print("No --sample-file given; using the documented analytic MERRA-2 grid.")
        print("Re-run with --sample-file once the first sample granule is downloaded to confirm.\n")
        result = snap_analytic(args.lat_min, args.lat_max, args.lon_min, args.lon_max)

    print(f"Requested box: lat [{args.lat_min}, {args.lat_max}], lon [{args.lon_min}, {args.lon_max}]")
    print(f"Grid-snapped lat bounds: {result['lat_bounds']} ({result['n_lat']} points)")
    print(f"Grid-snapped lon bounds: {result['lon_bounds']} ({result['n_lon']} points)")
    print(f"lat index slice: {result['lat_slice']}, lon index slice: {result['lon_slice']}")

    if args.patch_size:
        for n, axis in [(result["n_lat"], "lat"), (result["n_lon"], "lon")]:
            if n % args.patch_size != 0:
                print(
                    f"WARNING: n_{axis}={n} is not evenly divisible by "
                    f"patch size {args.patch_size}; widen the box or pad."
                )


if __name__ == "__main__":
    main()
