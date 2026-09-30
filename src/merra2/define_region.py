"""Compute a mask-unit-aligned Northeast-US bounding box on the native MERRA-2 grid.

Confirmed from the pretrained prithvi.wxc.2300m.v1 config.yaml (fetched
2026-09-29 from huggingface.co/Prithvi-WxC/prithvi.wxc.2300m.v1):

    mask_unit_size_px: [30, 32]   # [lat pixels, lon pixels]
    patch_size_px:     [2, 2]

The model asserts `n_lats_px % mask_unit_size_px[0] == 0` and likewise for
lon (see PrithviWxC/model.py __init__), so a regional crop's pixel counts
MUST be exact multiples of 30 (lat) and 32 (lon) -- not just of patch_size.
At MERRA-2's native 0.5 deg (lat) x 0.625 deg (lon) spacing that's a
mandatory 15 deg-lat x 20 deg-lon tile size.

None of the pretrained weight tensors are shaped by n_lats_px/n_lons_px
directly -- patch/static embeddings are ordinary spatially-agnostic conv
layers, and positional info is Fourier-encoded live from actual lat/lon
values (positional_encoding="fourier"), not a fixed-size lookup table. So a
regional crop is architecturally safe to load the pretrained state_dict into,
as long as this divisibility holds. (Running the model on the full global
360x576 grid with ~0 masking, by contrast, is not practical for a full
year -- the official demo needs mask_ratio=0.99 just to stay small; a
regional crop's ~50x fewer pixels is what makes a full-year pass feasible.)

Recommended default: a 2x2 mask-unit tile (30 deg lat x 40 deg lon),
covering all New England, NY, the Maritimes/southern Quebec, and a wide
western-Atlantic margin, with non-degenerate global attention (4 global
positions instead of 1 for a minimal 1x1 tile).
"""
import argparse

import numpy as np
import xarray as xr

MERRA2_LAT_STEP = 0.5
MERRA2_LON_STEP = 0.625

MASK_UNIT_LAT_PX = 30  # -> 15 deg
MASK_UNIT_LON_PX = 32  # -> 20 deg

# Center roughly on the New England / Quebec border; 2x2 mask units.
# Verified against the analytic grid below: 30.0 and -90.0 are exact MERRA-2
# grid points, and the box below spans exactly 60 lat x 64 lon pixels
# (2 x MASK_UNIT_LAT_PX, 2 x MASK_UNIT_LON_PX).
DEFAULT_LAT_MIN, DEFAULT_LAT_MAX = 30.0, 59.5
DEFAULT_LON_MIN, DEFAULT_LON_MAX = -90.0, -50.625
DEFAULT_MASK_UNITS_LAT = 2
DEFAULT_MASK_UNITS_LON = 2


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
    parser.add_argument("--mask-unit-lat-px", type=int, default=MASK_UNIT_LAT_PX)
    parser.add_argument("--mask-unit-lon-px", type=int, default=MASK_UNIT_LON_PX)
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

    ok = True
    for n, axis, unit in [
        (result["n_lat"], "lat", args.mask_unit_lat_px),
        (result["n_lon"], "lon", args.mask_unit_lon_px),
    ]:
        if n % unit != 0:
            ok = False
            print(
                f"ERROR: n_{axis}={n} is not evenly divisible by "
                f"mask_unit size {unit}px -- PrithviWxC's constructor will "
                f"assert-fail on this crop. Adjust bounds to a multiple of "
                f"{unit} pixels ({unit * (MERRA2_LAT_STEP if axis == 'lat' else MERRA2_LON_STEP)} deg)."
            )
    if ok:
        n_global_lat = result["n_lat"] // args.mask_unit_lat_px
        n_global_lon = result["n_lon"] // args.mask_unit_lon_px
        print(
            f"OK: divisible by mask unit size. global_shape_mu = "
            f"({n_global_lat}, {n_global_lon}) -> {n_global_lat * n_global_lon} "
            f"global (regional) tokens per timestamp."
        )


if __name__ == "__main__":
    main()
