"""Crop the official (global) Prithvi-WxC climatology files to our regional bbox.

`residual="climate"` is mandatory for the pretrained checkpoint -- the
static-embedding layer's weight shape was trained expecting a climate
tensor concatenated in (see PrithviWxC/model.py: patch_embedding_static's
input channels = in_channels + in_channels_static only under residual=
"climate"). So Merra2Dataset needs climate_surface_doyDDD_hourHH.nc /
climate_vertical_doyDDD_hourHH.nc files covering our region for every
(day-of-year, hour-of-day) we extract embeddings for.

These files are global and already downloaded in full by
cluster/sbatch/service/01_setup_prithvi.sbatch (into
<SCRATCH_ROOT>/weights/prithvi.wxc.2300m.v1/climatology/). This script just
crops each one to our lat/lon box and re-writes it under the project's own
data directory, preserving the exact filename convention Merra2Dataset
expects, so no code changes are needed on the Merra2Dataset side.
"""
import argparse
from pathlib import Path

import xarray as xr


def crop_and_write(src: Path, dst: Path, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> bool:
    if dst.exists():
        return True
    if not src.exists():
        return False
    ds = xr.open_dataset(src)
    cropped = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    dst.parent.mkdir(parents=True, exist_ok=True)
    cropped.to_netcdf(dst, engine="h5netcdf")
    ds.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--climatology-dir", type=Path, required=True, help="Global climatology/ dir from the Prithvi weights download")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lat-min", type=float, required=True)
    parser.add_argument("--lat-max", type=float, required=True)
    parser.add_argument("--lon-min", type=float, required=True)
    parser.add_argument("--lon-max", type=float, required=True)
    args = parser.parse_args()

    surf_files = sorted(args.climatology_dir.glob("climate_surface_doy*_hour*.nc"))
    vert_files = sorted(args.climatology_dir.glob("climate_vertical_doy*_hour*.nc"))
    print(f"Found {len(surf_files)} surface + {len(vert_files)} vertical climatology files")

    n_ok = 0
    for src in surf_files + vert_files:
        dst = args.out_dir / src.name
        if crop_and_write(src, dst, args.lat_min, args.lat_max, args.lon_min, args.lon_max):
            n_ok += 1
        if n_ok % 200 == 0 and n_ok > 0:
            print(f"...{n_ok} files cropped")

    print(f"Done. Cropped {n_ok}/{len(surf_files) + len(vert_files)} files into {args.out_dir}")


if __name__ == "__main__":
    main()
