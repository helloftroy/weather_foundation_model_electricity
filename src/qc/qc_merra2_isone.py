"""Compact QC report: timestamp coverage / gaps / units for MERRA-2 vs ISO-NE.

Run on morrill (CPU, no internet needed) after both downloads complete.
Does NOT interpolate or merge the two datasets -- just characterizes what
exists, per Phase 1 step 5.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import xarray as xr


def qc_merra2(merra2_dir: Path) -> dict:
    nc_files = sorted(merra2_dir.rglob("*.nc"))
    if not nc_files:
        return {"n_files": 0}

    timestamps = []
    variables = set()
    for f in nc_files:
        ds = xr.open_dataset(f)
        if "time" in ds.coords:
            timestamps.extend(pd.to_datetime(ds["time"].values))
        variables.update(ds.data_vars)
        ds.close()

    timestamps = sorted(set(timestamps))
    ts_series = pd.Series(timestamps)
    gaps = ts_series.diff().value_counts() if len(ts_series) > 1 else {}

    return {
        "n_files": len(nc_files),
        "n_timestamps": len(timestamps),
        "first_timestamp": str(timestamps[0]) if timestamps else None,
        "last_timestamp": str(timestamps[-1]) if timestamps else None,
        "variables": sorted(variables),
        "timestep_value_counts": {str(k): int(v) for k, v in gaps.items()} if len(gaps) else {},
    }


def qc_isone(isone_dir: Path) -> dict:
    index_files = sorted(isone_dir.glob("isone_*.csv"))
    report = {}
    for f in index_files:
        df = pd.read_csv(f)
        report[f.name] = {
            "n_rows": len(df),
            "zones": sorted(df["zone"].unique().tolist()) if "zone" in df else [],
            "date_min": df["date"].min() if "date" in df and len(df) else None,
            "date_max": df["date"].max() if "date" in df and len(df) else None,
            "days_covered": df["date"].nunique() if "date" in df else None,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merra2-dir", required=True, type=Path)
    parser.add_argument("--isone-dir", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()

    report = {
        "merra2": qc_merra2(args.merra2_dir),
        "isone": qc_isone(args.isone_dir),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
