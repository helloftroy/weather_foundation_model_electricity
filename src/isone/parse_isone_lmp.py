"""Parse raw ISO-NE hourly LMP JSON (from download_isone_lmp.py) into one
clean table per series.

Format confirmed against real responses (2026-10-07):
{"HourlyLmps": {"HourlyLmp": [{"BeginDate": "2024-01-01T00:00:00.000-05:00",
"Location": {"@LocId": "4000", "@LocType": "HUB", "$": ".H.INTERNAL_HUB"},
"LmpTotal": 28.3, "EnergyComponent": 28.21, "CongestionComponent": 0,
"LossComponent": 0.09}, ...]}}

Prices are $/MWh. BeginDate is Eastern local time with its UTC offset and
marks the start of the hour. As in parse_isone_demand.py, timestamp_parsed
is UTC (timezone-naive) and timestamp_local / utc_offset_hours keep the
local view.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from parse_isone_demand import _describe, _find_records

PRICE_FIELDS = {
    "LmpTotal": "lmp_total",
    "EnergyComponent": "lmp_energy",
    "CongestionComponent": "lmp_congestion",
    "LossComponent": "lmp_loss",
}


def parse_one_file(path: Path) -> list[dict]:
    text = path.read_text()
    if not text.strip():
        raise ValueError(f"{path}: file is empty")
    payload = json.loads(text)
    records = _find_records(payload)
    if not records:
        raise ValueError(f"{path}: no hourly records found. Structure: {_describe(payload)}")

    rows = []
    for rec in records:
        if "LmpTotal" not in rec:
            raise ValueError(f"{path}: record has no LmpTotal. Record was: {rec}")
        location = rec.get("Location", {})
        row = {
            "timestamp_raw": rec["BeginDate"],
            "location_id_raw": location.get("@LocId") if isinstance(location, dict) else None,
            "location_name_raw": location.get("$") if isinstance(location, dict) else None,
        }
        for src, dst in PRICE_FIELDS.items():
            row[dst] = float(rec[src]) if rec.get(src) is not None else float("nan")
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp-dir", type=Path, required=True, help="Output dir from download_isone_lmp.py")
    parser.add_argument("--series", required=True, choices=["dayahead_hourly_lmp", "realtime_hourly_lmp"])
    parser.add_argument("--out-parquet", type=Path, required=True)
    args = parser.parse_args()

    index_files = sorted(args.lmp_dir.glob(f"isone_{args.series}_*.csv"))
    if not index_files:
        raise SystemExit(f"No index CSV found for series '{args.series}' under {args.lmp_dir}")
    index_df = pd.concat([pd.read_csv(f) for f in index_files], ignore_index=True).drop_duplicates("raw_json_path")
    print(f"Index: {len(index_df)} (location, day) entries")

    all_rows, n_failed = [], 0
    for r in index_df.itertuples():
        try:
            parsed = parse_one_file(Path(r.raw_json_path))
        except (ValueError, KeyError, json.JSONDecodeError, OSError) as e:
            n_failed += 1
            if n_failed <= 5:
                print(f"WARN: {e}")
            continue
        for row in parsed:
            row["location"] = r.location
            if row["location_id_raw"] is not None and str(row["location_id_raw"]) != str(r.location_id):
                raise SystemExit(f"{r.raw_json_path}: file says location {row['location_id_raw']}, index says {r.location_id}")
        all_rows.extend(parsed)

    if n_failed:
        print(f"{n_failed}/{len(index_df)} files failed to parse (first 5 shown above).")
    if not all_rows:
        raise SystemExit(f"No price rows parsed for series '{args.series}'.")

    df = pd.DataFrame(all_rows)
    ts_utc = pd.to_datetime(df["timestamp_raw"], utc=True, errors="coerce")
    n_bad_ts = int(ts_utc.isna().sum())
    if n_bad_ts:
        print(f"WARNING: {n_bad_ts} rows had an unparseable timestamp_raw value.")
    df["timestamp_parsed"] = ts_utc.dt.tz_localize(None).astype("datetime64[ns]")
    ts_local = ts_utc.dt.tz_convert("America/New_York")
    df["timestamp_local"] = ts_local.dt.tz_localize(None).astype("datetime64[ns]")
    df["utc_offset_hours"] = ts_local.map(lambda x: x.utcoffset().total_seconds() / 3600 if pd.notna(x) else float("nan"))
    df["series"] = args.series

    n_dup = int(df.duplicated(["location", "timestamp_parsed"]).sum())
    if n_dup:
        print(f"WARNING: {n_dup} duplicate (location, hour) rows.")

    summary = df.groupby("location").agg(
        hours=("timestamp_parsed", "nunique"),
        first=("timestamp_parsed", "min"),
        last=("timestamp_parsed", "max"),
        mean_lmp=("lmp_total", "mean"),
        min_lmp=("lmp_total", "min"),
        max_lmp=("lmp_total", "max"),
        missing=("lmp_total", lambda s: int(s.isna().sum())),
    )
    print("Per location (UTC range; a full leap year is 8784 hours; prices in $/MWh):")
    print(summary.round(2).to_string())

    cols = ["timestamp_parsed", "timestamp_local", "utc_offset_hours", "location", "series", *PRICE_FIELDS.values(),
            "timestamp_raw", "location_id_raw", "location_name_raw"]
    df = df[cols].sort_values(["location", "timestamp_parsed"])
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet, index=False)
    print(f"Wrote {len(df)} rows to {args.out_parquet}")


if __name__ == "__main__":
    main()
