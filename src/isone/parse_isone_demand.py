"""Parse the raw per-day ISO-NE JSON (from download_isone_demand.py) into a
clean timestamp | zone | demand_MW | series table.

Format confirmed against a real response (2026-10-07):
{"HourlyRtDemands": {"HourlyRtDemand": [{"BeginDate":
"2024-01-01T00:00:00.000-05:00", "Location": {"$": ".Z.MAINE", "@LocId":
"4001"}, "Load": 1211.896}, ...]}}. BeginDate is local Eastern time with its
UTC offset; output timestamp_parsed is UTC (timezone-naive) and
timestamp_local / utc_offset_hours preserve the local view.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

DEMAND_WRAPPER_KEYS = ["HourlyRtDemands", "HourlyDaDemands"]
DEMAND_LIST_KEYS = ["HourlyRtDemand", "HourlyDaDemand"]
LOAD_KEYS = ["Load", "load", "RtDemandMw", "DaDemandMw", "DemandMw"]
BEGIN_DATE_KEYS = ["BeginDate", "beginDate", "begin_date"]
LOCATION_ID_KEYS = ["LocId", "@LocId", "LocationId"]


def _first_present(d: dict, keys: list[str]):
    for k in keys:
        if k in d:
            return d[k]
    return None


def _find_records(node) -> list[dict]:
    """Collect every dict that has a BeginDate-style key, at any depth, so the
    parser doesn't depend on the exact wrapper names ISO-NE's JSON uses."""
    found = []
    if isinstance(node, dict):
        if any(k in node for k in BEGIN_DATE_KEYS):
            found.append(node)
        else:
            for v in node.values():
                found.extend(_find_records(v))
    elif isinstance(node, list):
        for v in node:
            found.extend(_find_records(v))
    return found


def _describe(node, depth: int = 0, max_depth: int = 4) -> str:
    """Short structural summary of a JSON payload, for error messages."""
    if depth >= max_depth:
        return "..."
    if isinstance(node, dict):
        return "{" + ", ".join(f"{k}: {_describe(v, depth + 1)}" for k, v in list(node.items())[:8]) + "}"
    if isinstance(node, list):
        return f"[{len(node)} x {_describe(node[0], depth + 1) if node else 'empty'}]"
    return type(node).__name__


def parse_one_file(path: Path) -> list[dict]:
    text = path.read_text()
    if not text.strip():
        raise ValueError(f"{path}: file is empty")
    payload = json.loads(text)

    records = _find_records(payload)
    if not records:
        raise ValueError(
            f"{path}: no record with a BeginDate-style key found. Structure: {_describe(payload)}"
        )

    rows = []
    for rec in records:
        begin_date = _first_present(rec, BEGIN_DATE_KEYS)
        load = _first_present(rec, LOAD_KEYS)
        location = rec.get("Location", {})
        loc_id = _first_present(location, LOCATION_ID_KEYS) if isinstance(location, dict) else None
        if load is None:
            raise ValueError(f"{path}: record has no known load key ({LOAD_KEYS}). Record was: {rec}")
        rows.append({"timestamp_raw": begin_date, "demand_MW": float(load), "location_id_raw": loc_id})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isone-dir", type=Path, required=True, help="Output dir from download_isone_demand.py")
    parser.add_argument("--series", default="realtime_hourly_demand", choices=["realtime_hourly_demand", "dayahead_hourly_demand"])
    parser.add_argument("--out-parquet", type=Path, required=True)
    args = parser.parse_args()

    index_files = sorted(args.isone_dir.glob(f"isone_{args.series}_*.csv"))
    if not index_files:
        raise SystemExit(f"No index CSV found for series '{args.series}' under {args.isone_dir}")
    index_df = pd.concat([pd.read_csv(f) for f in index_files], ignore_index=True)
    print(f"Index: {len(index_df)} (zone, day) entries")

    all_rows = []
    n_failed = 0
    for _, r in index_df.iterrows():
        raw_path = Path(r["raw_json_path"])
        try:
            parsed = parse_one_file(raw_path)
        except (ValueError, json.JSONDecodeError, OSError) as e:
            n_failed += 1
            if n_failed <= 5:
                print(f"WARN: {e}")
            continue
        for row in parsed:
            row["zone"] = r["zone"]
        all_rows.extend(parsed)

    if n_failed:
        print(f"{n_failed}/{len(index_df)} files failed to parse -- see warnings above (first 5 shown).")

    if not all_rows:
        raise SystemExit(
            f"No demand rows parsed for series '{args.series}': index had {len(index_df)} "
            f"(zone, day) entries and {n_failed} files failed to parse. "
            + ("The index is empty, so the download wrote no raw files -- check the "
               "download job's log for HTTP errors or an unmatched-zones warning."
               if len(index_df) == 0 else "See the WARN lines above for the actual file structure.")
        )

    df = pd.DataFrame(all_rows)
    # BeginDate is local Eastern time with an explicit UTC offset, e.g.
    # "2024-01-01T00:00:00.000-05:00" (-04:00 during daylight saving), and
    # marks the START of the hour. Convert via the offset to UTC so it lines
    # up with MERRA-2 (UTC); keep the local wall-clock time and offset too.
    ts_utc = pd.to_datetime(df["timestamp_raw"], utc=True, errors="coerce")
    n_bad_ts = int(ts_utc.isna().sum())
    if n_bad_ts:
        print(f"WARNING: {n_bad_ts} rows had an unparseable timestamp_raw value.")
    df["timestamp_parsed"] = ts_utc.dt.tz_localize(None).astype("datetime64[ns]")  # UTC, timezone-naive like the weather tables
    ts_local = ts_utc.dt.tz_convert("America/New_York")
    df["timestamp_local"] = ts_local.dt.tz_localize(None).astype("datetime64[ns]")
    df["utc_offset_hours"] = ts_local.map(lambda x: x.utcoffset().total_seconds() / 3600 if pd.notna(x) else float("nan"))

    per_zone = df.groupby("zone")["timestamp_parsed"].agg(["count", "nunique", "min", "max"])
    print("Rows per zone (UTC range; a full leap year is 8784 hours):")
    print(per_zone.to_string())

    df = df[["timestamp_parsed", "timestamp_local", "utc_offset_hours", "timestamp_raw", "zone", "demand_MW", "location_id_raw"]].sort_values(["zone", "timestamp_parsed"])
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet, index=False)
    print(f"Wrote {len(df)} rows to {args.out_parquet}")


if __name__ == "__main__":
    main()
