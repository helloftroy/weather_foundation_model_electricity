"""Parse the raw per-day ISO-NE JSON (from download_isone_demand.py) into a
clean timestamp | zone | demand_MW | series table.

The exact JSON key casing from ISO-NE's XML->JSON auto-conversion is NOT
independently verifiable without a live sample (their docs only describe the
underlying XSD: HourlyRtDemand/HourlyDaDemand each have BeginDate, a Location
reference, and a Load value -- see docs/isone_demand_sources.md). This parser
tries the documented field names under a few plausible wrapper/casing
variants and fails loudly with the raw top-level keys if none match, so a
mismatch is a quick fix against one real file rather than a silent bad join.

VERIFY ON CLUSTER: run this against one real downloaded raw JSON file first
(a handful, not the whole year) and eyeball the output before trusting a
full-year parse.
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
    # NOT yet labeled UTC or Eastern -- ISO-NE's BeginDate timezone convention
    # is unconfirmed (see docs/isone_demand_sources.md). Parse as naive/local
    # to whatever the string says and require explicit confirmation before
    # this is treated as either zone, per the "don't silently convert
    # timestamps" instruction.
    df["timestamp_parsed"] = pd.to_datetime(df["timestamp_raw"], errors="coerce")
    n_bad_ts = df["timestamp_parsed"].isna().sum()
    if n_bad_ts:
        print(f"WARNING: {n_bad_ts} rows had an unparseable timestamp_raw value -- inspect timestamp_raw format by hand.")
    print(
        "NOTE: timestamp_parsed's timezone (UTC vs. Eastern, and DST handling) "
        "is NOT yet confirmed -- check a few known-hour values by hand against "
        "ISO-NE's dashboard before using this for modeling. See "
        "docs/isone_demand_sources.md."
    )

    df = df[["timestamp_parsed", "timestamp_raw", "zone", "demand_MW", "location_id_raw"]].sort_values(["zone", "timestamp_parsed"])
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet, index=False)
    print(f"Wrote {len(df)} rows to {args.out_parquet}")


if __name__ == "__main__":
    main()
