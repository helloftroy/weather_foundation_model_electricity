"""Download 2024 hourly zonal electricity demand from the ISO-NE Web Services API.

API docs: https://webservices.iso-ne.com/docs/v1.1/
Requires an ISO Express account (free registration at
https://www.iso-ne.com/isoexpress) -- the API uses HTTP Basic Auth over SSL.
Set credentials via env vars before running:

    export ISONE_WS_USERNAME="you@example.com"
    export ISONE_WS_PASSWORD="..."

Pulls BOTH the real-time hourly demand and the day-ahead cleared demand for
all New England load zones (not just one), since the source instructions say
not to throw away a second reasonable actual-load product if it's cheap to
also fetch. Real-time hourly demand is the primary "actual load" series;
day-ahead is kept alongside for comparison, clearly labeled by endpoint.

Zone location IDs are discovered dynamically from the API's /locations
endpoint rather than hardcoded, since the numeric location IDs are not
reliably documented and guessing them risks silently pulling the wrong zone.
"""
import argparse
import csv
import os
import time
from datetime import date, timedelta
from pathlib import Path

import requests

BASE_URL = "https://webservices.iso-ne.com/api/v1.1"

TARGET_ZONE_NAME_FRAGMENTS = [
    "MAINE",
    "NEW HAMPSHIRE",
    "VERMONT",
    "CONNECTICUT",
    "RHODE ISLAND",
    "NEMA",
    "SEMA",
    "WCMA",
]

ENDPOINTS = {
    "realtime_hourly_demand": "realtimehourlydemand",
    "dayahead_hourly_demand": "dayaheadhourlydemand",
}


def session_from_env() -> requests.Session:
    user = os.environ.get("ISONE_WS_USERNAME")
    pw = os.environ.get("ISONE_WS_PASSWORD")
    if not user or not pw:
        raise SystemExit("Set ISONE_WS_USERNAME and ISONE_WS_PASSWORD env vars first.")
    s = requests.Session()
    s.auth = (user, pw)
    s.headers.update({"Accept": "application/json"})
    return s


def discover_zone_locations(session: requests.Session) -> dict:
    """Return {zone_name: location_id} for the 8 New England load zones."""
    resp = session.get(f"{BASE_URL}/locations.json")
    resp.raise_for_status()
    payload = resp.json()
    locations = payload.get("Locations", {}).get("Location", [])

    found = {}
    for loc in locations:
        name = str(loc.get("LocationName", "")).upper()
        loc_id = loc.get("LocationId")
        for fragment in TARGET_ZONE_NAME_FRAGMENTS:
            if fragment in name and fragment not in found:
                found[fragment] = loc_id

    missing = set(TARGET_ZONE_NAME_FRAGMENTS) - set(found)
    if missing:
        print(f"WARNING: could not auto-match zones: {missing}")
        print("All locations returned by the API, for manual mapping:")
        for loc in locations:
            print(f"  {loc.get('LocationId')}: {loc.get('LocationName')}")
    return found


def daterange(start: date, end: date):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sleep-seconds", type=float, default=0.5, help="Throttle between requests")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    session = session_from_env()

    zones = discover_zone_locations(session)
    print(f"Resolved zones: {zones}")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    for series_name, endpoint in ENDPOINTS.items():
        out_path = args.out_dir / f"isone_{series_name}_{args.start}_{args.end}.csv"
        rows_written = 0
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["zone", "location_id", "date", "raw_json_path"])
            raw_dir = args.out_dir / "raw" / series_name
            raw_dir.mkdir(parents=True, exist_ok=True)

            for zone_name, loc_id in zones.items():
                for day in daterange(start, end):
                    day_str = day.strftime("%Y%m%d")
                    url = f"{BASE_URL}/{endpoint}/day/{day_str}/location/{loc_id}.json"
                    resp = session.get(url)
                    if resp.status_code != 200:
                        print(f"  {series_name} {zone_name} {day_str}: HTTP {resp.status_code}")
                        time.sleep(args.sleep_seconds)
                        continue
                    raw_path = raw_dir / f"{zone_name}_{day_str}.json"
                    raw_path.write_text(resp.text)
                    writer.writerow([zone_name, loc_id, day.isoformat(), str(raw_path)])
                    rows_written += 1
                    time.sleep(args.sleep_seconds)
        print(f"{series_name}: wrote {rows_written} day-zone raw files, index at {out_path}")

    print(
        "\nRaw per-day JSON kept under out-dir/raw/<series>/. Build the "
        "cleaned timestamp|zone|demand_MW table from these in a separate QC/"
        "cleaning step, preserving both UTC and local (with DST flag)."
    )


if __name__ == "__main__":
    main()
