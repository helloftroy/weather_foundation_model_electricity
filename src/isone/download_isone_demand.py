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

Zone location IDs are ISO-NE's fixed load-zone IDs (4001-4008). The script
prints the names the API reports for them (from /locations/all) so a wrong
mapping is visible in the log, tests one request before the full run, skips
files already downloaded, waits out HTTP 429 rate limits, and exits non-zero
if nothing was written.
"""
import argparse
import csv
import os
import time
from datetime import date, timedelta
from pathlib import Path

import requests

BASE_URL = "https://webservices.iso-ne.com/api/v1.1"

# ISO-NE load zone location IDs. Keys match ZONE_CENTROIDS in
# src/merra2/extract_conventional_weather.py -- the local join pairs each
# demand row with "<zone>_<var>" weather columns by this exact name.
ZONES = {
    "ME": 4001,
    "NH": 4002,
    "VT": 4003,
    "CT": 4004,
    "RI": 4005,
    "SEMA": 4006,
    "WCMA": 4007,
    "NEMA": 4008,
}

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


def report_zone_names(session: requests.Session) -> None:
    """Print what the API calls each of our location IDs. Informational only:
    a failure here doesn't stop the download."""
    try:
        resp = session.get(f"{BASE_URL}/locations/all.json", timeout=60)
        if resp.status_code != 200:
            print(f"Could not list locations (HTTP {resp.status_code}); continuing with fixed zone IDs.")
            return
        text = resp.text
        for zone, loc_id in ZONES.items():
            i = text.find(str(loc_id))
            snippet = text[max(0, i - 80): i + 80].replace("\n", " ") if i >= 0 else "NOT FOUND in locations list"
            print(f"  {zone} ({loc_id}): ...{snippet}...")
    except requests.RequestException as e:
        print(f"Could not list locations ({e}); continuing with fixed zone IDs.")


def probe(session: requests.Session, day_str: str) -> None:
    """One request up front, so bad credentials or a wrong URL fail in
    seconds with the server's own message instead of after a silent loop."""
    endpoint = ENDPOINTS["realtime_hourly_demand"]
    url = f"{BASE_URL}/{endpoint}/day/{day_str}/location/{ZONES['ME']}.json"
    resp = session.get(url, timeout=60)
    print(f"Probe {url} -> HTTP {resp.status_code}")
    print(f"Probe response (first 600 chars): {resp.text[:600]}")
    if resp.status_code in (401, 403):
        raise SystemExit(
            "ISO-NE rejected the credentials. Check ISONE_WS_USERNAME/ISONE_WS_PASSWORD, "
            "and that the ISO Express account has been approved for web services."
        )
    if resp.status_code != 200:
        raise SystemExit(f"Probe request failed with HTTP {resp.status_code}; not starting the full download.")


def fetch(session: requests.Session, url: str, max_rate_limit_waits: int = 12):
    """GET one URL. On HTTP 429 (rate limit), wait and retry the same request
    rather than skipping it: honour Retry-After if given, else back off from
    1 minute up to 15. Returns (ok, detail, response)."""
    resp = None
    for attempt in range(max_rate_limit_waits + 1):
        try:
            resp = session.get(url, timeout=60)
        except requests.RequestException as e:
            return False, str(e), None
        if resp.status_code != 429:
            return resp.status_code == 200, f"HTTP {resp.status_code}", resp
        retry_after = resp.headers.get("Retry-After", "")
        wait = int(retry_after) if retry_after.isdigit() else min(60 * 2 ** attempt, 900)
        print(f"  rate limited (HTTP 429); waiting {wait}s before retrying", flush=True)
        time.sleep(wait)
    return False, "HTTP 429 (still rate limited after repeated waits)", resp


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
    parser.add_argument("--sleep-seconds", type=float, default=1.5, help="Pause between requests (ISO-NE rate-limits bursts)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    session = session_from_env()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print("Zone names reported by the API for our location IDs:")
    report_zone_names(session)
    probe(session, start.strftime("%Y%m%d"))

    total_files = 0
    for series_name, endpoint in ENDPOINTS.items():
        out_path = args.out_dir / f"isone_{series_name}_{args.start}_{args.end}.csv"
        raw_dir = args.out_dir / "raw" / series_name
        raw_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        n_new = n_failed = consecutive_failures = 0

        for zone_name, loc_id in ZONES.items():
            for day in daterange(start, end):
                day_str = day.strftime("%Y%m%d")
                raw_path = raw_dir / f"{zone_name}_{day_str}.json"
                if raw_path.exists() and raw_path.stat().st_size > 0:
                    rows.append([zone_name, loc_id, day.isoformat(), str(raw_path)])
                    continue
                url = f"{BASE_URL}/{endpoint}/day/{day_str}/location/{loc_id}.json"
                ok, detail, resp = fetch(session, url)
                if not ok:
                    n_failed += 1
                    consecutive_failures += 1
                    print(f"  {series_name} {zone_name} {day_str}: {detail}", flush=True)
                    if consecutive_failures >= 5:
                        raise SystemExit(
                            f"5 requests in a row failed (last: {detail}); stopping. "
                            "Re-run to resume -- completed files are skipped."
                        )
                    time.sleep(args.sleep_seconds * 4)
                    continue
                consecutive_failures = 0
                raw_path.write_text(resp.text)
                rows.append([zone_name, loc_id, day.isoformat(), str(raw_path)])
                n_new += 1
                if n_new % 200 == 0:
                    print(f"  {series_name}: {n_new} new files so far (latest {zone_name} {day_str})", flush=True)
                time.sleep(args.sleep_seconds)

        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["zone", "location_id", "date", "raw_json_path"])
            writer.writerows(rows)
        total_files += len(rows)
        print(f"{series_name}: {len(rows)} day-zone files indexed ({n_new} new, {n_failed} failed), index at {out_path}")

    if total_files == 0:
        raise SystemExit("No files were downloaded for any series.")

    print(
        "\nRaw per-day JSON kept under out-dir/raw/<series>/. Build the "
        "cleaned timestamp|zone|demand_MW table from these in a separate QC/"
        "cleaning step, preserving both UTC and local (with DST flag)."
    )


if __name__ == "__main__":
    main()
