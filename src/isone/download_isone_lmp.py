"""Download hourly ISO-NE locational marginal prices (LMPs) for the load zones.

For the later price-prediction phase. Uses the same ISO Express credentials
and the same rate-limit handling as download_isone_demand.py:

    export ISONE_WS_USERNAME="you@example.com"
    export ISONE_WS_PASSWORD="..."

Two series, both final (settled) hourly prices:
    dayahead_hourly_lmp   /hourlylmp/da/final/day/{day}/location/{id}
    realtime_hourly_lmp   /hourlylmp/rt/final/day/{day}/location/{id}

Locations: the eight load zones plus the internal hub (4000), which is the
standard New England reference price.

Raw JSON is saved one file per (location, day), with an index CSV per
series. The response field names have not been seen yet: the first request
is printed so the parser can be written against a real sample.
"""
import argparse
import csv
import time
from datetime import date
from pathlib import Path

from download_isone_demand import BASE_URL, ZONES, daterange, fetch, session_from_env

LOCATIONS = {"HUB": 4000, **ZONES}

ENDPOINTS = {
    "dayahead_hourly_lmp": "hourlylmp/da/final",
    "realtime_hourly_lmp": "hourlylmp/rt/final",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--locations", nargs="+", default=list(LOCATIONS), choices=list(LOCATIONS))
    parser.add_argument("--sleep-seconds", type=float, default=1.5, help="Pause between requests (ISO-NE rate-limits bursts)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    session = session_from_env()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    # One request per series up front: fails fast on bad credentials or a
    # wrong path, and shows the real response format.
    for series_name, endpoint in ENDPOINTS.items():
        url = f"{BASE_URL}/{endpoint}/day/{start:%Y%m%d}/location/{LOCATIONS['HUB']}.json"
        ok, detail, resp = fetch(session, url)
        print(f"Probe {series_name}: {url} -> {detail}")
        print(f"Probe response (first 800 chars): {resp.text[:800] if resp is not None else ''}", flush=True)
        if not ok:
            raise SystemExit(f"Probe request for {series_name} failed ({detail}); not starting the download.")

    total_files = 0
    for series_name, endpoint in ENDPOINTS.items():
        raw_dir = args.out_dir / "raw" / series_name
        raw_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        n_new = n_failed = consecutive_failures = 0

        for loc_name in args.locations:
            loc_id = LOCATIONS[loc_name]
            for day in daterange(start, end):
                day_str = day.strftime("%Y%m%d")
                raw_path = raw_dir / f"{loc_name}_{day_str}.json"
                if raw_path.exists() and raw_path.stat().st_size > 0:
                    rows.append([loc_name, loc_id, day.isoformat(), str(raw_path)])
                    continue
                url = f"{BASE_URL}/{endpoint}/day/{day_str}/location/{loc_id}.json"
                ok, detail, resp = fetch(session, url)
                if not ok:
                    n_failed += 1
                    consecutive_failures += 1
                    print(f"  {series_name} {loc_name} {day_str}: {detail}", flush=True)
                    if consecutive_failures >= 5:
                        raise SystemExit(
                            f"5 requests in a row failed (last: {detail}); stopping. "
                            "Re-run to resume -- completed files are skipped."
                        )
                    time.sleep(args.sleep_seconds * 4)
                    continue
                consecutive_failures = 0
                raw_path.write_text(resp.text)
                rows.append([loc_name, loc_id, day.isoformat(), str(raw_path)])
                n_new += 1
                if n_new % 200 == 0:
                    print(f"  {series_name}: {n_new} new files so far (latest {loc_name} {day_str})", flush=True)
                time.sleep(args.sleep_seconds)

        index_path = args.out_dir / f"isone_{series_name}_{args.start}_{args.end}.csv"
        with index_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["location", "location_id", "date", "raw_json_path"])
            writer.writerows(rows)
        total_files += len(rows)
        expected = len(args.locations) * ((end - start).days + 1)
        print(f"{series_name}: {len(rows)}/{expected} location-day files indexed ({n_new} new, {n_failed} failed), index at {index_path}")

    if total_files == 0:
        raise SystemExit("No files were downloaded for any series.")


if __name__ == "__main__":
    main()
