"""Download ISO-NE's own demand forecasts, every published version, from the
ISO-NE Web Services API (same ISO Express credentials as
download_isone_demand.py: ISONE_WS_USERNAME / ISONE_WS_PASSWORD).

Two series, one raw JSON file per target day:
    reliability_region_forecast  /reliabilityregionloadforecast/day/{day}/all
                                 Three-Day Reliability Region Demand Forecast:
                                 hourly MW for the eight reliability regions,
                                 every version issued for that day (in-day and
                                 the two preceding days), so the forecast
                                 issued the day before can be picked out.
    system_forecast              /hourlyloadforecast/all/day/{day}
                                 the system-wide hourly forecast, all versions.
    estimated_zonal_load         /fiveminuteestimatedzonalload/day/{day}
                                 Five-Minute Estimated Zonal Load: actual
                                 demand by load zone, five-minute
                                 instantaneous samples, in two versions:
                                 ordinary demand and demand including
                                 estimated behind-the-meter solar. Used to
                                 match the forecast's own target definition.
                                 History depth in the API is not documented:
                                 the probe on --start shows whether it goes
                                 back that far.

--series picks which to fetch (default: all three; files already
downloaded are skipped, so re-running only fetches what is missing).

Also saved once: /reliabilityregionloadforecast/info and /locations/all, to
check the region names against our load zones before using the data.
Reliability regions "generally correspond" to load zones (ISO-NE's wording);
the mapping and the definition of the forecast quantity are verified
locally against the actual zonal demand before any comparison.

Raw responses only; parsing happens locally once the format is seen. Skips
files already downloaded, waits out rate limits, stops after 5 failures in a
row (re-run to resume).
"""
import argparse
import time
from datetime import date
from pathlib import Path

from download_isone_demand import BASE_URL, daterange, fetch, session_from_env

SERIES = {
    "reliability_region_forecast": ["reliabilityregionloadforecast/day/{day}/all", "reliabilityregionloadforecast/day/{day}"],
    "system_forecast": ["hourlyloadforecast/all/day/{day}", "hourlyloadforecast/day/{day}"],
    "estimated_zonal_load": ["fiveminuteestimatedzonalload/day/{day}"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sleep-seconds", type=float, default=1.5)
    parser.add_argument("--series", nargs="+", default=list(SERIES), choices=list(SERIES))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    session = session_from_env()

    for name, path in [("reliability_region_forecast_info", "reliabilityregionloadforecast/info"), ("locations_all", "locations/all")]:
        ok, detail, resp = fetch(session, f"{BASE_URL}/{path}.json")
        print(f"{path}: {detail}")
        if ok:
            (args.out_dir / f"{name}.json").write_text(resp.text)
            print(f"  first 800 chars: {resp.text[:800]}")
        elif resp is not None and resp.status_code in (401, 403):
            raise SystemExit("ISO-NE rejected the credentials (see download_isone_demand.py).")

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    for series, patterns in ((k, SERIES[k]) for k in args.series):
        raw_dir = args.out_dir / "raw" / series
        raw_dir.mkdir(parents=True, exist_ok=True)
        pattern = None
        day_str = start.strftime("%Y%m%d")
        for p in patterns:   # use the first URL form that works (the "all versions" one if available)
            ok, detail, resp = fetch(session, f"{BASE_URL}/{p.format(day=day_str)}.json")
            print(f"Probe {series}: {p.format(day=day_str)} -> {detail}; first 1500 chars:\n{resp.text[:1500] if resp is not None else ''}\n", flush=True)
            if ok:
                pattern = p
                break
        if pattern is None:
            # history may not reach --start: try the end date too, to tell "no history" from "wrong URL"
            day_end = end.strftime("%Y%m%d")
            ok, detail, resp = fetch(session, f"{BASE_URL}/{patterns[0].format(day=day_end)}.json")
            print(f"{series}: no URL form worked for {day_str}; same URL for {day_end} -> {detail}"
                  + (" (so the API's history does not reach the start date; try a later --start)" if ok else "") + "; skipping this series.")
            continue
        (raw_dir / "URL_FORM.txt").write_text(pattern + "\n")
        n_new = n_failed = consecutive = 0
        for day in daterange(start, end):
            day_str = day.strftime("%Y%m%d")
            out = raw_dir / f"{day_str}.json"
            if out.exists() and out.stat().st_size > 0:
                continue
            ok, detail, resp = fetch(session, f"{BASE_URL}/{pattern.format(day=day_str)}.json")
            if not ok:
                n_failed += 1
                consecutive += 1
                print(f"  {series} {day_str}: {detail}", flush=True)
                if consecutive >= 5:
                    raise SystemExit(f"5 requests in a row failed (last: {detail}); stopping. Re-run to resume.")
                time.sleep(args.sleep_seconds * 4)
                continue
            consecutive = 0
            tmp = out.with_suffix(".part")
            tmp.write_text(resp.text)
            tmp.rename(out)
            n_new += 1
            if n_new % 100 == 0:
                print(f"  {series}: {n_new} new days (latest {day_str})", flush=True)
            time.sleep(args.sleep_seconds)
        print(f"{series}: {n_new} new days, {n_failed} failed; files in {raw_dir}", flush=True)


if __name__ == "__main__":
    main()
