"""Download ERCOT day-ahead market (DAM) hourly settlement point prices for the hubs and load zones, and pack each
year into one parquet.

Source: ERCOT MIS public report 13060, "Historical DAM Load Zone and Hub Prices" (NP4-180-ER), one zip per year
(DAMLZHBSPP_YYYY.zip) holding an xlsx with one sheet per month; public, no login. Columns: Delivery Date, Hour Ending
(01:00..24:00, Central Prevailing Time), Repeated Hour Flag (Y on the second 01:00 when clocks fall back), Settlement
Point, Settlement Point Price ($/MWh). Settlement points: hubs HB_NORTH, HB_HOUSTON, HB_SOUTH, HB_WEST, HB_PAN,
HB_HUBAVG, HB_BUSAVG and load zones LZ_*; naming has been stable since the nodal market began (December 2010).

Output: data/compact/ercot_dam_spp_YYYY.parquet with timestamp (UTC, hour beginning, like the ISO-NE tables'
timestamp_parsed), timestamp_local (Central, hour beginning), utc_offset_hours, settlement_point, price.
The raw zips are kept in data/ercot/raw/.
"""
import argparse
import io
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

LISTING = "https://www.ercot.com/misapp/GetReports.do?reportTypeId=13060"
DOWNLOAD = "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId={}"
UA = {"User-Agent": "Mozilla/5.0 (research download; weather_foundation_model_electricity)"}


def fetch(url, tries=5):
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - retried
            print(f"  retry {attempt}/{tries}: {e}", flush=True)
            time.sleep(10 * attempt)
    raise RuntimeError(url)


def to_table(xlsx_bytes):
    d = pd.concat(pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name=None).values(), ignore_index=True)
    d.columns = [c.strip() for c in d.columns]
    day = pd.to_datetime(d["Delivery Date"], format="%m/%d/%Y")
    hour = d["Hour Ending"].astype(str).str.slice(0, 2).astype(int) - 1          # hour beginning, 0..23
    local = day + pd.to_timedelta(hour, unit="h")
    # Fall-back hour: the first 01:00 is still daylight time, the repeated one (flag Y) standard time.
    dst_first = (d["Repeated Hour Flag"].astype(str).str.upper() != "Y").to_numpy()
    aware = pd.DatetimeIndex(local).tz_localize("US/Central", ambiguous=dst_first, nonexistent="raise")
    return pd.DataFrame({"timestamp": aware.tz_convert("UTC").tz_localize(None), "timestamp_local": local.to_numpy(),
                         "utc_offset_hours": np.asarray([t.utcoffset().total_seconds() / 3600 for t in aware]),
                         "settlement_point": d["Settlement Point"].astype(str).to_numpy(),
                         "price": d["Settlement Point Price"].astype(float).to_numpy()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    raw = args.data_dir / "ercot" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (args.data_dir / "compact").mkdir(parents=True, exist_ok=True)
    listing = fetch(LISTING).decode("utf-8", "replace")
    ids = dict(re.findall(r"(DAMLZHBSPP_\d{4}\.zip).*?doclookupId=(\d+)", listing, re.S))
    for y in args.years:
        name = f"DAMLZHBSPP_{y}.zip"
        z = raw / name
        if not z.exists():
            if name not in ids:
                raise SystemExit(f"{name} not in the ERCOT listing ({sorted(ids)})")
            z.write_bytes(fetch(DOWNLOAD.format(ids[name])))
        with zipfile.ZipFile(z) as zf:
            xl = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]
            t = to_table(zf.read(xl[0]))
        t = t.sort_values(["settlement_point", "timestamp"]).reset_index(drop=True)
        dup = t.duplicated(["settlement_point", "timestamp"]).sum()
        per = t.groupby("settlement_point").size()
        out = args.data_dir / "compact" / f"ercot_dam_spp_{y}.parquet"
        t.to_parquet(out, index=False)
        hub = t[t["settlement_point"] == "HB_NORTH"]["price"]
        print(f"{y}: {len(t)} rows, {per.min()}-{per.max()} hours per point, {dup} duplicate hours; HB_NORTH median "
              f"${hub.median():.2f}, max ${hub.max():.0f}; {sorted(per.index)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
