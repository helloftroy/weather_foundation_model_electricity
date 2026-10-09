"""Build a daily natural gas price table for the price models. Runs locally.

Two public series, both in $/MMBtu:

  algonquin_weekly   Algonquin Citygate (the New England benchmark). No free
                     daily series exists; daily data is sold by NGI, Platts
                     and others. EIA's Natural Gas Weekly Update quotes the
                     price each report week in its text ("from $X/MMBtu last
                     Wednesday to $Y/MMBtu yesterday"). This script downloads
                     the weekly reports, extracts those values (roughly one
                     per Wednesday), and interpolates linearly to daily.
                     Price moves between Wednesdays are therefore NOT
                     captured: mid-week spikes are missed or smoothed.
  henry_hub          Henry Hub daily spot (national benchmark), from EIA's
                     RNGWHHDd.xls. A true daily series; weekends and holidays
                     are filled from the last trading day.

Interpolating between weekly points uses the following Wednesday's value,
so algonquin_weekly is suitable for explaining prices, not for a forecast.
"""
import argparse
import datetime as dt
import html
import re
import subprocess
import time
from pathlib import Path

import pandas as pd

NGWU_URL = "https://www.eia.gov/naturalgas/weekly/archivenew_ngwu/{d:%Y}/{d:%m_%d}/"
HENRY_HUB_URL = "https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDd.xls"
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SENTENCE = re.compile(
    # ".{0,60}?" allows for the change amount, which is sometimes in dollars ("went up $9.31 from ...").
    r"Algonquin Citygate, which serves Boston-area consumers\s*,.{0,60}?from \$(\d+\.\d+)[^$]*?"
    r"last (Wednesday|Thursday|Tuesday|Friday|week) to \$(\d+\.\d+)[^.]*?(yesterday|this Wednesday|this week)",
    re.S,
)


def fetch(url: str, dest: Path) -> str:
    if not dest.exists():
        r = subprocess.run(["curl", "-sL", "-m", "40", "-w", "%{http_code}", "-o", str(dest), url], capture_output=True, text=True)
        if r.stdout.strip() != "200":
            dest.write_text("")
        time.sleep(0.3)
    return dest.read_text(errors="ignore")


def previous_weekday(day: dt.date, name: str) -> dt.date:
    """Most recent `name` strictly before `day`."""
    back = (day.weekday() - WEEKDAYS.index(name) - 1) % 7 + 1
    return day - dt.timedelta(days=back)


def algonquin_points(cache: Path, start: dt.date, end: dt.date) -> pd.DataFrame:
    rows, last_pair = [], None
    thursday = start - dt.timedelta(days=(start.weekday() - 3) % 7 + 7)
    while thursday <= end + dt.timedelta(days=14):
        for offset in (0, -1, 1, -2):  # holidays move publication off Thursday
            published = thursday + dt.timedelta(days=offset)
            text = fetch(NGWU_URL.format(d=published), cache / f"{published:%Y_%m_%d}.html")
            plain = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text)))
            m = SENTENCE.search(plain)
            if not m:
                continue
            prev_price, prev_word, price, cur_word = float(m[1]), m[2], float(m[3]), m[4]
            if (prev_price, price) == last_pair:
                break  # EIA's archive sometimes serves the previous week's page again
            last_pair = (prev_price, price)
            current = published - dt.timedelta(days=1) if cur_word == "yesterday" else previous_weekday(published + dt.timedelta(days=1), "Wednesday")
            previous = current - dt.timedelta(days=7) if prev_word == "week" else previous_weekday(current, prev_word)
            rows.append({"date": current, "price": price, "report": published, "role": "current"})
            rows.append({"date": previous, "price": prev_price, "report": published, "role": "previous"})
            break
        thursday += dt.timedelta(days=7)

    pts = pd.DataFrame(rows)
    # A date can be quoted twice (as one report's "yesterday" and the next
    # report's "last Wednesday"). They should agree; report where they don't.
    spread = pts.groupby("date")["price"].agg(["min", "max", "count"])
    clash = spread[spread["max"] - spread["min"] > 0.005]
    if len(clash):
        print(f"WARNING: {len(clash)} dates quoted with two different prices (keeping the 'current' quote):")
        print(clash.to_string())
    pts = pts.sort_values(["date", "role"]).drop_duplicates("date", keep="first")  # "current" sorts before "previous"
    return pts[["date", "price"]].sort_values("date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/gas"))
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()
    start, end = dt.date(args.year, 1, 1), dt.date(args.year, 12, 31)
    (args.cache_dir / "ngwu").mkdir(parents=True, exist_ok=True)

    pts = algonquin_points(args.cache_dir / "ngwu", start, end)
    pts.to_csv(args.cache_dir / f"algonquin_weekly_points_{args.year}.csv", index=False)
    in_year = pts[(pts["date"] >= start) & (pts["date"] <= end)]
    gaps = pd.Series(pd.to_datetime(pts["date"])).diff().dt.days
    print(f"Algonquin: {len(in_year)} quoted dates in {args.year} ({pts['date'].min()} to {pts['date'].max()} overall); "
          f"longest gap between quotes {int(gaps.max())} days; range ${in_year['price'].min():.2f}-${in_year['price'].max():.2f}")

    days = pd.date_range(start, end, freq="D")
    alg = pd.Series(pts["price"].to_numpy(), index=pd.to_datetime(pts["date"]))
    alg = alg.reindex(alg.index.union(days)).interpolate(method="time").reindex(days)
    days_to_quote = pd.Series([min(abs((d.date() - q).days) for q in pts["date"]) for d in days], index=days)

    hh_path = args.cache_dir / "RNGWHHDd.xls"
    if not hh_path.exists():
        subprocess.run(["curl", "-sL", "-m", "60", "-o", str(hh_path), HENRY_HUB_URL], check=True)
    hh = pd.read_excel(hh_path, sheet_name="Data 1", skiprows=2)
    hh.columns = ["date", "henry_hub"]
    hh = hh.dropna().assign(date=lambda d: pd.to_datetime(d["date"])).set_index("date")["henry_hub"]
    n_trading = int(hh.reindex(days).notna().sum())
    hh = hh.reindex(hh.index.union(days)).ffill().reindex(days)

    out = pd.DataFrame({"date": days.date, "algonquin_weekly": alg.round(3).to_numpy(), "henry_hub": hh.to_numpy(),
                        "algonquin_days_to_nearest_quote": days_to_quote.to_numpy()})
    if out[["algonquin_weekly", "henry_hub"]].isna().any().any():
        raise SystemExit(f"Gas table has gaps:\n{out[out.isna().any(axis=1)]}")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    monthly = out.assign(month=pd.to_datetime(out["date"]).dt.month).groupby("month")[["algonquin_weekly", "henry_hub"]].mean().round(2)
    print(f"Henry Hub: {n_trading} trading days in {args.year}.")
    print("Monthly means ($/MMBtu):")
    print(monthly.T.to_string())
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
