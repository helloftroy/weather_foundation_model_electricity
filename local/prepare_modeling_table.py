"""Add calendar features and a per-row "own-zone weather" view to the
compact table transferred from the cluster. Runs locally -- no cluster
access needed.

Deliberately minimal calendar set per the source instructions (hour, day of
week, day of year, weekend, holiday, year, zone) -- not over-engineered.

NOTE: the timestamp column's timezone convention (UTC vs. Eastern, DST
handling) is not yet confirmed -- see docs/isone_demand_sources.md. Calendar
features (hour, weekend, holiday) are computed directly off whatever
"timestamp" contains; if that turns out to be UTC rather than Eastern local
time, re-derive these once a confirmed Eastern-local column exists, since
"hour" and "is_weekend" are meaningfully wrong if computed in the wrong
timezone for a demand model.
"""
import argparse
from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

BASELINE_WEATHER_VARS = ["T2M", "QV2M", "U10M", "V10M", "SWGNT"]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_year"] = ts.dt.dayofyear
    df["month"] = ts.dt.month
    df["year"] = ts.dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    holidays = USFederalHolidayCalendar().holidays(start=ts.min(), end=ts.max())
    df["is_holiday"] = ts.dt.normalize().isin(holidays).astype(int)
    return df


def add_own_zone_weather(df: pd.DataFrame) -> pd.DataFrame:
    for var in BASELINE_WEATHER_VARS:
        col_for_row = df["zone"].astype(str) + f"_{var}"
        available = set(df.columns)
        df[f"weather_{var}"] = [
            df.at[i, c] if c in available else float("nan")
            for i, c in col_for_row.items()
        ]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-parquet", type=Path, required=True, help="Output of assemble_compact_dataset.py, transferred locally")
    parser.add_argument("--out-parquet", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.compact_parquet)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = add_calendar_features(df)
    df = add_own_zone_weather(df)

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet, index=False)
    print(f"Wrote {len(df)} rows x {len(df.columns)} cols to {args.out_parquet}")


if __name__ == "__main__":
    main()
