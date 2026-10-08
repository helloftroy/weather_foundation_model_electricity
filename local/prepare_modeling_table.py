"""Add calendar features and a per-row "own-zone weather" view to the compact
table transferred from the cluster. Runs locally.

Calendar features (hour, day of week, weekend, holiday, ...) are computed
from Eastern LOCAL time (timestamp_local), since that is what drives demand;
the UTC timestamp is kept for joining and ordering.

Only the global-pooled Prithvi embedding (global_emb_*) is carried forward.
The per-region embeddings stay in the cluster output for later use. If the
embedding columns are entirely NaN they are dropped with a warning, so the
calendar and weather baselines can still be run.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas.tseries.holiday import USFederalHolidayCalendar

BASELINE_WEATHER_VARS = ["T2M", "QV2M", "U10M", "V10M", "SWGNT"]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp_local"]
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_year"] = ts.dt.dayofyear
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    holidays = USFederalHolidayCalendar().holidays(start=ts.min(), end=ts.max())
    df["is_holiday"] = ts.dt.normalize().isin(holidays).astype(int)
    return df


def add_own_zone_weather(df: pd.DataFrame) -> pd.DataFrame:
    for var in BASELINE_WEATHER_VARS:
        out = np.full(len(df), np.nan)
        for zone in df["zone"].unique():
            col = f"{zone}_{var}"
            if col in df.columns:
                mask = (df["zone"] == zone).to_numpy()
                out[mask] = df.loc[mask, col].to_numpy()
        df[f"weather_{var}"] = out
    df["weather_wind_speed"] = np.hypot(df["weather_U10M"], df["weather_V10M"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-parquet", type=Path, required=True, help="modeling_input_*.parquet from the cluster")
    parser.add_argument("--out-parquet", type=Path, required=True)
    args = parser.parse_args()

    all_cols = pq.ParquetFile(args.compact_parquet).schema.names
    emb_cols = [c for c in all_cols if c.startswith("global_emb_")]
    other_cols = [c for c in all_cols if "_emb_" not in c]

    df = pd.read_parquet(args.compact_parquet, columns=other_cols)
    for c in ("timestamp", "timestamp_local"):
        df[c] = pd.to_datetime(df[c])

    df = add_calendar_features(df)
    df = add_own_zone_weather(df)
    zone_cols = [c for c in df.columns if any(c.endswith(f"_{v}") for v in BASELINE_WEATHER_VARS) and not c.startswith("weather_")]
    df = df.drop(columns=zone_cols + ["timestamp_raw", "location_id_raw"], errors="ignore")

    n_no_weather = int(df["weather_T2M"].isna().sum())
    print(f"{len(df)} demand rows; {n_no_weather} have no weather within the join tolerance (dropped at training time).")

    if emb_cols:
        emb = pd.read_parquet(args.compact_parquet, columns=emb_cols)
        frac_nan = float(emb.isna().to_numpy().mean())
        if frac_nan == 1.0:
            print("WARNING: every Prithvi embedding value is NaN -- dropping the embedding columns. "
                  "Only the calendar and weather models can be trained from this table.")
        else:
            print(f"Prithvi global embedding: {len(emb_cols)} columns, {frac_nan:.2%} NaN.")
            df = pd.concat([df, emb.astype("float32")], axis=1)

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet, index=False)
    print(f"Wrote {len(df)} rows x {len(df.columns)} cols to {args.out_parquet}")


if __name__ == "__main__":
    main()
