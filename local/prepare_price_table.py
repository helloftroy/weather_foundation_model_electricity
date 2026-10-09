"""Join hourly ISO-NE prices onto the demand modeling table, for the price
models. Runs locally.

Input: one parsed LMP table (parse_isone_lmp.py output: day-ahead or
real-time) and the demand modeling table from prepare_modeling_table.py
(calendar, own-zone weather, demand, and Prithvi columns if present).

Output: one row per zone-hour with the price columns added, plus
system_demand_MW (the eight zones summed), since the energy price is set
system-wide rather than zone by zone. The hub has no demand or weather of
its own, so hub prices are not carried into this table.
"""
import argparse
from pathlib import Path

import pandas as pd

PRICE_COLS = ["lmp_total", "lmp_energy", "lmp_congestion", "lmp_loss"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp-parquet", type=Path, required=True)
    parser.add_argument("--modeling-table", type=Path, required=True)
    parser.add_argument("--out-parquet", type=Path, required=True)
    args = parser.parse_args()

    lmp = pd.read_parquet(args.lmp_parquet)
    series = sorted(lmp["series"].unique())
    if len(series) != 1:
        raise SystemExit(f"Expected one price series per file, found {series}")
    lmp = lmp.rename(columns={"timestamp_parsed": "timestamp", "location": "zone"})[["timestamp", "zone", *PRICE_COLS]]
    lmp["timestamp"] = pd.to_datetime(lmp["timestamp"]).astype("datetime64[ns]")

    table = pd.read_parquet(args.modeling_table)
    table["timestamp"] = pd.to_datetime(table["timestamp"]).astype("datetime64[ns]")
    table["system_demand_MW"] = table.groupby("timestamp")["demand_MW"].transform("sum")
    n_zones = table.groupby("timestamp")["zone"].transform("nunique")
    table.loc[n_zones < table["zone"].nunique(), "system_demand_MW"] = float("nan")

    out = table.merge(lmp, on=["timestamp", "zone"], how="left", validate="one_to_one")
    out["price_series"] = series[0]
    n_missing = int(out["lmp_total"].isna().sum())
    print(f"{series[0]}: {len(out)} zone-hours, {n_missing} without a price.")
    print(out.groupby("zone")["lmp_total"].describe(percentiles=[0.01, 0.5, 0.99]).round(2).to_string())
    print(f"Hours with a negative price: {int((out['lmp_total'] < 0).sum())}")

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out_parquet, index=False)
    print(f"Wrote {len(out)} rows x {len(out.columns)} cols to {args.out_parquet}")


if __name__ == "__main__":
    main()
