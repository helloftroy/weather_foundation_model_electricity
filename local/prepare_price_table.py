"""Build the hourly system-level table for the price models. Runs locally.

Zonal prices in ISO-NE are almost one series: in 2024 every load zone's
day-ahead price correlates 0.995 or higher with the hub, and differs from
it by about $1/MWh on average. So the price target is the HUB price, one
row per hour, not eight near-identical zonal rows.

Inputs: one parsed LMP table (parse_isone_lmp.py output) and the demand
modeling table from prepare_modeling_table.py.

Output columns per hour:
    lmp_total etc.        hub price, $/MWh
    system_demand_MW      the eight zones' demand summed
    weather_*             the eight zones' weather, averaged with fixed
                          weights equal to each zone's share of annual demand
    calendar features     from Eastern local time
    global_emb_*          Prithvi embedding, if present in the modeling table
    gas_algonquin,        daily natural gas prices ($/MMBtu) from
    gas_henry_hub         build_gas_prices.py, if --gas-csv is given, joined
                          on the Eastern local date
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_COLS = ["lmp_total", "lmp_energy", "lmp_congestion", "lmp_loss"]
CALENDAR_COLS = ["timestamp_local", "utc_offset_hours", "hour", "day_of_week", "day_of_year", "month", "is_weekend", "is_holiday"]
WEATHER_COLS = ["weather_T2M", "weather_QV2M", "weather_U10M", "weather_V10M", "weather_SWGNT", "weather_wind_speed"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp-parquet", type=Path, required=True)
    parser.add_argument("--modeling-table", type=Path, required=True)
    parser.add_argument("--gas-csv", type=Path, help="Daily gas prices from build_gas_prices.py (optional)")
    parser.add_argument("--out-parquet", type=Path, required=True)
    args = parser.parse_args()

    lmp = pd.read_parquet(args.lmp_parquet)
    series = sorted(lmp["series"].unique())
    if len(series) != 1:
        raise SystemExit(f"Expected one price series per file, found {series}")
    lmp["timestamp"] = pd.to_datetime(lmp["timestamp_parsed"]).astype("datetime64[ns]")
    wide = lmp.pivot(index="timestamp", columns="location", values="lmp_total")
    zones = [c for c in wide.columns if c != "HUB"]
    print(f"{series[0]}: zone-vs-hub correlation {wide[zones].corrwith(wide['HUB']).min():.4f} to {wide[zones].corrwith(wide['HUB']).max():.4f}; "
          f"mean |zone - hub| {wide[zones].sub(wide['HUB'], axis=0).abs().mean().mean():.2f} $/MWh")
    hub = lmp[lmp["location"] == "HUB"].set_index("timestamp")[PRICE_COLS]

    table = pd.read_parquet(args.modeling_table)
    table["timestamp"] = pd.to_datetime(table["timestamp"]).astype("datetime64[ns]")
    n_zones = table["zone"].nunique()
    weights = table.groupby("zone")["demand_MW"].mean()
    weights = weights / weights.sum()
    table["_w"] = table["zone"].map(weights)

    g = table.groupby("timestamp")
    out = g[CALENDAR_COLS].first()
    out["system_demand_MW"] = g["demand_MW"].sum().where(g["zone"].nunique() == n_zones)
    for col in WEATHER_COLS:
        out[col] = (table[col] * table["_w"]).groupby(table["timestamp"]).sum(min_count=n_zones)
    emb_cols = [c for c in table.columns if c.startswith("global_emb_")]
    if emb_cols:
        out = out.join(g[emb_cols].first())
    out = out.join(hub, how="left").reset_index()
    out["price_series"] = series[0]

    if args.gas_csv:
        gas = pd.read_csv(args.gas_csv, parse_dates=["date"]).rename(columns={"algonquin_weekly": "gas_algonquin", "henry_hub": "gas_henry_hub"})
        out["date"] = out["timestamp_local"].dt.normalize()
        out = out.merge(gas[["date", "gas_algonquin", "gas_henry_hub"]], on="date", how="left").drop(columns="date")
        print(f"Gas prices joined; {int(out['gas_algonquin'].isna().sum())} hours without one. "
              f"Daily correlation with the hub price: Algonquin {out.groupby(out['timestamp_local'].dt.date)[['lmp_total', 'gas_algonquin']].mean().corr().iloc[0, 1]:.2f}, "
              f"Henry Hub {out.groupby(out['timestamp_local'].dt.date)[['lmp_total', 'gas_henry_hub']].mean().corr().iloc[0, 1]:.2f}")

    print(f"{len(out)} hours; {int(out['lmp_total'].isna().sum())} without a price, {int(out['weather_T2M'].isna().sum())} without weather.")
    print(out["lmp_total"].describe(percentiles=[0.01, 0.5, 0.9, 0.99]).round(2).to_string())
    print(f"Hours with a negative price: {int((out['lmp_total'] < 0).sum())}. Zone demand weights: {weights.round(3).to_dict()}")
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out_parquet, index=False)
    print(f"Wrote {len(out)} rows x {len(out.columns)} cols to {args.out_parquet}")


if __name__ == "__main__":
    main()
