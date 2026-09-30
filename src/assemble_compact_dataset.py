"""Join Prithvi embeddings + conventional weather + ISO-NE demand into the
compact tables to transfer off the cluster.

Prithvi embeddings and the conventional weather baseline are both on the
MERRA-2 vertical-collection's native 3-hourly cadence (00,03,...,21 UTC);
ISO-NE demand is hourly. We attach each hourly demand row to its NEAREST
weather timestamp (default 90 min tolerance) rather than interpolating
weather to hourly -- interpolation is an explicit later modeling decision,
not something to bake into the compact export (per the source instructions:
"do not interpolate weather to hourly yet").

Output: one row per (timestamp, zone) with demand + weather-baseline columns
(zone-specific where available) + Prithvi embedding columns (shared across
zones, since the embedding isn't zone-specific -- it describes the whole
regional atmospheric state). This repeats the embedding across zones, which
is the deliberate, simple choice for a "one CatBoost model with zone as a
categorical feature" first pass; a per-zone-only model could instead drop
the duplication, but that's a Phase 3 modeling detail, not an export-format
constraint.

Does NOT include the raw MERRA-2 cube -- only these compact, already-small
tables leave the cluster.
"""
import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-parquet", type=Path, required=True)
    parser.add_argument("--conventional-weather-parquet", type=Path, required=True)
    parser.add_argument("--isone-demand-parquet", type=Path, required=True, help="Output of parse_isone_demand.py")
    parser.add_argument("--tolerance-minutes", type=int, default=90)
    parser.add_argument("--out-parquet", type=Path, required=True)
    args = parser.parse_args()

    emb = pd.read_parquet(args.embeddings_parquet)
    weather = pd.read_parquet(args.conventional_weather_parquet)
    demand = pd.read_parquet(args.isone_demand_parquet)

    weather_cols = [c for c in weather.columns if c != "timestamp"]
    combined_weather = emb.merge(weather, on="timestamp", how="inner")
    print(f"Embeddings: {len(emb)} rows. Weather: {len(weather)} rows. Combined weather+embedding: {len(combined_weather)} rows.")
    if len(combined_weather) < min(len(emb), len(weather)):
        print(
            "WARNING: embeddings and conventional weather don't fully overlap on timestamp -- "
            "check both were built from the same date range/cadence."
        )

    combined_weather = combined_weather.sort_values("timestamp")
    demand = demand.rename(columns={"timestamp_parsed": "timestamp"}).sort_values("timestamp")
    demand = demand.dropna(subset=["timestamp"])

    out_frames = []
    for zone, zone_demand in demand.groupby("zone"):
        zone_demand = zone_demand.sort_values("timestamp")
        merged = pd.merge_asof(
            zone_demand,
            combined_weather,
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=args.tolerance_minutes),
        )
        n_unmatched = merged[[c for c in combined_weather.columns if c != "timestamp"][0]].isna().sum()
        if n_unmatched:
            print(f"{zone}: {n_unmatched}/{len(merged)} demand rows had no weather match within {args.tolerance_minutes} min")
        out_frames.append(merged)

    out = pd.concat(out_frames, ignore_index=True).sort_values(["zone", "timestamp"])
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out_parquet, index=False)
    print(f"Wrote {len(out)} rows x {len(out.columns)} cols to {args.out_parquet}")
    print(f"Zones present: {sorted(out['zone'].unique().tolist())}")


if __name__ == "__main__":
    main()
