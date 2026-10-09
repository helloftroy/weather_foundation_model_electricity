"""Price models: how much of the hourly ISO-NE hub price do calendar,
weather, and demand explain?

Target: the hub lmp_total ($/MWh), one row per hour, for one price series
(day-ahead or real-time). The hub is used instead of the eight zones
because zonal prices are nearly identical to it (see prepare_price_table.py).
One CatBoost model per feature set, with the same settings and the same two
splits as the demand models (train_catboost_comparisons.py).

Feature sets:
    calendar                  hour, day_of_week, day_of_year, month,
                              is_weekend, is_holiday
    calendar+weather          + demand-weighted regional T2M, QV2M, wind
                              speed, U10M, V10M, SWGNT
    calendar+weather+demand   + ACTUAL system demand for the same hour. An
                              upper bound on what a demand prediction could
                              contribute, not a usable forecast input.
    calendar+weather+gas      + natural gas prices: Algonquin Citygate
    calendar+weather+demand+gas   (weekly quotes interpolated to daily) and
                              Henry Hub daily. Added when the price table
                              has gas columns (see build_gas_prices.py).
    calendar+weather+demand+henryhub   Henry Hub only, to show whether the
                              national gas price is enough.
Prithvi feature sets are added automatically when global_emb_* columns are
present.

These are explanatory models: every input is from the same hour as the
price. Day-ahead prices are actually set the previous day from forecasts,
and no past-price information is included. The Algonquin gas series is
interpolated between weekly quotes, which uses the following week's value.

Scored in $/MWh (MAE and median absolute error) and R2. Percentage error is
not used: prices come close to zero and real-time prices go negative.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score

from train_catboost_comparisons import CALENDAR_FEATURES as ZONE_CALENDAR_FEATURES
from train_catboost_comparisons import WEATHER_FEATURES, assign_split

CALENDAR_FEATURES = [f for f in ZONE_CALENDAR_FEATURES if f != "zone"]
DEMAND_FEATURES = ["system_demand_MW"]
GAS_FEATURES = ["gas_algonquin", "gas_henry_hub"]
TARGET = "lmp_total"


def build_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    prithvi = [c for c in df.columns if c.startswith("global_emb_")]
    sets = {
        "calendar": CALENDAR_FEATURES,
        "calendar+weather": CALENDAR_FEATURES + WEATHER_FEATURES,
        "calendar+weather+demand": CALENDAR_FEATURES + WEATHER_FEATURES + DEMAND_FEATURES,
    }
    if all(c in df.columns for c in GAS_FEATURES):
        sets["calendar+weather+gas"] = CALENDAR_FEATURES + WEATHER_FEATURES + GAS_FEATURES
        sets["calendar+weather+demand+gas"] = CALENDAR_FEATURES + WEATHER_FEATURES + DEMAND_FEATURES + GAS_FEATURES
        sets["calendar+weather+demand+henryhub"] = CALENDAR_FEATURES + WEATHER_FEATURES + DEMAND_FEATURES + ["gas_henry_hub"]
    if prithvi:
        sets["calendar+prithvi"] = CALENDAR_FEATURES + prithvi
        sets["calendar+weather+prithvi"] = CALENDAR_FEATURES + WEATHER_FEATURES + prithvi
    return sets


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    err = np.abs(pred - y)
    return {"n": len(y), "r2": r2_score(y, pred), "mae": float(err.mean()), "median_abs_err": float(np.median(err)),
            "rmse": float(np.sqrt(np.mean((pred - y) ** 2))), "bias": float(np.mean(pred - y))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-table", type=Path, required=True, help="Output of prepare_price_table.py")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["chronological", "blocked"])
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    df = pd.read_parquet(args.price_table)
    df = df[df["timestamp_local"].dt.year == 2024].dropna(subset=[TARGET, "weather_T2M", "system_demand_MW"]).reset_index(drop=True)
    series = df["price_series"].iloc[0]
    feature_sets = build_feature_sets(df)
    print(f"{series}: {len(df)} rows, feature sets: {list(feature_sets)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    overall, by_condition, pred_frames = [], [], []
    for scheme in args.splits:
        split = assign_split(df, scheme)
        tr, va, te = df[split == "train"], df[split == "val"], df[split == "test"]
        print(f"\n[{scheme}] train {len(tr)}  val {len(va)}  test {len(te)} hours; price mean train {tr[TARGET].mean():.1f}, test {te[TARGET].mean():.1f} (sd {te[TARGET].std():.1f}) $/MWh")
        preds = te[["timestamp", "timestamp_local", TARGET, "system_demand_MW", "weather_T2M"]].copy()
        preds["split_scheme"] = scheme
        y = te[TARGET].to_numpy()
        price_rank = te[TARGET].rank(pct=True).to_numpy()
        temp_rank = te["weather_T2M"].rank(pct=True).to_numpy()
        # Reference point: always predicting the training-period mean price.
        overall.append({"split_scheme": scheme, "feature_set": "(training mean price)", "n_features": 0, "trees_used": 0,
                        **metrics(y, np.full(len(y), tr[TARGET].mean()))})

        for name, features in feature_sets.items():
            model = CatBoostRegressor(
                iterations=args.iterations, learning_rate=args.learning_rate, depth=args.depth, random_seed=args.seed,
                loss_function="RMSE", verbose=False, early_stopping_rounds=100, thread_count=-1,
                train_dir=str(args.out_dir / "catboost_info"),
            )
            model.fit(Pool(tr[features], tr[TARGET]), eval_set=Pool(va[features], va[TARGET]), use_best_model=True)
            pred = model.predict(te[features])
            preds[f"pred_{name}"] = pred
            row = {"split_scheme": scheme, "feature_set": name, "n_features": len(features),
                   "trees_used": model.get_best_iteration() + 1, **metrics(y, pred)}
            overall.append(row)
            print(f"  {name:26s} R2 {row['r2']:.3f}   MAE {row['mae']:.2f}   median abs err {row['median_abs_err']:.2f}   bias {row['bias']:+.2f} $/MWh   ({row['trees_used']} trees)")

            for label, m in [("high price (top 10%)", price_rank >= 0.9), ("ordinary price (middle 80%)", (price_rank > 0.1) & (price_rank < 0.9)),
                             ("hot (top 10% T2M)", temp_rank >= 0.9), ("cold (bottom 10% T2M)", temp_rank <= 0.1)]:
                by_condition.append({"split_scheme": scheme, "feature_set": name, "condition": label, **metrics(y[m], pred[m])})

            pd.DataFrame({"feature": features, "importance": model.get_feature_importance()}).sort_values("importance", ascending=False).to_csv(
                args.out_dir / f"feature_importance_{scheme}_{name.replace('+', '_')}.csv", index=False)
        pred_frames.append(preds)

    pd.DataFrame(overall).to_csv(args.out_dir / "comparison_summary.csv", index=False)
    pd.DataFrame(by_condition).to_csv(args.out_dir / "comparison_by_condition.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_parquet(args.out_dir / "test_predictions.parquet", index=False)
    print(f"\nWrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
