"""Price models: how much of the hourly ISO-NE price do calendar, weather,
and demand explain?

Target: lmp_total ($/MWh) for one price series (day-ahead or real-time),
for the eight load zones. One pooled CatBoost model per feature set, same
settings and the same two splits as the demand models
(train_catboost_comparisons.py).

Feature sets:
    calendar
    calendar+weather
    calendar+weather+demand   adds the zone's ACTUAL demand and the system
                              total for the same hour. An upper bound on
                              what a demand prediction could contribute,
                              not a usable forecast input.
Prithvi feature sets are added automatically when global_emb_* columns are
present.

These are explanatory models: every input is from the same hour as the
price. Day-ahead prices are actually set the previous day from forecasts,
and no fuel-price or past-price information is included.

Scored in $/MWh (MAE and median absolute error) and R2. Percentage error is
not used: prices come close to zero and can be negative.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score

from train_catboost_comparisons import CALENDAR_FEATURES, CAT_FEATURES, WEATHER_FEATURES, assign_split

DEMAND_FEATURES = ["demand_MW", "system_demand_MW"]
TARGET = "lmp_total"


def build_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    prithvi = [c for c in df.columns if c.startswith("global_emb_")]
    sets = {
        "calendar": CALENDAR_FEATURES,
        "calendar+weather": CALENDAR_FEATURES + WEATHER_FEATURES,
        "calendar+weather+demand": CALENDAR_FEATURES + WEATHER_FEATURES + DEMAND_FEATURES,
    }
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

    overall, by_zone, by_condition, pred_frames = [], [], [], []
    for scheme in args.splits:
        split = assign_split(df, scheme)
        tr, va, te = df[split == "train"], df[split == "val"], df[split == "test"]
        print(f"\n[{scheme}] train {len(tr)}  val {len(va)}  test {len(te)} rows; test price mean {te[TARGET].mean():.1f}, sd {te[TARGET].std():.1f} $/MWh")
        preds = te[["timestamp", "timestamp_local", "zone", TARGET, "demand_MW", "weather_T2M"]].copy()
        preds["split_scheme"] = scheme
        y = te[TARGET].to_numpy()

        for name, features in feature_sets.items():
            model = CatBoostRegressor(
                iterations=args.iterations, learning_rate=args.learning_rate, depth=args.depth, random_seed=args.seed,
                loss_function="RMSE", verbose=False, early_stopping_rounds=100, thread_count=-1,
                train_dir=str(args.out_dir / "catboost_info"),
            )
            model.fit(Pool(tr[features], tr[TARGET], cat_features=CAT_FEATURES),
                      eval_set=Pool(va[features], va[TARGET], cat_features=CAT_FEATURES), use_best_model=True)
            pred = model.predict(te[features])
            preds[f"pred_{name}"] = pred

            zone_rows = [{"split_scheme": scheme, "feature_set": name, "zone": z, **metrics(y[(te["zone"] == z).to_numpy()], pred[(te["zone"] == z).to_numpy()])}
                         for z in sorted(te["zone"].unique())]
            by_zone.extend(zone_rows)
            zdf = pd.DataFrame(zone_rows)
            row = {"split_scheme": scheme, "feature_set": name, "n_features": len(features), "trees_used": model.get_best_iteration() + 1,
                   "mean_zone_r2": zdf["r2"].mean(), "mean_zone_mae": zdf["mae"].mean(), "mean_zone_median_abs_err": zdf["median_abs_err"].mean()}
            overall.append(row)
            print(f"  {name:26s} mean zone R2 {row['mean_zone_r2']:.3f}   MAE {row['mean_zone_mae']:.2f}   median abs err {row['mean_zone_median_abs_err']:.2f} $/MWh   ({row['trees_used']} trees)")

            price_rank = te.groupby("zone")[TARGET].rank(pct=True).to_numpy()
            temp_rank = te.groupby("zone")["weather_T2M"].rank(pct=True).to_numpy()
            for label, m in [("high price (top 10% per zone)", price_rank >= 0.9), ("ordinary price (middle 80%)", (price_rank > 0.1) & (price_rank < 0.9)),
                             ("hot (top 10% T2M per zone)", temp_rank >= 0.9), ("cold (bottom 10% T2M per zone)", temp_rank <= 0.1)]:
                by_condition.append({"split_scheme": scheme, "feature_set": name, "condition": label, **metrics(y[m], pred[m])})

            pd.DataFrame({"feature": features, "importance": model.get_feature_importance()}).sort_values("importance", ascending=False).to_csv(
                args.out_dir / f"feature_importance_{scheme}_{name.replace('+', '_')}.csv", index=False)
        pred_frames.append(preds)

    pd.DataFrame(overall).to_csv(args.out_dir / "comparison_summary.csv", index=False)
    pd.DataFrame(by_zone).to_csv(args.out_dir / "comparison_by_zone.csv", index=False)
    pd.DataFrame(by_condition).to_csv(args.out_dir / "comparison_by_condition.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_parquet(args.out_dir / "test_predictions.parquet", index=False)
    print(f"\nWrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
