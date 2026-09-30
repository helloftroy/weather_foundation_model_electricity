"""Phase 3: does the frozen Prithvi weather representation help predict
ISO-NE zonal electricity demand, beyond calendar and simple weather?

Trains one pooled CatBoost model (zone as a categorical feature) per feature
set, using a time-respecting train/test split (a chronological cutoff, not a
random row split -- demand is strongly autocorrelated). Runs locally.

Feature sets compared:
    1. calendar          -- hour, day_of_week, day_of_year, is_weekend,
                             is_holiday, month, year, zone
    2. calendar+weather   -- (1) + weather_T2M/QV2M/U10M/V10M/SWGNT
                             (each row's own zone's nearest-gridcell value)
    3. calendar+prithvi   -- (1) + global_emb_* (the compact global-pooled
                             Prithvi embedding)
    4. calendar+weather+prithvi -- (2) + (3)

Start simple, per the source instructions: one pooled model per feature set,
not per-zone models (easy to add later if useful).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CALENDAR_FEATURES = ["hour", "day_of_week", "day_of_year", "is_weekend", "is_holiday", "month", "year", "zone"]
WEATHER_FEATURES = ["weather_T2M", "weather_QV2M", "weather_U10M", "weather_V10M", "weather_SWGNT"]
CAT_FEATURES = ["zone"]


def prithvi_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("global_emb_")]


def build_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    prithvi_cols = prithvi_feature_columns(df)
    return {
        "calendar": CALENDAR_FEATURES,
        "calendar+weather": CALENDAR_FEATURES + WEATHER_FEATURES,
        "calendar+prithvi": CALENDAR_FEATURES + prithvi_cols,
        "calendar+weather+prithvi": CALENDAR_FEATURES + WEATHER_FEATURES + prithvi_cols,
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred),
        "n": len(y_true),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-table", type=Path, required=True, help="Output of prepare_modeling_table.py")
    parser.add_argument("--train-end", required=True, help="YYYY-MM-DD: last day INCLUDED in training; everything after is test")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    df = pd.read_parquet(args.modeling_table)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["demand_MW"])

    train_end = pd.Timestamp(args.train_end)
    train_df = df[df["timestamp"] <= train_end].copy()
    test_df = df[df["timestamp"] > train_end].copy()
    print(f"Train: {len(train_df)} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"Test:  {len(test_df)} rows ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
    if len(test_df) == 0:
        raise SystemExit("Empty test set -- pick an earlier --train-end.")

    feature_sets = build_feature_sets(df)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    by_zone_rows = []
    predictions = {}

    for name, features in feature_sets.items():
        missing = [f for f in features if f not in df.columns]
        if missing:
            print(f"Skipping '{name}': missing columns {missing}")
            continue

        train_pool = Pool(train_df[features], train_df["demand_MW"], cat_features=CAT_FEATURES)
        test_pool = Pool(test_df[features], test_df["demand_MW"], cat_features=CAT_FEATURES)

        model = CatBoostRegressor(
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            random_seed=args.seed,
            loss_function="RMSE",
            verbose=False,
        )
        model.fit(train_pool, eval_set=test_pool, use_best_model=True)

        y_pred = model.predict(test_pool)
        predictions[name] = y_pred
        metrics = evaluate(test_df["demand_MW"].values, y_pred)
        metrics["feature_set"] = name
        metrics["n_features"] = len(features)
        summary_rows.append(metrics)
        print(f"{name}: R2={metrics['r2']:.4f} RMSE={metrics['rmse']:.1f} MAE={metrics['mae']:.1f} ({len(features)} features)")

        test_df_copy = test_df.copy()
        test_df_copy["y_pred"] = y_pred
        for zone, zdf in test_df_copy.groupby("zone"):
            zmetrics = evaluate(zdf["demand_MW"].values, zdf["y_pred"].values)
            zmetrics["feature_set"] = name
            zmetrics["zone"] = zone
            by_zone_rows.append(zmetrics)

        if "prithvi" in name:
            importances = model.get_feature_importance(train_pool)
            imp_df = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False)
            imp_df.to_csv(args.out_dir / f"feature_importance_{name.replace('+', '_')}.csv", index=False)

        model.save_model(str(args.out_dir / f"model_{name.replace('+', '_')}.cbm"))

    summary_df = pd.DataFrame(summary_rows).sort_values("r2", ascending=False)
    summary_df.to_csv(args.out_dir / "comparison_summary.csv", index=False)
    print("\n=== Summary ===")
    print(summary_df[["feature_set", "r2", "rmse", "mae", "n_features"]].to_string(index=False))

    by_zone_df = pd.DataFrame(by_zone_rows)
    by_zone_df.to_csv(args.out_dir / "comparison_by_zone.csv", index=False)

    # Extreme-weather-period breakdown, using the own-zone T2M column as a
    # simple hot/cold proxy (top/bottom decile of test-period temperature).
    if "weather_T2M" in test_df.columns and "calendar+weather+prithvi" in predictions:
        temps = test_df["weather_T2M"]
        hot_mask = temps >= temps.quantile(0.9)
        cold_mask = temps <= temps.quantile(0.1)
        extreme_rows = []
        for label, mask in [("hot (top decile T2M)", hot_mask), ("cold (bottom decile T2M)", cold_mask)]:
            for name, y_pred in predictions.items():
                m = evaluate(test_df.loc[mask, "demand_MW"].values, y_pred[mask.values])
                m["condition"] = label
                m["feature_set"] = name
                extreme_rows.append(m)
        extreme_df = pd.DataFrame(extreme_rows)
        extreme_df.to_csv(args.out_dir / "comparison_extreme_weather.csv", index=False)
        print(f"\nWrote extreme-weather breakdown: {args.out_dir / 'comparison_extreme_weather.csv'}")

    # Observed vs predicted plot for the richest model, if matplotlib is available.
    try:
        import matplotlib.pyplot as plt

        best_name = "calendar+weather+prithvi" if "calendar+weather+prithvi" in predictions else summary_df.iloc[0]["feature_set"]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(test_df["demand_MW"], predictions[best_name], s=4, alpha=0.3)
        lims = [test_df["demand_MW"].min(), test_df["demand_MW"].max()]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("Observed demand (MW)")
        ax.set_ylabel("Predicted demand (MW)")
        ax.set_title(f"Observed vs predicted -- {best_name}")
        fig.tight_layout()
        fig.savefig(args.out_dir / "observed_vs_predicted.png", dpi=150)
        print(f"Wrote plot: {args.out_dir / 'observed_vs_predicted.png'}")
    except ImportError:
        print("matplotlib not installed -- skipping observed-vs-predicted plot.")


if __name__ == "__main__":
    main()
