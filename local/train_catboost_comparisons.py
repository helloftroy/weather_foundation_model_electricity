"""Phase 3: does the frozen Prithvi weather representation help predict
ISO-NE zonal electricity demand, beyond calendar and simple weather?

One pooled CatBoost model (zone as a categorical feature) per feature set:
    calendar                  hour, day_of_week, day_of_year, month,
                              is_weekend, is_holiday, zone
    calendar+weather          + each row's own-zone T2M, QV2M, wind speed,
                              U10M, V10M, SWGNT
    calendar+prithvi          + global_emb_* (global-pooled Prithvi embedding)
    calendar+weather+prithvi  everything
Feature sets whose columns are missing are skipped.

Two time-respecting splits, by Eastern local date (never a random row split):
    chronological  train Jan-Aug, validate Sep, test Oct-Dec. The headline
                   split. Caveat with one year of data: the test months are
                   later in the year than anything in training, so
                   day_of_year/month cannot help there.
    blocked        within every month: train days 1-18, validate 19-21,
                   test day 22 onward. All seasons appear in train and test.
The validation block is used only for early stopping; the test block is
never seen during fitting.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, r2_score

CALENDAR_FEATURES = ["hour", "day_of_week", "day_of_year", "month", "is_weekend", "is_holiday", "zone"]
WEATHER_FEATURES = ["weather_T2M", "weather_QV2M", "weather_wind_speed", "weather_U10M", "weather_V10M", "weather_SWGNT"]
CAT_FEATURES = ["zone"]


def build_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    prithvi_cols = [c for c in df.columns if c.startswith("global_emb_")]
    sets = {
        "calendar": CALENDAR_FEATURES,
        "calendar+weather": CALENDAR_FEATURES + WEATHER_FEATURES,
    }
    if prithvi_cols:
        sets["calendar+prithvi"] = CALENDAR_FEATURES + prithvi_cols
        sets["calendar+weather+prithvi"] = CALENDAR_FEATURES + WEATHER_FEATURES + prithvi_cols
    return sets


def assign_split(df: pd.DataFrame, scheme: str) -> pd.Series:
    local = df["timestamp_local"]
    if scheme == "chronological":
        month = local.dt.month
        return pd.Series(np.select([month <= 8, month == 9], ["train", "val"], default="test"), index=df.index)
    if scheme == "blocked":
        day = local.dt.day
        return pd.Series(np.select([day <= 18, day <= 21], ["train", "val"], default="test"), index=df.index)
    raise ValueError(scheme)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - y
    return {
        "n": len(y),
        "r2": r2_score(y, pred),
        "mae_MW": mean_absolute_error(y, pred),
        "rmse_MW": float(np.sqrt(np.mean(err**2))),
        "mape_pct": float(np.mean(np.abs(err) / y) * 100),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["chronological", "blocked"])
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    df = pd.read_parquet(args.modeling_table)
    df = df[(df["timestamp_local"].dt.year == 2024)].dropna(subset=["demand_MW", "weather_T2M"]).reset_index(drop=True)
    feature_sets = build_feature_sets(df)
    print(f"{len(df)} rows, zones {sorted(df['zone'].unique())}, feature sets: {list(feature_sets)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    overall, by_zone, by_condition, pred_frames = [], [], [], []

    for scheme in args.splits:
        split = assign_split(df, scheme)
        tr, va, te = df[split == "train"], df[split == "val"], df[split == "test"]
        print(f"\n[{scheme}] train {len(tr)}  val {len(va)}  test {len(te)} rows")
        preds = te[["timestamp", "timestamp_local", "zone", "demand_MW", "weather_T2M"]].copy()
        preds["split_scheme"] = scheme

        for name, features in feature_sets.items():
            model = CatBoostRegressor(
                iterations=args.iterations, learning_rate=args.learning_rate, depth=args.depth,
                random_seed=args.seed, loss_function="RMSE", verbose=False,
                early_stopping_rounds=100, thread_count=-1, train_dir=str(args.out_dir / "catboost_info"),
            )
            model.fit(
                Pool(tr[features], tr["demand_MW"], cat_features=CAT_FEATURES),
                eval_set=Pool(va[features], va["demand_MW"], cat_features=CAT_FEATURES),
                use_best_model=True,
            )
            pred = model.predict(te[features])
            preds[f"pred_{name}"] = pred

            zone_rows = []
            for zone, idx in te.groupby("zone").groups.items():
                pos = te.index.get_indexer(idx)
                zone_rows.append({"split_scheme": scheme, "feature_set": name, "zone": zone, **metrics(te.loc[idx, "demand_MW"].to_numpy(), pred[pos])})
            by_zone.extend(zone_rows)
            zdf = pd.DataFrame(zone_rows)
            # Pooled R2 across zones is inflated by the size difference between
            # zones (CT is ~5x VT), so the headline numbers are per-zone means.
            row = {
                "split_scheme": scheme, "feature_set": name, "n_features": len(features),
                "trees_used": model.get_best_iteration() + 1,
                "mean_zone_r2": zdf["r2"].mean(), "mean_zone_mape_pct": zdf["mape_pct"].mean(),
                "pooled_mae_MW": mean_absolute_error(te["demand_MW"], pred), "pooled_r2": r2_score(te["demand_MW"], pred),
            }
            overall.append(row)
            print(f"  {name:26s} mean zone R2 {row['mean_zone_r2']:.3f}   MAPE {row['mean_zone_mape_pct']:.2f}%   ({row['trees_used']} trees)")

            # Conditions are defined within each zone (zones differ in size and climate).
            zone_demand_rank = te.groupby("zone")["demand_MW"].rank(pct=True)
            zone_temp_rank = te.groupby("zone")["weather_T2M"].rank(pct=True)
            for label, mask in [
                ("high demand (top 10% per zone)", zone_demand_rank >= 0.9),
                ("hot (top 10% T2M per zone)", zone_temp_rank >= 0.9),
                ("cold (bottom 10% T2M per zone)", zone_temp_rank <= 0.1),
            ]:
                m = mask.to_numpy()
                y, p = te["demand_MW"].to_numpy()[m], pred[m]
                by_condition.append({
                    "split_scheme": scheme, "feature_set": name, "condition": label, "n": int(m.sum()),
                    "mape_pct": float(np.mean(np.abs(p - y) / y) * 100),
                    "mean_bias_pct": float(np.mean((p - y) / y) * 100),
                })

            imp = pd.DataFrame({"feature": features, "importance": model.get_feature_importance()}).sort_values("importance", ascending=False)
            imp.to_csv(args.out_dir / f"feature_importance_{scheme}_{name.replace('+', '_')}.csv", index=False)

        pred_frames.append(preds)

    pd.DataFrame(overall).to_csv(args.out_dir / "comparison_summary.csv", index=False)
    pd.DataFrame(by_zone).to_csv(args.out_dir / "comparison_by_zone.csv", index=False)
    pd.DataFrame(by_condition).to_csv(args.out_dir / "comparison_by_condition.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_parquet(args.out_dir / "test_predictions.parquet", index=False)
    print(f"\nWrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
