"""Phase 3: does the frozen Prithvi weather representation help predict
ISO-NE zonal electricity demand, beyond calendar and simple weather?

One pooled CatBoost model (zone as a categorical feature) per feature set:
    calendar                  hour, day_of_week, day_of_year, month,
                              is_weekend, is_holiday, zone
    calendar+weather          + each row's own-zone T2M, QV2M, wind speed,
                              U10M, V10M, SWGNT
    calendar+prithvi          + the global-pooled Prithvi embedding
    calendar+weather+prithvi  everything
Feature sets whose columns are missing are skipped.

The Prithvi embedding has 2,560 columns but only about 2,900 distinct
values per year (one per 3-hourly weather time, shared by all zones). By
default it is standardised and reduced to its leading principal components
(--prithvi-pca, default 50, about 95% of its variance), fitted on the
training rows of each split only. --prithvi-pca 0 uses all 2,560 columns.

Splits, by Eastern local date (never a random row split):
    year_ahead     train on every earlier year, test on the whole last
                   year. Needs two or more years. The headline split.
    chronological  last year only: train Jan-Aug, test Oct-Dec.
    blocked        last year only: train days 1-18 of each month, test day
                   22 onward.
September (chronological) and days 19-21 (blocked) are left out of both
training and testing, as in earlier runs where they were a validation block.

Every model is trained for a fixed number of trees (--iterations, default
3000) with no early stopping: on these data early stopping with a short
patience stopped too soon, and no overfitting was seen up to 10,000 trees.
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


def build_feature_sets(prithvi_cols: list[str]) -> dict[str, list[str]]:
    sets = {
        "calendar": CALENDAR_FEATURES,
        "calendar+weather": CALENDAR_FEATURES + WEATHER_FEATURES,
    }
    if prithvi_cols:
        sets["calendar+prithvi"] = CALENDAR_FEATURES + prithvi_cols
        sets["calendar+weather+prithvi"] = CALENDAR_FEATURES + WEATHER_FEATURES + prithvi_cols
    return sets


def prithvi_components(df: pd.DataFrame, emb_cols: list[str], train_mask: np.ndarray, k: int) -> tuple[pd.DataFrame, float]:
    """Standardise the embedding and project it on its top-k principal
    components, both fitted on the distinct weather times in the training
    rows only. Returns the component columns and the share of training
    variance they keep."""
    X = df[emb_cols].to_numpy(dtype=np.float32)
    train_unique = df.loc[train_mask].drop_duplicates("timestamp").index
    T = X[df.index.get_indexer(train_unique)].astype(np.float64)
    mu, sd = T.mean(0), T.std(0)
    sd[sd == 0] = 1.0
    _, s, vt = np.linalg.svd((T - mu) / sd, full_matrices=False)
    kept = float((s[:k] ** 2).sum() / (s ** 2).sum())
    comps = ((X - mu) / sd) @ vt[:k].T
    return pd.DataFrame(comps.astype(np.float32), index=df.index, columns=[f"prithvi_pc{i + 1}" for i in range(k)]), kept


def assign_split(df: pd.DataFrame, scheme: str) -> pd.Series:
    """Label each row train / val / test / unused. "val" rows are held out
    of both training and testing."""
    local = df["timestamp_local"]
    last_year = local.dt.year.max()
    in_last = (local.dt.year == last_year).to_numpy()
    if scheme == "year_ahead":
        if local.dt.year.nunique() < 2:
            raise SystemExit("year_ahead needs at least two years of data.")
        return pd.Series(np.where(in_last, "test", "train"), index=df.index)
    if scheme == "chronological":
        month = local.dt.month
        label = np.select([month <= 8, month == 9], ["train", "val"], default="test")
    elif scheme == "blocked":
        day = local.dt.day
        label = np.select([day <= 18, day <= 21], ["train", "val"], default="test")
    else:
        raise ValueError(scheme)
    return pd.Series(np.where(in_last, label, "unused"), index=df.index)


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
    parser.add_argument("--modeling-table", type=Path, nargs="+", required=True, help="One table per year from prepare_modeling_table.py")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=None, help="Default: year_ahead (if 2+ years), chronological, blocked")
    parser.add_argument("--feature-sets", nargs="+", default=None, help="Default: all available")
    parser.add_argument("--prithvi-pca", type=int, default=50, help="Principal components of the embedding to use; 0 = all raw columns")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    df = pd.concat([pd.read_parquet(p) for p in args.modeling_table], ignore_index=True)
    emb_cols = [c for c in df.columns if c.startswith("global_emb_")]
    df = df.dropna(subset=["demand_MW", "weather_T2M"] + emb_cols[:1]).reset_index(drop=True)
    years = sorted(df["timestamp_local"].dt.year.unique())
    splits = args.splits or ((["year_ahead"] if len(years) > 1 else []) + ["chronological", "blocked"])
    print(f"{len(df)} rows, years {years}, zones {sorted(df['zone'].unique())}, embedding columns: {len(emb_cols)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    overall, by_zone, by_condition, pred_frames = [], [], [], []

    for scheme in splits:
        split = assign_split(df, scheme)
        train_mask = (split == "train").to_numpy()
        work = df.drop(columns=emb_cols)
        prithvi_cols = []
        if emb_cols and args.prithvi_pca > 0:
            comps, kept = prithvi_components(df, emb_cols, train_mask, args.prithvi_pca)
            work = pd.concat([work, comps], axis=1)
            prithvi_cols = list(comps.columns)
            pca_note = f"; Prithvi: {args.prithvi_pca} components keep {kept:.1%} of training variance"
        elif emb_cols:
            work = df
            prithvi_cols = emb_cols
            pca_note = f"; Prithvi: all {len(emb_cols)} raw columns"
        else:
            pca_note = ""
        feature_sets = build_feature_sets(prithvi_cols)
        if args.feature_sets:
            feature_sets = {k: v for k, v in feature_sets.items() if k in args.feature_sets}

        tr, te = work[train_mask], work[(split == "test").to_numpy()]
        print(f"\n[{scheme}] train {len(tr)} rows ({tr['timestamp_local'].min():%Y-%m-%d} to {tr['timestamp_local'].max():%Y-%m-%d}), "
              f"test {len(te)} rows ({te['timestamp_local'].min():%Y-%m-%d} to {te['timestamp_local'].max():%Y-%m-%d}){pca_note}", flush=True)
        preds = te[["timestamp", "timestamp_local", "zone", "demand_MW", "weather_T2M"]].copy()
        preds["split_scheme"] = scheme

        for name, features in feature_sets.items():
            model = CatBoostRegressor(
                iterations=args.iterations, learning_rate=args.learning_rate, depth=args.depth,
                random_seed=args.seed, loss_function="RMSE", verbose=False, thread_count=-1,
                train_dir=str(args.out_dir / "catboost_info"),
            )
            model.fit(Pool(tr[features], tr["demand_MW"], cat_features=CAT_FEATURES))
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
            ape = np.abs(pred - te["demand_MW"].to_numpy()) / te["demand_MW"].to_numpy() * 100
            row = {
                "split_scheme": scheme, "feature_set": name, "n_features": len(features), "trees": args.iterations,
                "mean_zone_r2": zdf["r2"].mean(), "mean_zone_mape_pct": zdf["mape_pct"].mean(), "median_ape_pct": float(np.median(ape)),
                "pooled_mae_MW": mean_absolute_error(te["demand_MW"], pred), "pooled_r2": r2_score(te["demand_MW"], pred),
            }
            overall.append(row)
            print(f"  {name:26s} mean zone R2 {row['mean_zone_r2']:.3f}   MAPE {row['mean_zone_mape_pct']:.2f}%   median {row['median_ape_pct']:.2f}%", flush=True)

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
