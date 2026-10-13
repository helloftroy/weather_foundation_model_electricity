"""Nowcast vs. day-old information: does the ranking of feature sets change
when no model is told the weather at the hour being predicted?

For each lag L, every weather-derived input describes the atmosphere L hours
BEFORE the demand hour (L = 0 is the nowcast used everywhere else). Calendar
features always describe the demand hour itself. This is not a forecast
experiment: nothing here uses a weather forecast. It asks which description
of the atmosphere at t - L says more about demand at t.

Inputs per zone-hour:
    weather        the zone's six MERRA-2 numbers (T2M, QV2M, wind, SWGNT)
    prithvi_local  encoder tokens, mean of the 3 x 3 patches around the
                   zone, 50 principal components
    prithvi_pooled encoder tokens averaged over the whole crop, 50 components
PCA is fitted on training years only. One pooled CatBoost model per feature
set, 3,000 trees. Train on all years but the last, test on the last.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

import train_catboost_comparisons as dm

ZONES = {"ME": (45.0, -69.0), "NH": (43.7, -71.5), "VT": (44.3, -72.6), "CT": (41.6, -72.7), "RI": (41.7, -71.5),
         "NEMA": (42.36, -71.06), "SEMA": (41.9, -70.9), "WCMA": (42.3, -72.6)}
WEATHER_VARS = ["T2M", "QV2M", "U10M", "V10M", "SWGNT"]


def pca(V: np.ndarray, train_rows: np.ndarray, k: int) -> np.ndarray:
    """V: [time, ..., C]. Standardise and project on the top-k components of the training rows."""
    T = V[train_rows].reshape(-1, V.shape[-1]).astype(np.float64)
    mu, sd = T.mean(0), T.std(0)
    sd[sd == 0] = 1
    _, _, vt = np.linalg.svd((T - mu) / sd, full_matrices=False)
    return (((V - mu) / sd) @ vt[:k].T).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument("--lags", type=int, nargs="+", default=[0, 24, 48])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--blocks", nargs="+", default=["weather", "prithvi_local", "prithvi_pooled"], choices=["weather", "prithvi_local", "prithvi_pooled"])
    args = parser.parse_args()
    c = args.data_dir / "compact"
    test_year = max(args.years)

    # Weather and embeddings on one common 3-hourly time axis.
    weather = pd.concat([pd.read_parquet(c / f"conventional_weather_{y}.parquet") for y in args.years]).set_index("timestamp").sort_index()
    times = weather.index
    W = np.stack([np.stack([weather[f"{z}_{v}"].to_numpy() for v in WEATHER_VARS] + [np.hypot(weather[f"{z}_U10M"], weather[f"{z}_V10M"]).to_numpy()], axis=1)
                  for z in ZONES], axis=1).astype(np.float32)                                  # [time, zone, 6]
    weather_names = [f"weather_{v}" for v in WEATHER_VARS] + ["weather_wind_speed"]

    pooled = pd.concat([pd.read_parquet(c / f"prithvi_embeddings_{y}.parquet").filter(regex="^(timestamp|global_emb_)") for y in args.years]).set_index("timestamp")
    G = pooled.reindex(times).to_numpy(np.float32)                                              # [time, C], NaN where missing
    local = np.full((len(times), len(ZONES), G.shape[1]), np.nan, dtype=np.float32)
    for y in args.years:
        tok = np.asarray(np.load(c / f"prithvi_tokens_ne_{y}.npy", mmap_mode="r"), dtype=np.float32)
        meta = json.loads((c / f"prithvi_tokens_ne_{y}.json").read_text())
        lat_c, lon_c = np.array(meta["patch_lat_centers"]), np.array(meta["patch_lon_centers"])
        rows = times.get_indexer(pd.DatetimeIndex(meta["timestamps"]))
        for k, (la, lo) in enumerate(ZONES.values()):
            i, j = np.abs(lat_c - la).argmin(), np.abs(lon_c - lo).argmin()
            local[rows, k] = tok[:, max(i - 1, 0):i + 2, max(j - 1, 0):j + 2].mean(axis=(1, 2))
        del tok
    have = ~np.isnan(G[:, 0]) & ~np.isnan(local[:, 0, 0])
    train_times = have & (times.year < test_year)
    P_local = pca(np.where(have[:, None, None], local, 0), train_times, 50)
    P_pooled = pca(np.where(have[:, None], G, 0), train_times, 50)
    del local, G

    df = pd.concat([pd.read_parquet(args.data_dir / f"modeling_table_{y}.parquet", columns=["timestamp", "timestamp_local", "zone", "demand_MW"] + [f for f in dm.CALENDAR_FEATURES if f != "zone"])
                    for y in args.years], ignore_index=True)
    zone_idx = df["zone"].map({z: k for k, z in enumerate(ZONES)}).to_numpy()
    year = df["timestamp_local"].dt.year.to_numpy()
    y_all = df["demand_MW"].to_numpy()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for lag in args.lags:
        ti = times.get_indexer((df["timestamp"] - pd.Timedelta(hours=lag)).dt.round("3h"))
        ok = (ti >= 0) & have[np.clip(ti, 0, None)] & np.isin(year, args.years)
        tr, te = ok & (year < test_year), ok & (year == test_year)
        blocks = {
            "weather": pd.DataFrame(W[ti, zone_idx], columns=weather_names, index=df.index),
            "prithvi_local": pd.DataFrame(P_local[ti, zone_idx], columns=[f"pl{i}" for i in range(50)], index=df.index),
            "prithvi_pooled": pd.DataFrame(P_pooled[ti], columns=[f"pp{i}" for i in range(50)], index=df.index),
        }
        print(f"\nInformation lag {lag} h: train {tr.sum()} rows ({sorted(set(year[tr]))}), test {te.sum()} rows ({test_year})", flush=True)
        combos = [[], ["weather"], ["prithvi_local"], ["prithvi_pooled"], ["weather", "prithvi_local"], ["weather", "prithvi_pooled"], ["weather", "prithvi_local", "prithvi_pooled"]]
        for combo in [c for c in combos if set(c) <= set(args.blocks)]:
            if lag > 0 and not combo:
                continue  # calendar-only does not depend on the lag
            X = pd.concat([df[dm.CALENDAR_FEATURES]] + [blocks[b] for b in combo], axis=1)
            model = CatBoostRegressor(iterations=args.iterations, learning_rate=0.05, depth=6, random_seed=7, verbose=False, thread_count=-1,
                                      train_dir=str(args.out_csv.parent / "catboost_info"))
            model.fit(Pool(X[tr], y_all[tr], cat_features=dm.CAT_FEATURES))
            pred = model.predict(X[te])
            ape = np.abs(pred - y_all[te]) / y_all[te] * 100
            by_zone = pd.Series(ape).groupby(df["zone"].to_numpy()[te]).mean()
            name = "calendar" + "".join(f" + {b}" for b in combo)
            rows_out.append({"lag_hours": lag, "feature_set": name, "mean_zone_mape_pct": by_zone.mean(), "median_ape_pct": float(np.median(ape)), **{f"mape_{z}": by_zone[z] for z in ZONES}})
            print(f"  {name:52s} MAPE {by_zone.mean():5.2f}%   median {np.median(ape):4.2f}%", flush=True)
            args.out_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows_out).to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
