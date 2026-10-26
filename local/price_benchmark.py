"""Day-ahead hub price, 24 h ahead, year-ahead: what do weather forecasts
(GFS, the Prithvi decoder, Prithvi tokens) add on top of past prices?

Two targets:
  heat_rate   LMP / that day's Algonquin Citygate gas price ($/MWh per
              $/MMBtu, i.e. an implied heat rate in MMBtu/MWh). The model
              predicts the heat rate; the price is the prediction times that
              day's gas price. Gas is used only to normalise, never as a
              feature, so what is left to predict is mostly demand, weather
              and grid conditions. A ratio, not a log, because LMP can be
              near zero or negative.
              Caveat: "that day's" Algonquin price is interpolated between
              roughly weekly EIA quotes (build_gas_prices.py), so it partly
              uses the following quote; it is a normaliser, not a forecast.
  lmp         LMP directly, with no gas information at all: past prices and
              weather only. The scientific test: can the weather features
              anticipate weather-driven price spikes before past prices
              show them?

Errors are always on the price in $/MWh (for heat_rate, after multiplying
back), all feature sets on the same hours.

Every feature set has calendar features and past values of its target, all
known when the day-ahead market for the target day is bid: 24 h, 48 h and
7 days before the target hour, and the mean of the 24 hours ending 24 h
before it. Weather blocks are system-level: each zone's value weighted by
its share of the year's demand.
    weather (issue)   MERRA-2 at the zone cells 24 h before the target hour
    GFS area          GFS 24-27 h forecast over the zone areas
    Prithvi area      Prithvi decoder 24 h forecast over the zone areas
    Prithvi tokens    final-layer tokens averaged over the New England box,
                      50 principal components fitted on training times
    true area         actual MERRA-2 over the zone areas at the target hour:
                      perfect knowledge, the ceiling
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

import train_catboost_comparisons as dm
from analyze_global_year import ZONES, box_tokens, pca
from gfs_demand_benchmark import NAMES, gfs_for_targets, merra_features, prithvi_features, zone_features

CALENDAR = [f for f in dm.CALENDAR_FEATURES if f != "zone"]
LAGS = {"lag24h": pd.Timedelta(hours=24), "lag48h": pd.Timedelta(hours=48), "lag7d": pd.Timedelta(days=7)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=[2021, 2022, 2023, 2024])
    parser.add_argument("--test-years", type=int, nargs="+", default=[2023, 2024])
    parser.add_argument("--targets", nargs="+", default=["heat_rate", "lmp"], choices=["heat_rate", "lmp"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--prithvi-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    c = args.data_dir / "compact"
    zn = list(ZONES)

    # ---- system-level hourly table: calendar, hub price, gas ----
    rows, sys_blocks, token_paths, targets = [], {}, [], pd.DatetimeIndex([])
    for y in args.years:
        table = pd.read_parquet(args.data_dir / f"modeling_table_{y}.parquet", columns=["timestamp", "timestamp_local", "zone", "demand_MW"] + CALENDAR)
        share = table.groupby("zone")["demand_MW"].mean()
        share = (share / share.sum()).reindex(zn).to_numpy()
        hourly = table.groupby("timestamp")[["timestamp_local"] + CALENDAR].first()
        lmp = pd.read_parquet(c / f"isone_dayahead_hourly_lmp_{y}.parquet")
        lmp["timestamp"] = pd.to_datetime(lmp["timestamp_parsed"]).astype("datetime64[ns]")
        hourly = hourly.join(lmp[lmp["location"] == "HUB"].set_index("timestamp")["lmp_total"], how="inner")
        gas = pd.read_csv(args.data_dir / "gas" / f"gas_prices_daily_{y}.csv", parse_dates=["date"]).set_index("date")["algonquin_weekly"]
        hourly["gas"] = gas.reindex(hourly["timestamp_local"].dt.normalize()).to_numpy()
        rows.append(hourly.reset_index())

        # weather blocks for this year's 3-hourly targets, zone-weighted to system level
        t = pd.date_range(f"{y}-01-01", f"{y}-12-31 21:00", freq="3h")
        weather = pd.read_parquet(c / f"conventional_weather_{y}.parquet").set_index("timestamp")
        prev = c / f"conventional_weather_{y - 1}.parquet"
        if prev.exists():
            weather = pd.concat([pd.read_parquet(prev).set_index("timestamp"), weather])
            weather = weather[~weather.index.duplicated(keep="last")].sort_index()
        ds, pick = gfs_for_targets(args.data_dir / "gfs" / f"gfs_ne_{y}.nc", t)
        _, gfs_area = zone_features(ds, pick, t, zn)
        pv = prithvi_features(args.prithvi_dir, y, t, weather, zn)
        w = lambda a: (a * share[None, :, None]).sum(axis=1)  # noqa: E731  [T, zone, 6] -> [T, 6]
        year_blocks = {"weather (issue)": w(merra_features(weather, t - pd.Timedelta(hours=24), zn)), "GFS area": w(gfs_area),
                       "Prithvi area": w(pv["Prithvi area"]), "true area": w(pv["true area"])}
        # tokens for each target time: the sample issued 24 h before it
        meta = json.loads((args.prithvi_dir / f"prithvi_tokens_ne_{y}_lead24.json").read_text())
        issue = pd.DatetimeIndex(meta["timestamps"])
        tok = box_tokens(args.prithvi_dir / f"prithvi_tokens_ne_{y}_lead24.npy")
        pos = issue.get_indexer(t - pd.Timedelta(hours=24))
        tok_t = np.full((len(t), tok.shape[1]), np.nan, np.float32)
        tok_t[pos >= 0] = tok[pos[pos >= 0]]
        year_blocks["Prithvi tokens (raw)"] = tok_t
        for k, v in year_blocks.items():
            sys_blocks.setdefault(k, []).append(v)
        targets = targets.append(t)
        print(f"{y}: {len(hourly)} hours with a hub price; gas missing for {int(hourly['gas'].isna().sum())}", flush=True)
    sys_blocks = {k: np.concatenate(v) for k, v in sys_blocks.items()}

    df = pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    df["heat_rate"] = df["lmp_total"] / df["gas"]
    year = df["timestamp_local"].dt.year.to_numpy()
    ti = targets.get_indexer(df["timestamp"].dt.round("3h"))
    for target in args.targets:
        s = df.set_index("timestamp")["heat_rate" if target == "heat_rate" else "lmp_total"]
        day_mean = s.rolling("24h").mean()
        for k, d in LAGS.items():
            df[f"{target}_{k}"] = s.reindex(df["timestamp"] - d).to_numpy()
        df[f"{target}_mean24h_to_lag24h"] = day_mean.reindex(df["timestamp"] - pd.Timedelta(hours=24)).to_numpy()
    lag_cols = {t: [c for c in df.columns if c.startswith(f"{t}_")] for t in args.targets}

    # Same hours for everything: every block, lag and the gas normaliser present.
    ok = ti >= 0
    for v in sys_blocks.values():
        ok &= ~np.isnan(v[np.clip(ti, 0, None)]).any(axis=1)
    for t in args.targets:
        ok &= df[lag_cols[t]].notna().all(axis=1).to_numpy()
    ok &= df["gas"].notna().to_numpy() & (df["gas"] > 0).to_numpy()
    print(f"{ok.sum()} of {len(df)} hours usable; heat rate median {df.loc[ok, 'heat_rate'].median():.1f} MMBtu/MWh "
          f"(1-99%: {df.loc[ok, 'heat_rate'].quantile(0.01):.1f} to {df.loc[ok, 'heat_rate'].quantile(0.99):.1f})", flush=True)

    combos = [[], ["weather (issue)"], ["weather (issue)", "GFS area"], ["weather (issue)", "Prithvi area"],
              ["weather (issue)", "GFS area", "Prithvi area"], ["weather (issue)", "Prithvi tokens"],
              ["weather (issue)", "GFS area", "Prithvi tokens"], ["weather (issue)", "true area"]]
    results = []
    lmp_all, gas_all = df["lmp_total"].to_numpy(), df["gas"].to_numpy()
    for test_year in args.test_years:
        tr, te = ok & (year < test_year) & (year >= min(args.years)), ok & (year == test_year)
        train_times = np.zeros(len(targets), bool)
        train_times[np.unique(ti[tr])] = True
        blocks = {k: v for k, v in sys_blocks.items() if k != "Prithvi tokens (raw)"}
        blocks["Prithvi tokens"] = pca(sys_blocks["Prithvi tokens (raw)"], train_times, 50)[0]
        for target in args.targets:
            y_fit = df["heat_rate" if target == "heat_rate" else "lmp_total"].to_numpy()
            print(f"\n[{target}, test {test_year}, train {min(args.years)}-{test_year - 1}] train {tr.sum()} h, test {te.sum()} h; "
                  f"test price mean ${lmp_all[te].mean():.2f}, sd ${lmp_all[te].std():.2f}", flush=True)
            for combo in combos:
                X = df[CALENDAR + lag_cols[target]].copy()
                for b in combo:
                    names = [f"{b}_{n}" for n in (NAMES if b != "Prithvi tokens" else range(50))]
                    X = pd.concat([X, pd.DataFrame(blocks[b][np.clip(ti, 0, None)], columns=names, index=df.index)], axis=1)
                model = CatBoostRegressor(iterations=args.iterations, learning_rate=0.05, depth=6, random_seed=7, verbose=False,
                                          thread_count=-1, train_dir=str(args.out_dir / "catboost_info"))
                model.fit(Pool(X[tr], y_fit[tr]))
                pred = model.predict(X[te])
                price = pred * gas_all[te] if target == "heat_rate" else pred
                err = price - lmp_all[te]
                mae = float(np.abs(err).mean())
                r2 = float(1 - (err ** 2).sum() / ((lmp_all[te] - lmp_all[te].mean()) ** 2).sum())
                name = "calendar + past " + ("heat rate" if target == "heat_rate" else "price") + "".join(f" + {b}" for b in combo)
                results.append({"target": target, "test_year": test_year, "feature_set": name, "mae": mae, "r2": r2,
                                "rmse": float(np.sqrt((err ** 2).mean())), "n_test_hours": int(te.sum())})
                print(f"  {name:75s} MAE ${mae:6.2f}   R2 {r2:5.2f}", flush=True)
                pd.DataFrame(results).to_csv(args.out_dir / "price_comparison.csv", index=False)
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
