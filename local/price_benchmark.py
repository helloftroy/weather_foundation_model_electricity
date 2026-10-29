"""Day-ahead hub price, 24 h ahead, year-ahead: what do weather forecasts
(GFS, the Prithvi decoder, Prithvi tokens) add on top of past prices?

Targets. Each model predicts LMP divided by a normaliser N known when the
day-ahead market is bid (trade date D-1 for delivery day D), and the price
is the prediction times N. Errors are always on the price.
  hr_henryhub   N = Henry Hub spot for the last trading day on or before
                D-1 (main heat-rate version, A). A national gas price, so
                the New England cold-snap premium is left for the weather
                features to explain.
  hr_algonquin  N = the last published Algonquin Citygate quote (EIA weekly
                update) dated on or before D-2, carried forward, never
                interpolated (A', secondary check). Weekly, so it misses
                most spikes.
  lmp_rel7d     N = mean day-ahead hub LMP over the 7 days ending 24 h before
                the target hour (B normalised). Gas-free: only past prices.
                Takes out regime shifts such as the 2022 gas spike.
  lmp           N = 1: raw LMP, no gas information (B).
  heat_rate     (older version) N = that day's Algonquin, interpolated
                between weekly quotes; kept for comparison.

Former two-target description, for reference:
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

--gas-features adds the gas prices known at bid time as plain FEATURES in
every feature set (used with the raw lmp target): Henry Hub for the last
trading day on or before D-1, and/or the last Algonquin quote dated on or
before D-2. The test: once the model is told the fuel regime directly, do
the Prithvi tokens still help?

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
from eval_tools import hard_day_labels, summarize
from analyze_global_year import ZONES, box_pixels, box_tokens, pca
from regional_weather import gfs_regional, prithvi_regional
from gfs_demand_benchmark import NAMES, gfs_for_targets, merra_features, prithvi_features, zone_features

CALENDAR = [f for f in dm.CALENDAR_FEATURES if f != "zone"]
LAGS = {"lag24h": pd.Timedelta(hours=24), "lag48h": pd.Timedelta(hours=48), "lag7d": pd.Timedelta(days=7)}


def placebo_index(targets, valid, seeds=(0, 1, 2)):
    """For each target time, the index of the target time whose tokens it gets under each placebo.
    Shift +/-7 days (falling back to the opposite shift at the ends of the record); shuffle = a random
    permutation within each (year, month, 3-hour UTC slot) group of times that have tokens."""
    out = {}
    plus = targets.get_indexer(targets + pd.Timedelta(days=7))
    minus = targets.get_indexer(targets - pd.Timedelta(days=7))
    for name, a, b in [("tokens placebo shift +7d", plus, minus), ("tokens placebo shift -7d", minus, plus)]:
        src = np.where(a >= 0, a, b)
        src = np.where((src >= 0) & valid[np.clip(src, 0, None)], src, np.arange(len(targets)))   # rare gaps: keep own (NaN-free) row
        out[name] = src
    key = pd.DataFrame({"y": targets.year, "m": targets.month, "h": targets.hour})
    for seed in seeds:
        rng = np.random.default_rng(seed)
        src = np.arange(len(targets))
        for _, idx in key[valid].groupby(["y", "m", "h"]).indices.items():
            rows = np.where(valid)[0][idx]
            src[rows] = rng.permutation(rows)
        out[f"tokens placebo shuffle s{seed}"] = src
    return out


def spatial_tokens(token_paths, train_times, ks):
    """Keep every patch of the 8 x 7 box; reduce each patch's 2,560-d token to k components with ONE basis
    (standardised PCA over all training (time, patch) vectors), so a component means the same thing everywhere.
    token_paths: per year in order, (file, pos) with pos[i] = sample index for that year's i-th target time
    (-1 if none). Returns {k: [all target times, patches * k]}."""
    n, s, ss, xtx, start = 0, 0.0, 0.0, 0.0, 0
    for path, pos in token_paths:   # pass 1: training moments, streamed
        a = np.load(path, mmap_mode="r")
        use = pos[(pos >= 0) & train_times[start:start + len(pos)]]
        for i in range(0, len(use), 128):
            x = np.asarray(a[np.sort(use[i:i + 128])], np.float64).reshape(-1, a.shape[-1])
            n += len(x); s = s + x.sum(0); ss = ss + (x ** 2).sum(0); xtx = xtx + x.T @ x
        start += len(pos)
    mu = s / n
    sd = np.sqrt(np.maximum(ss / n - mu ** 2, 1e-12))
    corr = (xtx / n - np.outer(mu, mu)) / np.outer(sd, sd)
    w, v = np.linalg.eigh(corr)
    v = v[:, ::-1]; w = w[::-1]
    out = {}
    for k in ks:
        print(f"  spatial tokens: {k} shared components keep {w[:k].sum() / w.sum():.1%} of the (standardised) variance per patch", flush=True)
        proj = (v[:, :k] / sd[:, None]).astype(np.float32)
        parts = []
        for path, pos in token_paths:   # pass 2: project every target time
            a = np.load(path, mmap_mode="r")
            o = np.full((len(pos), a.shape[1] * a.shape[2] * k), np.nan, np.float32)
            idx = np.where(pos >= 0)[0]
            for i in range(0, len(idx), 128):
                x = np.asarray(a[pos[idx[i:i + 128]]], np.float32)
                o[idx[i:i + 128]] = ((x - mu.astype(np.float32)) @ proj).reshape(len(x), -1)
            parts.append(o)
        out[k] = np.concatenate(parts)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=[2021, 2022, 2023, 2024])
    parser.add_argument("--test-years", type=int, nargs="+", default=[2023, 2024])
    parser.add_argument("--targets", nargs="+", default=["hr_henryhub", "hr_algonquin", "lmp_rel7d", "lmp"],
                        choices=["hr_henryhub", "hr_algonquin", "lmp_rel7d", "lmp", "heat_rate"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--prithvi-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--gas-features", nargs="*", default=[], choices=["henryhub", "algonquin"],
                        help="Add gas prices known at bid time as FEATURES in every feature set: Henry Hub for the last "
                             "trading day <= D-1, the last Algonquin quote dated <= D-2 (carried forward)")
    parser.add_argument("--input-control", action="store_true",
                        help="Add a control block: Prithvi's own MERRA-2 input fields at issue time (all 160 channels, "
                             "averaged over the box, 50 PCs fitted on training times), built exactly like the tokens")
    parser.add_argument("--token-variants", action="store_true",
                        help="Compare token readouts: pooled box mean with all 2,560 dimensions (no PCA), pooled PCA-50, "
                             "and spatial tokens (all 8 x 7 patches kept, the 2,560 axis reduced to k = 10 or 25 components "
                             "with one basis shared by all patches, fitted on training times)")
    parser.add_argument("--regional", action="store_true",
                        help="Add regional weather outside New England (regional_weather.py: five areas x 6 variables, "
                             "from GFS, the Prithvi decoder and actual MERRA-2) and compare with and without it")
    parser.add_argument("--pc-sweep", type=int, nargs="*", default=None,
                        help="Token PCA sweep: pooled tokens with each of these numbers of principal components, plus the full "
                             "2,560 dimensions, each added to GFS and to GFS + GFS regional (e.g. --pc-sweep 10 25 50 100 250 500)")
    parser.add_argument("--placebo", action="store_true",
                        help="Wrong-time placebo for the tokens: the same 50-PC token representation (fitted on the real "
                             "training tokens) re-assigned to the wrong target times, in training and test alike, then the "
                             "model retrained: shifted +7 / -7 days, and shuffled within the same calendar month and the "
                             "same 3-hour time of day (3 seeds), which keeps season and time of day but not the day's "
                             "weather. Compared with the real tokens, on top of GFS and of GFS + GFS regional")
    args = parser.parse_args()
    if args.placebo:
        args.regional = True
    if args.pc_sweep is not None:
        args.regional = True
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
        if args.regional:
            year_blocks["GFS regional"] = gfs_regional(args.data_dir / "gfs" / f"gfs_wide_{y}.nc", t)
            year_blocks.update(prithvi_regional(args.prithvi_dir, y, t, weather))
        token_paths.append((args.prithvi_dir / f"prithvi_tokens_ne_{y}_lead24.npy", pos))
        if args.input_control:
            inp = box_pixels(args.prithvi_dir / f"prithvi_tokens_ne_{y}_lead24_inputs.npy")
            inp_t = np.full((len(t), inp.shape[1]), np.nan, np.float32)
            inp_t[pos >= 0] = inp[pos[pos >= 0]]
            year_blocks["MERRA-2 fields (raw)"] = inp_t
        for k, v in year_blocks.items():
            sys_blocks.setdefault(k, []).append(v)
        targets = targets.append(t)
        print(f"{y}: {len(hourly)} hours with a hub price; gas missing for {int(hourly['gas'].isna().sum())}", flush=True)
    sys_blocks = {k: np.concatenate(v) for k, v in sys_blocks.items()}

    df = pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    day = df["timestamp_local"].dt.normalize()
    hh = pd.read_excel(args.data_dir / "gas" / "RNGWHHDd.xls", sheet_name="Data 1", skiprows=2)
    hh.columns = ["date", "henry_hub"]
    hh = hh.dropna().assign(date=lambda d: pd.to_datetime(d["date"])).set_index("date")["henry_hub"].sort_index()
    alg = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in sorted((args.data_dir / "gas").glob("algonquin_weekly_points_*.csv"))])
    alg = alg.drop_duplicates("date").set_index("date")["price"].sort_index()
    lmp_s = df.set_index("timestamp")["lmp_total"]
    trailing = lmp_s.rolling("168h").mean()
    norm = {"hr_henryhub": hh.reindex((day - pd.Timedelta(days=1)).to_numpy(), method="ffill").to_numpy(),
            "hr_algonquin": alg.reindex((day - pd.Timedelta(days=2)).to_numpy(), method="ffill").to_numpy(),
            "lmp_rel7d": trailing.reindex(df["timestamp"] - pd.Timedelta(hours=24)).to_numpy(),
            "lmp": np.ones(len(df)), "heat_rate": df["gas"].to_numpy()}
    gas_cols = [f"gas_known_{g}" for g in args.gas_features]
    for g in args.gas_features:
        df[f"gas_known_{g}"] = norm["hr_" + g]
    for t in args.targets:
        df[f"N_{t}"] = norm[t]
        df[f"y_{t}"] = df["lmp_total"] / df[f"N_{t}"]
    df["heat_rate"] = df["lmp_total"] / df["gas"]
    year = df["timestamp_local"].dt.year.to_numpy()
    ti = targets.get_indexer(df["timestamp"].dt.round("3h"))
    lag_cols = {}
    for target in args.targets:
        s = df.set_index("timestamp")[f"y_{target}"]
        day_mean = s.rolling("24h").mean()
        cols = []
        for k, d in LAGS.items():
            df[f"past_{target}__{k}"] = s.reindex(df["timestamp"] - d).to_numpy()
            cols.append(f"past_{target}__{k}")
        df[f"past_{target}__mean24h_to_lag24h"] = day_mean.reindex(df["timestamp"] - pd.Timedelta(hours=24)).to_numpy()
        cols.append(f"past_{target}__mean24h_to_lag24h")
        lag_cols[target] = cols   # built explicitly: selecting columns by name prefix leaked the target once
    for t, cols in lag_cols.items():
        assert len(cols) == 4 and not {"lmp_total", "heat_rate", "gas"} & set(cols), cols

    # Same hours for everything: every block, lag and the gas normaliser present.
    ok = ti >= 0
    for v in sys_blocks.values():
        ok &= ~np.isnan(v[np.clip(ti, 0, None)]).any(axis=1)
    for t in args.targets:
        ok &= df[lag_cols[t]].notna().all(axis=1).to_numpy()
    for col in gas_cols:
        ok &= df[col].notna().to_numpy()
    for t in args.targets:
        ok &= df[f"N_{t}"].notna().to_numpy() & (df[f"N_{t}"] > 0).to_numpy()
    print(f"{ok.sum()} of {len(df)} hours usable; heat rate median {df.loc[ok, 'heat_rate'].median():.1f} MMBtu/MWh "
          f"(1-99%: {df.loc[ok, 'heat_rate'].quantile(0.01):.1f} to {df.loc[ok, 'heat_rate'].quantile(0.99):.1f})", flush=True)

    combos = [[], ["weather (issue)"], ["weather (issue)", "GFS area"], ["weather (issue)", "Prithvi area"],
              ["weather (issue)", "GFS area", "Prithvi area"], ["weather (issue)", "Prithvi tokens"],
              ["weather (issue)", "GFS area", "Prithvi tokens"], ["weather (issue)", "GFS area", "Prithvi area", "Prithvi tokens"],
              ["weather (issue)", "Prithvi area", "Prithvi tokens"], ["weather (issue)", "true area"]]
    if args.token_variants:
        variants = ["tokens pooled full", "Prithvi tokens", "tokens spatial k10", "tokens spatial k25"]
        combos = [[], ["weather (issue)", "GFS area"]] + [["weather (issue)", v] for v in variants] + [["weather (issue)", "GFS area", v] for v in variants]
    if args.regional:
        wi, G, D, T = "weather (issue)", "GFS area", "Prithvi area", "Prithvi tokens"
        combos = [[wi, G], [wi, G, "GFS regional"], [wi, D], [wi, D, "Prithvi regional"], [wi, G, D], [wi, G, D, "GFS regional", "Prithvi regional"],
                  [wi, G, T], [wi, G, "GFS regional", T], [wi, "true area"], [wi, "true area", "true regional"]]
    if args.pc_sweep is not None:
        wi, G, R = "weather (issue)", "GFS area", "GFS regional"
        sweep = [f"tokens PC{k}" for k in args.pc_sweep] + ["tokens pooled full"]
        combos = [[wi, G], [wi, G, R]] + [[wi, G, v] for v in sweep] + [[wi, G, R, v] for v in sweep]
    if args.placebo:
        wi, G, R = "weather (issue)", "GFS area", "GFS regional"
        placebos = ["Prithvi tokens", "tokens placebo shift +7d", "tokens placebo shift -7d"] + [f"tokens placebo shuffle s{k}" for k in range(3)]
        combos = [[wi, G], [wi, G, R]] + [[wi, G, v] for v in placebos] + [[wi, G, R, v] for v in placebos]
    if args.input_control:
        combos = [[], ["weather (issue)", "GFS area"], ["weather (issue)", "Prithvi tokens"], ["weather (issue)", "MERRA-2 fields"],
                  ["weather (issue)", "GFS area", "Prithvi tokens"], ["weather (issue)", "GFS area", "MERRA-2 fields"],
                  ["weather (issue)", "GFS area", "MERRA-2 fields", "Prithvi tokens"]]
    # Hard days from the actual temperature over the zone areas (system-weighted), thresholds from training years.
    t2m_true = sys_blocks["true area"][np.clip(ti, 0, None), 0]
    week = (df["timestamp_local"].dt.isocalendar().year.astype(str) + "-" + df["timestamp_local"].dt.isocalendar().week.astype(str)).to_numpy()
    all_preds = []
    results = []
    lmp_all, gas_all = df["lmp_total"].to_numpy(), df["gas"].to_numpy()
    for test_year in args.test_years:
        tr, te = ok & (year < test_year) & (year >= min(args.years)), ok & (year == test_year)
        train_times = np.zeros(len(targets), bool)
        train_times[np.unique(ti[tr])] = True
        blocks = {k: v for k, v in sys_blocks.items() if not k.endswith(" (raw)")}
        blocks["Prithvi tokens"], frac = pca(sys_blocks["Prithvi tokens (raw)"], train_times, 50)
        print(f"  pooled tokens: 50 PCs keep {frac:.1%} of the (standardised) variance", flush=True)
        if args.placebo:
            for name, src in placebo_index(targets, ~np.isnan(blocks["Prithvi tokens"][:, 0])).items():
                blocks[name] = blocks["Prithvi tokens"][src]
        if args.pc_sweep is not None:
            blocks["tokens pooled full"] = sys_blocks["Prithvi tokens (raw)"]
            for k in args.pc_sweep:
                blocks[f"tokens PC{k}"], frac = pca(sys_blocks["Prithvi tokens (raw)"], train_times, k)
                print(f"  pooled tokens: {k} PCs keep {frac:.1%} of the (standardised) variance", flush=True)
        if args.token_variants:
            blocks["tokens pooled full"] = sys_blocks["Prithvi tokens (raw)"]
            for k, v in spatial_tokens(token_paths, train_times, (10, 25)).items():
                blocks[f"tokens spatial k{k}"] = v
        if args.input_control:
            blocks["MERRA-2 fields"] = pca(sys_blocks["MERRA-2 fields (raw)"], train_times, 50)[0]
        hard = hard_day_labels(df["timestamp_local"], t2m_true, tr)
        print(f"  hard days (test {test_year}): thresholds {hard.attrs['thresholds_K']}; "
              f"{hard.loc[te, 'cold_snap'].sum()} cold-snap and {hard.loc[te, 'heat_wave'].sum()} heat-wave test hours", flush=True)
        for target in args.targets:
            y_fit = df[f"y_{target}"].to_numpy()
            print(f"\n[{target}, test {test_year}, train {min(args.years)}-{test_year - 1}] train {tr.sum()} h, test {te.sum()} h; "
                  f"test price mean ${lmp_all[te].mean():.2f}, sd ${lmp_all[te].std():.2f}", flush=True)
            for combo in combos:
                X = df[CALENDAR + lag_cols[target] + gas_cols].copy()
                for b in combo:
                    names = [f"{b}_{n}" for n in (NAMES if blocks[b].shape[1] == len(NAMES) else range(blocks[b].shape[1]))]
                    X = pd.concat([X, pd.DataFrame(blocks[b][np.clip(ti, 0, None)], columns=names, index=df.index)], axis=1)
                model = CatBoostRegressor(iterations=args.iterations, learning_rate=0.05, depth=6, random_seed=7, verbose=False,
                                          thread_count=-1, train_dir=str(args.out_dir / "catboost_info"))
                model.fit(Pool(X[tr], y_fit[tr]))
                pred = model.predict(X[te])
                price = pred * df[f"N_{target}"].to_numpy()[te]
                err = price - lmp_all[te]
                mae = float(np.abs(err).mean())
                r2 = float(1 - (err ** 2).sum() / ((lmp_all[te] - lmp_all[te].mean()) ** 2).sum())
                name = "calendar + past values" + (" + known gas" if gas_cols else "") + "".join(f" + {b}" for b in combo)
                all_preds.append(pd.DataFrame({"row": np.where(te)[0], "test_year": test_year, "target": target, "feature_set": name,
                                               "actual": lmp_all[te], "pred": price, "week": week[te],
                                               "cold_snap": hard["cold_snap"].to_numpy()[te], "heat_wave": hard["heat_wave"].to_numpy()[te]}))
                results.append({"target": target, "test_year": test_year, "feature_set": name, "mae": mae, "r2": r2,
                                "rmse": float(np.sqrt((err ** 2).mean())), "n_test_hours": int(te.sum())})
                print(f"  {name:75s} MAE ${mae:6.2f}   R2 {r2:5.2f}", flush=True)
                pd.DataFrame(results).to_csv(args.out_dir / "price_comparison.csv", index=False)
    preds = pd.concat(all_preds, ignore_index=True)
    preds.to_parquet(args.out_dir / "price_predictions.parquet", index=False)
    ref = "calendar + past values" + (" + known gas" if gas_cols else "") + " + weather (issue) + GFS area"
    summary = summarize(preds, ref, group_cols=("target", "test_year"))
    summary.to_csv(args.out_dir / "price_summary.csv", index=False)
    pd.set_option("display.width", 250)
    for (target, ty), g in summary.groupby(["target", "test_year"], sort=False):
        print(f"\n== {target}, test {ty}: MAE ($/MWh) overall / cold snaps / heat waves / top 5% price hours; "
              f"difference vs '+ old weather + GFS' with 95% week-bootstrap interval (negative = better than GFS)")
        show = g[["feature_set", "MAE", "MAE_cold_snap", "MAE_heat_wave", "MAE_top5pct", "dMAE_vs_ref_all", "dMAE_vs_ref_all_ci",
                  "dMAE_vs_ref_cold_snap", "dMAE_vs_ref_cold_snap_ci"]].copy()
        show["feature_set"] = show["feature_set"].str.replace("calendar + past values" + (" + known gas" if gas_cols else ""), "", regex=False).str.strip(" +").replace("", "(calendar + past values)")
        print(show.round(2).to_string(index=False))
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
