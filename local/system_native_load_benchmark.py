"""System demand, 24 h ahead, against ISO-NE's own prior-day forecast, on the
forecast's own target definition: ISO-NE system NativeLoad.

Why this target. ISO-NE's three-day system forecast matches the hourly
system NativeLoad (/hourlysysload) with no midday gap, no trend over years
and near-zero bias (2021-2024: mean +36 MW, MAPE 2.13%). Our zonal
HourlyRtDemand.Load summed over zones is not that quantity: it equals
NativeLoad + about 0.4 x ISO-NE's estimated behind-the-meter solar - about
180 MW (Jun-Dec 2024, correlation 0.988). So the clean comparison retrains
our models on NativeLoad, and scores ISO-NE's forecast as published, with
no correction fitted.

Folds: year-ahead (test 2023 trained on 2021-22; test 2024 trained on
2021-23). Features as elsewhere, system level: calendar; weather blocks
weighted by each zone's share of that year's demand (weather at issue,
GFS area, Prithvi decoder area, actual area = ceiling); Prithvi tokens
(final encoder layer averaged over the New England box, 50 PCs fitted on
training times); ISO-NE's prior-day forecast as a feature (the latest
version issued the day before, before 10:00 Eastern). No lagged load.
Errors: absolute percentage error of NativeLoad; hard days from the actual
area temperature (training-year thresholds); top 5% = highest NativeLoad
test hours; week-block bootstrap against ISO-NE's published forecast.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

import train_catboost_comparisons as dm
from analyze_global_year import ZONES, box_tokens, pca
from eval_tools import hard_day_labels, summarize
from gfs_demand_benchmark import NAMES, gfs_for_targets, merra_features, prithvi_features, zone_features

CALENDAR = [f for f in dm.CALENDAR_FEATURES if f != "zone"]
ET = "America/New_York"


def iso_prior_day_system(path):
    s = pd.read_parquet(path)
    created = s["creation_utc"].dt.tz_convert(ET)
    day = s["target_utc"].dt.tz_convert(ET).dt.normalize()
    s = s[(created.dt.normalize() == day - pd.Timedelta(days=1)) & (created.dt.hour < 10)]
    out = s.sort_values("creation_utc").groupby("target_utc")["LoadMw"].last()
    out.index = out.index.tz_convert("UTC").tz_localize(None)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=[2021, 2022, 2023, 2024])
    parser.add_argument("--test-years", type=int, nargs="+", default=[2023, 2024])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--prithvi-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    c, zn = args.data_dir / "compact", list(ZONES)

    rows, sys_blocks, targets = [], {}, pd.DatetimeIndex([])
    for y in args.years:
        table = pd.read_parquet(args.data_dir / f"modeling_table_{y}.parquet", columns=["timestamp", "timestamp_local", "zone", "demand_MW"] + CALENDAR)
        share = table.groupby("zone")["demand_MW"].mean().reindex(zn).to_numpy()
        share = share / share.sum()
        rows.append(table.groupby("timestamp")[["timestamp_local"] + CALENDAR].first().reset_index())
        t = pd.date_range(f"{y}-01-01", f"{y}-12-31 21:00", freq="3h")
        weather = pd.read_parquet(c / f"conventional_weather_{y}.parquet").set_index("timestamp")
        prev = c / f"conventional_weather_{y - 1}.parquet"
        if prev.exists():
            weather = pd.concat([pd.read_parquet(prev).set_index("timestamp"), weather])
            weather = weather[~weather.index.duplicated(keep="last")].sort_index()
        ds, pick = gfs_for_targets(args.data_dir / "gfs" / f"gfs_ne_{y}.nc", t)
        _, gfs_area = zone_features(ds, pick, t, zn)
        pv = prithvi_features(args.prithvi_dir, y, t, weather, zn)
        w = lambda a: (a * share[None, :, None]).sum(axis=1)  # noqa: E731
        blk = {"weather (issue)": w(merra_features(weather, t - pd.Timedelta(hours=24), zn)), "GFS area": w(gfs_area),
               "Prithvi area": w(pv["Prithvi area"]), "true area": w(pv["true area"])}
        meta = json.loads((args.prithvi_dir / f"prithvi_tokens_ne_{y}_lead24.json").read_text())
        issue = pd.DatetimeIndex(meta["timestamps"])
        tok = box_tokens(args.prithvi_dir / f"prithvi_tokens_ne_{y}_lead24.npy")
        pos = issue.get_indexer(t - pd.Timedelta(hours=24))
        tok_t = np.full((len(t), tok.shape[1]), np.nan, np.float32)
        tok_t[pos >= 0] = tok[pos[pos >= 0]]
        blk["Prithvi tokens (raw)"] = tok_t
        for k, v in blk.items():
            sys_blocks.setdefault(k, []).append(v)
        targets = targets.append(t)
        print(f"{y}: blocks built", flush=True)
    sys_blocks = {k: np.concatenate(v) for k, v in sys_blocks.items()}

    df = pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    sysload = pd.read_parquet(args.data_dir / "isone_forecast" / "isone_system_load_hourly.parquet").set_index("target_utc")
    sysload.index = sysload.index.tz_convert("UTC").tz_localize(None)
    df["native_load"] = sysload["NativeLoad"].reindex(df["timestamp"]).to_numpy()
    df["isone_fc"] = iso_prior_day_system(args.data_dir / "isone_forecast" / "isone_system_forecast.parquet").reindex(df["timestamp"]).to_numpy()
    ti = targets.get_indexer(df["timestamp"].dt.round("3h"))
    ok = (ti >= 0) & df["native_load"].notna().to_numpy() & df["isone_fc"].notna().to_numpy()
    for v in sys_blocks.values():
        ok &= ~np.isnan(v[np.clip(ti, 0, None)]).any(axis=1)
    print(f"{ok.sum()} of {len(df)} hours usable", flush=True)
    year = df["timestamp_local"].dt.year.to_numpy()
    iso = df["timestamp_local"].dt.isocalendar()
    week = (iso.year.astype(str) + "-" + iso.week.astype(str)).to_numpy()
    y_all, fc = df["native_load"].to_numpy(), df["isone_fc"].to_numpy()
    t2m = sys_blocks["true area"][np.clip(ti, 0, None), 0]

    wi, G, D, T, I = "weather (issue)", "GFS area", "Prithvi area", "Prithvi tokens", "ISO-NE forecast"
    combos = [[wi], [wi, G], [wi, D], [wi, T], [wi, G, D], [wi, G, T], [wi, "true area"],
              [I], [I, wi], [I, wi, G], [I, wi, D], [I, wi, T], [I, wi, G, D], [I, wi, G, T], [I, wi, "true area"]]
    preds, results = [], []
    for ty in args.test_years:
        tr, te = ok & (year < ty), ok & (year == ty)
        train_times = np.zeros(len(targets), bool)
        train_times[np.unique(ti[tr])] = True
        blocks = {k: v for k, v in sys_blocks.items() if not k.endswith("(raw)")}
        blocks[T] = pca(sys_blocks["Prithvi tokens (raw)"], train_times, 50)[0]
        hard = hard_day_labels(df["timestamp_local"], t2m, tr)
        top = te & (y_all >= np.quantile(y_all[te], 0.95))
        common = {"row": np.where(te)[0], "test_year": ty, "actual": y_all[te], "week": week[te], "top": top[te],
                  "cold_snap": hard["cold_snap"].to_numpy()[te], "heat_wave": hard["heat_wave"].to_numpy()[te]}
        print(f"\n[test {ty}] train {tr.sum()} h, test {te.sum()} h; {hard.loc[te, 'cold_snap'].sum() // 24} cold-snap and "
              f"{hard.loc[te, 'heat_wave'].sum() // 24} heat-wave days (approx.)", flush=True)
        preds.append(pd.DataFrame({**common, "feature_set": "ISO-NE forecast (published)", "pred": fc[te]}))
        print(f"  {'ISO-NE forecast (published)':60s} MAPE {np.mean(np.abs(fc[te] - y_all[te]) / y_all[te]) * 100:5.2f}%", flush=True)
        for combo in combos:
            X = df[CALENDAR].copy()
            for b in combo:
                if b == I:
                    X["isone_forecast_MW"] = df["isone_fc"]
                    continue
                cols = [f"{b}_{n}" for n in (NAMES if blocks[b].shape[1] == len(NAMES) else range(blocks[b].shape[1]))]
                X = pd.concat([X, pd.DataFrame(blocks[b][np.clip(ti, 0, None)], columns=cols, index=df.index)], axis=1)
            model = CatBoostRegressor(iterations=args.iterations, learning_rate=0.05, depth=6, random_seed=7, verbose=False,
                                      thread_count=-1, train_dir=str(args.out_dir / "catboost_info"))
            model.fit(Pool(X[tr], y_all[tr]))
            p = model.predict(X[te])
            name = "calendar + " + " + ".join(combo)
            preds.append(pd.DataFrame({**common, "feature_set": name, "pred": p}))
            mape = float(np.mean(np.abs(p - y_all[te]) / y_all[te]) * 100)
            results.append({"test_year": ty, "feature_set": name, "mape": mape})
            print(f"  {name:60s} MAPE {mape:5.2f}%", flush=True)
    preds = pd.concat(preds, ignore_index=True)
    preds.to_parquet(args.out_dir / "system_native_load_predictions.parquet", index=False)
    summary = summarize(preds, "ISO-NE forecast (published)", pct=True)
    summary.to_csv(args.out_dir / "system_native_load_summary.csv", index=False)
    pd.set_option("display.width", 250)
    for ty, g in summary.groupby("test_year"):
        print(f"\n== system NativeLoad, test {ty}: MAPE (%) overall / cold snaps / heat waves / top 5% hours; "
              f"difference vs ISO-NE's published forecast, 95% week-bootstrap interval (negative = better than ISO-NE)")
        show = g[["feature_set", "MAE", "MAE_cold_snap", "MAE_heat_wave", "MAE_top5pct", "dMAE_vs_ref_all", "dMAE_vs_ref_all_ci",
                  "dMAE_vs_ref_cold_snap", "dMAE_vs_ref_cold_snap_ci", "dMAE_vs_ref_heat_wave", "dMAE_vs_ref_heat_wave_ci"]].copy()
        show["feature_set"] = show["feature_set"].str.replace("calendar + ", "", regex=False)
        print(show.round(2).to_string(index=False))
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
