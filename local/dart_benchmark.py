"""DART spread (real-time minus day-ahead ISO-NE hub LMP) on the market clock.

Decision time: about 13:30 ET on day D-1, when ISO-NE publishes the day-ahead results for day D; one prediction
for every hour of D, before any of D's real-time hours. Only information available at that moment is used:

  day-ahead     the DA price for the target hour, and D's DA daily mean, max and min (all published together)
  spread lags   DART for the same hour on D-2 and D-7; for the same hour on D-1 only when that real-time hour
                ended by 13:00 ET (target hours 00-12), else missing; mean DART over D-1 00:00-13:00 ET; mean DART
                over all of D-2
  Henry Hub     last trading day <= D-1 (traded that morning, before the day-ahead deadline)
  calendar      hour, day of week, day of year, month, weekend, holiday (ET)
  weather       MERRA-2 at 12 UTC on D-1 (zone-weighted, as in the price runs)
  GFS           the 12 UTC run of D-1 (the newest run published by 13:30 ET: it is complete about 15:30-16:00 UTC,
                the 18 UTC run about 21:30 UTC), forecast hour 15-42 nearest the target hour
                (data/gfs/gfs_ne_YEAR_f015_r12.nc; 06_download_gfs.sbatch RUNS=12 FHOURS="15 18 ... 42")
  tokens        the lead-24 Prithvi sample issued at 12 UTC on D-1 (same analysis time as the GFS run; MERRA-2 at
                15 UTC would be earlier than 13:30 ET too, but an operational analysis for it would not be), New England
                box mean, 50 PCs fitted on training times

Loss: absolute error by default (--loss MAE), since the score is MAE and the spread is centred near zero with heavy
tails; the zero-spread forecast (real time = day ahead) is the reference. Folds as for price.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

sys.path.insert(0, str(Path(__file__).parent))
import train_catboost_comparisons as dm  # noqa: E402
from analyze_global_year import ZONES, box_tokens, pca  # noqa: E402
from eval_tools import block_bootstrap  # noqa: E402
from gfs_demand_benchmark import NAMES, merra_features, zone_features  # noqa: E402

CALENDAR = [f for f in dm.CALENDAR_FEATURES if f != "zone"]
ZERO = "zero spread (real time = day ahead)"


def hub(c, kind, y):
    d = pd.read_parquet(c / f"isone_{kind}_hourly_lmp_{y}.parquet")
    d = d[d["location"] == "HUB"].copy()
    d["timestamp"] = pd.to_datetime(d["timestamp_parsed"]).astype("datetime64[ns]")
    return d.set_index("timestamp")


def gfs_pick(path, issue, targets):
    """Index into the 12 UTC file for each target: run = issue (12 UTC on D-1), forecast hour nearest the target."""
    import xarray as xr
    ds = xr.open_dataset(path)
    key = pd.Series(np.arange(ds.sizes["sample"]), index=pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(ds.run_time.values), ds.lead_hours.values.astype(int)]))
    key = key[~key.index.duplicated()]
    lead = ((targets - issue) / pd.Timedelta(hours=3)).round().astype(int) * 3
    pos = key.reindex(pd.MultiIndex.from_arrays([issue, np.asarray(lead)])).to_numpy()
    pick = np.where(np.isnan(pos), -1, pos).astype(int)
    return ds, pick


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=[2021, 2022, 2023, 2024])
    parser.add_argument("--test-years", type=int, nargs="+", default=[2023, 2024])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--prithvi-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--loss", default="MAE", help="CatBoost loss: MAE (default) or RMSE")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    c = args.data_dir / "compact"
    zn = list(ZONES)

    rows, blocks, have_gfs = [], {}, True
    for y in args.years:
        da, rt = hub(c, "dayahead", y), hub(c, "realtime", y)
        h = pd.DataFrame({"timestamp_local": da["timestamp_local"], "da": da["lmp_total"]}).join(rt["lmp_total"].rename("rt"), how="inner")
        table = pd.read_parquet(args.data_dir / f"modeling_table_{y}.parquet", columns=["timestamp", "zone", "demand_MW"] + CALENDAR)
        h = h.join(table.groupby("timestamp")[CALENDAR].first(), how="left")
        share = table.groupby("zone")["demand_MW"].mean().reindex(zn).to_numpy()
        share = share / share.sum()
        h["day"] = h["timestamp_local"].dt.normalize()
        issue = pd.DatetimeIndex(h["day"] - pd.Timedelta(days=1)) + pd.Timedelta(hours=12)   # 12 UTC on D-1
        t = pd.DatetimeIndex(h.index)
        w = lambda a: (a * share[None, :, None]).sum(axis=1)  # noqa: E731
        weather = pd.read_parquet(c / f"conventional_weather_{y}.parquet").set_index("timestamp")
        prev = c / f"conventional_weather_{y - 1}.parquet"
        if prev.exists():
            weather = pd.concat([pd.read_parquet(prev).set_index("timestamp"), weather])
            weather = weather[~weather.index.duplicated(keep="last")].sort_index()
        yb = {"weather (issue)": w(merra_features(weather, issue, zn))}
        gpath = args.data_dir / "gfs" / f"gfs_ne_{y}_f015_r12.nc"
        if gpath.exists():
            ds, pick = gfs_pick(gpath, issue, t)
            yb["GFS area"] = w(zone_features(ds, pick, t, zn)[1])
        else:
            have_gfs = False
        meta = json.loads((args.prithvi_dir / f"prithvi_tokens_ne_{y}_lead24.json").read_text())
        ts = pd.DatetimeIndex(meta["timestamps"])
        tok = box_tokens(args.prithvi_dir / f"prithvi_tokens_ne_{y}_lead24.npy")
        pos = ts.get_indexer(issue)
        tt = np.full((len(t), tok.shape[1]), np.nan, np.float32)
        tt[pos >= 0] = tok[pos[pos >= 0]]
        yb["Prithvi tokens (raw)"] = tt
        for k, v in yb.items():
            blocks.setdefault(k, []).append(v)
        rows.append(h)
    if not have_gfs:
        blocks.pop("GFS area", None)
        print("No 12 UTC GFS files (gfs_ne_YEAR_f015_r12.nc) for every year: running without GFS", flush=True)
    blocks = {k: np.concatenate(v) for k, v in blocks.items()}
    df = pd.concat(rows).sort_index()
    order = np.argsort(np.concatenate([r.index.values for r in rows]), kind="stable")
    blocks = {k: v[order] for k, v in blocks.items()}
    df = df.reset_index().rename(columns={"index": "timestamp"})
    df["dart"] = df["rt"] - df["da"]

    # ---- features known at 13:30 ET on D-1 ----
    s = df.set_index("timestamp")["dart"]
    lt = df["timestamp_local"]
    df["dart_same_hour_d2"] = s.reindex(df["timestamp"] - pd.Timedelta(hours=48)).to_numpy()
    df["dart_same_hour_d7"] = s.reindex(df["timestamp"] - pd.Timedelta(days=7)).to_numpy()
    d1 = s.reindex(df["timestamp"] - pd.Timedelta(hours=24)).to_numpy()
    df["dart_same_hour_d1"] = np.where(lt.dt.hour <= 12, d1, np.nan)     # that real-time hour ended by 13:00 ET on D-1
    daily = df.assign(morning=lt.dt.hour <= 12)
    morning = daily[daily["morning"]].groupby("day")["dart"].mean()
    whole = daily.groupby("day")["dart"].mean()
    df["dart_d1_morning_mean"] = morning.reindex(df["day"] - pd.Timedelta(days=1)).to_numpy()
    df["dart_d2_mean"] = whole.reindex(df["day"] - pd.Timedelta(days=2)).to_numpy()
    dad = df.groupby("day")["da"].agg(["mean", "max", "min"])
    for k in ("mean", "max", "min"):
        df[f"da_day_{k}"] = dad[k].reindex(df["day"]).to_numpy()
    hh = pd.read_excel(args.data_dir / "gas" / "RNGWHHDd.xls", sheet_name="Data 1", skiprows=2)
    hh.columns = ["date", "henry_hub"]
    hh = hh.dropna().assign(date=lambda d: pd.to_datetime(d["date"])).set_index("date")["henry_hub"].sort_index()
    df["gas_known_henryhub"] = hh.reindex((df["day"] - pd.Timedelta(days=1)).to_numpy(), method="ffill").to_numpy()
    base_cols = CALENDAR + ["da", "da_day_mean", "da_day_max", "da_day_min", "dart_same_hour_d2", "dart_same_hour_d7",
                            "dart_same_hour_d1", "dart_d1_morning_mean", "dart_d2_mean", "gas_known_henryhub"]
    must = [cc for cc in base_cols if cc != "dart_same_hour_d1"]      # missing by design for afternoon hours
    ok = df[must].notna().all(axis=1).to_numpy() & df["dart"].notna().to_numpy()
    for v in blocks.values():
        ok &= ~np.isnan(v).any(axis=1)
    year = lt.dt.year.to_numpy()
    week = (lt.dt.isocalendar().year.astype(str) + "-" + lt.dt.isocalendar().week.astype(str)).to_numpy()
    print(f"{ok.sum()} of {len(df)} hours usable; loss {args.loss}", flush=True)

    wi, G, T = "weather (issue)", "GFS area", "Prithvi tokens"
    combos = [[], [wi], [wi, T]] + ([[wi, G], [wi, G, T]] if "GFS area" in blocks else [])
    y_all = df["dart"].to_numpy()
    results, preds = [], []
    for test_year in args.test_years:
        tr, te = ok & (year < test_year) & (year >= min(args.years)), ok & (year == test_year)
        fb = {k: v for k, v in blocks.items() if not k.endswith("(raw)")}
        fb[T] = pca(blocks["Prithvi tokens (raw)"], tr, 50)[0]
        sets = {ZERO: np.zeros(te.sum())}
        for combo in combos:
            X = df[base_cols].copy()
            for b in combo:
                wdt = fb[b].shape[1]
                X = pd.concat([X, pd.DataFrame(fb[b], columns=[f"{b}_{n}" for n in (NAMES if wdt == len(NAMES) else range(wdt))], index=df.index)], axis=1)
            model = CatBoostRegressor(iterations=args.iterations, learning_rate=0.05, depth=6, random_seed=7, verbose=False,
                                      loss_function=args.loss, thread_count=-1, train_dir=str(args.out_dir / "catboost_info"))
            model.fit(Pool(X[tr], y_all[tr]))
            sets["calendar + DA + spread lags + Henry Hub" + "".join(f" + {b}" for b in combo)] = model.predict(X[te])
        act = y_all[te]
        zero_ae = np.abs(act)
        for name, pr in sets.items():
            e = pr - act
            m, lo, hi = block_bootstrap(np.abs(e) - zero_ae, week[te]) if name != ZERO else (np.nan,) * 3
            big = np.abs(act) > 5
            results.append({"test_year": test_year, "feature_set": name, "loss": args.loss, "mae": np.abs(e).mean(),
                            "median_ae": np.median(np.abs(e)), "r2": 1 - (e ** 2).sum() / ((act - act.mean()) ** 2).sum(),
                            "skill_vs_zero": 1 - np.abs(e).mean() / zero_ae.mean(), "dMAE_vs_zero": m, "lo": lo, "hi": hi,
                            "sign_hit_abs_gt5": float((np.sign(pr[big]) == np.sign(act[big])).mean()), "n": int(te.sum())})
            preds.append(pd.DataFrame({"row": np.where(te)[0], "test_year": test_year, "feature_set": name, "actual": act,
                                       "pred": pr, "week": week[te]}))
        print(f"\n[test {test_year}] train {tr.sum()} h, test {te.sum()} h; spread mean {act.mean():+.2f}, median {np.median(act):+.2f}")
        for r in [r for r in results if r["test_year"] == test_year]:
            print(f"  {r['feature_set']:85s} MAE {r['mae']:6.2f}  median {r['median_ae']:5.2f}  R2 {r['r2']:+.3f}  "
                  f"skill vs zero {r['skill_vs_zero']:+.3f}  dMAE vs zero {r['dMAE_vs_zero']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]  "
                  f"sign {r['sign_hit_abs_gt5']:.2f}", flush=True)
    pd.DataFrame(results).to_csv(args.out_dir / "dart_comparison.csv", index=False)
    pd.concat(preds, ignore_index=True).to_parquet(args.out_dir / "dart_predictions.parquet", index=False)
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
