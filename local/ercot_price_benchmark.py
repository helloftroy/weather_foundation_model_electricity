"""ERCOT replication of the ISO-NE price benchmark (local/price_benchmark.py, raw-LMP setting with Henry Hub known).

Target: hourly ERCOT day-ahead settlement point price at each hub (DAM, $/MWh), 24 h ahead, one model per hub.
Same design as ISO-NE: year-ahead folds (test 2023 trained on 2021-22, test 2024 trained on 2021-23), CatBoost
(3,000 trees, lr 0.05, depth 6, seed 7), 3-hourly target times (the Prithvi samples are 3-hourly), every block lined
up on the target hour and issued 24 h before it; every hour uses the nearest 3-hourly sample, as for ISO-NE.

Features (as for ISO-NE):
  calendar      hour, day of week, day of year, month, weekend, US federal holiday (Central time)
  past prices   the hub's price 24 h, 48 h and 7 days earlier, and the mean of the 24 h ending 24 h earlier
  Henry Hub     last trading day <= D-1 (EIA RNGWHHDd)
  blocks (6 variables each: T2M, QV2M, U10M, V10M, SWGNT, wind speed), averaged over the hub's area:
    weather (issue)  MERRA-2 at the issue time (Prithvi's own inputs, pooled over the hub area)
    GFS area         GFS 24/27 h forecast for the target hour (data/gfs/gfs_texas_YEAR.nc)
    Prithvi area     Prithvi decoder output for the target hour (pooled over the hub area)
    true area        actual MERRA-2 at the target hour (ceiling)
  Prithvi tokens     final encoder layer pooled over the whole ERCOT footprint, 50 PCs fitted on training times
  tokens hub area    the same pooled over the hub's own area (variant)
  input state        Prithvi's 160 input channels pooled over ERCOT, 50 PCs (the input-state control)

Inputs: data/compact/ercot_dam_spp_YEAR.parquet (src/ercot/download_ercot_dam_prices.py), data/gfs/gfs_texas_YEAR.nc
(06_download_gfs.sbatch with the Texas box), data/prithvi_tokens_tx_YEAR_lead24.json + *_pooled_{tokens,inputs,recon}.npy
(02_extract_embeddings.sbatch with REGION=ercot POOLED=1 LEAD_TIME=24), data/gas/RNGWHHDd.xls.

Winter Storm Uri (February 2021, prices at the $9,000 cap) sits in the first training year. --exclude-uri drops
2021-02-10..2021-02-22 (local days) from training; run both ways and report both.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from catboost import CatBoostRegressor, Pool
from pandas.tseries.holiday import USFederalHolidayCalendar

from analyze_global_year import pca
from eval_tools import hard_day_labels, summarize
from gfs_demand_benchmark import GFS_VARS, NAMES

CALENDAR = ["hour", "day_of_week", "day_of_year", "month", "is_weekend", "is_holiday"]
LAGS = {"lag24h": pd.Timedelta(hours=24), "lag48h": pd.Timedelta(hours=48), "lag7d": pd.Timedelta(days=7)}
HUB_AREA = {"HB_NORTH": "north", "HB_HOUSTON": "houston", "HB_SOUTH": "south", "HB_WEST": "west", "HB_PAN": "panhandle",
            "HB_HUBAVG": "ercot"}
SFC = ["T2M", "QV2M", "U10M", "V10M", "SWGNT"]
URI = ("2021-02-10", "2021-02-22")


def six(fields_by_var):
    """[T, 5] in SFC order -> [T, 6] with wind speed appended."""
    return np.c_[fields_by_var, np.hypot(fields_by_var[:, 2], fields_by_var[:, 3])].astype(np.float32)


def gfs_area(path, targets, box):
    """GFS 24 h forecast for each target time (27 h where no 24 h one), averaged over the box -> [T, 6]."""
    ds = xr.open_dataset(path)
    vt, lead = pd.DatetimeIndex(ds.valid_time.values), ds.lead_hours.values
    pick = np.full(len(targets), -1)
    for want in (27, 24):
        idx = pd.Series(np.where(lead == want)[0], index=vt[lead == want])
        pos = idx[~idx.index.duplicated()].reindex(targets).to_numpy()
        pick[~np.isnan(pos)] = pos[~np.isnan(pos)].astype(int)
    lat, lon = ds.lat.values, ds.lon.values
    rows, cols = (lat >= box[0]) & (lat <= box[1]), (lon >= box[2]) & (lon <= box[3])
    out = np.full((len(targets), 5), np.nan, np.float32)
    ok = pick >= 0
    for m, v in enumerate(SFC):
        out[ok, m] = ds[GFS_VARS[v]].values[pick[ok]][:, rows][:, :, cols].mean(axis=(1, 2))
    return six(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=[2021, 2022, 2023, 2024])
    parser.add_argument("--test-years", type=int, nargs="+", default=[2023, 2024])
    parser.add_argument("--hubs", nargs="+", default=list(HUB_AREA))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--prithvi-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--exclude-uri", action="store_true", help=f"Drop local days {URI[0]}..{URI[1]} from training")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    c = args.data_dir / "compact"

    # ---- prices: one column per hub, UTC hour beginning ----
    prices = pd.concat([pd.read_parquet(c / f"ercot_dam_spp_{y}.parquet") for y in args.years])
    prices = prices[prices["settlement_point"].isin(args.hubs)]
    wide = prices.pivot_table(index="timestamp", columns="settlement_point", values="price")
    local = prices.drop_duplicates("timestamp").set_index("timestamp")["timestamp_local"].reindex(wide.index)

    # ---- 3-hourly target times and their blocks ----
    targets = pd.DatetimeIndex([])
    blocks = {}
    for y in args.years:
        t = pd.date_range(f"{y}-01-01", f"{y}-12-31 21:00", freq="3h")
        meta = json.loads((args.prithvi_dir / f"prithvi_tokens_tx_{y}_lead24.json").read_text())
        ext = list(meta["pooled"]["extents"])
        ch = meta["pooled"]["channels"]
        stem = args.prithvi_dir / f"prithvi_tokens_tx_{y}_lead24"
        tok = np.load(f"{stem}_pooled_tokens.npy", mmap_mode="r")
        inp = np.load(f"{stem}_pooled_inputs.npy", mmap_mode="r")
        rec = np.load(f"{stem}_pooled_recon.npy", mmap_mode="r")
        issue = pd.DatetimeIndex(meta["timestamps"])
        pos_issue = issue.get_indexer(t - pd.Timedelta(hours=24))   # sample issued 24 h before the target
        pos_now = issue.get_indexer(t)                              # sample whose inputs ARE the target time

        def at(a, pos):
            a = np.asarray(a, np.float32)
            out = np.full((len(t),) + a.shape[1:], np.nan, np.float32)
            out[pos >= 0] = a[pos[pos >= 0]]
            return out

        sfc = [ch.index(v) for v in SFC]
        yb = {"Prithvi tokens (raw)": at(tok[:, ext.index("ercot")], pos_issue),
              "input state (raw)": at(inp[:, ext.index("ercot")], pos_issue)}
        for hub in args.hubs:
            e = ext.index(HUB_AREA[hub])
            box = meta["pooled"]["extents"][HUB_AREA[hub]]
            yb[f"weather (issue)|{hub}"] = six(at(inp[:, e][:, sfc], pos_issue))
            yb[f"Prithvi area|{hub}"] = six(at(rec[:, e][:, sfc], pos_issue))
            yb[f"true area|{hub}"] = six(at(inp[:, e][:, sfc], pos_now))
            yb[f"GFS area|{hub}"] = gfs_area(args.data_dir / "gfs" / f"gfs_texas_{y}.nc", t, box)
            yb[f"tokens hub area (raw)|{hub}"] = at(tok[:, e], pos_issue)
        for k, v in yb.items():
            blocks.setdefault(k, []).append(v)
        targets = targets.append(t)
    blocks = {k: np.concatenate(v) for k, v in blocks.items()}

    # ---- hourly rows ----
    df = pd.DataFrame({"timestamp": wide.index, "timestamp_local": local.to_numpy()})
    lt = df["timestamp_local"]
    hol = USFederalHolidayCalendar().holidays(f"{min(args.years) - 1}-01-01", f"{max(args.years) + 1}-12-31")
    df["hour"], df["day_of_week"], df["day_of_year"], df["month"] = lt.dt.hour, lt.dt.dayofweek, lt.dt.dayofyear, lt.dt.month
    df["is_weekend"] = (lt.dt.dayofweek >= 5).astype(int)
    df["is_holiday"] = lt.dt.normalize().isin(hol).astype(int)
    day = lt.dt.normalize()
    hh = pd.read_excel(args.data_dir / "gas" / "RNGWHHDd.xls", sheet_name="Data 1", skiprows=2)
    hh.columns = ["date", "henry_hub"]
    hh = hh.dropna().assign(date=lambda d: pd.to_datetime(d["date"])).set_index("date")["henry_hub"].sort_index()
    df["gas_known_henryhub"] = hh.reindex((day - pd.Timedelta(days=1)).to_numpy(), method="ffill").to_numpy()
    year = lt.dt.year.to_numpy()
    ti = targets.get_indexer(df["timestamp"].dt.round("3h"))   # every hour takes the nearest 3-hourly sample, as for ISO-NE
    week = (lt.dt.isocalendar().year.astype(str) + "-" + lt.dt.isocalendar().week.astype(str)).to_numpy()
    uri = ((day >= URI[0]) & (day <= URI[1])).to_numpy()

    wi, G, D, T, TA, IS, TR = "weather (issue)", "GFS area", "Prithvi area", "Prithvi tokens", "tokens hub area", "input state", "true area"
    combos = [[], [wi], [wi, G], [wi, D], [wi, G, D], [wi, T], [wi, G, T], [wi, G, D, T], [wi, G, TA], [wi, IS], [wi, G, IS],
              [wi, G, IS, T], [wi, TR]]
    results, all_preds = [], []
    for hub in args.hubs:
        s = wide[hub]
        lag_cols = []
        for k, d in LAGS.items():
            df[f"past__{k}"] = s.reindex(df["timestamp"] - d).to_numpy()
            lag_cols.append(f"past__{k}")
        df["past__mean24h_to_lag24h"] = s.rolling("24h").mean().reindex(df["timestamp"] - pd.Timedelta(hours=24)).to_numpy()
        lag_cols.append("past__mean24h_to_lag24h")
        hb = {k.split("|")[0]: v for k, v in blocks.items() if k.endswith(f"|{hub}") or "|" not in k}
        ok = (ti >= 0) & df[lag_cols].notna().all(axis=1).to_numpy() & df["gas_known_henryhub"].notna().to_numpy() & s.notna().to_numpy()
        for v in hb.values():
            ok &= ~np.isnan(v[np.clip(ti, 0, None)]).any(axis=1)
        y_all = s.to_numpy()
        t2m_true = hb[TR][np.clip(ti, 0, None), 0]
        for test_year in args.test_years:
            tr = ok & (year < test_year) & (year >= min(args.years)) & ~(uri & args.exclude_uri)
            te = ok & (year == test_year)
            train_times = np.zeros(len(targets), bool)
            train_times[np.unique(ti[tr])] = True
            fb = {k: v for k, v in hb.items() if not k.endswith(" (raw)")}
            fb[T] = pca(hb["Prithvi tokens (raw)"], train_times, 50)[0]
            fb[TA] = pca(hb["tokens hub area (raw)"], train_times, 50)[0]
            fb[IS] = pca(hb["input state (raw)"], train_times, 50)[0]
            hard = hard_day_labels(lt, t2m_true, tr)
            print(f"\n[{hub}, test {test_year}] train {tr.sum()} h, test {te.sum()} h; test price mean ${y_all[te].mean():.2f}, "
                  f"median ${np.median(y_all[te]):.2f}, max ${y_all[te].max():.0f}", flush=True)
            for combo in combos:
                X = df[CALENDAR + lag_cols + ["gas_known_henryhub"]].copy()
                for b in combo:
                    w = fb[b].shape[1]
                    cols = [f"{b}_{n}" for n in (NAMES if w == len(NAMES) else range(w))]
                    X = pd.concat([X, pd.DataFrame(fb[b][np.clip(ti, 0, None)], columns=cols, index=df.index)], axis=1)
                model = CatBoostRegressor(iterations=args.iterations, learning_rate=0.05, depth=6, random_seed=7, verbose=False,
                                          thread_count=-1, train_dir=str(args.out_dir / "catboost_info"))
                model.fit(Pool(X[tr], y_all[tr]))
                pred = model.predict(X[te])
                err = pred - y_all[te]
                name = "calendar + past prices + Henry Hub" + "".join(f" + {b}" for b in combo)
                r2 = float(1 - (err ** 2).sum() / ((y_all[te] - y_all[te].mean()) ** 2).sum())
                results.append({"hub": hub, "test_year": test_year, "feature_set": name, "mae": float(np.abs(err).mean()),
                                "median_ae": float(np.median(np.abs(err))), "r2": r2, "rmse": float(np.sqrt((err ** 2).mean())),
                                "n_test_hours": int(te.sum()), "exclude_uri": args.exclude_uri})
                all_preds.append(pd.DataFrame({"row": np.where(te)[0], "hub": hub, "test_year": test_year, "feature_set": name,
                                               "actual": y_all[te], "pred": pred, "week": week[te],
                                               "cold_snap": hard["cold_snap"].to_numpy()[te], "heat_wave": hard["heat_wave"].to_numpy()[te]}))
                print(f"  {name:95s} MAE ${results[-1]['mae']:7.2f}  median ${results[-1]['median_ae']:6.2f}  R2 {r2:5.2f}", flush=True)
                pd.DataFrame(results).to_csv(args.out_dir / "ercot_price_comparison.csv", index=False)
    preds = pd.concat(all_preds, ignore_index=True)
    preds.to_parquet(args.out_dir / "ercot_price_predictions.parquet", index=False)
    ref = "calendar + past prices + Henry Hub + weather (issue) + GFS area"
    summary = summarize(preds, ref, group_cols=("hub", "test_year"))
    summary.to_csv(args.out_dir / "ercot_price_summary.csv", index=False)
    pd.set_option("display.width", 250)
    for (hub, ty), g in summary.groupby(["hub", "test_year"], sort=False):
        print(f"\n== {hub}, test {ty}: MAE ($/MWh); difference vs '+ old weather + GFS' (95% week bootstrap)")
        show = g[["feature_set", "MAE", "MAE_cold_snap", "MAE_heat_wave", "MAE_top5pct", "dMAE_vs_ref_all", "dMAE_vs_ref_all_ci"]].copy()
        show["feature_set"] = show["feature_set"].str.replace("calendar + past prices + Henry Hub", "", regex=False).str.strip(" +").replace("", "(calendar + past prices + Henry Hub)")
        print(show.round(2).to_string(index=False))
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
