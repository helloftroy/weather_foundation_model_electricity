"""ERCOT wind and solar headroom test: how much weather signal is left beyond GFS, and do Prithvi tokens recover it?

Target: hourly ERCOT (EIA-930 balancing area ERCO) wind or solar net generation as a capacity factor, 24 h ahead.
Capacity proxy: the 99th percentile of hourly generation over the 90 days ending 24 h before the target hour (known at
issue time; installed capacity from EIA-860M can replace it later). Same design as the price benchmark: year-ahead folds
(test 2023 trained on 2021-22, test 2024 on 2021-23), CatBoost (3,000 trees, lr 0.05, depth 6, seed 7), every hour
takes the nearest 3-hourly Prithvi sample, 95% week-block bootstrap.

Areas (Prithvi pooled extents): wind = West (30-34 N, 104.5-99 W) + Panhandle (34-36.5 N, 103-100 W); solar = West.
GFS and the full GFS state use wind 30-36.5 N, 104-98 W and solar 29-34 N, 106.5-99 W (West + Far West).

Feature sets (each adds to the one before, except where noted):
  1 calendar (hour, day of week, day of year, month, weekend, holiday; solar also cos solar zenith at the area centre)
  2 + weather at issue      MERRA-2 at the issue time: T2M, QV2M, 10 m wind, lowest model level (~60 m) wind, SWGNT,
                            cloud liquid and ice paths (TQL, TQI), wind speeds
  3 + GFS                   24 h forecast (27 h where missing), area mean: 2 m T and q, 10 m wind, downward and net
                            surface shortwave (the local 0.25 deg files have no 100 m wind or cloud cover; those are in 4)
  4 + full GFS state        150 fields, area mean, PCA-50 (--gfs-full; src/gfs/gfs_full_state_pca.py --region texas)
  5 + Prithvi tokens        pooled over the area, PCA-50 (on 3, and on 4 when --gfs-full)
  6 + actual weather        the block of 2 at the target time (on 3): the ceiling
Scored as MAE in % of capacity, on all hours and on large ramps (top 5% of |CF(t) - CF(t - 3 h)| in the test year).

Data: EIA-930 six-month BALANCE files (downloaded once to data/eia930/erco_wind_solar.parquet), Prithvi pooled files
data/prithvi_tokens_tx_YEAR_lead24*, GFS data/gfs/gfs_texas_YEAR.nc.
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from catboost import CatBoostRegressor, Pool
from pandas.tseries.holiday import USFederalHolidayCalendar

sys.path.insert(0, str(Path(__file__).parent))
from analyze_global_year import pca  # noqa: E402
from ercot_price_benchmark import gfs_area  # noqa: E402
from eval_tools import block_bootstrap  # noqa: E402

YEARS = (2021, 2022, 2023, 2024)
EXT = {"wind": ("west", "panhandle"), "solar": ("west",)}
BOX = {"wind": (30.0, 36.5, -104.0, -98.0), "solar": (29.0, 34.0, -106.5, -99.0)}
NEAR = ["T2M", "QV2M", "U10M", "V10M", "U_72", "V_72", "SWGNT", "TQL", "TQI"]
CALENDAR = ["hour", "day_of_week", "day_of_year", "month", "is_weekend", "is_holiday"]


def eia930(path):
    if path.exists():
        return pd.read_parquet(path)
    url = "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_{}_{}.csv"
    raw = path.parent / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    parts = []
    for y, h in [(2020, "Jul_Dec")] + [(y, h) for y in YEARS for h in ("Jan_Jun", "Jul_Dec")]:
        f = raw / f"EIA930_BALANCE_{y}_{h}.csv"
        if not f.exists():
            print(f"  EIA-930 {y} {h}", flush=True)
            r = requests.get(url.format(y, h), timeout=600)
            r.raise_for_status()
            f.write_bytes(r.content)
        d = pd.read_csv(f, thousands=",", low_memory=False)
        d = d[d["Balancing Authority"] == "ERCO"]
        # later files split each fuel into with / without integrated battery storage: sum all adjusted columns of a fuel
        out = pd.DataFrame({"end": d["UTC Time at End of Hour"]})
        for fuel in ("Wind", "Solar"):
            c = [k for k in d.columns if k.startswith(f"Net Generation (MW) from {fuel}") and k.endswith("(Adjusted)")]
            out[fuel.lower()] = d[c].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        parts.append(out)
    d = pd.concat(parts)
    d["timestamp"] = pd.to_datetime(d.pop("end"), format="%m/%d/%Y %I:%M:%S %p") - pd.Timedelta(hours=1)  # hour beginning, UTC
    d = d.drop_duplicates("timestamp").set_index("timestamp").sort_index().astype(float)
    path.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(path)
    return d


def cos_zenith(ts, lat, lon):
    """Cosine of the solar zenith angle at the middle of each UTC hour (NOAA approximation)."""
    t = ts + pd.Timedelta(minutes=30)
    g = 2 * np.pi / 365 * (t.dayofyear - 1 + (t.hour - 12) / 24)
    decl = 0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    ha = np.deg2rad((t.hour * 60 + t.minute + eqt + 4 * lon) / 4 - 180)
    la = np.deg2rad(lat)
    return np.asarray(np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha), np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+", default=["wind", "solar"], choices=["wind", "solar"])
    ap.add_argument("--gfs-full", type=Path, default=None, help="gfs_full_state_texas_features.npz (with wind_mean / solar_mean)")
    ap.add_argument("--out-dir", type=Path, default=Path("results/ercot/renewables"))
    ap.add_argument("--iterations", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gen = eia930(Path("data/eia930/erco_wind_solar.parquet"))
    gfull = np.load(args.gfs_full) if args.gfs_full else None

    # ---- 3-hourly blocks per area ----
    targets, blocks = pd.DatetimeIndex([]), {}
    for y in YEARS:
        t = pd.date_range(f"{y}-01-01", f"{y}-12-31 21:00", freq="3h")
        stem = Path("data") / f"prithvi_tokens_tx_{y}_lead24"
        meta = json.loads(Path(f"{stem}.json").read_text())
        ext, ch = list(meta["pooled"]["extents"]), meta["pooled"]["channels"]
        tok = np.load(f"{stem}_pooled_tokens.npy", mmap_mode="r")
        inp = np.load(f"{stem}_pooled_inputs.npy", mmap_mode="r")
        issue = pd.DatetimeIndex(meta["timestamps"])
        pos_issue, pos_now = issue.get_indexer(t - pd.Timedelta(hours=24)), issue.get_indexer(t)

        def at(a, pos):
            out = np.full((len(t),) + a.shape[1:], np.nan, np.float32)
            out[pos >= 0] = np.asarray(a[pos[pos >= 0]], np.float32)
            return out

        near = [ch.index(v) for v in NEAR]

        def weather(e, pos):
            a = at(inp[:, ext.index(e)][:, near], pos)
            return np.c_[a, np.hypot(a[:, 2], a[:, 3]), np.hypot(a[:, 4], a[:, 5])]

        yb = {}
        for tgt, exts in EXT.items():
            yb[f"weather (issue)|{tgt}"] = np.concatenate([weather(e, pos_issue) for e in exts], axis=1)
            yb[f"actual weather|{tgt}"] = np.concatenate([weather(e, pos_now) for e in exts], axis=1)
            yb[f"tokens (raw)|{tgt}"] = np.concatenate([at(tok[:, ext.index(e)], pos_issue) for e in exts], axis=1)
            g = gfs_area(Path("data/gfs") / f"gfs_texas_{y}.nc", t, BOX[tgt])          # T2M, QV2M, U10, V10, SW net, speed
            ds = xr.open_dataset(Path("data/gfs") / f"gfs_texas_{y}.nc")
            yb[f"GFS|{tgt}"] = np.c_[g, gfs_dswrf(ds, t, BOX[tgt])]
            if gfull is not None:
                vt, gl = pd.DatetimeIndex(gfull["valid_time"]), gfull["lead_hours"]
                pick = np.full(len(t), -1)
                for want in (27, 24):
                    ix = pd.Series(np.where(gl == want)[0], index=vt[gl == want])
                    gp = ix[~ix.index.duplicated()].reindex(t).to_numpy()
                    pick[~np.isnan(gp)] = gp[~np.isnan(gp)].astype(int)
                a = gfull[f"{tgt}_mean"]
                yb[f"full GFS state (raw)|{tgt}"] = np.where((pick >= 0)[:, None], a[np.clip(pick, 0, None)], np.nan).astype(np.float32)
        for k, v in yb.items():
            blocks.setdefault(k, []).append(v)
        targets = targets.append(t)
    blocks = {k: np.concatenate(v) for k, v in blocks.items()}

    # ---- hourly rows ----
    ts = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31 23:00", freq="h")
    df = pd.DataFrame({"timestamp": ts})
    lt = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("US/Central").dt.tz_localize(None)
    hol = USFederalHolidayCalendar().holidays(f"{YEARS[0] - 1}-01-01", f"{YEARS[-1] + 1}-12-31")
    df["hour"], df["day_of_week"], df["day_of_year"], df["month"] = lt.dt.hour, lt.dt.dayofweek, lt.dt.dayofyear, lt.dt.month
    df["is_weekend"] = (lt.dt.dayofweek >= 5).astype(int)
    df["is_holiday"] = lt.dt.normalize().isin(hol).astype(int)
    df["cos_zenith"] = cos_zenith(ts, 31.5, -102.75)
    year = lt.dt.year.to_numpy()
    week = (lt.dt.isocalendar().year.astype(str) + "-" + lt.dt.isocalendar().week.astype(str)).to_numpy()
    ti = targets.get_indexer(df["timestamp"].dt.round("3h"))

    wi, G, F, T, A = "weather (issue)", "GFS", "full GFS state", "tokens", "actual weather"
    combos = [[], [wi], [wi, G], [wi, G, T], [wi, G, A]]
    if gfull is not None:
        combos += [[wi, G, F], [wi, G, F, T]]
    rows, preds = [], []
    for tgt in args.targets:
        s = gen[tgt].clip(lower=0)
        cap = s.rolling("90D", min_periods=24 * 60).quantile(0.99)
        cap_t = cap.reindex(df["timestamp"] - pd.Timedelta(hours=24)).to_numpy()
        cf = s.reindex(df["timestamp"]).to_numpy() / cap_t
        ramp = np.abs(cf - pd.Series(cf).shift(3).to_numpy())
        cal = CALENDAR + (["cos_zenith"] if tgt == "solar" else [])
        hb = {k.split("|")[0]: v for k, v in blocks.items() if k.endswith(f"|{tgt}")}
        ok = (ti >= 0) & np.isfinite(cf) & (cap_t > 0)
        for v in hb.values():
            ok &= ~np.isnan(v[np.clip(ti, 0, None)]).any(axis=1)
        for ty in (2023, 2024):
            tr, te = ok & (year < ty), ok & (year == ty)
            train_times = np.zeros(len(targets), bool)
            train_times[np.unique(ti[tr])] = True
            fb = {k: v for k, v in hb.items() if not k.endswith("(raw)")}
            fb[T] = pca(hb["tokens (raw)"], train_times, 50)[0]
            if gfull is not None:
                fb[F] = pca(hb["full GFS state (raw)"], train_times, 50)[0]
            big = ramp[te] >= np.nanquantile(ramp[te], 0.95)
            print(f"\n[{tgt}, test {ty}] train {tr.sum()} h, test {te.sum()} h; mean CF {np.mean(cf[te]):.3f}, "
                  f"test capacity proxy {np.nanmean(cap_t[te]):.0f} MW", flush=True)
            for combo in combos:
                X = df[cal].copy()
                for b in combo:
                    X = pd.concat([X, pd.DataFrame(fb[b][np.clip(ti, 0, None)], index=df.index).add_prefix(f"{b}_")], axis=1)
                m = CatBoostRegressor(iterations=args.iterations, learning_rate=0.05, depth=6, random_seed=7, verbose=False,
                                      thread_count=args.threads, train_dir=str(args.out_dir / "catboost_info"))
                m.fit(Pool(X[tr], cf[tr]))
                p = m.predict(X[te])
                name = "calendar" + "".join(f" + {b}" for b in combo)
                ae = 100 * np.abs(p - cf[te])
                rows.append({"target": tgt, "test_year": ty, "feature_set": name, "MAE_pct_cap": ae.mean(), "MAE_ramp_pct_cap": ae[big].mean()})
                preds.append(pd.DataFrame({"target": tgt, "test_year": ty, "feature_set": name, "row": np.where(te)[0],
                                           "actual": cf[te], "pred": p, "week": week[te], "ramp": big}))
                print(f"  {name:60s} MAE {ae.mean():5.2f}%  ramps {ae[big].mean():5.2f}%", flush=True)
    pr = pd.concat(preds, ignore_index=True)
    pr.to_parquet(args.out_dir / "renewables_predictions.parquet", index=False)
    pd.DataFrame(rows).to_csv(args.out_dir / "renewables_summary.csv", index=False)

    # ---- contrasts: each set vs GFS (3), tokens vs full state, and the headroom to the ceiling ----
    pr["ae"] = 100 * (pr["pred"] - pr["actual"]).abs()
    pairs = [("calendar + weather (issue) + GFS + actual weather", "calendar + weather (issue) + GFS"),
             ("calendar + weather (issue) + GFS + tokens", "calendar + weather (issue) + GFS")]
    if gfull is not None:
        pairs += [("calendar + weather (issue) + GFS + full GFS state", "calendar + weather (issue) + GFS"),
                  ("calendar + weather (issue) + GFS + actual weather", "calendar + weather (issue) + GFS + full GFS state"),
                  ("calendar + weather (issue) + GFS + full GFS state + tokens", "calendar + weather (issue) + GFS + full GFS state")]
    out = []
    for (tgt, ty), g in pr.groupby(["target", "test_year"]):
        w = g.pivot_table(index="row", columns="feature_set", values="ae")
        wk = g.drop_duplicates("row").set_index("row")["week"].reindex(w.index).to_numpy()
        rp = g.drop_duplicates("row").set_index("row")["ramp"].reindex(w.index).to_numpy()
        for a, b in pairs:
            for sub, mask in (("all", np.ones(len(w), bool)), ("ramps", rp)):
                d, lo, hi = block_bootstrap((w[a] - w[b]).to_numpy()[mask], wk[mask])
                out.append({"target": tgt, "test_year": ty, "hours": sub, "model": a.replace("calendar + weather (issue) + GFS", "GFS"),
                            "minus": b.replace("calendar + weather (issue) + GFS", "GFS"), "dMAE_pct_cap": d, "lo": lo, "hi": hi,
                            "pct_of_minus_MAE": 100 * d / w[b].to_numpy()[mask].mean()})
    c = pd.DataFrame(out)
    c.to_csv(args.out_dir / "renewables_contrasts.csv", index=False)
    pd.set_option("display.width", 250)
    print(c.round(2).to_string(index=False))


def gfs_dswrf(ds, t, box):
    """Downward surface shortwave from the GFS file, area mean, 24 h forecast else 27 h -> [T, 1]."""
    vt, lead = pd.DatetimeIndex(ds.valid_time.values), ds.lead_hours.values
    pick = np.full(len(t), -1)
    for want in (27, 24):
        ix = pd.Series(np.where(lead == want)[0], index=vt[lead == want])
        pos = ix[~ix.index.duplicated()].reindex(t).to_numpy()
        pick[~np.isnan(pos)] = pos[~np.isnan(pos)].astype(int)
    r, c = (ds.lat.values >= box[0]) & (ds.lat.values <= box[1]), (ds.lon.values >= box[2]) & (ds.lon.values <= box[3])
    out = np.full((len(t), 1), np.nan, np.float32)
    ok = pick >= 0
    out[ok, 0] = ds["DSWRF_sfc"].values[pick[ok]][:, r][:, :, c].mean(axis=(1, 2))
    return out


if __name__ == "__main__":
    main()
