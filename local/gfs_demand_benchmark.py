"""Demand with real 24 h weather forecasts (GFS) against stale and actual
weather. No Prithvi here; this sets the bar the Prithvi decoder has to clear.

Times. Every row is lined up on its TARGET hour T. For each 3-hourly target
time the GFS forecast is the run at T - 24 h (forecast hour 24) or T - 27 h
(forecast hour 27), whichever exists, i.e. the latest run issued at least
24 h before T. This matches the lead-24 Prithvi setup (both ignore the few
hours a forecast takes to be published).

Feature blocks, per zone (all five variables plus wind speed):
    weather (target)  MERRA-2 at the zone's grid cell at T: perfect
                      knowledge, the ceiling
    weather (issue)   MERRA-2 at the zone's grid cell at T - 24 h:
                      persistence, what you know without a forecast
    GFS point         GFS forecast at the 0.25-degree cell nearest the zone
                      point
    GFS area          GFS forecast averaged over the zone's 3 x 3 Prithvi
                      patch window (the area the Prithvi decoder features
                      use), so the two forecasts can be compared like for like

Variables: T2M = TMP 2 m; QV2M = SPFH 2 m; U10M, V10M = 10 m winds;
SWGNT = DSWRF - USWRF at the surface. GFS shortwave is an average over the
preceding 6 h (hour 24) or 3 h (hour 27), MERRA-2 an hourly value, so the
two solar numbers are not the same quantity.

With --prithvi-dir (the lead-24 Prithvi files), also:
    Prithvi point     the decoder's 24 h forecast at the zone's MERRA-2 cell
    Prithvi area      the decoder's forecast averaged over the same window
    true area         the actual MERRA-2 values over the same window at T
                      (from the Prithvi run's own inputs: the sample issued
                      at T holds the truth at T). The fair area ceiling
The decoder works in normalised units; they are converted back with one
scale and offset per variable, recovered from the data (the model's input
pixel on a zone's cell matches the weather table exactly, correlation 1).

With --load-lags, the feature sets instead all include the zone's own
demand 24 h and 168 h (one week) before the target hour. Both are known when
a day-ahead forecast is made, so this asks what the forecasts add on top of
recent load.

Also reports GFS 2 m temperature error against MERRA-2 at the zone cells,
next to persistence, as a direct forecast-quality check.

Splits: blocked and chronological within each year, and year-ahead (train
on every year but the last, test on the last). CatBoost as elsewhere: one
pooled model with zone as a category, fixed 3,000 trees.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from catboost import CatBoostRegressor, Pool

import train_catboost_comparisons as dm
from analyze_global_year import WEATHER_VARS, ZONES, pca, per_zone_tokens, zone_windows
from eval_tools import hard_day_labels, summarize

PATCH_LAT = np.arange(40.25, 47.26, 1.0)               # Prithvi token box patch centres (see the .json sidecars)
PATCH_LON = -74.6875 + 1.25 * np.arange(7)
GFS_VARS = {"T2M": "TMP_2m", "QV2M": "SPFH_2m", "U10M": "UGRD_10m", "V10M": "VGRD_10m", "SWGNT": "SW_net_sfc"}
NAMES = WEATHER_VARS + ["wind_speed"]


def gfs_for_targets(path, targets):
    """GFS fields for each target time: the lead-24 forecast if there is one, else lead 27."""
    ds = xr.open_dataset(path)
    vt = pd.DatetimeIndex(ds.valid_time.values)
    lead = ds.lead_hours.values
    pick = np.full(len(targets), -1)
    for want in (27, 24):  # 24 overwrites 27 where both exist
        idx = pd.Series(np.where(lead == want)[0], index=vt[lead == want])
        idx = idx[~idx.index.duplicated()]
        pos = idx.reindex(targets).to_numpy()
        ok = ~np.isnan(pos)
        pick[ok] = pos[ok].astype(int)
    return ds, pick


def zone_features(ds, pick, targets, zn):
    """[T, zone, 6] at the nearest cell and over each zone's patch window; NaN where no forecast."""
    windows = zone_windows(PATCH_LAT, PATCH_LON)
    lat, lon = ds.lat.values, ds.lon.values
    fields = {v: ds[g].values for v, g in GFS_VARS.items()}                 # [sample, lat, lon]
    point = np.full((len(targets), len(zn), len(NAMES)), np.nan, np.float32)
    area = np.full_like(point, np.nan)
    ok = pick >= 0
    for k, z in enumerate(zn):
        la, lo = ZONES[z]
        i, j = int(np.abs(lat - la).argmin()), int(np.abs(lon - lo).argmin())
        r, c = windows[z]
        lat_lo, lat_hi = PATCH_LAT[r.start] - 0.5, PATCH_LAT[r.stop - 1] + 0.5
        lon_lo, lon_hi = PATCH_LON[c.start] - 0.625, PATCH_LON[c.stop - 1] + 0.625
        rows = (lat >= lat_lo) & (lat <= lat_hi)
        cols = (lon >= lon_lo) & (lon <= lon_hi)
        for m, v in enumerate(WEATHER_VARS):
            f = fields[v][pick[ok]]
            point[ok, k, m] = f[:, i, j]
            area[ok, k, m] = f[:, rows][:, :, cols].mean(axis=(1, 2))
        point[ok, k, 5] = np.hypot(point[ok, k, 2], point[ok, k, 3])
        area[ok, k, 5] = np.hypot(area[ok, k, 2], area[ok, k, 3])
    return point, area


def prithvi_features(data_dir, y, targets, weather, zn):
    """Decoder forecast (point, area) and true area weather for each target time, physical units; NaN if no sample."""
    import json
    stem = data_dir / f"prithvi_tokens_ne_{y}_lead24"
    meta = json.loads(stem.with_suffix(".json").read_text())
    issue = pd.DatetimeIndex(meta["timestamps"])
    plat, plon = np.array(meta["pixel_lats"]), np.array(meta["pixel_lons"])
    inp = np.load(data_dir / meta["fields"]["inputs"]["file"], mmap_mode="r")
    rec = np.load(data_dir / meta["fields"]["recon_sfc"]["file"], mmap_mode="r")
    ich = [meta["fields"]["inputs"]["channels"].index(v) for v in WEATHER_VARS]
    rch = [meta["fields"]["recon_sfc"]["channels"].index(v) for v in WEATHER_VARS]
    windows = zone_windows(PATCH_LAT, PATCH_LON)
    cells = {z: (int(np.abs(plat - la).argmin()), int(np.abs(plon - lo).argmin())) for z, (la, lo) in ZONES.items()}

    # Normalised -> physical: fit value = mu + sd * z on the zone cells, where the model's
    # input pixel and the weather table are the same MERRA-2 cell.
    w_issue = weather.reindex(issue)
    scale = []
    for m, v in enumerate(WEATHER_VARS):
        xs = np.concatenate([np.asarray(inp[:, ich[m], i, j], np.float64) for i, j in cells.values()])
        ys = np.concatenate([w_issue[f"{z}_{v}"].to_numpy() for z in cells])
        ok = ~np.isnan(ys)
        sd, mu = np.polyfit(xs[ok], ys[ok], 1)
        r = np.corrcoef(xs[ok], ys[ok])[0, 1]
        if r < 0.9999:
            raise SystemExit(f"{y} {v}: model input and weather table do not match (correlation {r:.5f}); check the grids.")
        scale.append((mu, sd))
    mu = np.array([a for a, _ in scale], np.float32)
    sd = np.array([b for _, b in scale], np.float32)

    def physical(a):  # [..., 5] normalised -> physical, plus wind speed
        a = a * sd + mu
        return np.concatenate([a, np.hypot(a[..., 2], a[..., 3])[..., None]], axis=-1)

    n_t, n_z = len(targets), len(zn)
    out = {k: np.full((n_t, n_z, len(NAMES)), np.nan, np.float32) for k in ("Prithvi point", "Prithvi area", "true area")}
    by_target = issue.get_indexer(targets - pd.Timedelta(hours=24))   # sample whose target is T
    by_issue = issue.get_indexer(targets)                              # sample issued at T: truth at T
    rec_c = np.asarray(rec[:, rch], np.float32)                        # [S, 5, lat_px, lon_px]
    inp_c = np.asarray(inp[:, ich], np.float32)
    for k, z in enumerate(zn):
        i, j = cells[z]
        r, c = windows[z]
        pr, pc = slice(r.start * 2, r.stop * 2), slice(c.start * 2, c.stop * 2)
        ok = by_target >= 0
        out["Prithvi point"][ok, k] = physical(rec_c[by_target[ok], :, i, j])
        out["Prithvi area"][ok, k] = physical(rec_c[by_target[ok], :, pr, pc].mean(axis=(2, 3)))
        ok = by_issue >= 0
        out["true area"][ok, k] = physical(inp_c[by_issue[ok], :, pr, pc].mean(axis=(2, 3)))
    return out


def merra_features(weather, at, zn):
    w = weather.reindex(at)
    return np.stack([np.stack([w[f"{z}_{v}"].to_numpy() for v in WEATHER_VARS] + [np.hypot(w[f"{z}_U10M"], w[f"{z}_V10M"]).to_numpy()], axis=1)
                     for z in zn], axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--splits", nargs="+", default=["blocked", "chronological", "year_ahead"])
    parser.add_argument("--test-years", type=int, nargs="+", default=None,
                        help="Year-ahead folds: test on each of these, training only on the earlier years in --years")
    parser.add_argument("--load-lags", action="store_true", help="Add demand 24 h and 168 h earlier to every feature set")
    parser.add_argument("--prithvi-dir", type=Path, default=None, help="Folder with prithvi_tokens_ne_YEAR_lead24* (adds Prithvi blocks)")
    parser.add_argument("--hard-days", action="store_true",
                        help="Year-ahead evaluation with Prithvi tokens (each zone's 3 x 3 patch window, 50 PCs fitted on training "
                             "times), weather-defined hard days, top-5%% system-demand hours and week-block bootstrap "
                             "against '+ weather (issue) + GFS area'; saves predictions and daily errors")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    zn = list(ZONES)
    c = args.data_dir / "compact"

    blocks, frames, skill = {}, [], []
    targets = pd.DatetimeIndex([])
    per_year = {}
    for y in args.years:
        t = pd.date_range(f"{y}-01-01", f"{y}-12-31 21:00", freq="3h")
        weather = pd.read_parquet(c / f"conventional_weather_{y}.parquet").set_index("timestamp")
        prev = c / f"conventional_weather_{y - 1}.parquet"
        if prev.exists():  # the first day's "24 h earlier" weather is in the previous year's table
            weather = pd.concat([pd.read_parquet(prev).set_index("timestamp"), weather])
            weather = weather[~weather.index.duplicated(keep="last")].sort_index()
        ds, pick = gfs_for_targets(args.data_dir / "gfs" / f"gfs_ne_{y}.nc", t)
        point, area = zone_features(ds, pick, t, zn)
        per_year[y] = {"weather (target)": merra_features(weather, t, zn),
                       "weather (issue)": merra_features(weather, t - pd.Timedelta(hours=24), zn),
                       "GFS point": point, "GFS area": area}
        if args.prithvi_dir:
            per_year[y].update(prithvi_features(args.prithvi_dir, y, t, weather, zn))
            # the true area at T - 24 h: persistence over the area
            sh = np.full_like(per_year[y]["true area"], np.nan)
            sh[8:] = per_year[y]["true area"][:-8]
            per_year[y]["true area (issue)"] = sh
        targets = targets.append(t)
        print(f"{y}: {(pick >= 0).sum()}/{len(t)} target times have a GFS 24-27 h forecast"
              + (f", {(~np.isnan(per_year[y]['Prithvi point'][:, 0, 0])).sum()} a Prithvi one" if args.prithvi_dir else ""))

        # Forecast check: 2 m temperature at the zone cells, K.
        checks = [("point", "GFS", point, per_year[y]["weather (target)"]),
                  ("point", "persistence (24 h earlier)", per_year[y]["weather (issue)"], per_year[y]["weather (target)"])]
        if args.prithvi_dir:
            pv = per_year[y]
            checks += [("point", "Prithvi decoder", pv["Prithvi point"], pv["weather (target)"]),
                       ("area", "GFS", area, pv["true area"]), ("area", "Prithvi decoder", pv["Prithvi area"], pv["true area"]),
                       ("area", "persistence (24 h earlier)", pv["true area (issue)"], pv["true area"])]
        common = np.ones(point.shape[:2], bool)
        for *_, est, truth in checks:
            common &= ~np.isnan(est[:, :, 0]) & ~np.isnan(truth[:, :, 0])
        for where, name, est, truth in checks:   # every forecast scored on the same times
            err = est[:, :, 0][common] - truth[:, :, 0][common]
            skill.append({"year": y, "where": where, "forecast": name, "n": int(common.sum()), "T2M_RMSE_K": float(np.sqrt((err ** 2).mean())),
                          "T2M_bias_K": float(err.mean()), "T2M_MAE_K": float(np.abs(err).mean())})
        frames.append(pd.read_parquet(args.data_dir / f"modeling_table_{y}.parquet",
                                      columns=["timestamp", "timestamp_local", "zone", "demand_MW"] + [f for f in dm.CALENDAR_FEATURES if f != "zone"]))
    for name in per_year[args.years[0]]:
        blocks[name] = np.concatenate([per_year[y][name] for y in args.years])
    if args.hard_days:   # final-layer tokens of the sample issued 24 h before each target, per zone window
        import json
        toks = []
        for y in args.years:
            stem = args.prithvi_dir / f"prithvi_tokens_ne_{y}_lead24"
            issue = pd.DatetimeIndex(json.loads(stem.with_suffix(".json").read_text())["timestamps"])
            a = per_zone_tokens(stem.with_suffix(".npy"), zone_windows(PATCH_LAT, PATCH_LON))
            t = pd.date_range(f"{y}-01-01", f"{y}-12-31 21:00", freq="3h")
            pos = issue.get_indexer(t - pd.Timedelta(hours=24))
            out = np.full((len(t), len(zn), a.shape[-1]), np.nan, np.float32)
            out[pos >= 0] = a[pos[pos >= 0]]
            toks.append(out)
            print(f"{y}: tokens for {(pos >= 0).sum()}/{len(t)} target times", flush=True)
        blocks["Prithvi tokens (raw)"] = np.concatenate(toks)
    skill = pd.DataFrame(skill)
    skill.to_csv(args.out_dir / "temperature_forecast_skill.csv", index=False)
    print("\n2 m temperature 24 h ahead against MERRA-2 (K); 'point' = the zone's grid cell, 'area' = the zone's patch window:")
    print(skill.round(2).to_string(index=False))

    df = pd.concat(frames, ignore_index=True)
    ti = targets.get_indexer(df["timestamp"].dt.round("3h"))
    zi = df["zone"].map({z: k for k, z in enumerate(zn)}).to_numpy()
    y_all, zones = df["demand_MW"].to_numpy(), df["zone"].to_numpy()
    lags = None
    if args.load_lags:
        series = df.set_index(["zone", "timestamp"])["demand_MW"]
        series = series[~series.index.duplicated()]
        lags = pd.DataFrame({f"demand_lag{h}h": series.reindex(pd.MultiIndex.from_arrays([df["zone"], df["timestamp"] - pd.Timedelta(hours=h)])).to_numpy()
                             for h in (24, 168)}, index=df.index)
        print(f"Load lags: {int(lags.isna().any(axis=1).sum())} rows without both lags (the first week)")
    # Score every feature set on the same hours: those where every block has a value.
    has_gfs = ti >= 0
    if lags is not None:
        has_gfs &= lags.notna().all(axis=1).to_numpy()
    for b in blocks.values():
        has_gfs &= ~np.isnan(b[np.clip(ti, 0, None), zi, 0])
    combos = [[], ["weather (target)"], ["weather (issue)"], ["GFS point"], ["GFS area"],
              ["weather (issue)", "GFS point"], ["weather (issue)", "GFS area"]]
    if args.prithvi_dir:
        combos += [["true area"], ["weather (issue)", "true area"], ["Prithvi point"], ["Prithvi area"],
                   ["weather (issue)", "Prithvi area"], ["weather (issue)", "GFS area", "Prithvi area"]]
    if args.hard_days:
        wi = "weather (issue)"
        combos = [[], [wi], [wi, "GFS area"], [wi, "Prithvi area"], [wi, "Prithvi tokens"], [wi, "Prithvi area", "Prithvi tokens"],
                  [wi, "GFS area", "Prithvi area"], [wi, "GFS area", "Prithvi tokens"], [wi, "GFS area", "Prithvi area", "Prithvi tokens"],
                  [wi, "true area"]]
        # hard days from the actual temperature over the zone areas, weighted by demand share (as for price)
        share = df.groupby("zone")["demand_MW"].mean().reindex(zn).to_numpy()
        t2m_sys = (blocks["true area"][:, :, 0] * (share / share.sum())).sum(axis=1)[np.clip(ti, 0, None)]
        system_demand = df.groupby("timestamp")["demand_MW"].transform("sum").to_numpy()
        iso = df["timestamp_local"].dt.isocalendar()
        week = (iso.year.astype(str) + "-" + iso.week.astype(str)).to_numpy()
        all_preds = []
    if lags is not None:
        combos = [[], ["weather (issue)"], ["weather (issue)", "GFS area"]]
        if args.prithvi_dir:
            combos += [["weather (issue)", "Prithvi area"], ["weather (issue)", "GFS area", "Prithvi area"], ["weather (issue)", "true area"]]

    results = []
    year = df["timestamp_local"].dt.year.to_numpy()
    for scheme in args.splits:
        if scheme == "year_ahead" and args.test_years:
            groups = [(f"test {ty}, train {min(args.years)}-{ty - 1}", None, ty) for ty in args.test_years]
        elif scheme == "year_ahead":
            groups = [("all", np.ones(len(df), bool), None)]
        else:
            groups = [(str(y), year == y, None) for y in args.years]
        for label, in_group, test_year in groups:
            tr = np.zeros(len(df), bool)
            te = np.zeros(len(df), bool)
            if test_year is not None:
                tr[:] = year < test_year
                te[:] = year == test_year
            else:
                split = dm.assign_split(df[in_group], scheme).to_numpy()
                tr[in_group] = split == "train"
                te[in_group] = split == "test"
            tr &= has_gfs
            te &= has_gfs   # every feature set is scored on the same hours
            if not tr.any() or not te.any():
                continue
            print(f"\n[{scheme}, {label}] train {tr.sum()} rows, test {te.sum()} rows", flush=True)
            fold_blocks = blocks
            if args.hard_days:
                train_times = np.zeros(len(targets), bool)
                train_times[np.unique(ti[tr])] = True
                fold_blocks = {k: v for k, v in blocks.items() if k != "Prithvi tokens (raw)"}
                fold_blocks["Prithvi tokens"] = pca(blocks["Prithvi tokens (raw)"], train_times, 50)[0]
                hard = hard_day_labels(df["timestamp_local"], t2m_sys, tr)
                top = te & (system_demand >= np.quantile(system_demand[te], 0.95))
                print(f"  hard days: thresholds {hard.attrs['thresholds_K']}; {hard.loc[te, 'cold_snap'].sum() // len(zn)} cold-snap and "
                      f"{hard.loc[te, 'heat_wave'].sum() // len(zn)} heat-wave test hours", flush=True)
            for combo in combos:
                X = df[dm.CALENDAR_FEATURES].copy()
                if lags is not None:
                    X = pd.concat([X, lags], axis=1)
                for b in combo:
                    tag = b.replace(" ", "_").replace("(", "").replace(")", "")
                    cols = [f"{tag}_{n}" for n in (range(50) if b == "Prithvi tokens" else NAMES)]
                    X = pd.concat([X, pd.DataFrame(fold_blocks[b][np.clip(ti, 0, None), zi], columns=cols, index=df.index)], axis=1)
                model = CatBoostRegressor(iterations=args.iterations, learning_rate=0.05, depth=6, random_seed=7, verbose=False, thread_count=-1,
                                          train_dir=str(args.out_dir / "catboost_info"))
                model.fit(Pool(X[tr], y_all[tr], cat_features=dm.CAT_FEATURES))
                pred = model.predict(X[te])
                byz = pd.Series(np.abs(pred - y_all[te]) / y_all[te] * 100).groupby(zones[te]).mean()
                name = "calendar" + (" + load lags" if lags is not None else "") + "".join(f" + {b}" for b in combo)
                results.append({"split": scheme, "group": label, "feature_set": name, "mean_zone_mape_pct": byz.mean(),
                                **{f"mape_{z}": byz.get(z, np.nan) for z in zn}})
                print(f"  {name:62s} MAPE {byz.mean():5.2f}%   VT {byz.get('VT', np.nan):5.1f}%", flush=True)
                if args.hard_days:
                    all_preds.append(pd.DataFrame({"row": np.where(te)[0], "test_year": test_year, "zone": zones[te], "feature_set": name,
                                                   "actual": y_all[te], "pred": pred, "week": week[te], "top": top[te],
                                                   "day": df["timestamp_local"].dt.normalize().to_numpy()[te],
                                                   "is_holiday": df["is_holiday"].to_numpy()[te],
                                                   "cold_snap": hard["cold_snap"].to_numpy()[te], "heat_wave": hard["heat_wave"].to_numpy()[te]}))
                pd.DataFrame(results).to_csv(args.out_dir / ("demand_with_load_lags.csv" if lags is not None else "demand_gfs_comparison.csv"), index=False)
    if args.hard_days:
        preds = pd.concat(all_preds, ignore_index=True)
        preds.to_parquet(args.out_dir / "demand_predictions.parquet", index=False)
        summary = summarize(preds, "calendar + weather (issue) + GFS area", pct=True)
        summary.to_csv(args.out_dir / "demand_summary.csv", index=False)
        preds["ape"] = (preds["pred"] - preds["actual"]).abs() / preds["actual"] * 100
        daily = preds.groupby(["test_year", "day", "feature_set"]).agg(mape=("ape", "mean"), cold_snap=("cold_snap", "max"),
                                                                        heat_wave=("heat_wave", "max"), is_holiday=("is_holiday", "max")).reset_index()
        daily.to_csv(args.out_dir / "demand_daily_mape.csv", index=False)
        pd.set_option("display.width", 250)
        for ty, g in summary.groupby("test_year"):
            print(f"\n== demand, test {ty}: MAPE (%) overall / cold snaps / heat waves / top 5% system-demand hours; "
                  f"difference vs '+ old weather + GFS' with 95% week-bootstrap interval (negative = better than GFS)")
            show = g[["feature_set", "MAE", "MAE_cold_snap", "MAE_heat_wave", "MAE_top5pct", "dMAE_vs_ref_all", "dMAE_vs_ref_all_ci",
                      "dMAE_vs_ref_cold_snap", "dMAE_vs_ref_cold_snap_ci", "dMAE_vs_ref_heat_wave", "dMAE_vs_ref_heat_wave_ci"]].copy()
            show["feature_set"] = show["feature_set"].str.replace("calendar", "", regex=False).str.strip(" +").replace("", "(calendar)")
            print(show.round(2).to_string(index=False))
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
