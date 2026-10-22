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
from analyze_global_year import WEATHER_VARS, ZONES, zone_windows

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
        targets = targets.append(t)
        print(f"{y}: {(pick >= 0).sum()}/{len(t)} target times have a GFS 24-27 h forecast")

        # Forecast check: 2 m temperature at the zone cells, K.
        truth = per_year[y]["weather (target)"][:, :, 0]
        for name, est in [("GFS point, 24-27 h", point[:, :, 0]), ("persistence (24 h earlier)", per_year[y]["weather (issue)"][:, :, 0])]:
            ok = ~np.isnan(est) & ~np.isnan(truth)
            err = est[ok] - truth[ok]
            skill.append({"year": y, "forecast": name, "T2M_RMSE_K": float(np.sqrt((err ** 2).mean())), "T2M_bias_K": float(err.mean()),
                          "T2M_MAE_K": float(np.abs(err).mean())})
        frames.append(pd.read_parquet(args.data_dir / f"modeling_table_{y}.parquet",
                                      columns=["timestamp", "timestamp_local", "zone", "demand_MW"] + [f for f in dm.CALENDAR_FEATURES if f != "zone"]))
    for name in ("weather (target)", "weather (issue)", "GFS point", "GFS area"):
        blocks[name] = np.concatenate([per_year[y][name] for y in args.years])
    skill = pd.DataFrame(skill)
    skill.to_csv(args.out_dir / "temperature_forecast_skill.csv", index=False)
    print("\n2 m temperature at the zone grid cells, against MERRA-2 (K):")
    print(skill.round(2).to_string(index=False))

    df = pd.concat(frames, ignore_index=True)
    ti = targets.get_indexer(df["timestamp"].dt.round("3h"))
    zi = df["zone"].map({z: k for k, z in enumerate(zn)}).to_numpy()
    y_all, zones = df["demand_MW"].to_numpy(), df["zone"].to_numpy()
    has_gfs = (ti >= 0) & ~np.isnan(blocks["GFS point"][np.clip(ti, 0, None), zi, 0])
    combos = [[], ["weather (target)"], ["weather (issue)"], ["GFS point"], ["GFS area"],
              ["weather (issue)", "GFS point"], ["weather (issue)", "GFS area"]]

    results = []
    for scheme in args.splits:
        groups = [(str(y), df["timestamp_local"].dt.year.to_numpy() == y) for y in args.years] if scheme != "year_ahead" else [("all", np.ones(len(df), bool))]
        for label, in_group in groups:
            sub = df[in_group]
            split = dm.assign_split(sub, scheme).to_numpy()
            tr = np.zeros(len(df), bool)
            te = np.zeros(len(df), bool)
            tr[in_group] = split == "train"
            te[in_group] = split == "test"
            tr &= has_gfs
            te &= has_gfs   # every feature set is scored on the same hours
            if not tr.any() or not te.any():
                continue
            print(f"\n[{scheme}, {label}] train {tr.sum()} rows, test {te.sum()} rows", flush=True)
            for combo in combos:
                X = df[dm.CALENDAR_FEATURES].copy()
                for b in combo:
                    tag = b.replace(" ", "_").replace("(", "").replace(")", "")
                    X = pd.concat([X, pd.DataFrame(blocks[b][np.clip(ti, 0, None), zi], columns=[f"{tag}_{n}" for n in NAMES], index=df.index)], axis=1)
                model = CatBoostRegressor(iterations=args.iterations, learning_rate=0.05, depth=6, random_seed=7, verbose=False, thread_count=-1,
                                          train_dir=str(args.out_dir / "catboost_info"))
                model.fit(Pool(X[tr], y_all[tr], cat_features=dm.CAT_FEATURES))
                pred = model.predict(X[te])
                byz = pd.Series(np.abs(pred - y_all[te]) / y_all[te] * 100).groupby(zones[te]).mean()
                name = "calendar" + "".join(f" + {b}" for b in combo)
                results.append({"split": scheme, "group": label, "feature_set": name, "mean_zone_mape_pct": byz.mean(),
                                **{f"mape_{z}": byz.get(z, np.nan) for z in zn}})
                print(f"  {name:45s} MAPE {byz.mean():5.2f}%   VT {byz.get('VT', np.nan):5.1f}%", flush=True)
                pd.DataFrame(results).to_csv(args.out_dir / "demand_gfs_comparison.csv", index=False)
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
