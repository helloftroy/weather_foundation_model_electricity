"""Where does the token gain happen? Splits the MAE reduction from adding Prithvi tokens to GFS (day-ahead hub price) by
weather regime of the target day, defined in advance from the same 24 h-ahead GFS 2 m temperature forecasts the models
see (no observed weather). Per local target day D: forecast daily mean and max T2M; regimes, in this priority order:
  cold   daily mean <= 10th percentile of the training days
  hot    daily max  >= 90th percentile of the training days
  ramp   |daily mean(D) - daily mean(D-1)| >= 90th percentile of the training days (fronts / swings)
  normal everything else
Percentiles come from each fold's training years (2021-22 for test 2023, 2021-23 for test 2024). ISO-NE: hub LMP,
GFS averaged over the eight load-zone grid points; models GFS vs GFS + New England tokens, and GFS + regional vs
+ tokens (results/regional/price_pooled_extents). ERCOT: hub average, GFS averaged over the four load-hub boxes;
GFS vs GFS + tokens, Uri fortnight out of training (results/ercot/without_uri). Writes results/price_regimes/."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from analyze_global_year import ZONES  # noqa: E402
from eval_tools import block_bootstrap  # noqa: E402

root = Path(__file__).resolve().parent.parent
c, OUT = root / "data/compact", root / "results/price_regimes"
OUT.mkdir(parents=True, exist_ok=True)
YEARS = (2021, 2022, 2023, 2024)
TX_BOXES = [(31.5, 34.5, -99.0, -95.0), (28.5, 31.0, -96.5, -94.0), (26.0, 31.0, -100.5, -96.5), (30.0, 34.0, -104.5, -99.0)]


def gfs_t2m(path, targets, cells):
    """24 h GFS forecast of 2 m temperature (27 h where no 24 h one) for each target time, mean over `cells` (K)."""
    ds = xr.open_dataset(path)
    vt, lead = pd.DatetimeIndex(ds.valid_time.values), ds.lead_hours.values
    pick = np.full(len(targets), -1)
    for want in (27, 24):  # the 24 h lead overwrites
        idx = pd.Series(np.where(lead == want)[0], index=vt[lead == want])
        pos = idx[~idx.index.duplicated()].reindex(targets).to_numpy()
        pick[~np.isnan(pos)] = pos[~np.isnan(pos)].astype(int)
    tmp = ds["TMP_2m"].values
    out = np.full(len(targets), np.nan)
    ok = pick >= 0
    out[ok] = np.nanmean(np.stack([tmp[pick[ok]][:, i, j] for i, j in cells], 1), axis=1)
    return pd.Series(out, index=targets)


def cells_points(ds, pts):
    lat, lon = ds.lat.values, ds.lon.values
    return [(int(np.abs(lat - a).argmin()), int(np.abs(lon - b).argmin())) for a, b in pts]


def cells_boxes(ds, boxes):
    lat, lon = ds.lat.values, ds.lon.values
    return [(i, j) for a, b, l0, l1 in boxes for i in np.where((lat >= a) & (lat <= b))[0] for j in np.where((lon >= l0) & (lon <= l1))[0]]


def forecast_series(stem, cell_fn, arg):
    parts = []
    for y in YEARS:
        p = root / "data/gfs" / f"{stem}_{y}.nc"
        t = pd.date_range(f"{y}-01-01", f"{y}-12-31 21:00", freq="3h")
        parts.append(gfs_t2m(p, t, cell_fn(xr.open_dataset(p), arg)))
    return pd.concat(parts)


def daily_regimes(t2m_3h, tz):
    """Forecast daily mean / max / day-over-day change by local day, and the regime per day for each test year."""
    loc = t2m_3h.copy()
    loc.index = loc.index.tz_localize("UTC").tz_convert(tz).tz_localize(None)
    d = loc.groupby(loc.index.normalize()).agg(["mean", "max", "count"])
    d = d[d["count"] >= 6]
    d["swing"] = d["mean"].diff().abs().where(d.index.to_series().diff() == pd.Timedelta(days=1))
    out = {}
    for test_year in (2023, 2024):
        tr = (d.index.year < test_year)
        thr = {"cold": d.loc[tr, "mean"].quantile(0.10), "hot": d.loc[tr, "max"].quantile(0.90),
               "ramp": d.loc[tr, "swing"].quantile(0.90)}
        r = pd.Series("normal", index=d.index)
        r[d["swing"] >= thr["ramp"]] = "ramp"
        r[d["max"] >= thr["hot"]] = "hot"
        r[d["mean"] <= thr["cold"]] = "cold"
        out[test_year] = (r, thr)
    return out


def split(pred, base, full, day_of_row, regimes):
    """pred: one hourly row per (row, feature_set) with test_year, actual, pred, week."""
    rows = []
    for y, g in pred.groupby("test_year"):
        g = g.assign(ae=(g["pred"] - g["actual"]).abs())
        w = g.pivot_table(index="row", columns="feature_set", values="ae")[[base, full]].dropna()
        wk = g.drop_duplicates("row").set_index("row")["week"].reindex(w.index).to_numpy()
        reg, thr = regimes[y]
        lab = reg.reindex(day_of_row.reindex(w.index).to_numpy()).fillna("no forecast").to_numpy()
        diff = (w[full] - w[base]).to_numpy()
        total_gain = -diff.sum()
        for name in ("cold", "hot", "ramp", "normal", "all"):
            m = np.ones(len(w), bool) if name == "all" else lab == name
            if m.sum() == 0:
                continue
            d, lo, hi = block_bootstrap(diff[m], wk[m])
            b = w[base].to_numpy()[m].mean()
            rows.append({"test_year": y, "regime": name, "hours": int(m.sum()), "share_of_hours": m.mean(),
                         "MAE_base": b, "MAE_tokens": w[full].to_numpy()[m].mean(), "dMAE": d, "lo": lo, "hi": hi,
                         "reduction_pct": -100 * d / b, "red_lo": -100 * hi / b, "red_hi": -100 * lo / b,
                         "share_of_total_gain": -diff[m].sum() / total_gain})
        rows[-1]["thresholds_K"] = {k: round(float(v), 2) for k, v in thr.items()}
    return pd.DataFrame(rows)


# ---- ISO-NE: rebuild price_benchmark.py's hourly row order (years 2021-2024, hours with a hub price, sorted) ----
hours = []
for y in YEARS:
    ts = pd.read_parquet(root / f"data/modeling_table_{y}.parquet", columns=["timestamp", "timestamp_local"]).groupby("timestamp").first()
    lmp = pd.read_parquet(c / f"isone_dayahead_hourly_lmp_{y}.parquet")
    lmp["timestamp"] = pd.to_datetime(lmp["timestamp_parsed"]).astype("datetime64[ns]")
    hours.append(ts.join(lmp[lmp["location"] == "HUB"].set_index("timestamp")["lmp_total"], how="inner").reset_index())
ne_rows = pd.concat(hours, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
p = pd.read_parquet(root / "results/regional/price_pooled_extents/price_predictions.parquet")
chk = p.drop_duplicates("row").set_index("row")["actual"]
assert np.allclose(chk.to_numpy(), ne_rows.loc[chk.index, "lmp_total"].to_numpy()), "ISO-NE row order mismatch"
ne_reg = daily_regimes(forecast_series("gfs_ne", cells_points, list(ZONES.values())), "US/Eastern")
ne_day = ne_rows["timestamp_local"].dt.normalize()
PRE = "calendar + past values + known gas + weather (issue) + "
res = []
for label, base, full in (("ISO-NE: GFS -> + tokens", "GFS area", "GFS area + tokens ne"),
                          ("ISO-NE: GFS + regional -> + tokens", "GFS area + GFS regional", "GFS area + GFS regional + tokens ne")):
    res.append(split(p, PRE + base, PRE + full, ne_day, ne_reg).assign(comparison=label))

# ---- ERCOT: the hub columns' hourly index, as in ercot_price_benchmark.py ----
pr = pd.concat([pd.read_parquet(c / f"ercot_dam_spp_{y}.parquet") for y in YEARS])
pr = pr[pr["settlement_point"] == "HB_HUBAVG"]
tx_rows = pd.read_parquet(root / "results/ercot/without_uri/ercot_price_predictions.parquet")
tx_rows = tx_rows[tx_rows["hub"] == "HB_HUBAVG"]
all_hubs = pd.concat([pd.read_parquet(c / f"ercot_dam_spp_{y}.parquet") for y in YEARS])
idx = all_hubs.pivot_table(index="timestamp", columns="settlement_point", values="price").index
local = all_hubs.drop_duplicates("timestamp").set_index("timestamp")["timestamp_local"].reindex(idx)
price = pr.drop_duplicates("timestamp").set_index("timestamp")["price"].reindex(idx)
chk = tx_rows.drop_duplicates("row").set_index("row")["actual"]
assert np.allclose(chk.to_numpy(), price.to_numpy()[chk.index]), "ERCOT row order mismatch"
tx_reg = daily_regimes(forecast_series("gfs_texas", cells_boxes, TX_BOXES), "US/Central")
tx_day = pd.Series(pd.DatetimeIndex(local.to_numpy()).normalize(), index=range(len(idx)))
B = "calendar + past prices + Henry Hub + weather (issue) + GFS area"
res.append(split(tx_rows, B, B + " + Prithvi tokens", tx_day, tx_reg).assign(comparison="ERCOT hub average: GFS -> + tokens"))

out = pd.concat(res, ignore_index=True)
out.to_csv(OUT / "gain_by_regime.csv", index=False)
pd.set_option("display.width", 250)
cols = ["comparison", "test_year", "regime", "hours", "share_of_hours", "MAE_base", "MAE_tokens", "dMAE", "lo", "hi",
        "reduction_pct", "share_of_total_gain"]
print(out[cols].round(3).to_string(index=False))
print(out.dropna(subset=["thresholds_K"])[["comparison", "test_year", "thresholds_K"]].to_string(index=False))
