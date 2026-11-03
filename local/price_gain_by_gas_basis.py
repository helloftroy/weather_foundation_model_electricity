"""Is the ISO-NE cold-day token gain concentrated when regional gas is constrained? Among forecast-cold days (regimes
as in price_gain_by_regime.py), splits hours by the Algonquin Citygate minus Henry Hub basis known before bidding:
the last weekly Algonquin quote dated on or before D-2 minus Henry Hub on that same date. High / low = above / at or
below the median (and, as a check, the upper quartile) of that basis over the training years' cold days. Token gain =
MAE(GFS + tokens) - MAE(baseline), with a 95% week-block bootstrap; the high-minus-low difference resamples weeks
jointly. Cold days of 2023 and 2024 are pooled (only 10 cold days in 2023) and also shown per year; all non-cold days
are split the same way as a control. Writes results/price_regimes/gain_by_gas_basis.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze_global_year import ZONES  # noqa: E402
from price_gain_by_regime import cells_points, daily_regimes, forecast_series, isone_rows  # noqa: E402

root = Path(__file__).resolve().parent.parent
g = root / "data/gas"
PRE = "calendar + past values + known gas + weather (issue) + "
COMPARISONS = (("GFS -> + tokens", "GFS area", "GFS area + tokens ne"),
               ("GFS + regional -> + tokens", "GFS area + GFS regional", "GFS area + GFS regional + tokens ne"))

hh = pd.read_excel(g / "RNGWHHDd.xls", sheet_name="Data 1", skiprows=2)
hh.columns = ["date", "henry_hub"]
hh = hh.dropna().assign(date=lambda d: pd.to_datetime(d["date"])).set_index("date")["henry_hub"].sort_index()
alg = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in sorted(g.glob("algonquin_weekly_points_*.csv"))])
alg = alg.drop_duplicates("date").set_index("date")["price"].sort_index()
basis_at_quote = alg - hh.reindex(alg.index, method="ffill")


def known_basis(days):
    """Basis of the last Algonquin quote dated <= D-2 (same-date Henry Hub)."""
    return basis_at_quote.reindex(pd.DatetimeIndex(days) - pd.Timedelta(days=2), method="ffill").to_numpy()


rows = isone_rows()
p = pd.read_parquet(root / "results/regional/price_pooled_extents/price_predictions.parquet")
reg = daily_regimes(forecast_series("gfs_ne", cells_points, list(ZONES.values())), "US/Eastern")
day = rows["timestamp_local"].dt.normalize()

# thresholds from the training years' cold days (per fold)
thr = {}
for y, (r, _) in reg.items():
    cold_train = r.index[(r.index.year < y) & (r == "cold")]
    b = known_basis(cold_train)
    thr[y] = {"median": float(np.nanmedian(b)), "upper quartile": float(np.nanquantile(b, 0.75))}


def boot(diff, week, mask_hi, n=2000, seed=0):
    """Means and 95% intervals of the gain (-diff) in high, low, and high minus low, resampling weeks jointly."""
    weeks, inv = np.unique(week, return_inverse=True)
    s_hi = np.bincount(inv, weights=diff * mask_hi, minlength=len(weeks))
    n_hi = np.bincount(inv, weights=mask_hi.astype(float), minlength=len(weeks))
    s_lo = np.bincount(inv, weights=diff * ~mask_hi, minlength=len(weeks))
    n_lo = np.bincount(inv, weights=(~mask_hi).astype(float), minlength=len(weeks))
    pick = np.random.default_rng(seed).integers(0, len(weeks), size=(n, len(weeks)))
    hi = -s_hi[pick].sum(1) / n_hi[pick].sum(1)
    lo = -s_lo[pick].sum(1) / n_lo[pick].sum(1)
    q = lambda a: (float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5)))  # noqa: E731
    return {"gain_high": -diff[mask_hi].mean(), "gain_high_ci": q(hi), "gain_low": -diff[~mask_hi].mean(), "gain_low_ci": q(lo),
            "high_minus_low": -diff[mask_hi].mean() + diff[~mask_hi].mean(), "high_minus_low_ci": q(hi - lo)}


out = []
for label, base, full in COMPARISONS:
    parts = []
    for y, gy in p.groupby("test_year"):
        gy = gy.assign(ae=(gy["pred"] - gy["actual"]).abs())
        w = gy.pivot_table(index="row", columns="feature_set", values="ae")[[PRE + base, PRE + full]].dropna()
        w.columns = ["base", "full"]
        w["week"] = gy.drop_duplicates("row").set_index("row")["week"].reindex(w.index)
        d = day.reindex(w.index).to_numpy()
        w["regime"] = reg[y][0].reindex(d).fillna("no forecast").to_numpy()
        w["basis"] = known_basis(d)
        w["test_year"] = y
        for k, v in thr[y].items():
            w[f"high_{k}"] = w["basis"] > v
        parts.append(w)
    w = pd.concat(parts)
    for subset, m in (("cold days", w["regime"] == "cold"), ("non-cold days", w["regime"] != "cold")):
        for years in ((2023, 2024), (2023,), (2024,)):
            s = w[m & w["test_year"].isin(years)]
            for k in ("median", "upper quartile"):
                hi = s[f"high_{k}"].to_numpy()
                if hi.all() or not hi.any():
                    continue
                r = boot((s["full"] - s["base"]).to_numpy(), s["week"].astype(str).to_numpy(), hi)
                out.append({"comparison": label, "days": subset, "test_years": "+".join(map(str, years)), "split": k,
                            "hours_high": int(hi.sum()), "hours_low": int((~hi).sum()),
                            "days_high": s.loc[hi].index.size // 24, "MAE_base_high": s.loc[hi, "base"].mean(),
                            "MAE_base_low": s.loc[~hi, "base"].mean(), "basis_high_mean": s.loc[hi, "basis"].mean(),
                            "basis_low_mean": s.loc[~hi, "basis"].mean(), **r,
                            "pct_high": 100 * r["gain_high"] / s.loc[hi, "base"].mean(),
                            "pct_low": 100 * r["gain_low"] / s.loc[~hi, "base"].mean()})
res = pd.DataFrame(out)
res.to_csv(root / "results/price_regimes/gain_by_gas_basis.csv", index=False)
fmt = lambda v, ci: f"{v:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"  # noqa: E731
res["high"] = [fmt(a, b) for a, b in zip(res["gain_high"], res["gain_high_ci"])]
res["low"] = [fmt(a, b) for a, b in zip(res["gain_low"], res["gain_low_ci"])]
res["high-low"] = [fmt(a, b) for a, b in zip(res["high_minus_low"], res["high_minus_low_ci"])]
pd.set_option("display.width", 250)
print("thresholds ($/MMBtu):", {y: {k: round(v, 2) for k, v in t.items()} for y, t in thr.items()})
print(res[["comparison", "days", "test_years", "split", "hours_high", "hours_low", "basis_high_mean", "basis_low_mean",
           "MAE_base_high", "MAE_base_low", "high", "low", "high-low", "pct_high", "pct_low"]].round(2).to_string(index=False))
