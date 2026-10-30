"""What weather do the most important token PCs encode? Uses the fold-2024 PCA (fitted on 2021-2023) from
price_benchmark.py --save-shap. For each chosen PC: its SHAP dependence (direction of its price effect), its
correlation with each of Prithvi's 160 input channels averaged over New England and over the wide box (both after
removing the month x time-of-day mean), and top-minus-bottom-decile composites of the issue-time input fields
(deseasonalised, in units of the model's input sigma). Writes results/price_token_shap/composites/."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent
D, S = root / "data", root / "results/price_token_shap/shap"
OUT = root / "results/price_token_shap/composites"
OUT.mkdir(parents=True, exist_ok=True)
PCS = [int(a) for a in sys.argv[1:]] or [23, 2, 19]
YEARS = (2021, 2022, 2023, 2024)

z = np.load(S / "token_pcs_2024.npz")
targets = pd.DatetimeIndex(z["targets"])
scores = z["scores"]
issue = targets - pd.Timedelta(hours=24)


def deseason(a, times):
    """Subtract the month x hour-of-day mean (over all four years) from a [T, ...] array."""
    key = pd.Series(times.month * 100 + times.hour)
    out = np.full_like(a, np.nan, dtype=np.float32)
    for _, idx in key.groupby(key).groups.items():
        idx = np.asarray(idx)
        out[idx] = a[idx] - np.nanmean(a[idx], axis=0)
    return out


# issue-time fields for every target time: wide-box 5 surface fields, New England 160-channel pixels (a subset),
# pooled 160 channels over ne and wide
meta = json.loads((D / "prithvi_tokens_ne_2024_lead24.json").read_text())
ch = meta["pooled"]["channels"]
ext = list(meta["pooled"]["extents"])
NE_CH = ["SLP", "T2M", "T_63", "T_53", "T_44", "H_51", "U_45", "V_45", "QV2M", "TQV", "CLOUD_56", "OMEGA_56", "LWGAB", "SWGNT"]
wide = np.full((len(targets), 5, 37, 42), np.nan, np.float32)
ne = np.full((len(targets), len(NE_CH), 16, 14), np.nan, np.float32)
pooled = np.full((len(targets), 2, 160), np.nan, np.float32)
for y in YEARS:
    stem = D / f"prithvi_tokens_ne_{y}_lead24"
    ts = pd.DatetimeIndex(json.loads((D / f"prithvi_tokens_ne_{y}_lead24.json").read_text())["timestamps"])
    pos = ts.get_indexer(issue)
    m = pos >= 0
    wide[m] = np.load(f"{stem}_wide_inputs5.npy", mmap_mode="r")[pos[m]]
    inp = np.load(f"{stem}_inputs.npy", mmap_mode="r")
    ne[m] = inp[pos[m]][:, [ch.index(c) for c in NE_CH]]
    pooled[m] = np.load(f"{stem}_pooled_inputs.npy", mmap_mode="r")[pos[m]][:, [ext.index("ne"), ext.index("wide")]]
ok = ~np.isnan(scores[:, 0]) & ~np.isnan(wide[:, 0, 0, 0]) & ~np.isnan(ne[:, 0, 0, 0])
print(f"{ok.sum()} target times with tokens and fields")
t = targets[ok]
sc, wide, ne, pooled = deseason(scores[ok], t), deseason(wide[ok], t), deseason(ne[ok], t), deseason(pooled[ok], t)
raw_sc = scores[ok]

# SHAP direction: correlation of the PC's value with its SHAP value on the test hours (both folds' models use their
# own PCA; the 2024 fold uses exactly these scores)
sh = pd.read_parquet(S / "shap_lmp_2024.parquet")
pos = targets.get_indexer(pd.DatetimeIndex(sh["timestamp"]).round("3h"))
rows, corr_rows = [], []
for k in PCS:
    v = scores[pos, k]
    good = ~np.isnan(v)
    s = sh[f"Prithvi tokens_{k}"].to_numpy()[good]
    hi = v[good] >= np.quantile(v[good], 0.9)
    lo = v[good] <= np.quantile(v[good], 0.1)
    # seasonality of the raw PC: share of its variance explained by month x hour means
    season_share = 1 - np.nanvar(sc[:, k]) / np.nanvar(raw_sc[:, k])
    month_mean = pd.Series(raw_sc[:, k]).groupby(t.month).mean()
    rows.append({"pc": k, "shap_vs_value_corr": np.corrcoef(v[good], s)[0, 1], "mean_shap_top_decile_$": s[hi].mean(),
                 "mean_shap_bottom_decile_$": s[lo].mean(), "seasonal_share_of_variance": season_share,
                 "peak_month": int(month_mean.idxmax()), "trough_month": int(month_mean.idxmin())})
    for e, name in ((0, "ne"), (1, "wide")):
        r = [np.corrcoef(sc[:, k], pooled[:, e, c])[0, 1] for c in range(160)]
        for c, rv in zip(ch, r):
            corr_rows.append({"pc": k, "extent": name, "channel": c, "r": rv})
    top, bot = sc[:, k] >= np.quantile(sc[:, k], 0.9), sc[:, k] <= np.quantile(sc[:, k], 0.1)
    np.savez(OUT / f"pc{k}_composite.npz", wide=wide[top].mean(0) - wide[bot].mean(0), ne=ne[top].mean(0) - ne[bot].mean(0),
             wide_channels=np.array(meta["wide"]["channels"]), ne_channels=np.array(NE_CH),
             wide_lats=np.array(meta["wide"]["pixel_lats"]), wide_lons=np.array(meta["wide"]["pixel_lons"]),
             ne_lats=np.array(meta["pixel_lats"]), ne_lons=np.array(meta["pixel_lons"]), n=int(top.sum()))
summary = pd.DataFrame(rows)
corr = pd.DataFrame(corr_rows)
summary.to_csv(OUT / "pc_summary.csv", index=False)
corr.to_csv(OUT / "pc_channel_correlations.csv", index=False)
pd.set_option("display.width", 200)
print(summary.round(3).to_string(index=False))
for k in PCS:
    for e in ("ne", "wide"):
        c = corr[(corr["pc"] == k) & (corr["extent"] == e)].reindex(columns=["channel", "r"])
        c = c.loc[c["r"].abs().sort_values(ascending=False).index].head(10)
        print(f"PC{k} vs {e} channels (deseasonalised):", ", ".join(f"{a} {b:+.2f}" for a, b in c.to_numpy()))


# How much of each (deseasonalised) PC is a linear function of the pooled input state? And does it drift by year?
from numpy.linalg import lstsq  # noqa: E402
X = np.concatenate([pooled[:, 0], pooled[:, 1]], axis=1)
X = X[:, np.nanstd(X, axis=0) > 0]
X = np.c_[np.ones(len(X)), (X - X.mean(0)) / X.std(0)]
tr = t.year <= 2023
for k in PCS:
    y = sc[:, k]
    b = lstsq(X[tr], y[tr], rcond=None)[0]
    r2 = 1 - ((y[~tr] - X[~tr] @ b) ** 2).sum() / ((y[~tr] - y[~tr].mean()) ** 2).sum()
    by_year = pd.Series(raw_sc[:, k]).groupby(t.year).mean().round(2).to_dict()
    print(f"PC{k}: R² from the pooled 2x160 input channels (fit 2021-23, scored 2024) {r2:.2f}; raw mean by year {by_year}")
