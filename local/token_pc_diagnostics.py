"""Companion to token_pc_composites.py: separates what each top token PC encodes from when the price model uses it.
(1) PC score vs its own SHAP value on the 2024 test hours: Pearson and Spearman correlation, mean SHAP by PC-score
decile, and the share of SHAP variance explained by the PC's own value (decile means; the rest is interactions with
other features). (2) Correlation of each deseasonalised PC score with a handful of interpretable deseasonalised
diagnostics (pooled Prithvi inputs at issue time), over all 2021-2024 target times. Writes
results/price_token_shap/composites/pc_shap_profile.csv and pc_diagnostics.csv."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


root = Path(__file__).resolve().parent.parent
D, S, OUT = root / "data", root / "results/price_token_shap/shap", root / "results/price_token_shap/composites"
PCS = [23, 2, 19]
# (label, extent, channel) or (label, (extent_a, extent_b), channel) for a difference a - b.
DIAG = [("T2M, New England", "ne", "T2M"), ("T2M, wide box", "wide", "T2M"),
        ("T2M gradient, Quebec–Mar minus Southeast", ("Quebec-Mar", "Southeast"), "T2M"),
        ("T2M, Great Lakes minus New England", ("GreatLakes", "ne"), "T2M"),
        ("SLP, New England", "ne", "SLP"), ("SLP, wide box", "wide", "SLP"),
        ("Height ≈500 hPa, New England", "ne", "H_51"), ("Height ≈500 hPa, wide box", "wide", "H_51"),
        ("T ≈600 hPa, wide box", "wide", "T_53"), ("T ≈250 hPa, New England", "ne", "T_44"),
        ("U wind ≈290 hPa, New England", "ne", "U_45"), ("V wind ≈290 hPa, New England", "ne", "V_45"),
        ("V wind 10 m, New England", "ne", "V10M"), ("Column water vapour, New England", "ne", "TQV"),
        ("Cloud ≈500 hPa, New England", "ne", "CLOUD_51")]



def deseason(a, times):
    """Subtract the month x hour-of-day mean (as in token_pc_composites.py)."""
    key = pd.Series(times.month * 100 + times.hour)
    out = np.full_like(a, np.nan, dtype=np.float32)
    for _, idx in key.groupby(key).groups.items():
        idx = np.asarray(idx)
        out[idx] = a[idx] - np.nanmean(a[idx], axis=0)
    return out


z = np.load(S / "token_pcs_2024.npz")
targets = pd.DatetimeIndex(z["targets"])
scores = z["scores"]
issue = targets - pd.Timedelta(hours=24)
meta = json.loads((D / "prithvi_tokens_ne_2024_lead24.json").read_text())
ch, ext = meta["pooled"]["channels"], list(meta["pooled"]["extents"])
pooled = np.full((len(targets), len(ext), len(ch)), np.nan, np.float32)
for y in (2021, 2022, 2023, 2024):
    stem = D / f"prithvi_tokens_ne_{y}_lead24"
    ts = pd.DatetimeIndex(json.loads(stem.with_suffix(".json").read_text())["timestamps"])
    pos = ts.get_indexer(issue)
    m = pos >= 0
    pooled[m] = np.load(f"{stem}_pooled_inputs.npy", mmap_mode="r")[pos[m]]
ok = ~np.isnan(scores[:, 0]) & ~np.isnan(pooled[:, 0, 0])
t = targets[ok]
sc, pooled = deseason(scores[ok], t), deseason(pooled[ok], t)

# (1) PC value vs SHAP on the test hours
sh = pd.read_parquet(S / "shap_lmp_2024.parquet")
pos = targets.get_indexer(pd.DatetimeIndex(sh["timestamp"]).round("3h"))
prof, summ = [], []
for k in PCS:
    v = scores[pos, k]
    g = ~np.isnan(v)
    v, s = v[g], sh[f"Prithvi tokens_{k}"].to_numpy()[g]
    dec = pd.qcut(v, 10, labels=False)
    by = pd.Series(s).groupby(dec)
    for d, grp in by:
        prof.append({"pc": k, "decile": int(d) + 1, "mean_shap": grp.mean(), "sd_shap": grp.std(),
                     "q10": grp.quantile(0.1), "q90": grp.quantile(0.9)})
    means = by.transform("mean").to_numpy()
    sign = np.sign(np.corrcoef(v, s)[0, 1])
    hi, lo = (dec == 9, dec == 0) if sign > 0 else (dec == 0, dec == 9)
    summ.append({"pc": k, "pearson": np.corrcoef(v, s)[0, 1], "spearman": pd.Series(v).corr(pd.Series(s), method="spearman"),
                 "shap_var_explained_by_own_value": 1 - np.var(s - means) / np.var(s),
                 "mean_shap_raising_decile": s[hi].mean(), "mean_shap_lowering_decile": s[lo].mean(),
                 "dSHAP_raising_minus_lowering": s[hi].mean() - s[lo].mean(), "mean_abs_shap": np.abs(s).mean()})
prof, summ = pd.DataFrame(prof), pd.DataFrame(summ)

# (2) PC score vs interpretable diagnostics (all target times, deseasonalised)
rows = []
for label, e, c in DIAG:
    ci = ch.index(c)
    x = pooled[:, ext.index(e[0]), ci] - pooled[:, ext.index(e[1]), ci] if isinstance(e, tuple) else pooled[:, ext.index(e), ci]
    for k in PCS:
        rows.append({"diagnostic": label, "pc": k, "r": np.corrcoef(sc[:, k], x)[0, 1]})
diag = pd.DataFrame(rows).pivot(index="diagnostic", columns="pc", values="r").reindex([d[0] for d in DIAG])[PCS]

summ.to_csv(OUT / "pc_shap_summary.csv", index=False)
prof.to_csv(OUT / "pc_shap_profile.csv", index=False)
diag.to_csv(OUT / "pc_diagnostics.csv")
pd.set_option("display.width", 200)
print(summ.round(3).to_string(index=False))
print(prof.pivot(index="decile", columns="pc", values="mean_shap")[PCS].round(2).to_string())
print(diag.round(2).to_string())
