"""Figure 3 (price): A, GFS vs GFS + Prithvi tokens pooled over New England / the Northeast / the wide box;
B, GFS + regional weather, + tokens, + Prithvi's full input state, + both. Out-of-sample R² of the hourly day-ahead
hub LMP, 2023 and 2024 as two lines. Reads results/regional/price_pooled_extents/; writes
results/figures/fig3_price.{png,pdf} and fig3_price_table.csv (MAE, R², ΔMAE vs the panel's first model with a
95% week-block bootstrap). Bare panels; caption text lives in the reply / PROJECT_RESULTS.md."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eval_tools import block_bootstrap

plt.rcParams["font.family"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 8.5
root = Path(__file__).resolve().parent.parent
PRE = "calendar + past values + known gas + weather (issue) + "
G, R = "GFS area", "GFS area + GFS regional"
PANELS = {
    "A": [("GFS", G), ("+ tokens\nNew England", G + " + tokens ne"), ("+ tokens\nNortheast", G + " + tokens northeast"),
          ("+ tokens\nwide box", G + " + tokens wide")],
    "B": [("GFS +\nregional", R), ("+ tokens", R + " + tokens ne"), ("+ input\nstate", R + " + MERRA-2 fields ne"),
          ("+ input state\n+ tokens", R + " + MERRA-2 fields ne + tokens ne")],
}
Y23, Y24 = "#1f4e9c", "#d9822b"

p = pd.read_parquet(root / "results/regional/price_pooled_extents/price_predictions.parquet")
p["fs"] = p["feature_set"].str.replace(PRE, "", regex=False)
p["ae"] = (p["pred"] - p["actual"]).abs()
rows = []
for panel, models in PANELS.items():
    for y, g in p.groupby("test_year"):
        w = g.pivot_table(index="row", columns="fs", values="ae")
        week = g.drop_duplicates("row").set_index("row")["week"].reindex(w.index).to_numpy()
        ref = models[0][1]
        for label, fs in models:
            h = g[g["fs"] == fs]
            e = h["pred"] - h["actual"]
            r2 = 1 - (e ** 2).sum() / ((h["actual"] - h["actual"].mean()) ** 2).sum()
            m, lo, hi = block_bootstrap((w[fs] - w[ref]).to_numpy(), week) if fs != ref else (np.nan,) * 3
            rows.append({"panel": panel, "model": label.replace("\n", " "), "feature_set": fs, "test_year": y, "R2": r2,
                         "MAE": h["ae"].mean(), "dMAE_vs_first": m, "lo": lo, "hi": hi})
t = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, gridspec_kw={"wspace": 0.12})
for ax, (panel, models) in zip(axes, PANELS.items()):
    x = np.arange(len(models))
    for y, color, marker in ((2023, Y23, "o"), (2024, Y24, "s")):
        v = t[(t["panel"] == panel) & (t["test_year"] == y)].set_index("feature_set").loc[[m[1] for m in models], "R2"].to_numpy()
        ax.plot(x[:2], v[:2], color=color, lw=1.6, zorder=2)  # the added step
        ax.plot(x[1:], v[1:], color=color, lw=1.0, ls=(0, (1.5, 2)), zorder=2)  # alternatives, not further steps
        ax.scatter(x, v, s=34, color=color, marker=marker, zorder=3, edgecolor="white", linewidth=0.6)
        ax.text(x[-1] + 0.18, v[-1], str(y), color=color, va="center", ha="left", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in models], fontsize=7.5)
    ax.set_xlim(-0.35, len(models) - 0.35)
    ax.grid(axis="y", color="#ececec", lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(-0.02, 1.02, panel, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="right", va="bottom")
axes[0].set_ylabel("Out-of-sample R² (hourly day-ahead price)")
axes[0].set_ylim(0.6, 0.86)
out = root / "results/figures/fig3_price"
fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
t.drop(columns="feature_set").to_csv(root / "results/figures/fig3_price_table.csv", index=False)
pd.set_option("display.width", 200)
print(t.drop(columns="feature_set").round(3).to_string(index=False))
