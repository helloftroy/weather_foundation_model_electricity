"""Figure 3 (price), ISO-NE and ERCOT. A, ISO-NE: GFS vs GFS + Prithvi tokens pooled over New England / the
Northeast / the wide box; B, ISO-NE: GFS + regional weather, + tokens, + Prithvi's MERRA-2 input state, + the full GFS
forecast state (150 fields, box mean, PCA-50), + GFS state + tokens
(out-of-sample R² of the hourly day-ahead hub LMP, 2023 and 2024 as two lines). C, ERCOT: % reduction in MAE from
adding Prithvi tokens to GFS at each hub (Uri fortnight dropped from training), 95% week-block bootstrap. D, both
markets against the full GFS state (% MAE reduction vs GFS from tokens, the full GFS state as box mean or gridded
PCA-50, and box state + tokens; ISO-NE hub and the ERCOT hub average, from results/regional/price_gfs_full and
results/ercot/gfs_full). Relative (%) changes are used across
markets because their price levels differ. Reads results/regional/price_pooled_extents/ (A, D), results/regional/price_gfs_full/ (B) and results/ercot/without_uri/;
writes results/figures/fig3_price.{png,pdf}, fig3_price_table.csv (A, B) and fig3_price_markets_table.csv (C, D).
Bare panels; caption text lives in the reply / PROJECT_RESULTS.md."""
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
    "B": [("GFS +\nregional", R), ("+ tokens", R + " + tokens ne"), ("+ MERRA-2\nstate\n(issue time)", R + " + MERRA-2 fields ne"),
          ("+ full GFS\nforecast\nstate", R + " + GFS full state box"), ("+ GFS state\n+ tokens", R + " + GFS full state box + tokens ne")],
}
# B comes from the full-GFS-state run (results/regional/price_gfs_full; 12 fewer usable hours, where 0.5 deg GFS files
# are missing), so every column of B is compared on the same hours.
SOURCE = {"A": "results/regional/price_pooled_extents", "B": "results/regional/price_gfs_full"}
Y23, Y24 = "#1f4e9c", "#d9822b"



def load(d):
    q = pd.read_parquet(root / d / "price_predictions.parquet")
    q["fs"] = q["feature_set"].str.replace(PRE, "", regex=False)
    q["ae"] = (q["pred"] - q["actual"]).abs()
    return q


src = {k: load(d) for k, d in SOURCE.items()}
p = src["A"]
rows = []
for panel, models in PANELS.items():
    for y, g in src[panel].groupby("test_year"):
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

METRIC = sys.argv[1] if len(sys.argv) > 1 else "R2"  # "R2" (fig3_price) or "MAE" (fig3_price_mae)
fig = plt.figure(figsize=(7.2, 6.4))
outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.42)
top = outer[0].subgridspec(1, 2, wspace=0.12, width_ratios=[4, 5])
bottom = outer[1].subgridspec(1, 2, width_ratios=[1, 1], wspace=1.05)
axes = [fig.add_subplot(top[0])]
axes.append(fig.add_subplot(top[1], sharey=axes[0]))
plt.setp(axes[1].get_yticklabels(), visible=False)


def tidy(ax, label):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(-0.02, 1.02, label, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="right", va="bottom")


for ax, (panel, models) in zip(axes, PANELS.items()):
    x = np.arange(len(models))
    for y, color, marker in ((2023, Y23, "o"), (2024, Y24, "s")):
        v = t[(t["panel"] == panel) & (t["test_year"] == y)].set_index("feature_set").loc[[m[1] for m in models], METRIC].to_numpy()
        # Unconnected points: every column is an alternative added to the first; a thin line marks the baseline level.
        ax.axhline(v[0], color=color, lw=0.7, ls=(0, (3, 2)), alpha=0.6, zorder=1)
        ax.scatter(x, v, s=34, color=color, marker=marker, zorder=3, edgecolor="white", linewidth=0.6)
        ax.text(x[-1] + 0.18, v[-1], str(y), color=color, va="center", ha="left", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in models], fontsize=7.5)
    ax.set_xlim(-0.35, len(models) - 0.35)
    ax.grid(axis="y", color="#ececec", lw=0.6, zorder=0)
    tidy(ax, panel)
if METRIC == "R2":
    axes[0].set_ylabel("ISO-NE out-of-sample R²\n(hourly day-ahead price)")
    axes[0].set_ylim(0.6, 0.86)
else:
    axes[0].set_ylabel("ISO-NE MAE, $/MWh\n(hourly day-ahead price)")
    axes[0].set_ylim(6.4, 8.8)

# C, D: % MAE reduction from adding tokens to GFS (positive = tokens help), each relative to its own GFS model.
B_TX = "calendar + past prices + Henry Hub + weather (issue) + GFS area"
HUBS = [("HB_NORTH", "North"), ("HB_HOUSTON", "Houston"), ("HB_SOUTH", "South"), ("HB_WEST", "West"),
        ("HB_PAN", "Panhandle"), ("HB_HUBAVG", "Hub average")]


def gain(g, base, full, week_col="week"):
    g = g.assign(ae=(g["pred"] - g["actual"]).abs())
    w = g.pivot_table(index="row", columns="feature_set", values="ae")
    wk = g.drop_duplicates("row").set_index("row")[week_col].reindex(w.index).to_numpy()
    m, lo, hi = block_bootstrap((w[full] - w[base]).to_numpy(), wk)
    ref = w[base].mean()
    return {"MAE_GFS": ref, "MAE_tokens": w[full].mean(), "dMAE": m, "lo": lo, "hi": hi,
            "gain_pct": -100 * m / ref, "gain_lo": -100 * hi / ref, "gain_hi": -100 * lo / ref}


tx = pd.read_parquet(root / "results/ercot/without_uri/ercot_price_predictions.parquet")
tx = tx[tx["feature_set"].isin([B_TX, B_TX + " + Prithvi tokens"])]
m_rows = []
for (hub, y), g in tx.groupby(["hub", "test_year"]):
    m_rows.append({"market": "ERCOT", "hub": dict(HUBS)[hub], "test_year": y, **gain(g, B_TX, B_TX + " + Prithvi tokens")})
for y, g in p[p["fs"].isin([G, G + " + tokens ne"])].groupby("test_year"):
    g = g.drop(columns="feature_set").rename(columns={"fs": "feature_set"})
    m_rows.append({"market": "ISO-NE", "hub": "Internal hub", "test_year": y, **gain(g, G, G + " + tokens ne")})
mt = pd.DataFrame(m_rows)

axc = fig.add_subplot(bottom[0])
ypos = {name: i + (0.6 if name == "Hub average" else 0) for i, (_, name) in enumerate(HUBS)}
for y, color, marker, off in ((2023, Y23, "o", -0.15), (2024, Y24, "s", 0.15)):
    d = mt[(mt["market"] == "ERCOT") & (mt["test_year"] == y)]
    yy = d["hub"].map(ypos).to_numpy() + off
    axc.hlines(yy, d["gain_lo"], d["gain_hi"], color=color, lw=1.2, zorder=2)
    axc.scatter(d["gain_pct"], yy, s=30, color=color, marker=marker, zorder=3, edgecolor="white", linewidth=0.6, label=str(y))
axc.axhline(len(HUBS) - 1 - 0.2, color="#cccccc", lw=0.6)
axc.axvline(0, color="#555555", lw=0.8, zorder=2.5)
axc.text(0, 1.01, "no token benefit", transform=axc.get_xaxis_transform(), ha="center", va="bottom", fontsize=6,
         color="#555555")
axc.set_yticks(list(ypos.values()))
axc.set_yticklabels(list(ypos.keys()))
axc.invert_yaxis()
axc.set_xlabel("MAE reduction from adding Prithvi tokens\nto GFS, ERCOT hubs (%)")
axc.grid(axis="x", color="#ececec", lw=0.6, zorder=0)
axc.legend(frameon=False, fontsize=7.5, loc="lower right", handletextpad=0.2, borderaxespad=0.1)
tidy(axc, "C")

# D: both markets against the full GFS state (from the full-GFS runs, so every row of a market is on the same hours):
# % MAE reduction vs GFS from adding tokens, the full GFS state (box mean or gridded PCA-50), or the box state + tokens.
ROWS = [("tokens", "+ Prithvi tokens"), ("box", "+ full GFS state, box"), ("grid", "+ full GFS state, grid"),
        ("box+tokens", "+ GFS state (box) + tokens")]
FS = {"ISO-NE": (G, {"tokens": " + tokens ne", "box": " + GFS full state box", "grid": " + GFS full state grid",
                     "box+tokens": " + GFS full state box + tokens ne"}),
      "ERCOT": (B_TX, {"tokens": " + Prithvi tokens", "box": " + GFS full state", "grid": " + GFS full grid",
                       "box+tokens": " + GFS full state + Prithvi tokens"})}
iso = load("results/regional/price_gfs_full").drop(columns="feature_set").rename(columns={"fs": "feature_set"})
erc = pd.read_parquet(root / "results/ercot/gfs_full/ercot_price_predictions.parquet")
erc = erc[erc["hub"] == "HB_HUBAVG"]
d_rows = []
for mk, src_df in (("ISO-NE", iso), ("ERCOT", erc)):
    base, add = FS[mk]
    for y, g in src_df.groupby("test_year"):
        for key, _ in ROWS:
            d_rows.append({"market": mk, "row": key, "test_year": y, **gain(g, base, base + add[key])})
dt = pd.DataFrame(d_rows)
axd = fig.add_subplot(bottom[1])
ylab, ypos = [], []
for m_i, (mk, title) in enumerate((("ISO-NE", "ISO-NE"), ("ERCOT", "ERCOT hub average"))):
    y0 = m_i * (len(ROWS) + 1.3)
    axd.text(-0.02, y0 - 0.85, title, transform=axd.get_yaxis_transform(), ha="right", va="center", fontsize=8, fontweight="bold")
    for k, (key, lab) in enumerate(ROWS):
        ylab.append(lab)
        ypos.append(y0 + k)
        for y, color, marker, off in ((2023, Y23, "o", -0.17), (2024, Y24, "s", 0.17)):
            r = dt[(dt["market"] == mk) & (dt["row"] == key) & (dt["test_year"] == y)].iloc[0]
            axd.hlines(y0 + k + off, r["gain_lo"], r["gain_hi"], color=color, lw=1.2, zorder=2)
            axd.scatter(r["gain_pct"], y0 + k + off, s=26, color=color, marker=marker, zorder=3, edgecolor="white", linewidth=0.5)
axd.axvline(0, color="#555555", lw=0.8, zorder=2.5)
axd.set_yticks(ypos)
axd.set_yticklabels(ylab, fontsize=7.5)
axd.set_ylim(max(ypos) + 0.6, -1.4)
axd.set_xlabel("MAE reduction vs GFS (%)")
axd.grid(axis="x", color="#ececec", lw=0.6, zorder=0)
tidy(axd, "D")
out = root / ("results/figures/fig3_price" + ("" if METRIC == "R2" else "_mae"))
fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
t.drop(columns="feature_set").to_csv(root / "results/figures/fig3_price_table.csv", index=False)
pd.set_option("display.width", 200)
print(t.drop(columns="feature_set").round(3).to_string(index=False))
mt.to_csv(root / "results/figures/fig3_price_markets_table.csv", index=False)
dt.to_csv(root / "results/figures/fig3_price_fullstate_table.csv", index=False)
print(dt.round(2).to_string(index=False))
print(mt.round(2).to_string(index=False))
