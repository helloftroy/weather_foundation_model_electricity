"""Figure 4C: what the most important token PCs look like in the weather. For each PC (rows), the difference
between the top and bottom deciles of the PC score itself (selected on the score, not on SHAP; deseasonalised
issue-time MERRA-2 input fields, in units of the model's input sigma), oriented by the sign of the score-SHAP
correlation so every row reads 'the price-raising end of the PC'. Row label: ΔSHAP, mean SHAP in the price-raising
score decile minus the price-lowering one (2024 test hours). Reads results/price_token_shap/composites/; writes
results/figures/fig4c_pc_composites.{png,pdf}. Bare panels; caption text lives in the reply."""
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 8
root = Path(__file__).resolve().parent.parent
C = root / "results/price_token_shap/composites"
summary = pd.read_csv(C / "pc_summary.csv").set_index("pc")
PCS = [23, 2, 19]
COLS = [  # (source, colour field, arrows (u, v) or None, column label)
    ("wide", "T2M", ("U10M", "V10M"), "2 m temperature,\n10 m wind"),
    ("ne", "SLP", None, "Sea-level\npressure"),
    ("ne", "H_51", ("U_45", "V_45"), "Height ≈500 hPa,\nwind ≈290 hPa"),
    ("ne", "T_53", None, "Temperature\n≈600 hPa"),
    ("ne", "T_44", None, "Temperature\n≈250 hPa"),
]
NE_BOX = (-75.0, -66.875, 40.0, 47.5)
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(8.6, 5.2))
gs = fig.add_gridspec(len(PCS) + 1, len(COLS) + 2, width_ratios=[1.5, 1, 1, 1, 1, 0.32, 1.15], height_ratios=[1] * len(PCS) + [0.07],
                      hspace=0.12, wspace=0.08)
lim = {}
comps = {k: np.load(C / f"pc{k}_composite.npz") for k in PCS}
sign = {k: np.sign(summary.loc[k, "shap_vs_value_corr"]) for k in PCS}


def field(k, src, name):
    z = comps[k]
    chans = list(z[f"{src}_channels"])
    return sign[k] * z[src][chans.index(name)].astype(float)


for j, (src, name, arrows, _) in enumerate(COLS):
    lim[j] = max(np.nanmax(np.abs(field(k, src, name))) for k in PCS)
for i, k in enumerate(PCS):
    z = comps[k]
    for j, (src, name, arrows, label) in enumerate(COLS):
        ax = fig.add_subplot(gs[i, j], projection=proj)
        lats, lons = z[f"{src}_lats"], z[f"{src}_lons"]
        f = field(k, src, name)
        step = np.diff(lons).mean() / 2, np.diff(lats).mean() / 2
        extent = [lons.min() - step[0], lons.max() + step[0], lats.min() - step[1], lats.max() + step[1]]
        im = ax.pcolormesh(lons, lats, f, cmap="RdBu_r", vmin=-lim[j], vmax=lim[j], transform=proj, shading="nearest")
        if arrows:
            u, v = field(k, src, arrows[0]), field(k, src, arrows[1])
            s = 4 if src == "wide" else 3
            ax.quiver(lons[::s], lats[::s], u[::s, ::s], v[::s, ::s], transform=proj, color="#222222", width=0.006,
                      scale=lim[j] * 0 + (12 if src == "wide" else 10), headwidth=3.5)
        ax.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.4, edgecolor="#333333")
        ax.add_feature(cf.STATES.with_scale("50m"), lw=0.25, edgecolor="#777777")
        ax.set_extent(extent, crs=proj)
        if src == "wide":
            x0, x1, y0, y1 = NE_BOX
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="#222222", lw=0.8, ls="--", transform=proj)
        if i == 0:
            ax.text(0.5, 1.04, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5)
        if j == 0:
            eff = summary.loc[k]
            d = sign[k] * (eff["mean_shap_top_decile_$"] - eff["mean_shap_bottom_decile_$"])
            ax.text(-0.06, 0.5, f"PC{k}\nΔSHAP\n+${d:.1f}/MWh", transform=ax.transAxes, ha="right", va="center", fontsize=8,
                    fontweight="bold")
        if i == len(PCS) - 1:
            cax = fig.add_subplot(gs[-1, j])
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=6.5)
            cb.locator = plt.MaxNLocator(3)
            cb.update_ticks()
# Last column: the PC's own price effect across its score deciles (2024 test hours), one shared y-axis.
prof = pd.read_csv(C / "pc_shap_profile.csv")
ymin, ymax = prof["mean_shap"].min(), prof["mean_shap"].max()
for i, k in enumerate(PCS):
    ax = fig.add_subplot(gs[i, -1])
    d = prof[prof["pc"] == k].sort_values("decile")
    up = 10 if sign[k] > 0 else 1  # the price-raising end, the decile the maps compare against the other end
    ax.axhline(0, color="#888888", lw=0.6, zorder=1)
    ax.plot(d["decile"], d["mean_shap"], color="#555555", lw=1.0, zorder=2)
    ax.scatter(d["decile"], d["mean_shap"], s=14, color="#555555", zorder=3)
    for dd, c in ((up, "#b2182b"), (11 - up, "#2166ac")):
        r = d[d["decile"] == dd].iloc[0]
        ax.scatter(dd, r["mean_shap"], s=30, color=c, zorder=4, edgecolor="white", linewidth=0.5)
    ax.set_ylim(ymin - 0.4, ymax + 0.4)
    ax.set_xlim(0.3, 10.7)
    ax.set_xticks([1, 5, 10])
    ax.tick_params(labelsize=6.5, length=2)
    if i < len(PCS) - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("PC-score decile", fontsize=7)
    if i == len(PCS) // 2:
        ax.set_ylabel("Mean SHAP of the PC, $/MWh", fontsize=7, labelpad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if i == 0:
        ax.text(0.5, 1.04, "Price effect by\nPC-score decile", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5)
fig.text(0.5, 0.0, "Difference, price-raising minus price-lowering decile of the PC score (σ units, deseasonalised)", ha="center", fontsize=7.5)
out = root / "results/figures/fig4c_pc_composites"
fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print(out)
