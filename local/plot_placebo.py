"""Figure: wrong-time placebo for the Prithvi tokens (raw day-ahead price, Henry Hub known).
Reads results/price_known_gas/placebo/placebo_increments.csv; writes results/figures/price_token_placebo.{png,pdf}.
Bare panels (no titles); caption text lives in PROJECT_RESULTS.md."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 9
root = Path(__file__).resolve().parent.parent
r = pd.read_csv(root / "results/price_known_gas/placebo/placebo_increments.csv")

REAL, PLACEBO, SEED = "#1f4e9c", "#6f6f6f", "#b3b3b3"
cats = ["Real", "Shuffled\n(month + hour)", "Shifted\n−7 d", "Shifted\n+7 d"]
panels = [(2023, "GFS"), (2023, "GFS + regional"), (2024, "GFS"), (2024, "GFS + regional")]
labels = {"GFS": "over GFS", "GFS + regional": "over GFS + regional"}

fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.4), sharey=True, sharex=True)
for ax, (year, base) in zip(axes.ravel(), panels):
    g = r[(r["year"] == year) & (r["base"] == base)].set_index("tokens")
    ax.axhline(0, color="#444444", lw=0.9, zorder=1)

    def whisker(x, name, color):
        row = g.loc[name]
        ax.errorbar(x, row["d_vs_base"], yerr=[[row["d_vs_base"] - row["lo"]], [row["hi"] - row["d_vs_base"]]], fmt="o",
                    color=color, ms=6, elinewidth=1.4, capsize=0, zorder=3)

    whisker(0, "Prithvi tokens", REAL)
    seeds = g.loc[[f"shuffle s{k}" for k in range(3)], "d_vs_base"]
    ax.vlines(1, seeds.min(), seeds.max(), color=PLACEBO, lw=1.4, zorder=2)          # range over the 3 seeds
    ax.scatter(1 + np.array([-0.12, 0, 0.12]), seeds, s=14, color=SEED, edgecolor="none", zorder=3)
    ax.scatter([1], [seeds.mean()], s=36, color=PLACEBO, zorder=4)
    whisker(2, "shift -7d", PLACEBO)
    whisker(3, "shift +7d", PLACEBO)

    ax.set_xticks(range(4))
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-1.42, 0.62)
    ax.text(0.97, 0.97, f"{year}, {labels[base]}", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#ececec", lw=0.6, zorder=0)
    if (year, base) == (2024, "GFS + regional"):
        real = g.loc["Prithvi tokens", "d_vs_base"]
        removed = 1 - seeds / real
        ax.text(0.97, 0.04, f"Shuffling removes {removed.min():.0%}–{removed.max():.0%}\nof the real-token gain",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#333333")
for ax in axes[:, 0]:
    ax.set_ylabel("Change in MAE from adding\ntokens ($/MWh); negative = helps")
fig.tight_layout(h_pad=1.2, w_pad=1.0)
out = root / "results/figures/price_token_placebo"
fig.savefig(out.with_suffix(".png"), dpi=300)
fig.savefig(out.with_suffix(".pdf"))
print(out)
