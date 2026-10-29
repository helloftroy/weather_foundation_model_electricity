"""Figure: token gain on raw day-ahead price vs number of token principal components.
Reads results/price_known_gas/pc_sweep/pc_sweep_increments.csv; writes results/figures/price_token_pc_sweep.{png,pdf}.
Bare panels (no titles); caption text lives in PROJECT_RESULTS.md."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 9
root = Path(__file__).resolve().parent.parent
r = pd.read_csv(root / "results/price_known_gas/pc_sweep/pc_sweep_increments.csv")
order = ["PC10", "PC25", "PC50", "PC100", "PC250", "PC500", "pooled full"]
x = np.array([10, 25, 50, 100, 250, 500, 2560])
labels = ["10", "25", "50", "100", "250", "500", "2,560\n(full)"]
style = {"GFS": ("#2a6fdb", "o", "added to old weather + GFS"),
         "GFS + regional": ("#d1495b", "s", "added to old weather + GFS + regional GFS")}

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, (year, g) in zip(axes, r.groupby("test_year")):
    ax.axhline(0, color="#888888", lw=0.8)
    for k, (base, (col, mk, lab)) in enumerate(style.items()):
        s = g[g["baseline"] == base].set_index("tokens").loc[order]
        xs = x * (1.06 if k else 1 / 1.06)   # small offset so the intervals do not overlap
        ax.errorbar(xs, s["dMAE"], yerr=[s["dMAE"] - s["lo"], s["hi"] - s["dMAE"]], color=col, marker=mk, ms=5,
                    lw=2, elinewidth=1.2, capsize=0, label=lab)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.minorticks_off()
    ax.text(0.03, 0.04, str(year), transform=ax.transAxes, fontsize=10, fontweight="bold")
    ax.set_xlabel("Token principal components")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
axes[0].set_ylabel("Change in MAE ($/MWh)\nnegative = tokens help")
axes[0].legend(frameon=False, loc="upper left", fontsize=8)
fig.tight_layout()
out = root / "results/figures/price_token_pc_sweep"
fig.savefig(out.with_suffix(".png"), dpi=300)
fig.savefig(out.with_suffix(".pdf"))
print(out)
