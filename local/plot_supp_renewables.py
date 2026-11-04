"""Supplementary figure: ERCOT wind and solar headroom (local/ercot_renewables_benchmark.py; results/ercot/renewables).
% MAE reduction vs GFS (10 m wind, shortwave; with calendar and weather at issue) from adding Prithvi tokens, the full
GFS state (150 fields, area mean, PCA-50), the full state + tokens, and the ceiling (actual MERRA-2 weather at the
target hour); columns wind / solar, rows all hours / top-5% 3-h ramps; 95% week-block bootstrap.
Writes results/figures/supp_renewables.{png,pdf} and supp_renewables_table.csv. Bare panels."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eval_tools import block_bootstrap  # noqa: E402

plt.rcParams["font.family"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 8.5
root = Path(__file__).resolve().parent.parent
BASE = "calendar + weather (issue) + GFS"
ROWS = [(" + tokens", "+ Prithvi tokens"), (" + full GFS state", "+ full GFS state"),
        (" + full GFS state + tokens", "+ full GFS state + tokens"), (" + actual weather", "Ceiling: actual weather")]
Y23, Y24 = "#1f4e9c", "#d9822b"

p = pd.read_parquet(root / "results/ercot/renewables/renewables_predictions.parquet")
p["ae"] = 100 * (p["pred"] - p["actual"]).abs()
rows = []
for (tgt, y), g in p.groupby(["target", "test_year"]):
    w = g.pivot_table(index="row", columns="feature_set", values="ae")
    one = g.drop_duplicates("row").set_index("row").reindex(w.index)
    for hours, mask in (("all", slice(None)), ("ramps", one["ramp"].to_numpy())):
        ref = w[BASE].to_numpy()[mask].mean()
        for suffix, _ in ROWS:
            m, lo, hi = block_bootstrap((w[BASE + suffix] - w[BASE]).to_numpy()[mask], one["week"].to_numpy()[mask])
            rows.append({"target": tgt, "test_year": y, "hours": hours, "added": suffix.strip(" +"), "MAE_GFS": ref,
                         "gain_pct": -100 * m / ref, "gain_lo": -100 * hi / ref, "gain_hi": -100 * lo / ref})
t = pd.DataFrame(rows)

fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharey=True, gridspec_kw={"hspace": 0.55, "wspace": 0.12})
for i, hours in enumerate(("all", "ramps")):
    for j, tgt in enumerate(("wind", "solar")):
        ax = axes[i, j]
        for k, (suffix, _) in enumerate(ROWS):
            for y, color, marker, off in ((2023, Y23, "o", -0.15), (2024, Y24, "s", 0.15)):
                r = t[(t["target"] == tgt) & (t["test_year"] == y) & (t["hours"] == hours) & (t["added"] == suffix.strip(" +"))].iloc[0]
                ax.hlines(k + off, r["gain_lo"], r["gain_hi"], color=color, lw=1.2, zorder=2)
                ax.scatter(r["gain_pct"], k + off, s=28, color=color, marker=marker, zorder=3, edgecolor="white", linewidth=0.5,
                           label=str(y) if k == 0 else None)
        ax.axhline(len(ROWS) - 1.5, color="#cccccc", lw=0.6)          # separates the ceiling from the forecasts
        ax.axvline(0, color="#555555", lw=0.8, zorder=2.5)
        ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels([lab for _, lab in ROWS])
        ax.set_ylim(len(ROWS) - 0.5, -0.5)
        ax.grid(axis="x", color="#ececec", lw=0.6, zorder=0)
        ax.set_xlabel(f"MAE reduction vs GFS (%), {tgt}, {'all hours' if hours == 'all' else 'large ramps'}")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.text(-0.02 if j == 0 else 0.0, 1.03, "ABCD"[2 * i + j], transform=ax.transAxes, fontsize=11, fontweight="bold",
                ha="right" if j == 0 else "left", va="bottom")
axes[0, 1].legend(frameon=False, fontsize=7.5, loc="upper right", handletextpad=0.2)
out = root / "results/figures/supp_renewables"
fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
t.to_csv(root / "results/figures/supp_renewables_table.csv", index=False)
pd.set_option("display.width", 200)
print(t.round(1).to_string(index=False))
