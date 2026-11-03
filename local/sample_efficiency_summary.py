"""Sample efficiency of the token gain (price_benchmark.py --sample-efficiency; results/regional/price_sample_efficiency):
test MAE of GFS, GFS + tokens (PCA-50) and GFS + raw state (Prithvi's MERRA-2 input fields, PCA-50) against the number
of labelled training months. For the random-month design the hourly absolute errors are averaged over the draws first
(the draw spread is reported separately); differences get a 95% week-block bootstrap on the test year.
Writes sample_efficiency.csv and results/figures/sample_efficiency.{png,pdf}."""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eval_tools import block_bootstrap  # noqa: E402

plt.rcParams["font.family"] = ["Arial", "Helvetica", "DejaVu Sans"]
root = Path(__file__).resolve().parent.parent
d = root / "results/regional/price_sample_efficiency"
p = pd.read_parquet(d / "price_predictions.parquet")
PRE = "calendar + past values + known gas + weather (issue) + GFS area"
ARMS = {"": "GFS", " + tokens ne": "tokens", " + MERRA-2 fields ne": "state"}


def parse(fs):
    m = re.match(re.escape(PRE) + r"(.*?)( \[train (recent|random) (\d+)m(?: d(\d+))?\])?$", fs)
    return ARMS[m.group(1)], m.group(3) or "all", int(m.group(4)) if m.group(4) else None, int(m.group(5)) if m.group(5) else 0


p[["arm", "design", "months", "draw"]] = pd.DataFrame([parse(f) for f in p["feature_set"]], index=p.index)
p["ae"] = (p["pred"] - p["actual"]).abs()
rows = []
for y, g in p.groupby("test_year"):
    n_all = (y - 2021) * 12
    g = g.assign(months=g["months"].fillna(n_all).astype(int))
    week = g.drop_duplicates("row").set_index("row")["week"]
    for design in ("random", "recent"):
        sub = g[g["design"].isin([design, "all"])]
        for n, s in sub.groupby("months"):
            draw_mae = s.groupby(["arm", "draw"])["ae"].mean().unstack()
            ae = s.groupby(["row", "arm"])["ae"].mean().unstack()            # averaged over draws
            wk = week.reindex(ae.index).to_numpy()
            r = {"test_year": y, "design": design, "months": n, "n_draws": s["draw"].nunique()}
            for a in ARMS.values():
                r[f"MAE_{a}"] = ae[a].mean()
                r[f"MAE_{a}_draw_min"], r[f"MAE_{a}_draw_max"] = draw_mae.loc[a].min(), draw_mae.loc[a].max()
            for a, b in (("tokens", "GFS"), ("state", "GFS"), ("tokens", "state")):
                m, lo, hi = block_bootstrap((ae[a] - ae[b]).to_numpy(), wk)
                r[f"d_{a}_vs_{b}"], r[f"d_{a}_vs_{b}_lo"], r[f"d_{a}_vs_{b}_hi"] = m, lo, hi
            # draws where tokens beat state
            r["draws_tokens_better_than_state"] = int((draw_mae.loc["tokens"] < draw_mae.loc["state"]).sum())
            rows.append(r)
out = pd.DataFrame(rows).sort_values(["design", "test_year", "months"])
out.to_csv(d / "sample_efficiency.csv", index=False)
pd.set_option("display.width", 250)
show = out[["design", "test_year", "months", "n_draws", "MAE_GFS", "MAE_state", "MAE_tokens"]].copy()
for a, b in (("tokens", "GFS"), ("state", "GFS"), ("tokens", "state")):
    show[f"{a}-{b}"] = [f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]" for m, lo, hi in
                        zip(out[f"d_{a}_vs_{b}"], out[f"d_{a}_vs_{b}_lo"], out[f"d_{a}_vs_{b}_hi"])]
show["tok<state draws"] = out["draws_tokens_better_than_state"].astype(str) + "/" + out["n_draws"].astype(str)
print(show.round(2).to_string(index=False))

# ---- figure: top, MAE vs training months (random-month design, draw-averaged; whiskers = draw range);
#      bottom, MAE reduction vs GFS at the same training length (%), both designs ----
C = {"GFS": "#6b6b6b", "state": "#d08c2a", "tokens": "#2a6fb0"}
LAB = {"GFS": "GFS", "state": "+ raw state (PCA-50)", "tokens": "+ Prithvi tokens (PCA-50)"}
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True, gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.12, "wspace": 0.28})
for j, y in enumerate((2023, 2024)):
    ax = axes[0, j]
    o = out[(out["design"] == "random") & (out["test_year"] == y)]
    for k, a in enumerate(("GFS", "state", "tokens")):
        x = o["months"] * (1 + 0.03 * (k - 1))
        ax.errorbar(x, o[f"MAE_{a}"], yerr=[o[f"MAE_{a}"] - o[f"MAE_{a}_draw_min"], o[f"MAE_{a}_draw_max"] - o[f"MAE_{a}"]],
                    color=C[a], marker="o", ms=4.5, lw=1.6, capsize=0, elinewidth=0.9, label=LAB[a])
    ax.set_ylabel("Test MAE ($/MWh)" if j == 0 else "")
    ax.text(0.97, 0.95, f"Test {y}", transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#333")
    ax2 = axes[1, j]
    for a in ("state", "tokens"):
        for design, ls, fill in (("random", "-", True), ("recent", "--", False)):
            o = out[(out["design"] == design) & (out["test_year"] == y)]
            base = o["MAE_GFS"]
            v, lo, hi = (-100 * o[f"d_{a}_vs_GFS"] / base, -100 * o[f"d_{a}_vs_GFS_hi"] / base, -100 * o[f"d_{a}_vs_GFS_lo"] / base)
            x = o["months"] * (1.03 if a == "tokens" else 0.97) * (1.0 if design == "random" else 1.015)
            ax2.errorbar(x, v, yerr=[v - lo, hi - v], color=C[a], ls=ls, marker="o", ms=4.5, lw=1.4, elinewidth=0.9,
                         mfc=C[a] if fill else "white", mec=C[a], capsize=0)
    ax2.axhline(0, color="#999", lw=0.8)
    ax2.set_ylabel("MAE reduction vs GFS (%)" if j == 0 else "")
    ax2.set_xscale("log")
    ticks = sorted(out.loc[out["test_year"] == y, "months"].unique())
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([str(t) for t in ticks])
    ax2.minorticks_off()
    ax2.set_xlabel("Labelled training months")
for ax in axes.flat:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)
    ax.yaxis.label.set_size(9)
    ax.xaxis.label.set_size(9)
for ax, l in zip(axes.flat, "ABCD"):
    ax.text(-0.16, 1.02, l, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
h, lab = axes[0, 0].get_legend_handles_labels()
from matplotlib.lines import Line2D  # noqa: E402
h += [Line2D([], [], color="#555", ls="-", marker="o", ms=4.5, lw=1.4), Line2D([], [], color="#555", ls="--", marker="o", ms=4.5, mfc="white", lw=1.4)]
lab += ["random months", "most recent months"]
fig.legend(h, lab, loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02), handlelength=2.2, columnspacing=1.2)
fig.subplots_adjust(bottom=0.15, top=0.95, left=0.1, right=0.98)
for ext in ("png", "pdf"):
    fig.savefig(root / f"results/figures/sample_efficiency.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
print("Wrote", d / "sample_efficiency.csv", "and results/figures/sample_efficiency.png")
