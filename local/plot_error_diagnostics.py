"""Two diagnostics for the blocked split (train days 1-18, validate 19-21,
test day 22 onward of each month):

  left   how hourly percentage errors are distributed for one model
         (is the mean a typical error, or a few huge ones?)
  right  mean error against days to the nearest training day, for every
         model in the predictions file

Days to the nearest training day: back to the 18th of the same month, or
forward to the 1st of the next, whichever is closer. December has no
following training month, so its late days would be the only ones more than
7 days away; those bins are a single month and are left off the plot.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE, INK, INK_2, MUTED, HAIRLINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# Fixed colour per model, in the palette's categorical order.
MODEL_COLOURS = {"calendar+weather": "#2a78d6", "calendar": "#eb6834",
                 "calendar+prithvi": "#1baf7a", "calendar+weather+prithvi": "#8a5fd6"}
MODEL_LABELS = {"calendar": "Calendar only", "calendar+weather": "Calendar + weather",
                "calendar+prithvi": "Calendar + Prithvi", "calendar+weather+prithvi": "Calendar + weather + Prithvi"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(HAIRLINE)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.grid(axis="y", color=HAIRLINE, linewidth=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--feature-set", default="calendar+weather", help="Model shown in the distribution panel")
    parser.add_argument("--out-png", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.predictions)
    df = df[df["split_scheme"] == "blocked"].copy()
    models = [c[5:] for c in df.columns if c.startswith("pred_")]
    for m in models:
        df[f"ape_{m}"] = (df[f"pred_{m}"] - df["demand_MW"]).abs() / df["demand_MW"] * 100

    local = df["timestamp_local"]
    day, days_in_month = local.dt.day, local.dt.days_in_month
    back = day - 18
    forward = (days_in_month - day + 1).where(local.dt.month < 12, np.inf)
    df["days_to_training"] = np.minimum(back, forward)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.06, right=0.97, top=0.76, bottom=0.14, wspace=0.2)

    # ---- left: distribution of hourly errors ----
    ape = df[f"ape_{args.feature_set}"].to_numpy()
    colour = MODEL_COLOURS.get(args.feature_set, "#2a78d6")
    cap = 30
    edges = np.arange(0, cap + 2)
    counts, _ = np.histogram(np.minimum(ape, cap + 0.5), bins=edges)
    share = counts / len(ape) * 100
    style(ax1)
    ax1.bar(edges[:-1] + 0.5, share, width=0.86, color=colour, zorder=2)
    ax1.set_xlim(-0.3, cap + 1.3)
    ax1.set_xticks([0, 5, 10, 15, 20, 25, cap + 0.5], labels=["0", "5", "10", "15", "20", "25", f"{cap}+"])
    ax1.set_xlabel("Absolute error in one zone-hour (% of actual demand)", color=INK_2, fontsize=11)
    ax1.set_ylabel("Share of test hours (%)", color=INK_2, fontsize=11)
    median, mean = float(np.median(ape)), float(ape.mean())
    top = share.max()
    for value, name, dy in [(median, "Median", 1.0), (mean, "Mean", 0.84)]:
        # The mean line stops short so it doesn't run through the median label.
        ax1.plot([value, value], [0, top * (dy + 0.02)], color=INK, linewidth=1, linestyle=(0, (4, 3)), zorder=3)
        ax1.text(value + 0.35, top * dy, f"{name} {value:.1f}%", color=INK, fontsize=11, va="top")
    facts = (f"{(ape < 2).mean() * 100:.0f}% of hours are within 2%\n"
             f"{(ape < 5).mean() * 100:.0f}% within 5%\n"
             f"{(ape > 10).mean() * 100:.0f}% are off by more than 10%\n"
             f"{(ape > 20).mean() * 100:.1f}% by more than 20%\n"
             f"{(ape > 50).mean() * 100:.1f}% by more than 50% (largest {ape.max():.0f}%)")
    ax1.text(0.56, 0.62, facts, transform=ax1.transAxes, color=INK_2, fontsize=11, va="top", linespacing=1.6)
    ax1.set_title(f"How large are the errors?  {MODEL_LABELS.get(args.feature_set, args.feature_set)}",
                  loc="left", fontsize=13, color=INK, pad=10)

    # ---- right: error vs. days to nearest training day ----
    style(ax2)
    g = df[df["days_to_training"] <= 7].groupby("days_to_training")
    ymax = 0
    for m in models:
        series = g[f"ape_{m}"].mean()
        c = MODEL_COLOURS.get(m, MUTED)
        ax2.plot(series.index, series.values, color=c, linewidth=2, marker="o", markersize=7,
                 markeredgecolor=SURFACE, markeredgewidth=2, label=MODEL_LABELS.get(m, m), zorder=3)
        ax2.text(series.index[-1] + 0.15, series.values[-1], MODEL_LABELS.get(m, m), color=INK, fontsize=11, va="center")
        ymax = max(ymax, series.max())
    ax2.set_xlim(0.6, 9.4)
    ax2.set_xticks(range(1, 8))
    ax2.set_ylim(0, np.ceil(ymax / 5) * 5)
    ax2.set_xlabel("Days to the nearest training day", color=INK_2, fontsize=11)
    ax2.set_ylabel("Mean absolute error (%)", color=INK_2, fontsize=11)
    ax2.set_title("Does error grow with distance from the training days?", loc="left", fontsize=13, color=INK, pad=10)
    ax2.legend(loc="lower left", frameon=False, fontsize=10, labelcolor=INK_2)

    fig.text(0.06, 0.93, "Demand prediction errors on the blocked split", fontsize=18, color=INK, weight="bold")
    fig.text(0.06, 0.875, f"Held-out test hours: day 22 onward of every month, 2024, all eight ISO-NE load zones ({len(df):,} zone-hours). "
             "Training days are the 1st-18th.", fontsize=12, color=INK_2)
    fig.text(0.06, 0.03, "Days to nearest training day: back to the 18th or forward to the 1st of the next month, whichever is closer. "
             "Late-December days (more than 7 days away, one month only) are not shown.", fontsize=9, color=MUTED)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=200, facecolor=SURFACE)
    print(f"{args.feature_set}: median {median:.2f}%, mean {mean:.2f}%, mean without errors >20%: {ape[ape <= 20].mean():.2f}%")
    print(g[[f"ape_{m}" for m in models]].mean().round(2).assign(n=g.size()).to_string())
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
