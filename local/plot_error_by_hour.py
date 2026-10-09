"""Demand and prediction error by hour of day, for two load zones side by side.

Top row: mean actual and predicted demand at each local hour over the test
hours. Bottom row: mean absolute percentage error at each hour, on a shared
scale so the two zones can be compared directly.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE, INK, INK_2, MUTED, HAIRLINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
PREDICTED, ACTUAL = "#2a78d6", "#0b0b0b"
ZONE_NAMES = {"CT": "Connecticut", "ME": "Maine", "NH": "New Hampshire", "RI": "Rhode Island", "VT": "Vermont",
              "NEMA": "NE Massachusetts", "SEMA": "SE Massachusetts", "WCMA": "W/Central Massachusetts"}
MODEL_LABELS = {"calendar": "calendar only", "calendar+weather": "calendar + simple weather",
                "calendar+prithvi": "calendar + Prithvi embedding", "calendar+weather+prithvi": "calendar + weather + Prithvi embedding"}
HOUR_TICKS = [0, 3, 6, 9, 12, 15, 18, 21]
HOUR_LABELS = ["12am", "3am", "6am", "9am", "noon", "3pm", "6pm", "9pm"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(HAIRLINE)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.grid(axis="y", color=HAIRLINE, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.7, 23.7)
    ax.set_xticks(HOUR_TICKS, labels=HOUR_LABELS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", default="blocked", choices=["chronological", "blocked"])
    parser.add_argument("--feature-set", default="calendar+weather")
    parser.add_argument("--zones", nargs=2, default=["VT", "NH"])
    parser.add_argument("--out-png", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.predictions)
    df = df[df["split_scheme"] == args.split].copy()
    col = f"pred_{args.feature_set}"
    df["hour"] = df["timestamp_local"].dt.hour
    df["ape"] = (df[col] - df["demand_MW"]).abs() / df["demand_MW"] * 100
    by_hour = {z: df[df["zone"] == z].groupby("hour").agg(actual=("demand_MW", "mean"), predicted=(col, "mean"), mape=("ape", "mean"))
               for z in args.zones}
    overall = {z: float(df.loc[df["zone"] == z, "ape"].mean()) for z in args.zones}
    err_max = float(np.ceil(max(t["mape"].max() for t in by_hour.values()) / 5) * 5)

    fig, axes = plt.subplots(2, 2, figsize=(16, 8.6), facecolor=SURFACE, gridspec_kw={"height_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.06, right=0.97, top=0.8, bottom=0.11, wspace=0.14, hspace=0.42)

    for j, zone in enumerate(args.zones):
        t = by_hour[zone]
        ax = axes[0, j]
        style(ax)
        ax.plot(t.index, t["actual"], color=ACTUAL, linewidth=2, label="Actual")
        ax.plot(t.index, t["predicted"], color=PREDICTED, linewidth=2, label="Predicted")
        ax.set_ylim(0, float(np.ceil(max(t["actual"].max(), t["predicted"].max()) * 1.12 / 100) * 100))
        ax.set_title(f"{ZONE_NAMES.get(zone, zone)}: mean demand by hour (MW)", loc="left", fontsize=13, color=INK, pad=8)
        ax.legend(loc="lower left", frameon=False, fontsize=10, labelcolor=INK_2, ncols=2)
        # Direct labels at the hour where the two lines are furthest apart.
        gap_hour = int((t["predicted"] - t["actual"]).abs().idxmax())
        hi, lo = ("Predicted", "Actual") if t.loc[gap_hour, "predicted"] > t.loc[gap_hour, "actual"] else ("Actual", "Predicted")
        span = ax.get_ylim()[1]
        ax.text(gap_hour, max(t.loc[gap_hour, "predicted"], t.loc[gap_hour, "actual"]) + span * 0.035, hi, ha="center", va="bottom", fontsize=10, color=INK)
        ax.text(gap_hour, min(t.loc[gap_hour, "predicted"], t.loc[gap_hour, "actual"]) - span * 0.035, lo, ha="center", va="top", fontsize=10, color=INK)

        ax = axes[1, j]
        style(ax)
        ax.bar(t.index, t["mape"], width=0.8, color=PREDICTED, zorder=2)
        ax.set_ylim(0, err_max)
        ax.set_title(f"{ZONE_NAMES.get(zone, zone)}: mean absolute error by hour (%)", loc="left", fontsize=13, color=INK, pad=8)
        ax.set_xlabel("Hour of day (Eastern local time)", color=INK_2, fontsize=11)
        peak = int(t["mape"].idxmax())
        ax.text(peak, t.loc[peak, "mape"] + err_max * 0.02, f"{t.loc[peak, 'mape']:.0f}%", ha="center", va="bottom", fontsize=10, color=INK)
        ax.text(0.99, 0.93, f"All hours: {overall[zone]:.1f}%", transform=ax.transAxes, ha="right", va="top", fontsize=11, color=INK_2)

    period = "day 22 onward of every month, 2024" if args.split == "blocked" else "Oct 1 to Dec 31, 2024"
    names = " and ".join(ZONE_NAMES.get(z, z) for z in args.zones)
    fig.text(0.06, 0.94, f"Demand prediction error by hour of day: {names}", fontsize=18, color=INK, weight="bold")
    fig.text(0.06, 0.89, f"CatBoost, model inputs: {MODEL_LABELS.get(args.feature_set, args.feature_set)}. Held-out test hours: {period}.",
             fontsize=12, color=INK_2)
    fig.text(0.06, 0.025, "Each point is the mean over all test days at that local hour. Error panels share one scale; demand panels do not (the zones differ in size).",
             fontsize=9, color=MUTED)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=200, facecolor=SURFACE)
    for z in args.zones:
        print(z, by_hour[z].round(1).T.to_string())
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
