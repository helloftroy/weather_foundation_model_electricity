"""Does training longer help? Error against number of trees, with early
stopping switched off.

Fits one CatBoost model per split with a fixed, large number of trees, then
scores the training and test hours after every tree. The dashed line marks
where early stopping on the validation block would have stopped. Uses the
stripped-down price model (calendar + weather + demand, no gas prices) by
default.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from train_catboost_comparisons import assign_split
from train_price_catboost import CALENDAR_FEATURES, DEMAND_FEATURES, TARGET, WEATHER_FEATURES

SURFACE, INK, INK_2, MUTED, HAIRLINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
COLOURS = {"Test hours": "#2a78d6", "Training hours": "#eb6834"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-table", type=Path, required=True)
    parser.add_argument("--trees", type=int, default=10000)
    parser.add_argument("--out-png", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.price_table)
    df = df[df["timestamp_local"].dt.year == 2024].dropna(subset=[TARGET, "weather_T2M", "system_demand_MW"]).reset_index(drop=True)
    features = CALENDAR_FEATURES + WEATHER_FEATURES + DEMAND_FEATURES
    series = "Day-ahead" if "dayahead" in df["price_series"].iloc[0] else "Real-time"

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.4), facecolor=SURFACE, sharey=True)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.76, bottom=0.14, wspace=0.08)
    rows = []
    for ax, scheme in zip(axes, ["chronological", "blocked"]):
        split = assign_split(df, scheme)
        tr, va, te = df[split == "train"], df[split == "val"], df[split == "test"]
        model = CatBoostRegressor(iterations=args.trees, learning_rate=0.05, depth=6, random_seed=7, loss_function="RMSE",
                                  verbose=False, thread_count=-1, train_dir=str(args.out_png.parent / "catboost_info"))
        model.fit(Pool(tr[features], tr[TARGET]))
        curves = {}
        for name, part in [("Training hours", tr), ("Validation", va), ("Test hours", te)]:
            y = part[TARGET].to_numpy()
            curves[name] = np.array([np.abs(p - y).mean() for p in model.staged_predict(part[features])])
        n = np.arange(1, args.trees + 1)
        stop = int(curves["Validation"].argmin()) + 1
        best_test = int(curves["Test hours"].argmin()) + 1
        reference = float(np.abs(tr[TARGET].mean() - te[TARGET]).mean())
        for k in sorted({100, 300, 1000, 3000, args.trees, stop, best_test}):
            if k <= args.trees:
                rows.append({"split": scheme, "trees": k, "train_mae": curves["Training hours"][k - 1], "test_mae": curves["Test hours"][k - 1],
                             "note": ("early stop " if k == stop else "") + ("best test" if k == best_test else "")})

        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(HAIRLINE)
        ax.tick_params(colors=MUTED, labelsize=10, length=0)
        ax.grid(axis="y", color=HAIRLINE, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.axhline(reference, color=MUTED, linewidth=1, linestyle=(0, (2, 3)), zorder=1)
        ax.text(args.trees, reference, " Guessing the training average", color=INK_2, fontsize=10, va="bottom", ha="right")
        for name in ("Test hours", "Training hours"):
            ax.plot(n, curves[name], color=COLOURS[name], linewidth=2, label=name, zorder=3)
            ax.text(args.trees * 1.02, curves[name][-1], name, color=INK, fontsize=11, va="center")
        ax.axvline(stop, color=INK, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        ax.text(stop * 1.12, ax.get_ylim()[1] * 0.02 + 0.6, f"Lowest validation error: {stop:,} trees", color=INK, fontsize=10, rotation=90, va="bottom")
        ax.set_xscale("log")
        ax.set_xlim(1, args.trees * 3.2)
        ax.set_xticks([1, 10, 100, 1000, 10000][: int(np.log10(args.trees)) + 1], labels=["1", "10", "100", "1,000", "10,000"][: int(np.log10(args.trees)) + 1])
        ax.set_ylim(0, None)
        ax.set_xlabel("Number of trees (log scale)", color=INK_2, fontsize=11)
        title = "Chronological split: train Jan-Aug, test Oct-Dec" if scheme == "chronological" else "Blocked split: train days 1-18, test day 22 onward"
        ax.set_title(title, loc="left", fontsize=13, color=INK, pad=10)
    axes[0].set_ylabel("Mean absolute error ($/MWh)", color=INK_2, fontsize=11)
    axes[0].legend(loc="lower left", frameon=False, fontsize=10, labelcolor=INK_2)

    fig.text(0.06, 0.93, f"Does training longer help? {series} hub price, error against number of trees", fontsize=18, color=INK, weight="bold")
    fig.text(0.06, 0.875, "CatBoost, calendar + weather + actual demand (no gas prices), 2024. Early stopping switched off; "
             "each line is the error after that many trees.", fontsize=12, color=INK_2)
    fig.text(0.06, 0.03, "Dashed vertical line: the tree count with the lowest error on the validation block. Learning rate 0.05, depth 6.", fontsize=9, color=MUTED)
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=200, facecolor=SURFACE)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
