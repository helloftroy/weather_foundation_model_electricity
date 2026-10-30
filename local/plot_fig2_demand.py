"""Figure 2 (demand): A, out-of-sample R² by feature set; B, daily error by day type for the best forecast model,
the same plus Prithvi tokens, and actual weather. Reads results/demand_hard_days/; writes
results/figures/fig2_demand.{png,pdf} and fig2_demand_table.csv. Bare panels; caption text lives in the reply /
PROJECT_RESULTS.md."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 8.5
root = Path(__file__).resolve().parent.parent
src = root / "results/demand_hard_days"
C, W = "calendar", "calendar + weather (issue)"
ROWS = [  # (label, feature set), top to bottom
    ("Calendar only", C),
    ("+ weather at issue time", W),
    ("+ Prithvi tokens", W + " + Prithvi tokens"),
    ("+ Prithvi decoder", W + " + Prithvi area"),
    ("+ GFS", W + " + GFS area"),
    ("+ GFS + tokens", W + " + GFS area + Prithvi tokens"),
    ("+ GFS + decoder", W + " + GFS area + Prithvi area"),
]
ACTUAL = W + " + true area"
BEST, TOKENS = W + " + GFS area + Prithvi area", W + " + GFS area + Prithvi area + Prithvi tokens"
Y23, Y24, GREY = "#1f4e9c", "#d9822b", "#7a7a7a"

# ---- panel A: R² within each zone (hourly, test year), averaged over the 8 zones; MAPE likewise ----
p = pd.read_parquet(src / "demand_predictions.parquet", columns=["test_year", "zone", "feature_set", "actual", "pred"])
p["e"] = p["pred"] - p["actual"]
z = p.groupby(["feature_set", "test_year", "zone"]).agg(sse=("e", lambda e: (e ** 2).sum()), sst=("actual", lambda a: ((a - a.mean()) ** 2).sum()),
                                                       mape=("e", lambda e: np.nan))
z["mape"] = (p.assign(ape=p["e"].abs() / p["actual"] * 100).groupby(["feature_set", "test_year", "zone"])["ape"].mean())
z["r2"] = 1 - z["sse"] / z["sst"]
stats = z.groupby(["feature_set", "test_year"])[["r2", "mape"]].mean().unstack()
table = pd.DataFrame({"label": [r[0] for r in ROWS] + ["+ actual weather (ceiling)"], "feature_set": [r[1] for r in ROWS] + [ACTUAL]})
for y in (2023, 2024):
    table[f"R2_{y}"] = table["feature_set"].map(stats[("r2", y)])
    table[f"MAPE_{y}"] = table["feature_set"].map(stats[("mape", y)])

# ---- panel B: daily MAPE (mean over zones) by day type, 2023 and 2024 pooled ----
d = pd.read_csv(src / "demand_daily_mape.csv", parse_dates=["day"])


def bridge(day):
    """Days next to a holiday that the federal-holiday flag misses."""
    m, dd, wd = day.month, day.day, day.weekday()
    after_thanksgiving = m == 11 and wd == 4 and 23 <= dd <= 29
    return (m, dd) in [(12, 24), (12, 26), (12, 31)] or after_thanksgiving or (m == 7 and dd == 3 and wd < 5) or (m == 7 and dd == 5 and wd == 4)


d["cat"] = np.select([(d["is_holiday"] == 1) | d["day"].map(bridge), d["cold_snap"] | d["heat_wave"], d["day"].dt.weekday >= 5],
                     ["Holidays &\nbridge days", "Cold snaps &\nheat waves", "Weekends"], "Other\nweekdays")
CATS = ["Holidays &\nbridge days", "Cold snaps &\nheat waves", "Weekends", "Other\nweekdays"]
best = d[d["feature_set"] == BEST]
worst = best.groupby("test_year", group_keys=False).apply(lambda g: g.nlargest(int(round(len(g) * 0.05)), "mape"))
n_days = best.groupby("cat").size()
n_worst = worst.groupby("cat").size().reindex(CATS, fill_value=0)
rng = np.random.default_rng(0)
MODELS = [("GFS + decoder (best forecast)", BEST, Y23, "o"), ("+ Prithvi tokens", TOKENS, Y24, "s"), ("Actual weather (ceiling)", ACTUAL, GREY, "D")]
rows_b = []
for name, fs, *_ in MODELS:
    g = d[d["feature_set"] == fs]
    for cat in CATS:
        v = g.loc[g["cat"] == cat, "mape"].to_numpy()
        boots = rng.choice(v, size=(2000, len(v))).mean(axis=1)
        rows_b.append({"model": name, "cat": cat, "mape": v.mean(), "lo": np.percentile(boots, 2.5), "hi": np.percentile(boots, 97.5),
                       "n_days": len(v), "n_worst5pct": int(n_worst[cat])})
pb = pd.DataFrame(rows_b)

# ---- figure ----
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.5})
ys = np.arange(len(ROWS))[::-1]
ceiling = table.loc[table["feature_set"] == ACTUAL, ["R2_2023", "R2_2024"]].to_numpy().mean()
ax.axvline(ceiling, color=GREY, lw=1, ls=(0, (3, 2)), zorder=1)
ax.text(ceiling - 0.008, ys[0] + 1.0, "actual weather", ha="right", va="center", fontsize=7, color="#555555")
for (label, fs), yv in zip(ROWS, ys):
    r = table.set_index("feature_set").loc[fs]
    ax.plot([r["R2_2023"], r["R2_2024"]], [yv, yv], color="#d0d0d0", lw=1.2, zorder=1)
    ax.scatter(r["R2_2023"], yv, s=34, color=Y23, marker="o", zorder=3, edgecolor="white", linewidth=0.6)
    ax.scatter(r["R2_2024"], yv, s=30, color=Y24, marker="s", zorder=3, edgecolor="white", linewidth=0.6)
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in ROWS])
ax.set_xlim(0.4, 0.95)
ax.set_ylim(ys[-1] - 0.6, ys[0] + 1.4)
ax.set_xlabel("Out-of-sample R² (hourly zonal demand)")
ax.scatter([], [], s=34, color=Y23, marker="o", label="2023")
ax.scatter([], [], s=30, color=Y24, marker="s", label="2024")
ax.legend(loc="lower left", frameon=False, fontsize=7.5, handletextpad=0.2)
ax.text(-0.02, 1.02, "A", transform=ax.transAxes, fontsize=11, fontweight="bold", ha="right", va="bottom")

xs = np.arange(len(CATS))
for k, (name, fs, color, marker) in enumerate(MODELS):
    g = pb[pb["model"] == name].set_index("cat").loc[CATS]
    x = xs + (k - 1) * 0.22
    bx.errorbar(x, g["mape"], yerr=[g["mape"] - g["lo"], g["hi"] - g["mape"]], fmt=marker, color=color, ms=5, elinewidth=1.1,
                capsize=0, label=name, zorder=3, markeredgecolor="white", markeredgewidth=0.5)
bx.set_xticks(xs)
bx.set_xticklabels(CATS, fontsize=7)
for x, c in zip(xs, CATS):
    bx.text(x, 3.62, f"{n_days[c]} days\n{n_worst[c]} of {len(worst)} worst", ha="center", va="bottom", fontsize=6.5, color="#555555")
bx.set_ylabel("Daily error, MAPE (%)")
bx.set_ylim(3.5, 7.2)
bx.set_xlim(-0.5, len(CATS) - 0.5)
bx.legend(loc="upper right", frameon=False, fontsize=7, handletextpad=0.2)
bx.text(-0.02, 1.02, "B", transform=bx.transAxes, fontsize=11, fontweight="bold", ha="right", va="bottom")
for a in (ax, bx):
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
ax.grid(axis="x", color="#ececec", lw=0.6, zorder=0)
bx.grid(axis="y", color="#ececec", lw=0.6, zorder=0)
out = root / "results/figures/fig2_demand"
fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
table.drop(columns="feature_set").to_csv(root / "results/figures/fig2_demand_table.csv", index=False)
pb.to_csv(root / "results/figures/fig2_demand_panelB.csv", index=False)
pd.set_option("display.width", 200)
print(table.drop(columns="feature_set").round(3).to_string(index=False))
print(pb.round(2).to_string(index=False))
print(out)
