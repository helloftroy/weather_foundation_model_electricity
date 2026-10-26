"""Evaluation beyond overall MAE: weather-defined hard days, the top price
hours, and week-block bootstrap confidence intervals on paired gains.

Hard days are defined from the weather, not from model errors, and with
thresholds taken from the TRAINING years only:
    cold snap   a day whose mean temperature (the given temperature series,
                averaged over the local calendar day) is at or below the
                5th percentile of training days
    heat wave   a day whose maximum temperature is at or above the 95th
                percentile of training-day maxima

The bootstrap resamples calendar weeks with replacement (keeping each
week's hours together, since errors within a week are correlated), and
recomputes the mean paired difference in absolute error between a feature
set and a reference set on the same hours. Reported: the observed
difference and a 95% interval. Negative = the feature set has lower error.
"""
import numpy as np
import pandas as pd


def hard_day_labels(local_time: pd.Series, temp_k: np.ndarray, is_train: np.ndarray, cold_q=0.05, hot_q=0.95) -> pd.DataFrame:
    """Label every row's local day as cold snap / heat wave using training-day thresholds."""
    day = local_time.dt.normalize()
    daily = pd.DataFrame({"day": day, "t": temp_k, "train": is_train}).groupby("day").agg(t_mean=("t", "mean"), t_max=("t", "max"), train=("train", "max"))
    cold_thr = daily.loc[daily["train"], "t_mean"].quantile(cold_q)
    hot_thr = daily.loc[daily["train"], "t_max"].quantile(hot_q)
    daily["cold_snap"] = daily["t_mean"] <= cold_thr
    daily["heat_wave"] = daily["t_max"] >= hot_thr
    out = daily.reindex(day)[["cold_snap", "heat_wave"]].reset_index(drop=True)
    out.attrs["thresholds_K"] = {"cold_snap_daily_mean_le": float(cold_thr), "heat_wave_daily_max_ge": float(hot_thr)}
    return out.fillna(False).astype(bool)


def block_bootstrap(diff: np.ndarray, week: np.ndarray, n_boot=2000, seed=0):
    """Mean of `diff` and a 95% interval from resampling weeks."""
    weeks, inv = np.unique(week, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(weeks), size=(n_boot, len(weeks)))
    boots = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarize(pred: pd.DataFrame, reference: str, value_col="actual", group_cols=("test_year",), top_q=0.95, unit="$"):
    """pred: one row per (feature_set, hour) with columns feature_set, actual, pred, week, cold_snap, heat_wave, test_year.
    Returns a table per group and feature set: MAE overall, on cold snaps, heat waves, top-5% hours,
    and the bootstrap difference in MAE against `reference` (overall and on the hard days)."""
    rows = []
    for keys, g in pred.groupby(list(group_cols)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        ref = g[g["feature_set"] == reference].set_index("row")
        top_thr = ref[value_col].quantile(top_q)
        for name, s in g.groupby("feature_set", sort=False):
            s = s.set_index("row")
            ae = (s["pred"] - s[value_col]).abs()
            d = ae - (ref["pred"] - ref[value_col]).abs().reindex(s.index)
            row = dict(zip(group_cols, keys))
            row.update({"feature_set": name, "MAE": ae.mean(), "MAE_cold_snap": ae[s["cold_snap"]].mean(),
                        "MAE_heat_wave": ae[s["heat_wave"]].mean(), f"MAE_top{int(round((1 - top_q) * 100))}pct": ae[s[value_col] >= top_thr].mean(),
                        "n_cold_snap_hours": int(s["cold_snap"].sum()), "n_heat_wave_hours": int(s["heat_wave"].sum())})
            for label, m in [("all", np.ones(len(s), bool)), ("cold_snap", s["cold_snap"].to_numpy()), ("heat_wave", s["heat_wave"].to_numpy())]:
                if m.sum() == 0 or name == reference:
                    continue
                mean, lo, hi = block_bootstrap(d.to_numpy()[m], s["week"].to_numpy()[m])
                row[f"dMAE_vs_ref_{label}"] = mean
                row[f"dMAE_vs_ref_{label}_ci"] = f"[{lo:+.2f}, {hi:+.2f}]"
            rows.append(row)
    return pd.DataFrame(rows)
