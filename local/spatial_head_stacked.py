"""Spatial readout heads, stacked into CatBoost (second design; the residual-correction version in spatial_head.py failed:
its training signal was dominated by the 2022 gas-price level).

Stage 1, heads. Target: the gas-normalised price, LMP / Henry Hub (last trading day <= D-1), so the head learns weather
and calendar effects on the heat rate, not gas levels. Head inputs: one representation (below) + context (calendar and
GFS area; no gas). Same conv head as spatial_head.py (1x1 conv to 32, two 3x3 convs, flatten, linear) for every
representation, L1 loss, early stopping. Out-of-fold over 5 interleaved calendar-month folds of the training years:
for fold k the head trains on folds other than k and k+1 (k+1 = early-stopping validation), with every row within
--buffer days of a fold-k day removed, and predicts fold k; the test-year feature is the mean of the 5 fold heads, all
averaged over --seeds seeds.
Representations: none (context only, the control); tokens_pca50 (the headline pooled readout, tokens over New England,
PCA-50, as a 1 x 1 x 50 input: the POSITIVE CONTROL, which should recover the known ~10% gain); tokens / merra / gfs_full
on the 8 x 7 patch grid, each 64 numbers per patch from one shared PCA basis, as "grid" and as the patch mean ("pooled").

Stage 2, CatBoost: the GFS feature set (calendar + past prices + Henry Hub + weather at issue + GFS area) + the one
head feature, raw LMP target, same settings as price_benchmark.py; references refit here: GFS alone and GFS + pooled
tokens PCA-50 jointly (the headline). Paired week-block bootstrap contrasts.

  .venv/bin/python local/price_benchmark.py --targets lmp --gas-features henryhub --export-residuals --out-dir results/regional/spatial_head
  .venv/bin/python local/spatial_head_stacked.py --dir results/regional/spatial_head --threads 4
Writes spatial_stacked_{base}_summary.csv, _contrasts.csv, _predictions.parquet and _head_features.npz in --dir.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor, Pool

sys.path.insert(0, str(Path(__file__).parent))
from eval_tools import block_bootstrap  # noqa: E402
from spatial_head import cell_pca, gfs_full_cells, merra_cells, root, token_cells, train_head  # noqa: E402


def pooled_tokens_pca50(targets, years, train_times):
    """The headline readout: tokens averaged over the New England extent, standardised PCA-50 on training times."""
    out = np.full((len(targets), 2560), np.nan, np.float32)
    for y in years:
        stem = root / f"data/prithvi_tokens_ne_{y}_lead24"
        meta = json.loads(stem.with_suffix(".json").read_text())
        issue = pd.DatetimeIndex(meta["timestamps"])
        a = np.load(f"{stem}_pooled_tokens.npy", mmap_mode="r")[:, list(meta["pooled"]["extents"]).index("ne")]
        sel = np.where(targets.year == y)[0]
        pos = issue.get_indexer(targets[sel] - pd.Timedelta(hours=24))
        out[sel[pos >= 0]] = a[pos[pos >= 0]]
    T = out[train_times & ~np.isnan(out[:, 0])].astype(np.float64)
    mu, sd = T.mean(0), T.std(0)
    sd[sd == 0] = 1
    _, _, vt = np.linalg.svd((T - mu) / sd, full_matrices=False)
    return (((out - mu) / sd) @ vt[:50].T).astype(np.float32)


def buffered(day, held, buffer_days):
    """Rows whose local day is within buffer_days of any held-out day."""
    days = np.asarray(day).astype("datetime64[D]")
    hd = np.unique(days[held])
    near = (hd[:, None] + np.arange(-buffer_days, buffer_days + 1)).ravel()
    return np.isin(days, near)


def catboost(X, y, tr, te, threads):
    m = CatBoostRegressor(iterations=3000, learning_rate=0.05, depth=6, random_seed=7, verbose=False, thread_count=threads,
                          allow_writing_files=False)
    m.fit(Pool(X[tr], y[tr]))
    return m.predict(X[te])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--base", default="gfs", choices=["gfs", "gfs_regional"])
    ap.add_argument("--test-years", type=int, nargs="+", default=[2023, 2024])
    ap.add_argument("--reps", nargs="+", default=["none", "tokens_pca50", "tokens", "merra", "gfs_full"])
    ap.add_argument("--views", nargs="+", default=["grid", "pooled"])
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--buffer", type=int, default=3, help="days dropped around each held-out fold")
    ap.add_argument("--gfs-full-dir", type=Path, default=root / "data/gfs")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    summary, contrasts, frames, feats = [], [], [], {}
    for test_year in args.test_years:
        z = np.load(args.dir / f"residuals_{test_year}_{args.base}.npz", allow_pickle=True)   # our own file (string arrays)
        targets = pd.DatetimeIndex(z["targets"])
        ti, tr, te, y, week, fold = z["ti"], z["tr"], z["te"], z["y"], z["week"], z["month_fold"]
        X, xn = z["X"], list(z["x_names"])
        day = z["local_day"]
        hh = X[:, xn.index("gas_known_henryhub")]
        hr = y / hh
        train_times = np.zeros(len(targets), bool)
        train_times[np.unique(ti[tr])] = True
        cn = list(z["context_names"])
        cal = z["context"][:, [i for i, c in enumerate(cn) if not c.startswith("gas_known")]]
        h, doy = cal[:, cn.index("hour")], cal[:, cn.index("day_of_year")]
        ctx = np.c_[cal, z["gfs_area"], np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                    np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)].astype(np.float32)
        ctx = ((ctx - ctx[tr].mean(0)) / (ctx[tr].std(0) + 1e-6)).astype(np.float32)
        r_mu, r_sd = np.nanmean(hr[tr]), np.nanstd(hr[tr])
        r = np.where(tr, (hr - r_mu) / r_sd, 0.0)
        print(f"\n[test {test_year}, base {args.base}] train {tr.sum()} h, test {te.sum()} h; heat-rate target mean {r_mu:.1f}, "
              f"sd {r_sd:.1f} MMBtu/MWh", flush=True)
        # folds: fit on not-k, not-v, outside the buffer around k; validate on v
        splits = []
        for k in range(5):
            v = (k + 1) % 5
            near = buffered(day, tr & (fold == k), args.buffer)
            splits.append((tr & (fold == k), tr & (fold != k) & (fold != v) & ~near, tr & (fold == v) & ~near))
        print("  folds: fit rows " + ", ".join(str(int(f.sum())) for _, f, _ in splits) + f" (buffer {args.buffer} d)", flush=True)

        tok50 = pooled_tokens_pca50(targets, sorted(set(targets.year)), train_times)
        preds = {"GFS (CatBoost)": catboost(X, y, tr, te, args.threads),
                 "GFS + pooled tokens PCA-50 (CatBoost, headline)": catboost(np.c_[X, tok50[np.clip(ti, 0, None)]], y, tr, te, args.threads)}
        for name, pr in preds.items():
            print(f"  {name:48s} MAE {np.abs(pr - y[te]).mean():6.2f}", flush=True)
        for rep in args.reps:
            t0 = time.time()
            if rep == "none":
                grids = {"-": np.zeros((len(targets), 1, 1, 1), np.float32)}
            elif rep == "tokens_pca50":
                grids = {"pooled": np.nan_to_num(tok50)[:, None, None, :]}
            else:
                cells = {"tokens": lambda: token_cells(targets, sorted(set(targets.year))),
                         "merra": lambda: merra_cells(targets, sorted(set(targets.year))),
                         "gfs_full": lambda: gfs_full_cells(targets, args.gfs_full_dir)}[rep]()
                g, frac = cell_pca(cells, train_times, args.k)
                del cells
                g = np.nan_to_num(g)
                grids = {v: (g if v == "grid" else g.mean(axis=(1, 2), keepdims=True)) for v in args.views}
                print(f"  {rep}: {args.k} per-patch components keep {frac:.1%} ({time.time() - t0:.0f} s)", flush=True)
            for view, g in grids.items():
                name = "context only (control)" if rep == "none" else f"{rep} {view}"
                feat = np.full(len(y), np.nan)
                test_f = []
                for seed in range(args.seeds):
                    oof = np.full(len(y), np.nan)
                    for held, fit, val in splits:
                        p, _, ep, n_par = train_head(g, ctx, ti, r, fit, val, held | te, "conv", rep != "none", seed, "cpu")
                        rows = np.where(held | te)[0]
                        out = np.full(len(y), np.nan)
                        out[rows] = p
                        oof[held] = out[held]
                        test_f.append(out[te])
                    feat = oof if seed == 0 else feat + oof
                feat = feat / args.seeds
                feat[te] = np.mean(test_f, 0)
                feat = feat * r_sd + r_mu
                feats[f"{test_year} {name}"] = feat
                corr_tr = np.corrcoef(feat[tr], hr[tr])[0, 1]
                pr = catboost(np.c_[X, feat], y, tr, te, args.threads)
                preds[f"GFS + head: {name}"] = pr
                mae = np.abs(pr - y[te]).mean()
                print(f"  head {name:26s} params {n_par:6d}; OOF corr with heat rate {corr_tr:.3f}, test {np.corrcoef(feat[te], hr[te])[0, 1]:.3f}; "
                      f"CatBoost MAE {mae:6.2f}", flush=True)
                summary.append({"test_year": test_year, "base": args.base, "model": f"GFS + head: {name}", "rep": rep, "view": view,
                                "head_params": n_par, "MAE": mae, "oof_corr_heat_rate": corr_tr,
                                "test_corr_heat_rate": np.corrcoef(feat[te], hr[te])[0, 1]})
        for name in ("GFS (CatBoost)", "GFS + pooled tokens PCA-50 (CatBoost, headline)"):
            summary.append({"test_year": test_year, "base": args.base, "model": name, "MAE": np.abs(preds[name] - y[te]).mean()})
        ae = {k: np.abs(v - y[te]) for k, v in preds.items()}
        wk = week[te]
        G, C = "GFS (CatBoost)", "GFS + head: context only (control)"
        pairs = [(m, G) for m in ae if m != G]
        pairs += [(m, C) for m in ae if m.startswith("GFS + head:") and m != C]
        for v in args.views:
            for other in ("merra", "gfs_full"):
                pairs.append((f"GFS + head: tokens {v}", f"GFS + head: {other} {v}"))
        for rep in ("tokens", "merra", "gfs_full"):
            pairs.append((f"GFS + head: {rep} grid", f"GFS + head: {rep} pooled"))
        pairs.append(("GFS + head: tokens_pca50 pooled", "GFS + pooled tokens PCA-50 (CatBoost, headline)"))
        for a, b in pairs:
            if a in ae and b in ae:
                m, lo, hi = block_bootstrap(ae[a] - ae[b], wk)
                contrasts.append({"test_year": test_year, "model": a, "minus": b, "dMAE": m, "lo": lo, "hi": hi,
                                  "pct_of_second": -100 * m / ae[b].mean()})
        frames.append(pd.DataFrame({"row": np.where(te)[0], "test_year": test_year, "actual": y[te], "week": wk, **preds}))
    d, b = args.dir, args.base
    pd.DataFrame(summary).to_csv(d / f"spatial_stacked_{b}_summary.csv", index=False)
    c = pd.DataFrame(contrasts)
    c.to_csv(d / f"spatial_stacked_{b}_contrasts.csv", index=False)
    pd.concat(frames).to_parquet(d / f"spatial_stacked_{b}_predictions.parquet", index=False)
    np.savez(d / f"spatial_stacked_{b}_head_features.npz", **{k.replace(" ", "_"): v for k, v in feats.items()})
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    c["dMAE [95%]"] = [f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]" for m, lo, hi in zip(c["dMAE"], c["lo"], c["hi"])]
    print(c.pivot_table(index=["model", "minus"], columns="test_year", values="dMAE [95%]", aggfunc="first", sort=False).to_string())


if __name__ == "__main__":
    main()
