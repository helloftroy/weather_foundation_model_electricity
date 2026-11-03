"""Gridded principal components of the full GFS state over New England, one basis per year-ahead fold.

Input: gfs_full_{year}_meta.npz / _ne_grid.npy from download_gfs_full_state.py (150 fields x 17 x 18 points at 0.5 deg).
For each test year (2023, 2024) the basis is fitted on the forecasts valid in the training years (2021 to test year - 1):
every (field, point) is centred on its training mean and divided by its field's training standard deviation pooled over
the box (so each of the 150 fields carries equal weight and near-constant points are not inflated); the top 50
components come from the training Gram matrix. All samples are projected on each fold's basis.

Output (small, to copy to the local data/gfs/): gfs_full_state_features.npz with run_time, lead_hours, valid_time,
names, ne_mean and northeast_mean (box means [sample, 150]), grid_pc_2023 / grid_pc_2024 [sample, 50] and the variance
fraction each keeps.
"""
import argparse
from pathlib import Path

import numpy as np

YEARS = (2021, 2022, 2023, 2024)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--k", type=int, default=50)
    args = p.parse_args()
    metas = [np.load(args.dir / f"gfs_full_{y}_meta.npz") for y in YEARS]
    grids = [np.load(args.dir / f"gfs_full_{y}_ne_grid.npy", mmap_mode="r") for y in YEARS]
    yr = np.concatenate([np.full(len(m["lead_hours"]), y) for m, y in zip(metas, YEARS)])
    good = np.concatenate([~np.isnan(m["ne_mean"]).any(axis=1) for m in metas])
    n_f = grids[0].shape[1]
    X = np.concatenate([np.asarray(g, np.float32).reshape(len(g), -1) for g in grids])   # [sample, field * points]
    field = np.repeat(np.arange(n_f), X.shape[1] // n_f)
    out = {k: np.concatenate([m[k] for m in metas]) for k in ("run_time", "lead_hours", "valid_time", "ne_mean", "northeast_mean")}
    out["names"] = metas[0]["names"]
    for test in (2023, 2024):
        tr = good & (yr < test)
        mu = X[tr].mean(0, dtype=np.float64)
        A = X[tr] - mu
        sd = np.sqrt(np.bincount(field, weights=(A.astype(np.float64) ** 2).sum(0)) / (tr.sum() * np.bincount(field)))
        sd[sd == 0] = 1
        A = (A / sd[field]).astype(np.float64)
        w, u = np.linalg.eigh(A @ A.T)
        w, u = w[::-1], u[:, ::-1]
        V = (A.T @ u[:, :args.k]) / np.sqrt(w[:args.k])                       # [features, k], unit columns
        scores = np.full((len(X), args.k), np.nan, np.float32)
        for i in range(0, len(X), 2000):
            scores[i:i + 2000] = (((X[i:i + 2000] - mu) / sd[field]) @ V).astype(np.float32)
        scores[~good] = np.nan
        out[f"grid_pc_{test}"] = scores
        out[f"grid_frac_{test}"] = np.float32(w[:args.k].sum() / w.sum())
        print(f"test {test}: basis from {tr.sum()} training forecasts; {args.k} PCs keep {w[:args.k].sum() / w.sum():.1%}", flush=True)
    np.savez(args.dir / "gfs_full_state_features.npz", **out)
    print(f"Wrote {args.dir / 'gfs_full_state_features.npz'}: {len(X)} forecasts, {good.sum()} complete", flush=True)


if __name__ == "__main__":
    main()
