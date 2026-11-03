"""Spatial readout heads: is the token grid a better representation than the raw state grid, at equal readout capacity?

Pooled PCA (the headline readout) averages the tokens over New England before CatBoost sees them. Here every
representation keeps its 8 x 7 grid (Prithvi's token patches over New England, 1 deg x 1.25 deg) and goes through the
SAME small trained head:
  tokens   Prithvi final-layer tokens, 2,560 per patch (lead-24 sample issued at t - 24 h)
  merra    Prithvi's own MERRA-2 input fields at issue time, 160 channels x 2 x 2 pixels = 640 per patch
  gfs_full the GFS 24-27 h forecast of 150 fields, averaged per patch (gfs_full_state_patch_grid.npy, optional)
  none     no grid (context only): what the head alone gets from calendar, gas and GFS area
Each grid is reduced to the same k (default 64) numbers per patch with ONE PCA basis shared by all patches (standardised,
fitted on training target times), so the heads have identical parameter counts. Views: "grid" (8 x 7 x k) and "pooled"
(the patch mean, 1 x 1 x k), to measure what spatial structure adds for each representation.

Target: the residual of the CatBoost GFS model (price_benchmark.py --export-residuals): out-of-fold residuals on the
training rows, the head's correction is added to the test predictions. Heads: "conv" (1x1 conv to 32, two 3x3 convs,
flatten, linear to 32) or "attn" (linear to 32 + learned position embedding, 2 transformer layers, 4 heads, CLS token);
both concatenated with the context (calendar, known gas, GFS area; 16-d projection) and an MLP to the correction. L1 loss
(the metric is MAE), AdamW, early stopping on one of the five interleaved month folds of the training years, an ensemble
of --seeds seeds. Paired contrasts with a 95% week-block bootstrap. Writes spatial_head_{base}_predictions.parquet,
spatial_head_{base}_summary.csv and _contrasts.csv in --residuals-dir.

  .venv/bin/python local/price_benchmark.py --targets lmp --gas-features henryhub --export-residuals --out-dir results/regional/spatial_head
  .venv/bin/python local/spatial_head.py --residuals-dir results/regional/spatial_head [--base gfs_regional] [--gfs-full-dir data/gfs]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from eval_tools import block_bootstrap  # noqa: E402

root = Path(__file__).resolve().parent.parent
H, W = 8, 7


def cell_pca(cells, train_times, k, chunk=512):
    """cells: [T, 8, 7, D] (NaN rows where missing) -> [T, 8, 7, k] on one standardised basis shared by all patches,
    fitted on training target times; components scaled to unit training variance."""
    T, D = len(cells), cells.shape[-1]
    use = np.where(train_times & ~np.isnan(cells[:, 0, 0, 0]))[0]
    n, s, ss, xtx = 0, np.zeros(D), np.zeros(D), np.zeros((D, D))
    for i in range(0, len(use), chunk):
        x = np.asarray(cells[use[i:i + chunk]], np.float64).reshape(-1, D)
        n += len(x); s += x.sum(0); ss += (x ** 2).sum(0); xtx += x.T @ x
    mu = s / n
    sd = np.sqrt(np.maximum(ss / n - mu ** 2, 1e-12))
    corr = (xtx / n - np.outer(mu, mu)) / np.outer(sd, sd)
    w, v = np.linalg.eigh(corr)
    w, v = w[::-1][:k], v[:, ::-1][:, :k]
    proj = (v / sd[:, None] / np.sqrt(np.maximum(w, 1e-12))).astype(np.float32)
    out = np.full((T, H, W, k), np.nan, np.float32)
    for i in range(0, T, chunk):
        x = np.asarray(cells[i:i + chunk], np.float32)
        out[i:i + chunk] = (x - mu.astype(np.float32)) @ proj
    frac = w.sum() / np.trace(corr)
    return out, frac


def token_cells(targets, years):
    """Tokens for each target time (sample issued 24 h before), [T, 8, 7, 2560] float16 (memory-mapped reads)."""
    out = np.full((len(targets), H, W, 2560), np.nan, np.float16)
    for y in years:
        stem = root / f"data/prithvi_tokens_ne_{y}_lead24"
        issue = pd.DatetimeIndex(json.loads(stem.with_suffix(".json").read_text())["timestamps"])
        a = np.load(f"{stem}.npy", mmap_mode="r")
        sel = np.where(targets.year == y)[0]
        pos = issue.get_indexer(targets[sel] - pd.Timedelta(hours=24))
        ok = pos >= 0
        for i in range(0, ok.sum(), 256):
            out[sel[ok][i:i + 256]] = a[np.sort(pos[ok][i:i + 256])][np.argsort(np.argsort(pos[ok][i:i + 256]))]
    return out


def merra_cells(targets, years):
    """Prithvi's input fields at issue time per patch: [T, 8, 7, 160 * 2 * 2]."""
    out = np.full((len(targets), H, W, 640), np.nan, np.float16)
    for y in years:
        stem = root / f"data/prithvi_tokens_ne_{y}_lead24"
        issue = pd.DatetimeIndex(json.loads(stem.with_suffix(".json").read_text())["timestamps"])
        a = np.load(f"{stem}_inputs.npy", mmap_mode="r")                          # [S, 160, 16, 14]
        sel = np.where(targets.year == y)[0]
        pos = issue.get_indexer(targets[sel] - pd.Timedelta(hours=24))
        ok = pos >= 0
        x = np.asarray(a[pos[ok]], np.float16).reshape(ok.sum(), 160, H, 2, W, 2).transpose(0, 2, 4, 1, 3, 5)
        out[sel[ok]] = x.reshape(ok.sum(), H, W, 640)
    return out


def gfs_full_cells(targets, d, leads=(24, 27)):
    """GFS full state per patch for each target: the 24 h forecast valid then, else the 27 h one. [T, 8, 7, 150]."""
    f = np.load(d / "gfs_full_state_features.npz")
    g = np.load(d / "gfs_full_state_patch_grid.npy", mmap_mode="r")
    vt, gl = pd.DatetimeIndex(f["valid_time"]), f["lead_hours"]
    pick = np.full(len(targets), -1)
    for want in leads[::-1]:
        ix = pd.Series(np.where(gl == want)[0], index=vt[gl == want])
        gp = ix[~ix.index.duplicated()].reindex(targets).to_numpy()
        pick[~np.isnan(gp)] = gp[~np.isnan(gp)].astype(int)
    out = np.full((len(targets), H, W, g.shape[1]), np.nan, np.float32)
    ok = pick >= 0
    out[ok] = np.asarray(g[pick[ok]], np.float32).transpose(0, 2, 3, 1)
    return out


class Head(nn.Module):
    def __init__(self, k, h, w, n_ctx, kind, use_grid=True, d=32):
        super().__init__()
        self.kind, self.use_grid = kind, use_grid
        if use_grid and kind == "conv":
            self.enc = nn.Sequential(nn.Conv2d(k, d, 1), nn.GELU(), nn.Conv2d(d, d, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(d, d, 3, padding=1), nn.GELU(), nn.Flatten(), nn.Linear(d * h * w, d))
        elif use_grid:
            self.inp = nn.Linear(k, d)
            self.pos = nn.Parameter(torch.zeros(1, h * w, d))
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, 2)
        self.ctx = nn.Linear(n_ctx, 16)
        self.mlp = nn.Sequential(nn.Linear((d if use_grid else 0) + 16, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, grid, ctx):
        parts = [self.ctx(ctx)]
        if self.use_grid:
            if self.kind == "conv":
                parts.insert(0, self.enc(grid.permute(0, 3, 1, 2)))
            else:
                z = self.inp(grid.flatten(1, 2)) + self.pos
                z = self.enc(torch.cat([self.cls.expand(len(z), -1, -1), z], 1))
                parts.insert(0, z[:, 0])
        return self.mlp(torch.cat(parts, 1)).squeeze(1)


def train_head(grid, ctx, ti, r, fit, val, pred_rows, kind, use_grid, seed, device, max_epochs=100, patience=10, bs=512):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    G = torch.tensor(grid, device=device)
    C = torch.tensor(ctx, dtype=torch.float32, device=device)
    R = torch.tensor(r, dtype=torch.float32, device=device)
    TI = torch.tensor(ti, device=device)
    model = Head(grid.shape[-1], grid.shape[1], grid.shape[2], ctx.shape[1], kind, use_grid).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    fit_idx, val_idx = np.where(fit)[0], torch.tensor(np.where(val)[0], device=device)
    best, best_state, bad = np.inf, None, 0

    def predict(idx):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(G[TI[idx[i:i + 4096]]], C[idx[i:i + 4096]]) for i in range(0, len(idx), 4096)])

    for epoch in range(max_epochs):
        model.train()
        perm = rng.permutation(fit_idx)
        for i in range(0, len(perm), bs):
            b = torch.tensor(perm[i:i + bs], device=device)
            loss = (model(G[TI[b]], C[b]) - R[b]).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = (predict(val_idx) - R[val_idx]).abs().mean().item()
        if v < best - 1e-4:
            best, bad, best_state = v, 0, {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    n_par = sum(p.numel() for p in model.parameters())
    return predict(torch.tensor(np.where(pred_rows)[0], device=device)).cpu().numpy(), best, epoch + 1, n_par


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--residuals-dir", type=Path, required=True)
    ap.add_argument("--base", default="gfs", choices=["gfs", "gfs_regional"])
    ap.add_argument("--test-years", type=int, nargs="+", default=[2023, 2024])
    ap.add_argument("--reps", nargs="+", default=["none", "tokens", "merra"], help="add gfs_full once its patch grid is in --gfs-full-dir")
    ap.add_argument("--heads", nargs="+", default=["conv", "attn"])
    ap.add_argument("--views", nargs="+", default=["grid", "pooled"])
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gfs-full-dir", type=Path, default=root / "data/gfs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for torch (keep low to leave the machine usable)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rows, contrasts = [], []
    pred_frames = []
    for test_year in args.test_years:
        z = np.load(args.residuals_dir / f"residuals_{test_year}_{args.base}.npz", allow_pickle=False)
        targets = pd.DatetimeIndex(z["targets"])
        ti, tr, te, y, oof, test_pred = z["ti"], z["tr"], z["te"], z["y"], z["oof"], z["test_pred"]
        week, mfold = z["week"], z["month_fold"]
        train_times = np.zeros(len(targets), bool)
        train_times[np.unique(ti[tr])] = True
        ctx = np.c_[z["context"], z["gfs_area"]].astype(np.float32)
        names = list(z["context_names"])
        h, doy = ctx[:, names.index("hour")], ctx[:, names.index("day_of_year")]
        ctx = np.c_[ctx, np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]
        mu, sd = ctx[tr].mean(0), ctx[tr].std(0) + 1e-6
        ctx = ((ctx - mu) / sd).astype(np.float32)
        r = (y - oof)
        r_sd = np.nanstd(r[tr])
        r = np.where(tr, r / r_sd, 0.0)
        val, fit = tr & (mfold == 0), tr & (mfold != 0)
        ae_base = np.abs(test_pred[te] - y[te])
        print(f"\n[test {test_year}, base {args.base}] train {tr.sum()} h (fit {fit.sum()}, val {val.sum()}), test {te.sum()} h; "
              f"base MAE {ae_base.mean():.2f}; out-of-fold residual sd {r_sd:.2f}", flush=True)
        years = sorted(set(targets.year))
        preds = {"base": test_pred[te]}
        for rep in args.reps:
            if rep == "none":
                grid = np.zeros((len(targets), 1, 1, 1), np.float32)
            else:
                t0 = time.time()
                cells = {"tokens": token_cells, "merra": merra_cells}[rep](targets, years) if rep != "gfs_full" else gfs_full_cells(targets, args.gfs_full_dir)
                grid, frac = cell_pca(cells, train_times, args.k)
                del cells
                print(f"  {rep}: {args.k} shared per-patch components keep {frac:.1%} of the standardised variance ({time.time() - t0:.0f} s)", flush=True)
            missing = np.isnan(grid).reshape(len(grid), -1).any(1)
            grid = np.nan_to_num(grid)
            for view in (["-"] if rep == "none" else args.views):
                g = grid if view != "pooled" else grid.mean(axis=(1, 2), keepdims=True)
                for kind in (["mlp"] if rep == "none" else args.heads):
                    name = "context only" if rep == "none" else f"{rep} {view} {kind}"
                    corr, info = [], []
                    for seed in range(args.seeds):
                        c, best, ep, n_par = train_head(g, ctx, ti, r, fit & ~missing[ti], val & ~missing[ti], te, kind if rep != "none" else "conv",
                                                        rep != "none", seed, args.device)
                        corr.append(c)
                        info.append((best, ep))
                    pr = test_pred[te] + np.mean(corr, 0) * r_sd
                    preds[name] = pr
                    mae = np.abs(pr - y[te]).mean()
                    rows.append({"test_year": test_year, "base": args.base, "model": name, "rep": rep, "view": view, "head": kind,
                                 "params": n_par, "MAE": mae, "MAE_base": ae_base.mean(), "dMAE_vs_base": mae - ae_base.mean(),
                                 "val_L1": np.mean([b for b, _ in info]), "epochs": [e for _, e in info]})
                    print(f"  {name:22s} params {n_par:7d}  MAE {mae:6.2f}  (base {ae_base.mean():.2f}; epochs {[e for _, e in info]})", flush=True)
        wk = week[te]
        ae = {k: np.abs(v - y[te]) for k, v in preds.items()}
        pairs = [(m, "base") for m in ae if m != "base"]
        for view in args.views:
            for kind in args.heads:
                for other in ("merra", "gfs_full"):
                    if f"tokens {view} {kind}" in ae and f"{other} {view} {kind}" in ae:
                        pairs.append((f"tokens {view} {kind}", f"{other} {view} {kind}"))
        for rep in [x for x in args.reps if x != "none"]:
            for kind in args.heads:
                if f"{rep} grid {kind}" in ae and f"{rep} pooled {kind}" in ae:
                    pairs.append((f"{rep} grid {kind}", f"{rep} pooled {kind}"))
        for a, b in pairs:
            m, lo, hi = block_bootstrap(ae[a] - ae[b], wk)
            contrasts.append({"test_year": test_year, "base": args.base, "model": a, "minus": b, "dMAE": m, "lo": lo, "hi": hi,
                              "pct": -100 * m / ae[b].mean()})
        pred_frames.append(pd.DataFrame({"row": np.where(te)[0], "test_year": test_year, "actual": y[te], "week": wk, **preds}))
    out = args.residuals_dir
    pd.concat(pred_frames).to_parquet(out / f"spatial_head_{args.base}_predictions.parquet", index=False)
    pd.DataFrame(rows).to_csv(out / f"spatial_head_{args.base}_summary.csv", index=False)
    c = pd.DataFrame(contrasts)
    c.to_csv(out / f"spatial_head_{args.base}_contrasts.csv", index=False)
    pd.set_option("display.width", 250)
    c["dMAE [95%]"] = [f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]" for m, lo, hi in zip(c["dMAE"], c["lo"], c["hi"])]
    print(c.pivot_table(index=["model", "minus"], columns="test_year", values="dMAE [95%]", aggfunc="first", sort=False).to_string())


if __name__ == "__main__":
    main()
