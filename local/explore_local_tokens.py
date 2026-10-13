"""Exploration: do Prithvi's un-pooled patch tokens help demand prediction?

Each zone is given the encoder token(s) at its own location instead of the
crop-wide average, four ways:
    box_mean   mean of all 56 New England patches (same for every zone)
    nearest    the single patch nearest the zone's reference point
    bilinear   the four surrounding patches, interpolated to the point
    mean3x3    mean of the 3 x 3 patches around the point
Tokens are standardised and reduced to 50 principal components fitted on
2023, then one pooled CatBoost model is trained on 2023 and tested on 2024
(3,000 trees). Also prints how well each zone's own temperature can be read
linearly from the components.

Needs data/compact/prithvi_tokens_ne_<year>.npy/.json for 2023 and 2024 and
the modeling tables from prepare_modeling_table.py. Run from the repo root:
    .venv/bin/python local/explore_local_tokens.py
"""
import argparse
import json
import time

import pandas as pd, numpy as np
from catboost import CatBoostRegressor, Pool
import train_catboost_comparisons as dm
ZONES = {"ME": (45.0, -69.0), "NH": (43.7, -71.5), "VT": (44.3, -72.6), "CT": (41.6, -72.7), "RI": (41.7, -71.5), "NEMA": (42.36, -71.06), "SEMA": (41.9, -70.9), "WCMA": (42.3, -72.6)}
zn = list(ZONES)
toks, times = [], []
for y in (2023, 2024):
    toks.append(np.load(f"data/compact/prithvi_tokens_ne_{y}.npy", mmap_mode="r")); m = json.load(open(f"data/compact/prithvi_tokens_ne_{y}.json")); times += m["timestamps"]
lat_c, lon_c = np.array(m["patch_lat_centers"]), np.array(m["patch_lon_centers"]); times = pd.DatetimeIndex(times)
print("patch for each zone (nearest):", {z: (float(lat_c[np.abs(lat_c - la).argmin()]), float(lon_c[np.abs(lon_c - lo).argmin()])) for z, (la, lo) in ZONES.items()})
def zone_vectors(kind):
    out = []
    for a in toks:
        a = np.asarray(a, dtype=np.float32)                      # [T, 8, 7, C]
        per_zone = []
        for z, (la, lo) in ZONES.items():
            i, j = np.abs(lat_c - la).argmin(), np.abs(lon_c - lo).argmin()
            if kind == "nearest": v = a[:, i, j]
            elif kind == "mean3x3": v = a[:, max(i-1,0):i+2, max(j-1,0):j+2].mean(axis=(1, 2))
            elif kind == "bilinear":
                fi = np.clip((la - lat_c[0]) / (lat_c[1] - lat_c[0]), 0, len(lat_c) - 1.001); fj = np.clip((lo - lon_c[0]) / (lon_c[1] - lon_c[0]), 0, len(lon_c) - 1.001)
                i0, j0 = int(fi), int(fj); wi, wj = fi - i0, fj - j0
                v = (1-wi)*(1-wj)*a[:, i0, j0] + (1-wi)*wj*a[:, i0, j0+1] + wi*(1-wj)*a[:, i0+1, j0] + wi*wj*a[:, i0+1, j0+1]
            elif kind == "box_mean": v = a.mean(axis=(1, 2))
            per_zone.append(v)
        out.append(np.stack(per_zone, axis=1))                    # [T, 8, C]
        del a
    return np.concatenate(out)                                    # [Tall, 8, C]

base_cols = ["timestamp", "timestamp_local", "zone", "demand_MW"] + [c for c in dm.CALENDAR_FEATURES if c != "zone"] + dm.WEATHER_FEATURES
df = pd.concat([pd.read_parquet(f"data/modeling_table_{y}.parquet", columns=base_cols) for y in (2023, 2024)], ignore_index=True).dropna(subset=["weather_T2M"]).reset_index(drop=True)
df["wt"] = df.timestamp.dt.round("3h"); df = df[df.wt.isin(times)].reset_index(drop=True)
ti = times.get_indexer(df.wt); zi = df.zone.map({z: k for k, z in enumerate(zn)}).to_numpy()
tr = (df.timestamp_local.dt.year == 2023).to_numpy(); te = ~tr; y = df.demand_MW.to_numpy(); zones = df.zone.to_numpy()
train_t = times.year == 2023
# zone temperature at each weather time, for the readout check
wt = df[["wt", "zone", "weather_T2M"]].drop_duplicates(["wt", "zone"]); T2 = wt.pivot(index="wt", columns="zone", values="weather_T2M").reindex(times)[zn].to_numpy()

def fit_pca(V, k):
    T = V[train_t].reshape(-1, V.shape[-1]).astype(np.float64); mu, sd = T.mean(0), T.std(0); sd[sd == 0] = 1
    _, s, vt = np.linalg.svd((T - mu) / sd, full_matrices=False)
    return (((V - mu) / sd) @ vt[:k].T).astype(np.float32), float((s[:k]**2).sum() / (s**2).sum())
def run(P, label, with_weather, trees=3000):
    F = dm.CALENDAR_FEATURES + (dm.WEATHER_FEATURES if with_weather else []); Xd = df[F].copy()
    if P is not None:
        Z = P[ti, zi]
        Xd = pd.concat([Xd, pd.DataFrame(Z, columns=[f"p{i}" for i in range(Z.shape[1])], index=Xd.index)], axis=1)
    t0 = time.time(); m = CatBoostRegressor(iterations=trees, learning_rate=0.05, depth=6, random_seed=7, verbose=False, thread_count=-1, train_dir="results/phase3_2023_2024/catboost_info").fit(Pool(Xd[tr], y[tr], cat_features=["zone"]))
    p = m.predict(Xd[te]); ape = np.abs(p - y[te]) / y[te] * 100; byz = pd.Series(ape).groupby(zones[te]).mean()
    print(f"  {label:52s} MAPE {byz.mean():5.2f}%  median {np.median(ape):4.2f}%  VT {byz['VT']:5.1f}%  ({time.time()-t0:.0f}s)", flush=True)
    return p
def readout(P, label):
    res = []
    for k in range(len(zn)):
        A = np.c_[P[:, k], np.ones(len(P))]; t = T2[:, k]; ok = ~np.isnan(t); a, b = train_t & ok, ~train_t & ok
        coef = np.linalg.solve(A[a].T @ A[a] + np.eye(A.shape[1]), A[a].T @ t[a]); res.append(np.sqrt(np.mean((A[b] @ coef - t[b])**2)))
    print(f"  zone temperature read from {label}: RMSE {np.mean(res):.2f} K (range {min(res):.2f}-{max(res):.2f})", flush=True)

print("\nyear-ahead (train 2023, test 2024), 3,000 trees")
run(None, "calendar", False); run(None, "calendar + weather", True)
for kind in ("box_mean", "nearest", "bilinear", "mean3x3"):
    V = zone_vectors(kind); P, kept = fit_pca(V, 50); print(f"[{kind}] 50 PCs keep {kept:.0%}"); readout(P, kind)
    run(P, f"calendar + Prithvi local ({kind})", False); run(P, f"calendar + weather + Prithvi local ({kind})", True)
    if kind == "nearest":
        for k in (20, 200):
            Pk, kk = fit_pca(V, k); run(Pk, f"calendar + Prithvi local (nearest, {k} PCs, {kk:.0%})", False)
    del V
