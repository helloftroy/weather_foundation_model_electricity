"""One year of global-grid Prithvi output: what does it know about local
temperature, and does it help predict demand and price?

Uses the files written by extract_embeddings.py on the global grid for one
year and one lead time: the New England patch tokens (final encoder layer and
any saved earlier layers), the model's normalised inputs for the same box,
and its decoder output and climatology for the surface variables.

Times. Each sample is ISSUED at its timestamp (the model's latest input time)
and describes the TARGET time, issue + lead. Everything below is lined up on
the target time: a row for hour H uses the sample whose target is the 3-hourly
time nearest H. At lead 0 the two are the same (a nowcast). At lead 24 every
Prithvi feature, and the "issue-time weather" baseline, is about a day old.

Each zone is represented by the 3 x 3 patches around its reference point
(about 3 x 3.75 degrees): tokens are averaged over those patches, input,
decoder and climatology channels over the corresponding 6 x 6 pixels.
System-level features (for price) average over the whole 8 x 7 patch box.

Feature blocks:
    weather (target)      the zone's six MERRA-2 numbers at the target time:
                          perfect knowledge, the ceiling
    weather (issue)       the same six numbers at the issue time: persistence,
                          what you would know without a forecast (lead > 0 only)
    raw inputs            the model's 160 input channels at the issue time,
                          50 principal components
    tokens <layer>        each saved encoder layer, 50 principal components
    decoder forecast      the decoder's 20 surface channels for the target time
    decoder weather       only the decoder's T2M, QV2M, U10M, V10M, SWGNT: the
                          same five variables as the weather baseline
    true weather, area    the same five variables, TRUE values at the target
                          time, averaged over the same zone area as the decoder
                          (taken from the lead-0 run's inputs). The fair
                          ceiling for the decoder: same variables, same area,
                          same units, perfect knowledge
    gas (known)           (price only) natural gas, $/MMBtu, as known when the
                          day-ahead market for target day D is bid (morning
                          of D-1): Henry Hub spot for the last trading day
                          on or before D-2, and the last Algonquin Citygate
                          quote on or before D-2, held flat (no interpolation)
    gas (same day)        (price only) day D's values as in the 2024 price
                          models: Henry Hub for D, Algonquin interpolated
                          between weekly quotes (uses the NEXT quote, so an
                          upper bound, not a forecast)
    past price            (price only) the hub price 24 h, 48 h and 7 days
                          before the target hour, and the mean of the 24 hours
                          ending 24 h before it: all known when the day-ahead
                          market for the target day is bid

Part 1, temperature probe: ridge fit of each zone's 2 m temperature at the
target time on each block. Part 2, demand: one pooled CatBoost model per
feature set (zone as a categorical feature). Part 3, hub day-ahead price:
one row per hour, system-level features. Splits: blocked (train days 1-18,
test day 22 onward) and chronological (train Jan-Aug, test Oct-Dec), fixed
3,000 trees. Principal components are fitted on training times only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

import train_catboost_comparisons as dm

ZONES = {"ME": (45.0, -69.0), "NH": (43.7, -71.5), "VT": (44.3, -72.6), "CT": (41.6, -72.7), "RI": (41.7, -71.5),
         "NEMA": (42.36, -71.06), "SEMA": (41.9, -70.9), "WCMA": (42.3, -72.6)}
WEATHER_VARS = ["T2M", "QV2M", "U10M", "V10M", "SWGNT"]
WEATHER_NAMES = [f"weather_{v}" for v in WEATHER_VARS] + ["weather_wind_speed"]
PRICE_CALENDAR = [f for f in dm.CALENDAR_FEATURES if f != "zone"]


def readable(path):
    """True if the .npy file is complete (a copy still in progress is shorter than its header says)."""
    try:
        np.load(path, mmap_mode="r")
        return True
    except (OSError, ValueError):
        print(f"  skipping {Path(path).name}: missing or incomplete")
        return False


def zone_windows(lat_c, lon_c):
    """For each zone, the 3 x 3 patch window around its nearest patch, as slices."""
    out = {}
    for z, (la, lo) in ZONES.items():
        i, j = int(np.abs(lat_c - la).argmin()), int(np.abs(lon_c - lo).argmin())
        out[z] = (slice(max(i - 1, 0), i + 2), slice(max(j - 1, 0), j + 2))
    return out


def per_zone_tokens(path, windows):
    a = np.load(path, mmap_mode="r")
    return np.stack([np.asarray(a[:, r, c], dtype=np.float32).mean(axis=(1, 2)) for r, c in windows.values()], axis=1)   # [T, zone, C]


def box_tokens(path):
    a = np.load(path, mmap_mode="r")
    return np.concatenate([np.asarray(a[i:i + 256], dtype=np.float32).mean(axis=(1, 2)) for i in range(0, len(a), 256)])  # [T, C]


def per_zone_pixels(path, windows, patch=2):
    """Pixel fields [T, ch, lat_px, lon_px] averaged over each zone's window -> [T, zone, ch]."""
    a = np.load(path, mmap_mode="r")
    out = []
    for r, c in windows.values():
        pr, pc = slice(r.start * patch, r.stop * patch), slice(c.start * patch, c.stop * patch)
        out.append(np.asarray(a[:, :, pr, pc], dtype=np.float32).mean(axis=(2, 3)))
    return np.stack(out, axis=1)


def box_pixels(path):
    return np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32).mean(axis=(2, 3))   # [T, ch]


def pca(V, train_t, k):
    """Project V ([T, ..., d]) on the top k principal components of its training times."""
    T = V[train_t].reshape(-1, V.shape[-1]).astype(np.float64)
    mu, sd = T.mean(0), T.std(0)
    sd[sd == 0] = 1
    Z = (T - mu) / sd
    _, s, vt = np.linalg.svd(Z, full_matrices=False)
    return (((V - mu) / sd) @ vt[:k].T).astype(np.float32), float((s[:k] ** 2).sum() / (s ** 2).sum())


def probe(F, target, train_t, test_t, lam=1.0):
    """Ridge fit of target on features F (both [T]), returns test RMSE and R2."""
    ok = ~np.isnan(target)
    a, b = train_t & ok, test_t & ok
    X = np.c_[F, np.ones(len(F))]
    mu, sd = X[a].mean(0), X[a].std(0)
    sd[-1], mu[-1] = 1, 0
    sd[sd == 0] = 1
    Xs = (X - mu) / sd
    pen = lam * np.eye(Xs.shape[1])
    pen[-1, -1] = 0  # do not shrink the intercept
    coef = np.linalg.solve(Xs[a].T @ Xs[a] + pen, Xs[a].T @ target[a])
    err = Xs[b] @ coef - target[b]
    return float(np.sqrt(np.mean(err ** 2))), float(1 - (err ** 2).sum() / ((target[b] - target[b].mean()) ** 2).sum())


def zone_weather(weather, at, zn):
    w = weather.reindex(at)
    return np.stack([np.stack([w[f"{z}_{v}"].to_numpy() for v in WEATHER_VARS] + [np.hypot(w[f"{z}_U10M"], w[f"{z}_V10M"]).to_numpy()], axis=1)
                     for z in zn], axis=1).astype(np.float32)                                 # [T, zone, 6]


def fit_catboost(X, y, tr, te, out_dir, cat_features):
    model = CatBoostRegressor(iterations=ARGS.iterations, learning_rate=0.05, depth=6, random_seed=7, verbose=False, thread_count=-1,
                              train_dir=str(out_dir / "catboost_info"))
    model.fit(Pool(X[tr], y[tr], cat_features=cat_features))
    return model.predict(X[te])


def main() -> None:
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--lead", type=int, default=0, help="Lead time of the run to analyse (file suffix _leadN when non-zero)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--tokens-dir", type=Path, default=None, help="Where prithvi_tokens_ne_YEAR* are (default: DATA_DIR/compact)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--parts", nargs="+", default=["probe", "demand", "price"], choices=["probe", "demand", "price", "readout"])
    parser.add_argument("--only", nargs="+", default=None, help="Run only feature sets whose name contains one of these strings")
    parser.add_argument("--price-market", default="dayahead", choices=["dayahead", "realtime"])
    args = ARGS = parser.parse_args()
    c = args.data_dir / "compact"
    tok_dir = args.tokens_dir or c
    stem = tok_dir / (f"prithvi_tokens_ne_{args.year}" + (f"_lead{args.lead}" if args.lead else ""))
    meta = json.loads(stem.with_suffix(".json").read_text())
    if meta["lead_time_hours"] != args.lead:
        raise SystemExit(f"{stem.name}.json says lead {meta['lead_time_hours']} h, not {args.lead} h.")
    issue = pd.DatetimeIndex(meta["timestamps"])
    target = issue + pd.Timedelta(hours=args.lead)
    lat_c, lon_c = np.array(meta["patch_lat_centers"]), np.array(meta["patch_lon_centers"])
    windows = zone_windows(lat_c, lon_c)
    zn = list(ZONES)
    print(f"{args.year}, lead {args.lead} h: {len(issue)} samples, issued {issue[0]} .. {issue[-1]}, targets {target[0]} .. {target[-1]}; "
          f"grid {meta.get('model_grid')}, block order {meta.get('block_order')}")

    layers = {"final": stem.with_suffix(".npy")}
    for name, fname in meta.get("layers", {}).items():
        layers[name] = tok_dir / fname
    order = ["embed", "0", "4", "8", "12", "16", "20", "final"]
    layers = {k: layers[k] for k in order if k in layers and readable(layers[k])}
    fields = meta["fields"]
    i_t2m = fields["inputs"]["channels"].index("T2M")
    j_t2m = fields["climate_sfc"]["channels"].index("T2M")
    global DEC_IDX
    DEC_IDX = [fields["recon_sfc"]["channels"].index(v) for v in WEATHER_VARS]
    has_recon = readable(tok_dir / fields["recon_sfc"]["file"])

    weather = pd.read_parquet(c / f"conventional_weather_{args.year}.parquet").set_index("timestamp")
    W_target, W_issue = zone_weather(weather, target, zn), zone_weather(weather, issue, zn)
    day = target.day.to_numpy()
    train_t, test_t = day <= 18, day >= 22
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = per_zone_pixels(tok_dir / fields["inputs"]["file"], windows)                         # [T, zone, 160], issue time
    recon = per_zone_pixels(tok_dir / fields["recon_sfc"]["file"], windows) if has_recon else None  # [T, zone, 20], target time
    clim = per_zone_pixels(tok_dir / fields["climate_sfc"]["file"], windows)

    # True weather over the zone area at the target time, from the lead-0 run's inputs.
    area_true = area_true_box = None
    stem0 = tok_dir / f"prithvi_tokens_ne_{args.year}"
    if args.lead and stem0.with_suffix(".json").exists():
        meta0 = json.loads(stem0.with_suffix(".json").read_text())
        f0 = tok_dir / meta0["fields"]["inputs"]["file"]
        if readable(f0):
            idx0 = [meta0["fields"]["inputs"]["channels"].index(v) for v in WEATHER_VARS]
            pos = pd.DatetimeIndex(meta0["timestamps"]).get_indexer(target)
            ok = pos >= 0
            z = per_zone_pixels(f0, windows)[:, :, idx0]
            area_true = np.full((len(target), len(zn), len(idx0)), np.nan, np.float32)
            area_true[ok] = z[pos[ok]]
            b0 = box_pixels(f0)[:, idx0]
            area_true_box = np.full((len(target), len(idx0)), np.nan, np.float32)
            area_true_box[ok] = b0[pos[ok]]
            print(f"True weather over the zone areas: from {f0.name}, {ok.sum()}/{len(target)} target times matched")
    T2M = W_target[:, :, 0]   # each zone's MERRA-2 2 m temperature at the target time, K

    # ---------------- Part 1: temperature probe ----------------
    if "probe" in args.parts:
        sources = {f"tokens: {n}": pca(per_zone_tokens(p, windows), train_t, 50) for n, p in layers.items()}
        sources["raw inputs (160 channels, issue time)"] = (raw, 1.0)
        sources["raw inputs: T2M channel only (issue time)"] = (raw[:, :, [i_t2m]], 1.0)
        sources["climatology for the target time: T2M"] = (clim[:, :, [j_t2m]], 1.0)
        if recon is not None:
            sources["decoder output for the target time: T2M"] = (recon[:, :, [j_t2m]], 1.0)
        if args.lead:
            sources["MERRA-2 T2M at the zone, issue time (persistence)"] = (W_issue[:, :, [0]], 1.0)
        rows = []
        for name, (V, kept) in sources.items():
            rmse, r2 = zip(*(probe(V[:, k], T2M[:, k], train_t, test_t) for k in range(len(zn))))
            rows.append({"source": name, "dims": V.shape[-1], "pca_variance_kept": round(kept, 3),
                         "zone_T2M_RMSE_K_mean": float(np.mean(rmse)), "zone_T2M_R2_mean": float(np.mean(r2)),
                         **{f"RMSE_{z}": r for z, r in zip(zn, rmse)}})
        probe_df = pd.DataFrame(rows)
        probe_df.to_csv(args.out_dir / "temperature_probe.csv", index=False)
        print(f"\nZone 2 m temperature at the target time (issue + {args.lead} h), linear fit; train days 1-18, test day 22+:")
        print(probe_df[["source", "dims", "pca_variance_kept", "zone_T2M_RMSE_K_mean", "zone_T2M_R2_mean"]].round(3).to_string(index=False))
        if recon is not None and args.lead == 0:
            print(f"Decoder T2M vs the model's own input (normalised units, mean |difference|): decoder "
                  f"{np.abs(recon[:, :, j_t2m] - raw[:, :, i_t2m]).mean():.4f}, climatology {np.abs(clim[:, :, j_t2m] - raw[:, :, i_t2m]).mean():.4f}")

    def wanted(combo):
        name = "calendar" + "".join(f" + {b}" for b in combo)
        return args.only is None or any(o in name for o in args.only)

    def zone_blocks(train_times):
        b = {"weather (target)": (W_target, [f"{n}" for n in WEATHER_NAMES]),
             "raw inputs": (pca(raw, train_times, 50)[0], [f"in{i}" for i in range(50)])}
        if args.lead:
            b["weather (issue)"] = (W_issue, [f"{n}_issue" for n in WEATHER_NAMES])
        for n, p in layers.items():
            b[f"tokens {n}"] = (pca(per_zone_tokens(p, windows), train_times, 50)[0], [f"tok_{n}_{i}" for i in range(50)])
        if recon is not None:
            b["decoder forecast"] = (recon, [f"dec_{ch}" for ch in fields["recon_sfc"]["channels"]])
            b["decoder weather"] = (recon[:, :, DEC_IDX], [f"dec_{v}" for v in WEATHER_VARS])
        if area_true is not None:
            b["true weather, area"] = (area_true, [f"area_{v}" for v in WEATHER_VARS])
        return b

    def combos():
        known = "weather (issue)" if args.lead else "weather (target)"
        out = [[], ["weather (target)"]]
        if args.lead:
            out.append(["weather (issue)"])
        out += [["raw inputs"]] + [[f"tokens {n}"] for n in layers]
        if recon is not None:
            out += [["decoder forecast"], [known, "decoder forecast"], ["decoder weather"], [known, "decoder weather"]]
        out += [[known, "tokens final"], [known, "raw inputs"]]
        if area_true is not None:
            out += [["true weather, area"], [known, "true weather, area"]]
        return out

    # ---------------- Part 2: demand ----------------
    if "demand" in args.parts:
        df = pd.read_parquet(args.data_dir / f"modeling_table_{args.year}.parquet",
                             columns=["timestamp", "timestamp_local", "zone", "demand_MW"] + PRICE_CALENDAR)
        ti = target.get_indexer(df["timestamp"].dt.round("3h"))
        zi = df["zone"].map({z: k for k, z in enumerate(zn)}).to_numpy()
        valid = ti >= 0
        y, zones = df["demand_MW"].to_numpy(), df["zone"].to_numpy()
        results = []
        for scheme in ("blocked", "chronological"):
            split = dm.assign_split(df, scheme).to_numpy()
            tr, te = (split == "train") & valid, (split == "test") & valid
            if not tr.any() or not te.any():
                print(f"\n[demand, {scheme}] no train or test rows; skipped")
                continue
            train_times = np.zeros(len(target), bool)
            train_times[np.unique(ti[tr])] = True
            blocks = zone_blocks(train_times)
            print(f"\n[demand, {scheme}] train {tr.sum()} rows, test {te.sum()} rows", flush=True)
            for combo in combos():
                if not wanted(combo):
                    continue
                X = df[dm.CALENDAR_FEATURES].copy()
                for b in combo:
                    V, names = blocks[b]
                    X = pd.concat([X, pd.DataFrame(V[np.clip(ti, 0, None), zi], columns=names, index=df.index)], axis=1)
                pred = fit_catboost(X, y, tr, te, args.out_dir, dm.CAT_FEATURES)
                ape = np.abs(pred - y[te]) / y[te] * 100
                byz = pd.Series(ape).groupby(zones[te]).mean()
                name = "calendar" + "".join(f" + {b}" for b in combo)
                results.append({"lead_h": args.lead, "split": scheme, "feature_set": name, "mean_zone_mape_pct": byz.mean(),
                                "median_ape_pct": float(np.median(ape)), **{f"mape_{z}": byz.get(z, np.nan) for z in zn}})
                print(f"  {name:55s} MAPE {byz.mean():5.2f}%   median {np.median(ape):4.2f}%   VT {byz.get('VT', np.nan):5.1f}%", flush=True)
                pd.DataFrame(results).to_csv(args.out_dir / "demand_comparison.csv", index=False)

    # ---------------- Part 3: hub price ----------------
    if "price" in args.parts:
        table = pd.read_parquet(args.data_dir / f"modeling_table_{args.year}.parquet",
                                columns=["timestamp", "timestamp_local", "zone", "demand_MW"] + PRICE_CALENDAR)
        share = table.groupby("zone")["demand_MW"].mean()
        share = (share / share.sum()).reindex(zn).to_numpy()                                  # fixed weights: each zone's share of annual demand
        g = table.groupby("timestamp")
        df = g[["timestamp_local"] + PRICE_CALENDAR].first()
        df["system_demand_MW"] = g["demand_MW"].sum()
        lmp = pd.read_parquet(c / f"isone_{args.price_market}_hourly_lmp_{args.year}.parquet")
        lmp["timestamp"] = pd.to_datetime(lmp["timestamp_parsed"]).astype("datetime64[ns]")
        hub = lmp[lmp["location"] == "HUB"].set_index("timestamp")["lmp_total"].sort_index()
        df = df.join(hub, how="inner").reset_index()
        day_mean = hub.rolling("24h").mean()
        past = {"price_lag24h": hub.reindex(df["timestamp"] - pd.Timedelta(hours=24)),
                "price_lag48h": hub.reindex(df["timestamp"] - pd.Timedelta(hours=48)),
                "price_lag7d": hub.reindex(df["timestamp"] - pd.Timedelta(days=7)),
                "price_mean_24h_to_lag24h": day_mean.reindex(df["timestamp"] - pd.Timedelta(hours=24))}
        past = pd.DataFrame({k: v.to_numpy() for k, v in past.items()}, index=df.index)
        gas_known = gas_same = None
        gas_csv = args.data_dir / "gas" / f"gas_prices_daily_{args.year}.csv"
        if gas_csv.exists():
            daily = pd.read_csv(gas_csv, parse_dates=["date"]).set_index("date")
            pts = pd.read_csv(args.data_dir / "gas" / f"algonquin_weekly_points_{args.year}.csv", parse_dates=["date"]).set_index("date")["price"]
            hh_raw = pd.read_excel(args.data_dir / "gas" / "RNGWHHDd.xls", sheet_name="Data 1", skiprows=2)
            hh_raw.columns = ["date", "henry_hub"]
            hh_raw = hh_raw.dropna().assign(date=lambda d: pd.to_datetime(d["date"])).set_index("date")["henry_hub"].sort_index()
            day = df["timestamp_local"].dt.tz_localize(None).dt.normalize() if df["timestamp_local"].dt.tz is not None else df["timestamp_local"].dt.normalize()
            cutoff = (day - pd.Timedelta(days=2)).to_numpy()
            gas_known = pd.DataFrame({"henry_hub_known": hh_raw.reindex(cutoff, method="ffill").to_numpy(),
                                      "algonquin_known": pts.sort_index().reindex(cutoff, method="ffill").to_numpy()}, index=df.index)
            gas_same = pd.DataFrame({"henry_hub_same_day": daily["henry_hub"].reindex(day).to_numpy(),
                                     "algonquin_same_day": daily["algonquin_weekly"].reindex(day).to_numpy()}, index=df.index)
            print(f"Gas: {gas_csv.name}; known-in-advance missing for {int(gas_known.isna().any(axis=1).sum())} hours")
        ti = target.get_indexer(df["timestamp"].dt.round("3h"))
        valid = ti >= 0
        y = df["lmp_total"].to_numpy()
        sys_w = lambda W: (W * share[None, :, None]).sum(axis=1)                               # noqa: E731
        results = []
        for scheme in ("blocked", "chronological"):
            split = dm.assign_split(df, scheme).to_numpy()
            tr, te = (split == "train") & valid, (split == "test") & valid
            if not tr.any() or not te.any():
                continue
            train_times = np.zeros(len(target), bool)
            train_times[np.unique(ti[tr])] = True
            blocks = {"weather (target)": (sys_w(W_target), WEATHER_NAMES),
                      "raw inputs": (pca(box_pixels(tok_dir / fields["inputs"]["file"]), train_times, 50)[0], [f"in{i}" for i in range(50)])}
            if args.lead:
                blocks["weather (issue)"] = (sys_w(W_issue), [f"{n}_issue" for n in WEATHER_NAMES])
            for n, p in layers.items():
                blocks[f"tokens {n}"] = (pca(box_tokens(p), train_times, 50)[0], [f"tok_{n}_{i}" for i in range(50)])
            if recon is not None:
                dec_box = box_pixels(tok_dir / fields["recon_sfc"]["file"])
                blocks["decoder forecast"] = (dec_box, [f"dec_{ch}" for ch in fields["recon_sfc"]["channels"]])
                blocks["decoder weather"] = (dec_box[:, DEC_IDX], [f"dec_{v}" for v in WEATHER_VARS])
            if area_true_box is not None:
                blocks["true weather, area"] = (area_true_box, [f"area_{v}" for v in WEATHER_VARS])
            print(f"\n[price ({args.price_market} hub), {scheme}] train {tr.sum()} hours, test {te.sum()} hours; "
                  f"test-period mean ${y[te].mean():.2f}, sd ${y[te].std():.2f}", flush=True)
            known = "weather (issue)" if args.lead else "weather (target)"
            sets = combos() + [["weather (target)", "system demand"], ["past price"], [known, "past price"],
                               ["weather (target)", "past price"]]
            if recon is not None:
                sets += [[known, "decoder weather", "past price"]]
            sets += [[known, "tokens final", "past price"]]
            if gas_known is not None:
                sets += [[known, "gas (known)"], [known, "past price", "gas (known)"],
                         ["weather (target)", "past price", "gas (known)"]]
                if recon is not None:
                    sets += [[known, "decoder weather", "past price", "gas (known)"]]
                sets += [[known, "tokens final", "past price", "gas (known)"],
                         [known, "gas (same day)"], [known, "past price", "gas (same day)"]]
            for combo in sets:
                if not wanted(combo):
                    continue
                X = df[PRICE_CALENDAR].copy()
                for b in combo:
                    if b == "system demand":
                        X["system_demand_MW"] = df["system_demand_MW"]
                        continue
                    if b == "past price":
                        X = pd.concat([X, past], axis=1)
                        continue
                    if b in ("gas (known)", "gas (same day)"):
                        X = pd.concat([X, gas_known if b == "gas (known)" else gas_same], axis=1)
                        continue
                    V, names = blocks[b]
                    X = pd.concat([X, pd.DataFrame(V[np.clip(ti, 0, None)], columns=names, index=df.index)], axis=1)
                pred = fit_catboost(X, y, tr, te, args.out_dir, [])
                err = pred - y[te]
                mae, r2 = float(np.abs(err).mean()), float(1 - (err ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
                name = "calendar" + "".join(f" + {b}" for b in combo)
                results.append({"lead_h": args.lead, "market": args.price_market, "split": scheme, "feature_set": name, "mae": mae, "r2": r2,
                                "rmse": float(np.sqrt((err ** 2).mean()))})
                print(f"  {name:55s} MAE ${mae:6.2f}   R2 {r2:5.2f}", flush=True)
                pd.DataFrame(results).to_csv(args.out_dir / f"price_{args.price_market}_comparison.csv", index=False)
    # ---------------- Part 4: where to read the tokens from ----------------
    # Final-layer tokens only, each read out four ways for every zone.
    if "readout" in args.parts:
        a = np.load(layers["final"], mmap_mode="r")
        nearest = {z: (int(np.abs(lat_c - la).argmin()), int(np.abs(lon_c - lo).argmin())) for z, (la, lo) in ZONES.items()}
        box = box_tokens(layers["final"])
        readouts = {
            "single nearest patch": np.stack([np.asarray(a[:, i, j], dtype=np.float32) for i, j in nearest.values()], axis=1),
            "zone 3 x 3 patches": per_zone_tokens(layers["final"], windows),
            "New England box mean (same for all zones)": np.repeat(box[:, None], len(zn), axis=1),
        }
        for cand in [args.data_dir / f"prithvi_embeddings_{args.year}{'_lead' + str(args.lead) if args.lead else ''}.parquet",
                     c / f"prithvi_embeddings_{args.year}{'_lead' + str(args.lead) if args.lead else ''}.parquet"]:
            if cand.exists():
                pooled = pd.read_parquet(cand)
                pooled = pooled.set_index(pd.to_datetime(pooled["timestamp"])).drop(columns="timestamp").reindex(issue).to_numpy(np.float32)
                readouts["30-60N, 90-50W pooled vector (same for all zones)"] = np.repeat(pooled[:, None], len(zn), axis=1)
                break
        print("\nNearest patch per zone (row, col):", nearest)
        rows = [{"readout": n, **dict(zip(["zone_T2M_RMSE_K_mean", "zone_T2M_R2_mean"],
                                          np.mean([probe(pca(V, train_t, 50)[0][:, k], T2M[:, k], train_t, test_t) for k in range(len(zn))], axis=0)))}
                for n, V in readouts.items()]
        print("\nTemperature probe by readout (final layer, 50 components):")
        print(pd.DataFrame(rows).round(3).to_string(index=False))
        df = pd.read_parquet(args.data_dir / f"modeling_table_{args.year}.parquet",
                             columns=["timestamp", "timestamp_local", "zone", "demand_MW"] + PRICE_CALENDAR)
        ti = target.get_indexer(df["timestamp"].dt.round("3h"))
        zi = df["zone"].map({z: k for k, z in enumerate(zn)}).to_numpy()
        valid = ti >= 0
        y, zones = df["demand_MW"].to_numpy(), df["zone"].to_numpy()
        results = []
        for scheme in ("blocked", "chronological"):
            split = dm.assign_split(df, scheme).to_numpy()
            tr, te = (split == "train") & valid, (split == "test") & valid
            train_times = np.zeros(len(target), bool)
            train_times[np.unique(ti[tr])] = True
            print(f"\n[readout, demand, {scheme}]", flush=True)
            for n, V in readouts.items():
                for with_weather in (False, True):
                    P = pca(V, train_times, 50)[0]
                    X = df[dm.CALENDAR_FEATURES].copy()
                    if with_weather:
                        X = pd.concat([X, pd.DataFrame(W_target[np.clip(ti, 0, None), zi], columns=WEATHER_NAMES, index=df.index)], axis=1)
                    X = pd.concat([X, pd.DataFrame(P[np.clip(ti, 0, None), zi], columns=[f"tok{i}" for i in range(50)], index=df.index)], axis=1)
                    pred = fit_catboost(X, y, tr, te, args.out_dir, dm.CAT_FEATURES)
                    byz = pd.Series(np.abs(pred - y[te]) / y[te] * 100).groupby(zones[te]).mean()
                    label = f"{n}{' + weather' if with_weather else ''}"
                    results.append({"lead_h": args.lead, "split": scheme, "readout": n, "with_weather": with_weather,
                                    "mean_zone_mape_pct": byz.mean(), **{f"mape_{z}": byz.get(z, np.nan) for z in zn}})
                    print(f"  calendar + {label:62s} MAPE {byz.mean():5.2f}%   VT {byz.get('VT', np.nan):5.1f}%", flush=True)
                    pd.DataFrame(results).to_csv(args.out_dir / "readout_comparison.csv", index=False)
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
