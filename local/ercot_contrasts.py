"""Paired contrasts for the ERCOT price runs (results/ercot/{with_uri,without_uri}): tokens over GFS, the decoder over
GFS, the input state over GFS, tokens on top of the input state, all with a 95% week-block bootstrap; plus median
absolute error. Writes ercot_contrasts.csv next to each run."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eval_tools import block_bootstrap

root = Path(__file__).resolve().parent.parent
B = "calendar + past prices + Henry Hub + weather (issue)"
PAIRS = [(" + GFS area + Prithvi tokens", " + GFS area", "tokens over GFS"),
         (" + GFS area + Prithvi area", " + GFS area", "decoder over GFS"),
         (" + GFS area + tokens hub area", " + GFS area", "hub-area tokens over GFS"),
         (" + GFS area + input state", " + GFS area", "input state over GFS"),
         (" + GFS area + input state + Prithvi tokens", " + GFS area + input state", "tokens over GFS + input state"),
         (" + GFS area", "", "GFS over weather at issue"),
         (" + Prithvi tokens", "", "tokens over weather at issue")]
for v in ("with_uri", "without_uri"):
    p = pd.read_parquet(root / f"results/ercot/{v}/ercot_price_predictions.parquet")
    p["ae"] = (p["pred"] - p["actual"]).abs()
    rows = []
    for (hub, y), g in p.groupby(["hub", "test_year"], sort=False):
        w = g.pivot_table(index="row", columns="feature_set", values="ae")
        wk = g.drop_duplicates("row").set_index("row")["week"].reindex(w.index).to_numpy()
        for a, b, label in PAIRS:
            m, lo, hi = block_bootstrap((w[B + a] - w[B + b]).to_numpy(), wk)
            rows.append({"run": v, "hub": hub, "test_year": y, "contrast": label, "dMAE": m, "lo": lo, "hi": hi,
                         "rel_pct": 100 * m / w[B + b].mean(),
                         "dMedianAE": float(np.median(w[B + a]) - np.median(w[B + b]))})
    out = pd.DataFrame(rows)
    out.to_csv(root / f"results/ercot/{v}/ercot_contrasts.csv", index=False)
    pd.set_option("display.width", 250)
    t = out.assign(s=out.apply(lambda r: f"{r.dMAE:+.2f} [{r.lo:+.2f},{r.hi:+.2f}] {r.rel_pct:+.0f}%", axis=1))
    print(f"===== {v}")
    print(t.pivot_table(index=["contrast"], columns=["test_year", "hub"], values="s", aggfunc="first", sort=False).T.to_string())
