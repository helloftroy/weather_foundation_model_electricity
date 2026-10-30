"""Paired contrasts for the pooled-extents price run (results/regional/price_pooled_extents): wider token extents
against New England tokens, and tokens on top of the upper-air (input-field) control. Week-block bootstrap."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eval_tools import block_bootstrap

root = Path(__file__).resolve().parent.parent
d = root / "results/regional/price_pooled_extents"
p = pd.read_parquet(d / "price_predictions.parquet")
p["fs"] = p["feature_set"].str.replace("calendar + past values + known gas + weather (issue) + ", "", regex=False)
p["ae"] = (p["pred"] - p["actual"]).abs()
pairs = []
for base in ("GFS area", "GFS area + GFS regional"):
    for e in ("northeast", "wide", "5 regions"):
        pairs.append((f"{base} + tokens {e}", f"{base} + tokens ne", "wider tokens vs NE tokens"))
    for f in ("ne", "wide"):
        pairs.append((f"{base} + MERRA-2 fields {f} + tokens {f}", f"{base} + MERRA-2 fields {f}", "tokens on top of input fields"))
    pairs.append((f"{base} + MERRA-2 fields wide + tokens ne", f"{base} + MERRA-2 fields wide", "tokens on top of input fields"))
    pairs.append((f"{base} + tokens ne", f"{base} + MERRA-2 fields ne", "NE tokens vs NE input fields"))
    pairs.append((f"{base} + MERRA-2 fields wide", f"{base} + MERRA-2 fields ne", "wide fields vs NE fields"))
rows = []
for y, g in p.groupby("test_year"):
    w = g.pivot_table(index="row", columns="fs", values="ae")
    week = g.drop_duplicates("row").set_index("row")["week"].reindex(w.index).to_numpy()
    for a, b, kind in pairs:
        m, lo, hi = block_bootstrap((w[a] - w[b]).to_numpy(), week)
        rows.append({"test_year": y, "contrast": kind, "model": a, "minus": b, "dMAE": m, "lo": lo, "hi": hi})
out = pd.DataFrame(rows)
out.to_csv(d / "pooled_extents_contrasts.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
print(out.round(2).to_string(index=False))
