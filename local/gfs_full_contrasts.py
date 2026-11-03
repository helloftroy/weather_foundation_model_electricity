"""Paired contrasts for the full-GFS-state control (price_benchmark.py --gfs-full; results/regional/price_gfs_full):
does a 50-PC description of the GFS forecast atmosphere (all 150 fields over New England, as box means or gridded)
match the Prithvi tokens, and do the tokens still add on top of it? Delta MAE with a 95% week-block bootstrap
(negative = the first model is better). Writes gfs_full_contrasts.csv next to the predictions."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eval_tools import block_bootstrap  # noqa: E402

root = Path(__file__).resolve().parent.parent
d = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "results/regional/price_gfs_full"
p = pd.read_parquet(d / "price_predictions.parquet")
p["fs"] = p["feature_set"].str.replace("calendar + past values + known gas + weather (issue) + ", "", regex=False)
p["ae"] = (p["pred"] - p["actual"]).abs()
T, M, Fb, Fg = "tokens ne", "MERRA-2 fields ne", "GFS full state box", "GFS full state grid"
pairs = []
for base, short in (("GFS area", "over GFS"), ("GFS area + GFS regional", "over GFS + regional")):
    for x in (T, M, Fb, Fg):
        pairs.append((short, f"+ {x} vs nothing", f"{base} + {x}", base))
    for f in (Fb, Fg):
        pairs.append((short, f"tokens vs {f}", f"{base} + {T}", f"{base} + {f}"))
        pairs.append((short, f"tokens on top of {f}", f"{base} + {f} + {T}", f"{base} + {f}"))
    pairs.append((short, f"{Fg} vs MERRA-2 fields", f"{base} + {Fg}", f"{base} + {M}"))
rows = []
for y, g in p.groupby("test_year"):
    w = g.pivot_table(index="row", columns="fs", values="ae")
    week = g.drop_duplicates("row").set_index("row")["week"].reindex(w.index).to_numpy()
    for short, kind, a, b in pairs:
        m, lo, hi = block_bootstrap((w[a] - w[b]).to_numpy(), week)
        rows.append({"test_year": y, "baseline": short, "contrast": kind, "dMAE": m, "lo": lo, "hi": hi,
                     "MAE_first": w[a].mean(), "MAE_second": w[b].mean()})
out = pd.DataFrame(rows)
out.to_csv(d / "gfs_full_contrasts.csv", index=False)
out["dMAE [95%]"] = [f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]" for m, lo, hi in zip(out["dMAE"], out["lo"], out["hi"])]
pd.set_option("display.width", 250)
print(out.pivot_table(index=["baseline", "contrast"], columns="test_year", values="dMAE [95%]", aggfunc="first", sort=False).to_string())
