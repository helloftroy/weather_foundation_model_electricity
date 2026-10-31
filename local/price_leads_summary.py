"""Summarise the ISO-NE price runs at leads of 1, 3, 5 and 7 days (price_benchmark.py --lead-days L --regional) on
the hours usable at every lead. Writes results/price_leads/leads_summary.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eval_tools import block_bootstrap

root = Path(__file__).resolve().parent.parent
D = root / "results/price_leads"
PRE = "calendar + past values + known gas"
p = pd.concat([pd.read_parquet(D / f"lead{L}/price_predictions.parquet").assign(lead=L) for L in (1, 3, 5, 7)])
p["fs"] = p["feature_set"].str.replace(PRE, "base", regex=False)
common = p.groupby(["test_year", "row"])["lead"].nunique()
common = common[common == 4].reset_index()[["test_year", "row"]]
p = p.merge(common, on=["test_year", "row"])
p["ae"] = (p["pred"] - p["actual"]).abs()
rows = []
CONTRASTS = [("base + weather (issue) + GFS area + Prithvi tokens", "base + weather (issue) + GFS area", "tokens over GFS"),
             ("base + weather (issue) + GFS area + GFS regional + Prithvi tokens", "base + weather (issue) + GFS area + GFS regional", "tokens over GFS + regional"),
             ("base + weather (issue) + GFS area", "base + weather (issue)", "GFS over weather at issue"),
             ("base + weather (issue) + Prithvi tokens", "base + weather (issue)", "tokens over weather at issue")]
for (y, L), g in p.groupby(["test_year", "lead"]):
    w = g.pivot_table(index="row", columns="fs", values="ae")
    act = g.drop_duplicates("row").set_index("row")["actual"].reindex(w.index)
    pr = g.pivot_table(index="row", columns="fs", values="pred")
    wk = g.drop_duplicates("row").set_index("row")["week"].reindex(w.index).to_numpy()
    for fs in w.columns:
        e = pr[fs] - act
        rows.append({"test_year": y, "lead": L, "kind": "model", "name": fs, "MAE": w[fs].mean(),
                     "R2": 1 - (e ** 2).sum() / ((act - act.mean()) ** 2).sum(), "n": len(w)})
    for a, b, label in CONTRASTS:
        m, lo, hi = block_bootstrap((w[a] - w[b]).to_numpy(), wk)
        rows.append({"test_year": y, "lead": L, "kind": "contrast", "name": label, "dMAE": m, "lo": lo, "hi": hi,
                     "rel_pct": 100 * m / w[b].mean(), "n": len(w)})
out = pd.DataFrame(rows)
out.to_csv(D / "leads_summary.csv", index=False)
pd.set_option("display.width", 220)
m = out[out["kind"] == "model"].pivot_table(index="name", columns=["test_year", "lead"], values="MAE").round(2)
r = out[out["kind"] == "model"].pivot_table(index="name", columns=["test_year", "lead"], values="R2").round(3)
print(f"common hours: {common.groupby('test_year').size().to_dict()}\nMAE"); print(m.to_string()); print("R2"); print(r.to_string())
c = out[out["kind"] == "contrast"]
print(c.assign(ci=c.apply(lambda x: f"{x.dMAE:+.2f} [{x.lo:+.2f}, {x.hi:+.2f}] ({x.rel_pct:+.0f}%)", axis=1))
       .pivot_table(index="name", columns=["test_year", "lead"], values="ci", aggfunc="first").to_string())
