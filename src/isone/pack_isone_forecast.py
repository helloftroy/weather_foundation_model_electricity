"""Pack the per-day raw JSON from download_isone_rr_forecast.py into one
parquet file per series (all forecast versions, long format), so only a few
files need copying.

    python src/isone/pack_isone_forecast.py --raw-dir data/isone_forecast/raw --out-dir data/isone_forecast

Output columns: target_utc (start of the forecast hour, or the sample time
for actuals), creation_utc (when that version was issued; forecasts only),
every other field of the record as given, nested fields flattened (LoadMw,
ReliabilityRegion, ReliabilityRegionLoadPercentage, NetLoadMw, ...).
Writes isone_<series>.parquet; checks every input file parsed.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def records(path: Path) -> list:
    doc = json.loads(path.read_text())
    outer = next(iter(doc.values()))          # e.g. {"ReliabilityRegionLoadForecast": [...]}
    if not outer:
        return []
    rec = next(iter(outer.values()))
    return rec if isinstance(rec, list) else [rec]   # a single record comes as a dict


def pack(series_dir: Path, out_dir: Path) -> None:
    files = sorted(series_dir.glob("*.json"))
    if not files:
        print(f"{series_dir}: no files")
        return
    rows, empty = [], 0
    for f in files:
        r = records(f)
        empty += not r
        rows.extend(r)
    df = pd.json_normalize(rows)                # nested fields (e.g. Location.$) become their own columns
    df["target_utc"] = pd.to_datetime(df.pop("BeginDate"), utc=True)
    first = ["target_utc"]
    if "CreationDate" in df:                    # forecasts have versions; actuals do not
        df["creation_utc"] = pd.to_datetime(df.pop("CreationDate"), utc=True)
        first.append("creation_utc")
    df = df[first + [c for c in df.columns if c not in first]]
    df = df.sort_values(first).reset_index(drop=True)
    out = out_dir / f"isone_{series_dir.name}.parquet"
    df.to_parquet(out, index=False)
    print(f"{series_dir.name}: {len(files)} files ({empty} empty), {len(df)} rows, "
          f"targets {df['target_utc'].min()} to {df['target_utc'].max()}, columns {list(df.columns)} -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True, help="Folder holding one subfolder per series")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for d in sorted(p for p in args.raw_dir.iterdir() if p.is_dir()):
        pack(d, args.out_dir)


if __name__ == "__main__":
    main()
