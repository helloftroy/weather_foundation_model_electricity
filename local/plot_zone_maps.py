"""Map of CatBoost demand predictions by ISO-NE load zone.

Three panels for one model and one split, over the held-out test hours:
actual mean demand, predicted mean demand, and mean absolute percentage
error, each as one value per load zone.

The models predict one number per zone per hour, so the map is by zone, not
by grid cell. Zone shapes: the five single-state zones are state outlines;
the three Massachusetts zones are approximated by whole counties (ISO-NE's
real boundaries follow utility service areas and cut through some counties,
notably Norfolk).

Needs data/geo/counties.json (US county outlines):
  curl -sL https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json -o data/geo/counties.json
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from shapely.geometry import shape
from shapely.ops import unary_union

STATE_ZONES = {"09": "CT", "23": "ME", "33": "NH", "44": "RI", "50": "VT"}
MA_COUNTY_ZONES = {
    "WCMA": ["Berkshire", "Franklin", "Hampshire", "Hampden", "Worcester"],
    "NEMA": ["Essex", "Middlesex", "Suffolk"],
    "SEMA": ["Norfolk", "Bristol", "Plymouth", "Barnstable", "Dukes", "Nantucket"],
}
ZONE_NAMES = {"CT": "Connecticut", "ME": "Maine", "NH": "New Hampshire", "RI": "Rhode Island", "VT": "Vermont",
              "NEMA": "NE Mass.", "SEMA": "SE Mass.", "WCMA": "W/Central Mass."}
# Label anchor (lon, lat). Small coastal zones are labelled offshore with a leader line.
LABEL_AT = {"ME": (-69.2, 45.3), "NH": (-71.55, 43.75), "VT": (-72.7, 44.1), "CT": (-72.7, 41.6), "WCMA": (-72.45, 42.35),
            "NEMA": (-69.2, 42.85), "SEMA": (-68.9, 41.75), "RI": (-71.0, 40.75)}
LEADER_TO = {"NEMA": (-70.95, 42.55), "SEMA": (-70.75, 41.9), "RI": (-71.5, 41.6)}

SURFACE, INK, INK_2, MUTED, HAIRLINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE = ["#fbe6da", "#f6c3a8", "#f09a70", "#eb6834", "#c44e1f", "#96390f", "#6b2706"]


def load_zone_shapes(counties_json: Path) -> dict:
    features = json.load(open(counties_json))["features"]
    parts: dict[str, list] = {}
    county_to_zone = {c: z for z, cs in MA_COUNTY_ZONES.items() for c in cs}
    for f in features:
        state, name = f["properties"]["STATE"], f["properties"]["NAME"]
        zone = STATE_ZONES.get(state) or (county_to_zone.get(name) if state == "25" else None)
        if zone:
            parts.setdefault(zone, []).append(shape(f["geometry"]))
    return {z: unary_union(g) for z, g in parts.items()}


def draw_zone(ax, geom, **kw):
    for poly in getattr(geom, "geoms", [geom]):
        x, y = poly.exterior.xy
        ax.fill(x, y, **kw)


def text_colour(fill) -> str:
    r, g, b = to_rgb(fill)
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 else "#ffffff"


def panel(ax, shapes, values: dict, cmap, norm, title: str, fmt: str):
    ax.set_facecolor(SURFACE)
    for zone, geom in shapes.items():
        draw_zone(ax, geom, facecolor=cmap(norm(values[zone])), edgecolor=SURFACE, linewidth=1.6, zorder=2)
    for zone, (lx, ly) in LABEL_AT.items():
        label = f"{zone}\n{format(values[zone], fmt)}"
        if zone in LEADER_TO:
            tx, ty = LEADER_TO[zone]
            ax.plot([lx, tx], [ly, ty], color=MUTED, linewidth=0.8, zorder=3)
            ax.plot([tx], [ty], marker="o", markersize=2.5, color=MUTED, zorder=3)
            ax.text(lx, ly, label, ha="center", va="center", fontsize=11, color=INK, zorder=4,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
        else:
            ax.text(lx, ly, label, ha="center", va="center", fontsize=11, color=text_colour(cmap(norm(values[zone]))), zorder=4)
    ax.set_xlim(-73.9, -66.6)
    ax.set_ylim(40.3, 47.6)
    ax.set_aspect(1 / np.cos(np.deg2rad(44)))
    ax.set_xticks([-73, -71, -69, -67], labels=["73°W", "71°W", "69°W", "67°W"])
    ax.set_yticks([41, 43, 45, 47], labels=["41°N", "43°N", "45°N", "47°N"])
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(color=HAIRLINE, linewidth=0.6, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="test_predictions.parquet from train_catboost_comparisons.py")
    parser.add_argument("--split", default="chronological", choices=["chronological", "blocked"])
    parser.add_argument("--feature-set", default="calendar+weather")
    parser.add_argument("--counties-json", type=Path, default=Path("data/geo/counties.json"))
    parser.add_argument("--out-png", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.predictions)
    df = df[df["split_scheme"] == args.split]
    col = f"pred_{args.feature_set}"
    if col not in df.columns:
        raise SystemExit(f"No predictions for feature set '{args.feature_set}'. Available: {[c[5:] for c in df.columns if c.startswith('pred_')]}")
    df = df.assign(ape=(df[col] - df["demand_MW"]).abs() / df["demand_MW"] * 100)
    z = df.groupby("zone").agg(actual=("demand_MW", "mean"), predicted=(col, "mean"), mape=("ape", "mean"))
    local = df["timestamp_local"]
    period = f"{local.min():%b %-d} to {local.max():%b %-d, %Y}" if args.split == "chronological" else "day 22 onward of every month, 2024"

    shapes = load_zone_shapes(args.counties_json)
    blue = LinearSegmentedColormap.from_list("blue", BLUE)
    orange = LinearSegmentedColormap.from_list("orange", ORANGE)
    demand_norm = Normalize(0, float(np.ceil(max(z["actual"].max(), z["predicted"].max()) / 500) * 500))
    mape_norm = Normalize(0, float(np.ceil(z["mape"].max() / 5) * 5))

    fig, axes = plt.subplots(1, 3, figsize=(16, 8.2), facecolor=SURFACE)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.83, bottom=0.17, wspace=0.12)
    panel(axes[0], shapes, z["actual"].to_dict(), blue, demand_norm, "Actual mean demand (MW)", ",.0f")
    panel(axes[1], shapes, z["predicted"].to_dict(), blue, demand_norm, "Predicted mean demand (MW)", ",.0f")
    panel(axes[2], shapes, z["mape"].to_dict(), orange, mape_norm, "Hourly prediction error (MAPE, %)", ".1f")

    for ax_pair, cmap, norm, label in [((axes[0], axes[1]), blue, demand_norm, "Mean demand over test hours (MW)"),
                                       ((axes[2],), orange, mape_norm, "Mean absolute percentage error (%)")]:
        x0, x1 = ax_pair[0].get_position().x0, ax_pair[-1].get_position().x1
        cax = fig.add_axes([x0 + 0.02, 0.09, (x1 - x0) - 0.04, 0.018])
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=INK_2, labelsize=10, length=0)
        cb.set_label(label, color=INK_2, fontsize=10)

    features = {"calendar": "calendar features only", "calendar+weather": "calendar + simple weather",
                "calendar+prithvi": "calendar + Prithvi embedding",
                "calendar+weather+prithvi": "calendar + weather + Prithvi embedding"}.get(args.feature_set, args.feature_set)
    fig.text(0.04, 0.945, "CatBoost hourly electricity demand by ISO New England load zone", fontsize=18, color=INK, weight="bold")
    fig.text(0.04, 0.9, f"Model inputs: {features}. Held-out test hours: {period}. Mean error across zones: {z['mape'].mean():.1f}%.",
             fontsize=12, color=INK_2)
    fig.text(0.04, 0.012, "One value per load zone (the model predicts by zone, not by grid cell). "
             "Massachusetts zone boundaries are approximated by county. Demand: ISO-NE real-time hourly load. Weather: MERRA-2.",
             fontsize=9, color=MUTED)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=200, facecolor=SURFACE)
    print(z.round(1).to_string())
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
