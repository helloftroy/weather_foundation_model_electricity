"""Figure 1 (overview). A, the two study regions in continental context (ISO-NE, ERCOT); B, Northeast zoom: the New
England token-pooling box, the five regional weather boxes and the two wider pooling extents; C, Texas zoom: the
ERCOT pooling box, the five hub areas and the wider Texas extent; D, pipeline: global MERRA-2 state -> frozen Prithvi
WxC -> encoder tokens / decoder fields -> CatBoost (with GFS and the other known inputs) -> demand / price. Box
coordinates come from src/prithvi/extract_embeddings.py. Writes results/figures/fig1_overview.{png,pdf}. Bare panels;
caption text lives in the reply."""
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cf
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src/prithvi"))
plt.rcParams["font.family"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 7.5

# Same numbers as POOL_EXTENTS / POOL_EXTENTS_ERCOT in extract_embeddings.py (lat0, lat1, lon0, lon1).
NE = {"ne": (40.0, 48.0, -75.0, -66.25), "northeast": (38.0, 50.0, -80.0, -62.0), "wide": (32.0, 50.0, -90.0, -64.0)}
NE_REG = {"NY-PA": (40.0, 45.0, -80.0, -75.0), "Mid-Atlantic": (36.0, 40.0, -80.0, -74.0),
          "Great Lakes": (38.0, 46.0, -90.0, -80.0), "Southeast": (32.0, 36.0, -90.0, -75.0),
          "Quebec–Maritimes": (45.0, 50.0, -80.0, -64.0)}
TX = {"ercot": (26.0, 36.5, -106.5, -93.5), "texas_wide": (24.0, 40.0, -110.0, -88.0)}
TX_HUB = {"North": (31.5, 34.5, -99.0, -95.0), "Houston": (28.5, 31.0, -96.5, -94.0), "South": (26.0, 31.0, -100.5, -96.5),
          "West": (30.0, 34.0, -104.5, -99.0), "Panhandle": (34.0, 36.5, -103.0, -100.0)}
NE_STATES = {"Maine", "New Hampshire", "Vermont", "Massachusetts", "Rhode Island", "Connecticut"}
C_NE, C_TX, INK, MUTED = "#1f4e9c", "#b5541c", "#222222", "#6b6b6b"
PC = ccrs.PlateCarree()
states = list(shpreader.Reader(shpreader.natural_earth("50m", "cultural", "admin_1_states_provinces_lakes")).records())


def base(ax, extent):
    ax.set_extent(extent, crs=PC)
    ax.add_feature(cf.LAND.with_scale("110m"), facecolor="#f4f2ee", edgecolor="none")
    ax.add_feature(cf.OCEAN.with_scale("110m"), facecolor="#e8eef4", edgecolor="none")
    ax.add_feature(cf.STATES.with_scale("50m"), lw=0.3, edgecolor="#a8a8a8")
    ax.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.4, edgecolor="#777777")
    for s in ax.spines.values():
        s.set_edgecolor("#999999")
        s.set_linewidth(0.6)


def shade(ax, names, color, alpha):
    geoms = [r.geometry for r in states if r.attributes.get("name") in names and r.attributes.get("admin") == "United States of America"]
    ax.add_geometries(geoms, PC, facecolor=color, edgecolor="none", alpha=alpha, zorder=1.5)


def box(ax, b, **kw):
    lat0, lat1, lon0, lon1 = b
    ax.add_patch(Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0, transform=PC, **kw))


def letter(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="right", va="bottom")


fig = plt.figure(figsize=(7.4, 5.3))
gs = fig.add_gridspec(3, 3, height_ratios=[1.55, 0.13, 2.45], width_ratios=[1.25, 1, 1], hspace=0.12, wspace=0.07)

# ---- A: context ----
lcc = ccrs.LambertConformal(central_longitude=-92, central_latitude=38, standard_parallels=(30, 46))
axa = fig.add_subplot(gs[0, 0], projection=lcc)
base(axa, [-118, -62, 22, 52])
shade(axa, NE_STATES, C_NE, 0.55)
shade(axa, {"Texas"}, C_TX, 0.45)
box(axa, NE["ne"], fill=False, edgecolor=C_NE, lw=1.2, zorder=3)
box(axa, TX["ercot"], fill=False, edgecolor=C_TX, lw=1.2, zorder=3)
axa.text(-67.5, 38.5, "ISO-NE", transform=PC, color=C_NE, fontweight="bold", fontsize=8.5, ha="left")
axa.text(-92, 25.5, "ERCOT", transform=PC, color=C_TX, fontweight="bold", fontsize=8.5, ha="left")
letter(axa, "A")

# ---- B: Northeast ----
axb = fig.add_subplot(gs[0, 1], projection=PC)
base(axb, [-92, -60, 30, 52])
shade(axb, NE_STATES, C_NE, 0.35)
box(axb, NE["wide"], fill=False, edgecolor=MUTED, lw=0.8, ls=(0, (1, 1.5)), zorder=3)
box(axb, NE["northeast"], fill=False, edgecolor=MUTED, lw=0.8, ls=(0, (4, 2)), zorder=3)
for name, b in NE_REG.items():
    box(axb, b, facecolor="#9fb3d1", alpha=0.22, edgecolor="#7f95b8", lw=0.5, zorder=2)
box(axb, NE["ne"], fill=False, edgecolor=C_NE, lw=1.6, zorder=4)
lab = {"NY-PA": (-77.6, 41.4), "Mid-Atlantic": (-77, 37.2), "Great Lakes": (-85, 42), "Southeast": (-82.5, 33.2),
       "Quebec–Maritimes": (-79.5, 48.9)}
for name, (x, y) in lab.items():
    axb.text(x, y, name, transform=PC, fontsize=5.8, color="#3d4f6e",
             ha="left" if name.startswith("Q") else "center", va="center", zorder=5)
letter(axb, "B")

# ---- C: Texas ----
axc = fig.add_subplot(gs[0, 2], projection=PC)
base(axc, [-112, -86, 23, 41])
shade(axc, {"Texas"}, C_TX, 0.22)
box(axc, TX["texas_wide"], fill=False, edgecolor=MUTED, lw=0.8, ls=(0, (4, 2)), zorder=3)
for name, b in TX_HUB.items():
    box(axc, b, facecolor="#e2a77f", alpha=0.35, edgecolor="#c47b4c", lw=0.5, zorder=2)
    lat0, lat1, lon0, lon1 = b
    x, y, ha = {"Houston": (-93.0, 29.2, "left"), "South": (-98.5, 27.3, "center")}.get(name, ((lon0 + lon1) / 2, (lat0 + lat1) / 2, "center"))
    axc.text(x, y, name, transform=PC, fontsize=5.8, color="#6a3412", ha=ha, va="center", zorder=5)
box(axc, TX["ercot"], fill=False, edgecolor=C_TX, lw=1.6, zorder=4)
axc.text(-100, 37.2, "ERCOT tokens", transform=PC, fontsize=6.5, color=C_TX, fontweight="bold", ha="center", zorder=5)
letter(axc, "C")

# Shared map key under B and C.
kx = fig.add_subplot(gs[1, :])
kx.axis("off")
items = [("solid", C_NE, 1.6, "token pooling box (ISO-NE / ERCOT)"), ((0, (4, 2)), MUTED, 0.8, "wider pooling extent"),
         ((0, (1, 1.5)), MUTED, 0.8, "wide box")]
x = 0.03
for ls, c, lw, t in items:
    kx.plot([x, x + 0.045], [0.5, 0.5], ls=ls, color=c, lw=lw, transform=kx.transAxes)
    kx.text(x + 0.055, 0.5, t, transform=kx.transAxes, va="center", fontsize=6.3, color=INK)
    x += 0.055 + 0.0085 * len(t) + 0.03
kx.add_patch(Rectangle((x, 0.25), 0.03, 0.5, transform=kx.transAxes, facecolor="#b9b9b9", edgecolor="#8a8a8a", lw=0.5))
kx.text(x + 0.04, 0.5, "regional weather boxes / hub areas", transform=kx.transAxes, va="center", fontsize=6.3, color=INK)

# ---- D: pipeline ----
axd = fig.add_subplot(gs[2, :])
axd.set_xlim(0, 100)
axd.set_ylim(0, 40)
axd.axis("off")
letter(axd, "D", x=0.0, y=0.97)


def node(x, y, w, h, title, body, fc="#ffffff", ec="#888888", tc=INK, bold=True):
    axd.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", facecolor=fc, edgecolor=ec,
                                 lw=0.8))
    axd.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=6.9, color=tc,
             fontweight="bold" if bold else "normal")
    if body:
        axd.text(x + w / 2, y + h - 4.2, body, ha="center", va="top", fontsize=5.9, color="#444444", linespacing=1.25)


def arrow(x0, y0, x1, y1, c="#555555"):
    axd.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=c,
                                  shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=0"))


PR = "#eef2f9"
GREY = "#f5f5f5"
node(0.5, 19, 16, 20, "Global atmosphere", "")
glob = axd.inset_axes([2.2, 24.8, 12.6, 9.0], transform=axd.transData, projection=ccrs.Robinson(central_longitude=-80))
glob.set_global()
glob.add_feature(cf.LAND.with_scale("110m"), facecolor="#d9d4ca", edgecolor="none")
glob.add_feature(cf.OCEAN.with_scale("110m"), facecolor="#e8eef4", edgecolor="none")
box(glob, NE["ne"], facecolor=C_NE, edgecolor=C_NE, lw=0.6)
box(glob, TX["ercot"], facecolor=C_TX, edgecolor=C_TX, lw=0.6)
glob.spines["geo"].set_edgecolor("#999999")
glob.spines["geo"].set_linewidth(0.4)
axd.text(8.5, 24.0, "MERRA-2 at t − 24 h\n160 channels", ha="center", va="top", fontsize=5.9, color="#444444")
node(21, 19, 17, 20, "Prithvi WxC 2.3B", "frozen, no fine-tuning;\nglobal forward pass,\n24 h lead to target t", fc=PR, ec=C_NE)
node(43, 30.5, 18, 8.5, "Prithvi tokens", "final encoder layer, region\nmean → PCA-50", fc=PR, ec=C_NE)
node(43, 20.5, 18, 8.5, "Prithvi decoder output", "forecast weather at t,\nregion mean", fc=PR, ec=C_NE)
node(43, 10.5, 18, 8.5, "GFS forecast", "issued t − 24 h, region\n(+ regional box) means", fc=GREY)
node(43, 0.5, 18, 8.5, "Known inputs", "calendar, past load / prices,\nweather at issue, gas price", fc=GREY)
node(66, 8, 13, 24, "CatBoost", "one model per\ntarget;\nyear-ahead tests\n2023 and 2024")
node(84, 22, 15.5, 15, "Day-ahead price", "ISO-NE internal hub\nERCOT: 5 hubs +\nhub average", ec=C_TX)
node(84, 3, 15.5, 15, "Demand", "ISO-NE,\n8 load zones", ec=C_NE)
arrow(16.9, 29, 20.6, 29)
arrow(38.4, 31, 42.6, 34.7)
arrow(38.4, 27, 42.6, 24.7)
for y0, y1 in ((34.7, 26), (24.7, 22), (14.7, 18), (4.7, 14)):
    arrow(61.4, y0, 65.6, y1)
arrow(79.4, 23, 83.6, 29.5)
arrow(79.4, 17, 83.6, 10.5)
# Timing strip: everything the model sees is available at issue time t - 24 h.
axd.plot([2, 36], [7, 7], color=MUTED, lw=0.8)
for x, t in ((3, "issue time t − 24 h"), (35, "target hour t")):
    axd.plot([x, x], [5.8, 8.2], color=MUTED, lw=0.8)
    axd.text(x, 9.3, t, ha="center", va="bottom", fontsize=5.9, color=MUTED)
axd.text(19, 4.6, "all inputs known at issue time", ha="center", va="top", fontsize=5.9, color=MUTED, style="italic")

out = root / "results/figures/fig1_overview"
fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print(out)
