"""
Butterfly Diagram Generator
Cleaned-up version of butterfly_diagram.py
Outputs: solar_forecast/website/home/butterfly.png
"""

import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Data paths ───────────────────────────────────────────────────────────────
DIRECTORY_PATH = r"C:\Users\Alfie\NN\RGO_NOAA1874_2013"
OUTPUT_PATH = Path(__file__).parent / "butterfly.png"

BG_DARK  = "#080d14"
AMBER    = "#f59e0b"
AMBER_MID = "#fbbf24"
AMBER_DIM = "#92400e"

# ── Parsing ───────────────────────────────────────────────────────────────────
def parse_line(line):
    """Parse one line of the RGO fixed-width format into a list of fields."""
    year  = line[:4]
    rest  = line[4:]
    month = rest[1] if rest[0] == " " else rest[:2]
    rest  = rest[2:]
    pattern = re.compile(r"\s?(\d+\.\d{3})(.*)")
    match   = pattern.match(rest)
    day     = match.group(1)
    tail    = match.group(2)
    return [year, month, day] + tail.split()


def load_dataframe(filepath):
    with open(filepath, "r") as f:
        rows = [parse_line(line) for line in f if line.strip()]
    return pd.DataFrame(rows)


def load_all_files(directory):
    frames = []
    for fp in Path(directory).rglob("*.txt"):
        frames.append(load_dataframe(fp))
    return pd.concat(frames, ignore_index=True)


# ── Load & clean ──────────────────────────────────────────────────────────────
print("Loading RGO data...")
raw = load_all_files(DIRECTORY_PATH)

COLUMN_NAMES = [
    "Year", "Month", "DayTime", "NA1", "NA2",
    "UmbralArea", "WholeSpotArea",
    "CorrectedUmbralArea", "CorrectedWholeSpotArea",
    "Distance", "PositionAngle",
    "CarringtonLongitude", "Latitude", "CMD",
]
raw.columns = COLUMN_NAMES

KEEP = ["Year", "Month", "DayTime",
        "UmbralArea", "WholeSpotArea",
        "CorrectedUmbralArea", "CorrectedWholeSpotArea",
        "Distance", "PositionAngle",
        "CarringtonLongitude", "Latitude", "CMD"]
df = raw[KEEP].copy()

# Build Date column
df["Day"] = df["DayTime"].str.split(".").str[0].astype(int)
df["Date"] = pd.to_datetime(
    df[["Year", "Month", "Day"]].astype(str).agg("-".join, axis=1),
    errors="coerce"
)
df = df.drop(columns=["Year", "Month", "DayTime", "Day"])
df = df.dropna(subset=["Date"])

# Numeric conversion
for col in df.columns.difference(["Date"]):
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Latitude", "WholeSpotArea"])
df = df[df["WholeSpotArea"] > 0]           # remove zero-area entries
df = df[df["WholeSpotArea"] < 15_000]      # remove outliers
df = df.reset_index(drop=True)

print(f"  {len(df):,} observations loaded, {df['Date'].min().year}-{df['Date'].max().year}")

# ── Colour assignment ─────────────────────────────────────────────────────────
# Three tiers by WholeSpotArea:  small -> dim amber, medium -> amber, large -> bright
bins   = [0, 250, 2500, 15_000]
labels = [AMBER_DIM, AMBER, AMBER_MID]
df["color"] = pd.cut(df["WholeSpotArea"], bins=bins, labels=labels, include_lowest=True).astype(str)

# ── Plot ───────────────────────────────────────────────────────────────────────
print("Rendering butterfly diagram...")

fig, ax = plt.subplots(figsize=(18, 7))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(BG_DARK)

# Plot each colour tier separately so legend works cleanly
tier_labels = ["Small group (<250 MSH)", "Medium group (<2500 MSH)", "Large group (>=2500 MSH)"]
for col, lbl in zip(labels, tier_labels):
    mask = df["color"] == col
    sub  = df[mask]
    size = np.clip(sub["WholeSpotArea"] * 0.05, 0.3, 30)
    ax.scatter(
        sub["Date"], sub["Latitude"],
        s=size, c=col, alpha=0.25, linewidths=0,
        label=lbl, rasterized=True
    )

# ── Axes styling ──────────────────────────────────────────────────────────────
TEXT_MUTED = "#7a90ae"
GRID_COL   = "#0f1928"
BORDER_COL = "#192438"

ax.set_ylim(-50, 50)
ax.set_xlim(df["Date"].min(), df["Date"].max())

ax.set_xlabel("Year", color=TEXT_MUTED, fontsize=11, labelpad=8)
ax.set_ylabel("Heliographic Latitude (deg)", color=TEXT_MUTED, fontsize=11, labelpad=8)

ax.tick_params(colors=TEXT_MUTED, labelsize=10)
for spine in ax.spines.values():
    spine.set_edgecolor(BORDER_COL)

ax.xaxis.label.set_color(TEXT_MUTED)
ax.yaxis.label.set_color(TEXT_MUTED)
ax.tick_params(axis="both", colors=TEXT_MUTED)

ax.grid(True, color=GRID_COL, linewidth=0.6, linestyle="--", alpha=0.8)
ax.axhline(0, color=BORDER_COL, linewidth=0.8, linestyle="-")

# ── Titles ────────────────────────────────────────────────────────────────────
ax.set_title(
    "Maunder Butterfly Diagram  |  Solar Cycles 12-24  (RGO/NOAA 1874-2013)",
    color="#e2e8f2", fontsize=13, fontweight="700", pad=14, loc="left"
)

# Subtle overline
fig.text(0.012, 0.96,
         "HELIOGRAPHIC LATITUDE OF SUNSPOT GROUPS",
         color=AMBER, fontsize=7.5, fontweight="600",
         ha="left", va="top", transform=fig.transFigure,
         alpha=0.75, style="normal",
         fontfamily="DejaVu Sans")

# ── Legend ────────────────────────────────────────────────────────────────────
leg = ax.legend(
    loc="upper right", frameon=True,
    facecolor="#0d1520", edgecolor=BORDER_COL,
    labelcolor=TEXT_MUTED, fontsize=9,
    markerscale=3, scatterpoints=1,
    handlelength=1.5,
)
for lh in leg.legend_handles:
    lh.set_alpha(0.9)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=1.2)
fig.savefig(OUTPUT_PATH, dpi=150, facecolor=BG_DARK, bbox_inches="tight")
plt.close(fig)

size_kb = OUTPUT_PATH.stat().st_size // 1024
print(f"Saved: {OUTPUT_PATH}  ({size_kb} KB)")
