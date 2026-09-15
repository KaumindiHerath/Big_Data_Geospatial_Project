"""Shared plotting style: fixed categorical palette per location (colorblind-
safe seaborn palette, assigned in a fixed order so a location's color never
changes across charts), sequential/diverging cmaps for magnitude/polarity.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import ORIGINAL_6

sns.set_theme(style="whitegrid", context="talk", font_scale=0.75)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# Fixed categorical order/colors, colorblind-safe (seaborn "colorblind").
_PALETTE = sns.color_palette("colorblind", 10)
LOCATION_ORDER = ORIGINAL_6  # extend automatically supported by fallback below
LOCATION_COLORS = {loc: _PALETTE[i % len(_PALETTE)] for i, loc in enumerate(LOCATION_ORDER)}

SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"


def color_for(location_id):
    if location_id not in LOCATION_COLORS:
        LOCATION_COLORS[location_id] = _PALETTE[len(LOCATION_COLORS) % len(_PALETTE)]
    return LOCATION_COLORS[location_id]


def location_palette(location_ids):
    return {loc: color_for(loc) for loc in location_ids}
