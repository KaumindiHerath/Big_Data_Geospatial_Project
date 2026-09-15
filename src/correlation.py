"""Task 10: Correlation analysis (Pearson + Spearman), row-level (hourly obs).

Row-level correlations (n=306,888) are statistically well-powered but MIX
two very different sources of variation: (1) genuine within-location temporal
variation (day/night, seasons) and (2) the handful of between-location
differences. For elevation/latitude/longitude specifically, this means a
"significant" row-level correlation can be driven almost entirely by only 6
distinct x-values repeated tens of thousands of times each - the p-value
looks tiny because of the repeated hourly sampling, not because the
relationship is well-estimated across genuinely independent locations. Both
the row-level and location-level (n=6) results are reported side by side so
this distinction is visible rather than hidden behind a single misleadingly
confident number.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from config import PROCESSED, TABLES, FIGURES
from plot_style import DIVERGING_CMAP

DF_PATH = PROCESSED / "analytical_dataset.parquet"

VARS = ["temperature_2m", "apparent_temperature", "precipitation", "wind_speed_10m",
        "relative_humidity_2m", "pressure_msl", "surface_pressure", "cloud_cover",
        "dew_point_2m", "elevation_m", "latitude", "longitude"]

FOCUS_PAIRS = [
    ("elevation_m", "temperature_2m"), ("elevation_m", "precipitation"),
    ("elevation_m", "wind_speed_10m"), ("elevation_m", "relative_humidity_2m"),
    ("elevation_m", "pressure_msl"),
    ("latitude", "temperature_2m"), ("latitude", "precipitation"), ("latitude", "wind_speed_10m"),
    ("longitude", "temperature_2m"), ("longitude", "precipitation"), ("longitude", "wind_speed_10m"),
    ("relative_humidity_2m", "temperature_2m"), ("pressure_msl", "elevation_m"),
]


def load():
    return pd.read_parquet(DF_PATH)


def heatmap(df):
    pearson = df[VARS].corr(method="pearson")
    spearman = df[VARS].corr(method="spearman")
    pearson.to_csv(TABLES / "task10_pearson_matrix.csv")
    spearman.to_csv(TABLES / "task10_spearman_matrix.csv")

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(pearson, cmap=DIVERGING_CMAP, center=0, annot=True, fmt=".2f", ax=axes[0],
                cbar_kws={"label": "Pearson r"}, vmin=-1, vmax=1)
    axes[0].set_title("Pearson correlation (row-level, n={:,})".format(len(df)))
    sns.heatmap(spearman, cmap=DIVERGING_CMAP, center=0, annot=True, fmt=".2f", ax=axes[1],
                cbar_kws={"label": "Spearman rho"}, vmin=-1, vmax=1)
    axes[1].set_title("Spearman correlation (row-level, n={:,})".format(len(df)))
    fig.tight_layout()
    fig.savefig(FIGURES / "20_correlation_heatmap.png")
    plt.close(fig)
    return pearson, spearman


def focused_pairs(df):
    rows = []
    for x, y in FOCUS_PAIRS:
        pear_r, pear_p = stats.pearsonr(df[x], df[y])
        spear_r, spear_p = stats.spearmanr(df[x], df[y])
        n_unique_x = df[x].nunique()
        rows.append({"x": x, "y": y, "pearson_r": pear_r, "pearson_p": pear_p,
                      "spearman_r": spear_r, "spearman_p": spear_p,
                      "n_distinct_x_values": n_unique_x, "n_rows": len(df)})
    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "task10_focused_pair_correlations.csv", index=False)
    print("=== ROW-LEVEL CORRELATIONS FOR KEY VARIABLE PAIRS ===")
    print(res.round(4).to_string(index=False))
    return res


def location_level_pairs(df):
    agg = df.groupby("location_id").agg(
        elevation_m=("elevation_m", "first"), latitude=("latitude", "first"), longitude=("longitude", "first"),
        temperature_2m=("temperature_2m", "mean"), precipitation=("precipitation", "mean"),
        wind_speed_10m=("wind_speed_10m", "mean"), relative_humidity_2m=("relative_humidity_2m", "mean"),
        pressure_msl=("pressure_msl", "mean"),
    ).reset_index()
    rows = []
    for x, y in FOCUS_PAIRS:
        if x not in agg.columns or y not in agg.columns:
            continue
        if agg[x].nunique() < 3:
            continue
        pear_r, pear_p = stats.pearsonr(agg[x], agg[y])
        rows.append({"x": x, "y": y, "pearson_r_location_level": pear_r, "pearson_p_location_level": pear_p,
                      "n_locations": len(agg)})
    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "task10_location_level_correlations.csv", index=False)
    print(f"\n=== LOCATION-LEVEL CORRELATIONS (n={len(agg)} locations - the honest sample size for geography) ===")
    print(res.round(4).to_string(index=False))
    return res


def main():
    df = load()
    heatmap(df)
    focused_pairs(df)
    location_level_pairs(df)
    print("\nCorrelation analysis complete.")


if __name__ == "__main__":
    main()
