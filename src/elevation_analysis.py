"""Task 8 & 9: Elevation analysis - the project's CORE research question.

IMPORTANT CONTEXT: the 6 original study locations span elevation 3-10m only
(3 of them share an identical 9.0m ERA5 grid elevation). That is not enough
elevation variation to detect a real lapse-rate signal - moving between
9m and 10m produces no physically meaningful temperature difference, and any
correlation found across just 6 points with 5 nearly-identical x-values would
be driven almost entirely by the 1-2 locations that differ (Galle 3m,
Matara 6m), i.e. essentially an n=2-3 comparison dressed up as a
correlation. This script:
  1. Reports elevation vs weather relationships honestly (correlation +
     regression + per-location aggregates), for whatever locations are on
     disk.
  2. If only the original low-relief 6 are present, explicitly states that
     the result is not a meaningful test of orographic/lapse-rate effects.
  3. If additional highland locations have been added (see
     fetch_additional_locations.py), the same code produces a real elevation
     gradient analysis (Sri Lanka's actual lapse rate is close to the moist
     adiabatic ~5-6 C/km, e.g. Nuwara Eliya at 1888m is famously ~8-10 C
     cooler than the coast).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from config import PROCESSED, TABLES, FIGURES, ORIGINAL_6
from plot_style import LOCATION_COLORS

DF_PATH = PROCESSED / "analytical_dataset.parquet"
ELEV_VARS = ["temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m", "pressure_msl"]


def load():
    return pd.read_parquet(DF_PATH)


def location_level_elevation_table(df):
    agg = df.groupby(["location_id", "name", "elevation_m"]).agg(
        mean_temp=("temperature_2m", "mean"),
        mean_rain_per_year=("precipitation", lambda s: s.sum() / df["year"].nunique()),
        mean_wind=("wind_speed_10m", "mean"),
        mean_humidity=("relative_humidity_2m", "mean"),
        mean_pressure=("pressure_msl", "mean"),
    ).reset_index().sort_values("elevation_m")
    agg.to_csv(TABLES / "task8_elevation_location_summary.csv", index=False)
    return agg


def elevation_range_check(agg):
    elev_range = agg.elevation_m.max() - agg.elevation_m.min()
    n_unique = agg.elevation_m.nunique()
    print(f"Elevation range across {len(agg)} locations: {agg.elevation_m.min():.0f}m - "
          f"{agg.elevation_m.max():.0f}m (span = {elev_range:.0f}m), {n_unique} distinct elevation values")
    if elev_range < 100:
        print("\n*** WARNING: elevation span is under 100m. This is NOT sufficient to detect a "
              "real orographic/lapse-rate effect. Any correlation below should be read as a "
              "location-comparison artifact, not evidence of an elevation effect. ***")
        return "insufficient"
    return "sufficient"


def scatter_with_regression(agg, col, ylabel, fname, title_prefix):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [LOCATION_COLORS.get(l) for l in agg.location_id]
    ax.scatter(agg.elevation_m, agg[col], c=colors, s=220, edgecolor="black", zorder=3)
    for _, r in agg.iterrows():
        ax.annotate(r["name"], (r.elevation_m, r[col]), textcoords="offset points", xytext=(6, 6), fontsize=9)

    if agg.elevation_m.nunique() > 2:
        slope, intercept, r, p, se = stats.linregress(agg.elevation_m, agg[col])
        xs = np.linspace(agg.elevation_m.min(), agg.elevation_m.max(), 50)
        ax.plot(xs, intercept + slope * xs, "--", color="gray", zorder=2,
                label=f"OLS fit: slope={slope:.4f}, r={r:.2f}, p={p:.3f}")
        ax.legend()
    ax.set_xlabel("Elevation (m)"); ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix}: Elevation vs {ylabel} (n={len(agg)} locations)")
    fig.tight_layout()
    fig.savefig(FIGURES / fname)
    plt.close(fig)


def elevation_relationships(agg, status):
    specs = [
        ("mean_temp", "Mean temperature (deg C)", "13_elevation_vs_temperature.png", "Elevation vs Temperature"),
        ("mean_rain_per_year", "Mean annual rainfall (mm)", "14_elevation_vs_rainfall.png", "Elevation vs Rainfall"),
        ("mean_wind", "Mean wind speed (m/s)", "15_elevation_vs_wind.png", "Elevation vs Wind"),
        ("mean_humidity", "Mean relative humidity (%)", "16_elevation_vs_humidity.png", "Elevation vs Humidity"),
        ("mean_pressure", "Mean sea-level pressure (hPa)", "17_elevation_vs_pressure.png", "Elevation vs Pressure"),
    ]
    results = []
    for col, ylabel, fname, title in specs:
        scatter_with_regression(agg, col, ylabel, fname, title)
        if agg.elevation_m.nunique() > 2:
            pear_r, pear_p = stats.pearsonr(agg.elevation_m, agg[col])
            spear_r, spear_p = stats.spearmanr(agg.elevation_m, agg[col])
        else:
            pear_r = pear_p = spear_r = spear_p = np.nan
        results.append({"variable": col, "pearson_r": pear_r, "pearson_p": pear_p,
                         "spearman_r": spear_r, "spearman_p": spear_p, "n_locations": len(agg),
                         "elevation_variation_status": status})
    res_df = pd.DataFrame(results)
    res_df.to_csv(TABLES / "task8_elevation_correlations.csv", index=False)
    print("\n=== ELEVATION CORRELATIONS (location-level means, n={}) ===".format(len(agg)))
    print(res_df.round(4).to_string(index=False))
    return res_df


def monthly_consistency(df, agg, status):
    # Is the elevation-temperature relationship consistent month to month?
    monthly = df.groupby(["location_id", "month"])["temperature_2m"].mean().reset_index()
    monthly = monthly.merge(agg[["location_id", "elevation_m", "name"]], on="location_id")
    corr_by_month = monthly.groupby("month").apply(
        lambda g: stats.pearsonr(g.elevation_m, g.temperature_2m)[0] if g.elevation_m.nunique() > 2 else np.nan
    )
    corr_by_month.name = "pearson_r_elev_temp"
    corr_by_month.to_csv(TABLES / "task8_elevation_temp_corr_by_month.csv")
    print("\n=== Elevation-Temperature correlation by month (consistency check) ===")
    print(corr_by_month.round(3).to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    for loc in monthly.location_id.unique():
        sub = monthly[monthly.location_id == loc].sort_values("month")
        ax.plot(sub.month, sub.temperature_2m, marker="o", label=sub.name.iloc[0], color=LOCATION_COLORS.get(loc))
    ax.set_xlabel("Month"); ax.set_ylabel("Mean temperature (deg C)")
    ax.set_title(f"D: Monthly temperature by location, ordered by elevation ({status} elevation variation)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "18_monthly_temp_by_elevation_consistency.png")
    plt.close(fig)


def elevation_grouping(agg, status):
    print("\n=== TASK 9: ELEVATION GROUPING ASSESSMENT ===")
    if status == "insufficient":
        print(f"""
Elevation values present: {sorted(agg.elevation_m.unique())}
All {len(agg)} locations fall within a {agg.elevation_m.max() - agg.elevation_m.min():.0f}m band
(coastal/near-coastal lowland). Creating "low/medium/high" elevation groups
from a 3-10m range would be arbitrary and would not correspond to any real
climatological distinction (there is no boundary layer, no orographic
lifting, and no meaningful lapse-rate difference over a 7m range). Per the
project's own instructions ("If the six locations do not provide enough
elevation variation to justify grouping, explicitly say so and use
continuous elevation instead"), NO elevation groups are created for the
original 6-location dataset. Continuous elevation (and, more informatively,
latitude/coastal-exposure) is used instead wherever a location grouping is
needed.
""")
        return None
    else:
        # Enough range to justify tertile-based grouping.
        q1, q2 = agg.elevation_m.quantile([1/3, 2/3])
        agg = agg.copy()
        agg["elevation_group"] = pd.cut(agg.elevation_m, bins=[-np.inf, q1, q2, np.inf],
                                         labels=["Low", "Medium", "High"])
        group_summary = agg.groupby("elevation_group")[["mean_temp", "mean_rain_per_year", "mean_wind"]].mean().round(2)
        group_summary.to_csv(TABLES / "task9_elevation_group_summary.csv")
        print(group_summary.to_string())

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=agg, x="elevation_group", y="mean_temp", ax=ax, palette="viridis",
                    order=["Low", "Medium", "High"])
        ax.set_title("Mean temperature by elevation group (tertile split)")
        fig.savefig(FIGURES / "19_elevation_group_comparison.png")
        plt.close(fig)
        return agg


def main():
    df = load()
    agg = location_level_elevation_table(df)
    print("=== ELEVATION / WEATHER SUMMARY BY LOCATION ===")
    print(agg.to_string(index=False))

    status = elevation_range_check(agg)
    elevation_relationships(agg, status)
    monthly_consistency(df, agg, status)
    elevation_grouping(agg, status)

    print("\nElevation analysis complete. See reports/tables/task8_* and task9_* for full numbers.")


if __name__ == "__main__":
    main()
