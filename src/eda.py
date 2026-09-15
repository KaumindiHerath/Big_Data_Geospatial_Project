"""Task 6: Exploratory Data Analysis - overall, temporal, and location-based.

Every figure is tied to a specific question (see comments above each plot
call). Key variables were chosen because they are the ones the project brief
asks about (temperature, rainfall, wind) plus humidity/pressure/cloud cover,
which matter for the elevation story later.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import PROCESSED, TABLES, FIGURES, ORIGINAL_6
from plot_style import LOCATION_COLORS, SEQUENTIAL_CMAP

DF_PATH = PROCESSED / "analytical_dataset.parquet"

KEY_VARS = ["temperature_2m", "apparent_temperature", "precipitation", "wind_speed_10m",
            "relative_humidity_2m", "pressure_msl", "cloud_cover"]

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load():
    df = pd.read_parquet(DF_PATH)
    df["location_name"] = df["name"]
    return df


# ---------------------------------------------------------------- 1. OVERALL
def overall_summary(df):
    summary = df[KEY_VARS].describe(percentiles=[.25, .5, .75]).T
    summary["skew"] = df[KEY_VARS].skew()
    summary.to_csv(TABLES / "task6_overall_summary_stats.csv")
    print("=== OVERALL SUMMARY STATISTICS ===")
    print(summary.round(2).to_string())

    # Q: What do the distributions of the core weather variables look like -
    # are any skewed, and where do extremes sit?
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()
    for i, col in enumerate(KEY_VARS):
        sns.histplot(df[col], bins=50, ax=axes[i], color="#4C72B0", kde=True)
        axes[i].set_title(col)
        axes[i].set_xlabel("")
    axes[-1].axis("off")
    fig.suptitle("Distributions of core weather variables (all locations, 2020-2025 hourly)")
    fig.tight_layout()
    fig.savefig(FIGURES / "01_overall_distributions_hist.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    df_z = df[KEY_VARS].apply(lambda s: (s - s.mean()) / s.std())
    sns.boxplot(data=df_z, ax=ax, palette="colorblind")
    ax.set_title("Z-scored boxplots of core weather variables (spread & outliers, standardized for comparability)")
    ax.set_ylabel("Standard deviations from mean")
    plt.xticks(rotation=30, ha="right")
    fig.savefig(FIGURES / "02_overall_boxplots_zscored.png")
    plt.close(fig)


# --------------------------------------------------------------- 2. TEMPORAL
def temporal_eda(df):
    # Q: Is there a detectable annual temperature trend 2020-2025?
    annual = df.groupby(["year", "location_name"])["temperature_2m"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 6))
    for loc in df["location_id"].unique():
        name = df.loc[df.location_id == loc, "name"].iloc[0]
        sub = annual[annual.location_name == name]
        ax.plot(sub.year, sub.temperature_2m, marker="o", label=name, color=LOCATION_COLORS.get(loc))
    ax.set_title("Annual mean temperature by location, 2020-2025")
    ax.set_xlabel("Year"); ax.set_ylabel("Mean temperature (deg C)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.savefig(FIGURES / "03_annual_temperature_trend.png")
    plt.close(fig)
    annual.to_csv(TABLES / "task6_annual_temperature_by_location.csv", index=False)

    # Q: What is the monthly temperature seasonality, and does it differ by location?
    monthly_temp = df.groupby(["month", "location_name"])["temperature_2m"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 6))
    for loc in df["location_id"].unique():
        name = df.loc[df.location_id == loc, "name"].iloc[0]
        sub = monthly_temp[monthly_temp.location_name == name].sort_values("month")
        ax.plot(sub.month, sub.temperature_2m, marker="o", label=name, color=LOCATION_COLORS.get(loc))
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTH_NAMES)
    ax.set_title("Monthly mean temperature by location (seasonality)")
    ax.set_ylabel("Mean temperature (deg C)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.savefig(FIGURES / "04_monthly_temperature_seasonality.png")
    plt.close(fig)
    monthly_temp.to_csv(TABLES / "task6_monthly_temperature_by_location.csv", index=False)

    # Q: Monthly rainfall pattern - do monsoon months stand out?
    monthly_rain = df.groupby(["month", "location_name"])["precipitation"].sum().reset_index()
    monthly_rain_avg = df.groupby(["month", "location_name", "year"])["precipitation"].sum().reset_index()
    monthly_rain_avg = monthly_rain_avg.groupby(["month", "location_name"])["precipitation"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 6))
    for loc in df["location_id"].unique():
        name = df.loc[df.location_id == loc, "name"].iloc[0]
        sub = monthly_rain_avg[monthly_rain_avg.location_name == name].sort_values("month")
        ax.plot(sub.month, sub.precipitation, marker="o", label=name, color=LOCATION_COLORS.get(loc))
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTH_NAMES)
    ax.set_title("Average monthly total rainfall by location (2020-2025 mean)")
    ax.set_ylabel("Mean monthly rainfall (mm)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.savefig(FIGURES / "05_monthly_rainfall_seasonality.png")
    plt.close(fig)
    monthly_rain_avg.to_csv(TABLES / "task6_monthly_rainfall_by_location.csv", index=False)

    # Q: Monthly wind pattern - does wind pick up in a particular monsoon phase?
    monthly_wind = df.groupby(["month", "location_name"])["wind_speed_10m"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 6))
    for loc in df["location_id"].unique():
        name = df.loc[df.location_id == loc, "name"].iloc[0]
        sub = monthly_wind[monthly_wind.location_name == name].sort_values("month")
        ax.plot(sub.month, sub.wind_speed_10m, marker="o", label=name, color=LOCATION_COLORS.get(loc))
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTH_NAMES)
    ax.set_title("Monthly mean wind speed by location")
    ax.set_ylabel("Mean wind speed (m/s)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.savefig(FIGURES / "06_monthly_wind_seasonality.png")
    plt.close(fig)
    monthly_wind.to_csv(TABLES / "task6_monthly_wind_by_location.csv", index=False)

    # Season table (monsoon regime means)
    season_summary = df.groupby("season")[["temperature_2m", "precipitation", "wind_speed_10m",
                                            "relative_humidity_2m"]].mean().round(2)
    season_summary.to_csv(TABLES / "task6_season_summary.csv")
    print("\n=== MEAN WEATHER BY MONSOON SEASON ===")
    print(season_summary.to_string())


# ------------------------------------------------------------- 3. LOCATION
def location_eda(df):
    order = [df.loc[df.location_id == l, "name"].iloc[0] for l in ORIGINAL_6 if l in df.location_id.unique()]

    # Q: Which locations have distinctly different temperature/rainfall/wind/humidity?
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    for ax, col, title in zip(
        axes,
        ["temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m"],
        ["Temperature (deg C)", "Precipitation (mm/hr)", "Wind speed (m/s)", "Relative humidity (%)"],
    ):
        sns.boxplot(data=df, x="location_name", y=col, order=order, ax=ax,
                    palette=[LOCATION_COLORS.get(l) for l in ORIGINAL_6 if l in df.location_id.unique()])
        ax.set_title(title); ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Distribution of key weather variables by location")
    fig.tight_layout()
    fig.savefig(FIGURES / "07_location_boxplot_comparison.png")
    plt.close(fig)

    loc_summary = df.groupby("location_name")[KEY_VARS].mean().round(2).reindex(order)
    loc_summary.to_csv(TABLES / "task6_location_summary.csv")
    print("\n=== MEAN WEATHER VARIABLES BY LOCATION ===")
    print(loc_summary.to_string())

    # Q: heatmap of monthly mean temperature x location - visualize seasonal+spatial pattern together
    pivot = df.pivot_table(index="location_name", columns="month", values="temperature_2m", aggfunc="mean")
    pivot = pivot.reindex(order)
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(pivot, cmap=SEQUENTIAL_CMAP, annot=True, fmt=".1f", ax=ax, cbar_kws={"label": "Mean temp (C)"})
    ax.set_xticklabels(MONTH_NAMES)
    ax.set_title("Monthly mean temperature heatmap by location")
    fig.savefig(FIGURES / "08_location_month_temp_heatmap.png")
    plt.close(fig)

    # Ranked comparison: mean temperature and rainfall, ranked
    ranked = loc_summary[["temperature_2m", "precipitation", "wind_speed_10m"]].sort_values("temperature_2m")
    fig, ax = plt.subplots(figsize=(9, 5))
    ranked["temperature_2m"].plot(kind="barh", ax=ax, color=[LOCATION_COLORS.get(l) for l in ORIGINAL_6
                                                               if df.loc[df.location_id == l, "name"].iloc[0] in ranked.index])
    ax.set_xlabel("Mean temperature (deg C)")
    ax.set_title("Locations ranked by mean temperature (2020-2025)")
    fig.savefig(FIGURES / "09_location_temp_ranked.png")
    plt.close(fig)


def main():
    df = load()
    overall_summary(df)
    temporal_eda(df)
    location_eda(df)
    print("\nEDA complete. Figures in reports/figures/, tables in reports/tables/.")


if __name__ == "__main__":
    main()
