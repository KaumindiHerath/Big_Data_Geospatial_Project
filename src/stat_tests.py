"""Task 11: Statistical hypothesis testing.

Each test below follows: research question -> H0/H1 -> assumption check ->
test choice -> statistic/p-value -> conclusion -> practical meaning. Tests
are run at daily-mean resolution (not raw hourly) for the location-comparison
tests (H1/H2) to avoid pseudo-replication from autocorrelated hourly
observations inflating power artificially; elevation/geography tests (H3-H5)
use the n=6 location-level means, the only non-pseudo-replicated sample size
available for a between-location geographic effect.
"""
import numpy as np
import pandas as pd
from scipy import stats

from config import PROCESSED, TABLES

DF_PATH = PROCESSED / "analytical_dataset.parquet"


def load():
    df = pd.read_parquet(DF_PATH)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    return df


def daily_means(df, col):
    return df.groupby(["location_id", "date"])[col].mean().reset_index()


def normality_and_variance_check(groups, label):
    print(f"\n-- Assumption checks for {label} --")
    normal = True
    for name, g in groups.items():
        if len(g) >= 8:
            stat, p = stats.shapiro(g.sample(min(len(g), 4999), random_state=42) if len(g) > 4999 else g)
            print(f"  Shapiro-Wilk normality ({name}): W={stat:.4f}, p={p:.4g} -> {'normal' if p > 0.05 else 'NOT normal'}")
            if p <= 0.05:
                normal = False
    lev_stat, lev_p = stats.levene(*groups.values())
    print(f"  Levene's test for equal variances: stat={lev_stat:.4f}, p={lev_p:.4g} -> "
          f"{'equal variances' if lev_p > 0.05 else 'UNEQUAL variances'}")
    return normal, lev_p > 0.05


def h1_temperature_by_location(df):
    print("\n" + "=" * 80)
    print("H1: Does mean daily temperature differ significantly between the six locations?")
    print("H0: All locations have the same mean daily temperature.")
    print("H1: At least one location's mean daily temperature differs.")
    dm = daily_means(df, "temperature_2m")
    groups = {loc: g["temperature_2m"] for loc, g in dm.groupby("location_id")}
    normal, equal_var = normality_and_variance_check(groups, "daily mean temperature by location")

    if normal and equal_var:
        stat, p = stats.f_oneway(*groups.values())
        test_used = "One-way ANOVA"
    else:
        stat, p = stats.kruskal(*groups.values())
        test_used = "Kruskal-Wallis H-test (non-parametric, chosen due to non-normality/unequal variance)"
    print(f"Test used: {test_used}")
    print(f"Statistic = {stat:.3f}, p-value = {p:.3g}")
    conclusion = "REJECT H0 - significant difference between locations" if p < 0.05 else "FAIL TO REJECT H0"
    print(f"Conclusion: {conclusion} (alpha=0.05)")
    print("Practical meaning: with n=6 locations across ~6 degrees of latitude, this tests whether "
          "day-to-day temperature regimes genuinely differ between sites - not whether the effect "
          "size is large.")
    return {"test": "H1_temperature_by_location", "test_used": test_used, "statistic": stat, "p_value": p,
            "conclusion": conclusion}


def h2_rainfall_by_location(df):
    print("\n" + "=" * 80)
    print("H2: Does daily rainfall differ significantly between locations?")
    print("H0: All locations have the same distribution of daily rainfall.")
    print("H1: At least one location's daily rainfall distribution differs.")
    dm = daily_means(df, "precipitation")
    # Rainfall is heavily right-skewed / zero-inflated -> normality will fail; use Kruskal-Wallis directly.
    groups = {loc: g["precipitation"] for loc, g in dm.groupby("location_id")}
    print("Rainfall is zero-inflated and right-skewed by nature (many dry hours/days) - "
          "ANOVA's normality assumption is not appropriate; using Kruskal-Wallis directly.")
    stat, p = stats.kruskal(*groups.values())
    print(f"Test used: Kruskal-Wallis H-test")
    print(f"Statistic = {stat:.3f}, p-value = {p:.3g}")
    conclusion = "REJECT H0 - significant difference between locations" if p < 0.05 else "FAIL TO REJECT H0"
    print(f"Conclusion: {conclusion} (alpha=0.05)")
    return {"test": "H2_rainfall_by_location", "test_used": "Kruskal-Wallis", "statistic": stat, "p_value": p,
            "conclusion": conclusion}


def h3_elevation_temperature(df):
    print("\n" + "=" * 80)
    print("H3: Is there a significant relationship between elevation and temperature?")
    print("H0: rho(elevation, temperature) = 0")
    print("H1: rho(elevation, temperature) != 0")
    agg = df.groupby("location_id").agg(elevation_m=("elevation_m", "first"),
                                         temperature_2m=("temperature_2m", "mean")).reset_index()
    print(f"n = {len(agg)} locations (elevation range {agg.elevation_m.min():.0f}-{agg.elevation_m.max():.0f}m)")
    r, p = stats.pearsonr(agg.elevation_m, agg.temperature_2m)
    rho, ps = stats.spearmanr(agg.elevation_m, agg.temperature_2m)
    print(f"Pearson r = {r:.3f}, p = {p:.3g} | Spearman rho = {rho:.3f}, p = {ps:.3g}")
    conclusion = "REJECT H0" if p < 0.05 else "FAIL TO REJECT H0"
    print(f"Conclusion: {conclusion} (alpha=0.05)")
    print("Practical meaning: with n=6 and a 7m elevation span, this test has very low power and "
          "the elevation values are not physically distinct enough to represent a real altitude "
          "gradient - a non-significant (or even significant) result here should NOT be "
          "interpreted as evidence about true orographic effects.")
    return {"test": "H3_elevation_temperature", "test_used": "Pearson correlation", "statistic": r, "p_value": p,
            "conclusion": conclusion}


def h4_elevation_rainfall(df):
    print("\n" + "=" * 80)
    print("H4: Is there a significant relationship between elevation and rainfall?")
    print("H0: rho(elevation, rainfall) = 0  |  H1: rho(elevation, rainfall) != 0")
    agg = df.groupby("location_id").agg(elevation_m=("elevation_m", "first"),
                                         precipitation=("precipitation", "mean")).reset_index()
    r, p = stats.pearsonr(agg.elevation_m, agg.precipitation)
    print(f"n = {len(agg)}. Pearson r = {r:.3f}, p = {p:.3g}")
    conclusion = "REJECT H0" if p < 0.05 else "FAIL TO REJECT H0"
    print(f"Conclusion: {conclusion} (alpha=0.05)")
    return {"test": "H4_elevation_rainfall", "test_used": "Pearson correlation", "statistic": r, "p_value": p,
            "conclusion": conclusion}


def h5_elevation_wind(df):
    print("\n" + "=" * 80)
    print("H5: Is there a significant relationship between elevation and wind speed?")
    print("H0: rho(elevation, wind) = 0  |  H1: rho(elevation, wind) != 0")
    agg = df.groupby("location_id").agg(elevation_m=("elevation_m", "first"),
                                         wind_speed_10m=("wind_speed_10m", "mean")).reset_index()
    r, p = stats.pearsonr(agg.elevation_m, agg.wind_speed_10m)
    print(f"n = {len(agg)}. Pearson r = {r:.3f}, p = {p:.3g}")
    conclusion = "REJECT H0" if p < 0.05 else "FAIL TO REJECT H0"
    print(f"Conclusion: {conclusion} (alpha=0.05)")
    return {"test": "H5_elevation_wind", "test_used": "Pearson correlation", "statistic": r, "p_value": p,
            "conclusion": conclusion}


def h6_latitude_temperature(df):
    print("\n" + "=" * 80)
    print("H6 (added - EDA/correlation flagged latitude, not elevation, as the strongest geographic "
          "signal): Is there a significant relationship between latitude and temperature?")
    print("H0: rho(latitude, temperature) = 0  |  H1: rho(latitude, temperature) != 0")
    agg = df.groupby("location_id").agg(latitude=("latitude", "first"),
                                         temperature_2m=("temperature_2m", "mean")).reset_index()
    r, p = stats.pearsonr(agg.latitude, agg.temperature_2m)
    print(f"n = {len(agg)}. Pearson r = {r:.3f}, p = {p:.3g}")
    conclusion = "REJECT H0" if p < 0.05 else "FAIL TO REJECT H0"
    print(f"Conclusion: {conclusion} (alpha=0.05)")
    print("Practical meaning: unlike elevation, latitude spans a genuine ~3.7 degrees (5.95N-9.66N) "
          "across these 6 locations, so this test has more physical meaning - it captures the "
          "dry-zone (north/east) vs wet-zone (south/west) climatic contrast in Sri Lanka.")
    return {"test": "H6_latitude_temperature", "test_used": "Pearson correlation", "statistic": r, "p_value": p,
            "conclusion": conclusion}


def main():
    df = load()
    results = [
        h1_temperature_by_location(df),
        h2_rainfall_by_location(df),
        h3_elevation_temperature(df),
        h4_elevation_rainfall(df),
        h5_elevation_wind(df),
        h6_latitude_temperature(df),
    ]
    res_df = pd.DataFrame(results)
    res_df.to_csv(TABLES / "task11_hypothesis_test_results.csv", index=False)
    print("\n\n=== SUMMARY OF ALL HYPOTHESIS TESTS ===")
    print(res_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
