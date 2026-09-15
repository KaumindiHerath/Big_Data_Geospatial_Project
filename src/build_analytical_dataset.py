"""Task 2 & 3: clean location reference table + merged weather+location dataset.

Reads locations.parquet, filters to the locations actually present in the
weather partitions (config.STUDY_LOCATION_IDS), and left-joins onto the
concatenated weather data built by inspect_dataset.py. Verifies the merge
doesn't duplicate or drop rows, and writes the final analytical dataset used
by every later stage of the pipeline.
"""
import glob
from pathlib import Path

import pandas as pd

from config import RAW, WEATHER_ROOT, LOCATIONS_FILE, PROCESSED, TABLES, STUDY_LOCATION_IDS, ORIGINAL_6


def parse_partition(path):
    parts = Path(path).parts
    loc = [p for p in parts if p.startswith("location_id=")][0].split("=", 1)[1]
    year = [p for p in parts if p.startswith("year=")][0].split("=", 1)[1]
    return loc, int(year)


def load_all_weather():
    files = sorted(glob.glob(str(WEATHER_ROOT / "location_id=*" / "year=*" / "weather.parquet")))
    dfs = []
    for f in files:
        loc, year = parse_partition(f)
        d = pd.read_parquet(f)
        d["location_id"] = loc
        d["year_partition"] = year
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def build_location_reference():
    loc_df = pd.read_parquet(LOCATIONS_FILE)
    ref = loc_df[loc_df["location_id"].isin(STUDY_LOCATION_IDS)].copy()
    ref["is_original_6"] = ref["location_id"].isin(ORIGINAL_6)
    ref = ref.sort_values("elevation_m").reset_index(drop=True)
    ref.to_csv(TABLES / "task2_location_reference.csv", index=False)
    print("=== CLEAN LOCATION REFERENCE TABLE (locations with downloaded weather data) ===")
    print(ref[["location_id", "name", "latitude", "longitude", "elevation_m", "region", "is_original_6"]]
          .to_string(index=False))
    return ref


def merge_weather_locations(weather, loc_ref):
    rows_before = len(weather)
    merged = weather.merge(
        loc_ref[["location_id", "name", "latitude", "longitude", "elevation_m", "region", "country"]],
        on="location_id", how="left", validate="many_to_one",
    )
    rows_after = len(merged)
    lost = rows_before - rows_after
    dup_after = merged.duplicated(subset=["location_id", "timestamp"]).sum()
    unmatched = merged["latitude"].isna().sum()

    print("\n=== MERGE VERIFICATION ===")
    print(f"Rows before merge (weather only): {rows_before}")
    print(f"Rows after merge: {rows_after}")
    print(f"Rows lost: {lost}")
    print(f"Duplicate (location_id, timestamp) after merge: {dup_after}")
    print(f"Unmatched rows (no location metadata found): {unmatched}")
    assert rows_before == rows_after, "Merge changed row count!"
    assert dup_after == 0, "Merge introduced duplicates!"
    assert unmatched == 0, "Some weather rows failed to match a location!"

    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    merged["date"] = merged["timestamp"].dt.date
    merged["year"] = merged["timestamp"].dt.year
    merged["month"] = merged["timestamp"].dt.month
    merged["day"] = merged["timestamp"].dt.day
    merged["hour"] = merged["timestamp"].dt.hour

    def season_of(m):
        # Sri Lanka's two monsoon regimes: Southwest monsoon (May-Sep) and
        # Northeast monsoon (Dec-Feb), with two inter-monsoon transition
        # periods (Mar-Apr, Oct-Nov).
        if m in (12, 1, 2):
            return "Northeast Monsoon"
        if m in (3, 4):
            return "First Inter-monsoon"
        if m in (5, 6, 7, 8, 9):
            return "Southwest Monsoon"
        return "Second Inter-monsoon"

    merged["season"] = merged["month"].apply(season_of)

    return merged


def main():
    loc_ref = build_location_reference()
    weather = load_all_weather()
    merged = merge_weather_locations(weather, loc_ref)

    out_path = PROCESSED / "analytical_dataset.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"\nSaved merged analytical dataset: {out_path} shape={merged.shape}")

    summary = pd.DataFrame([{
        "rows_before_merge": len(weather),
        "rows_after_merge": len(merged),
        "rows_lost": len(weather) - len(merged),
        "duplicate_records_after_merge": merged.duplicated(subset=["location_id", "timestamp"]).sum(),
        "n_locations": merged["location_id"].nunique(),
        "n_years": merged["year"].nunique(),
        "date_min": str(merged["timestamp"].min()),
        "date_max": str(merged["timestamp"].max()),
    }])
    summary.to_csv(TABLES / "task3_merge_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
