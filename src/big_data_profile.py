"""Task 5: Big Data scale / structure profiling.

Honest assessment of whether this dataset is "big data" in volume terms,
plus the storage/partitioning rationale and a note on when PySpark becomes
justified as the data scales (see pyspark_demo.py for a working demo).
"""
import glob
import os

import pandas as pd

from config import RAW, WEATHER_ROOT, PROCESSED, TABLES

DF_PATH = PROCESSED / "analytical_dataset.parquet"


def dir_size_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def main():
    df = pd.read_parquet(DF_PATH)

    n_rows = len(df)
    n_cols = df.shape[1]
    n_locations = df["location_id"].nunique()
    n_years = df["year"].nunique()

    parquet_files = glob.glob(str(WEATHER_ROOT / "location_id=*" / "year=*" / "weather.parquet"))
    raw_parquet_bytes = sum(os.path.getsize(f) for f in parquet_files)
    raw_total_bytes = dir_size_bytes(RAW)
    processed_bytes = os.path.getsize(DF_PATH)
    csv_equiv_bytes = df.memory_usage(deep=True).sum()  # approx in-memory (uncompressed-ish) size

    stats = {
        "total_rows": n_rows,
        "total_locations": n_locations,
        "total_years": n_years,
        "total_variables_columns": n_cols,
        "n_weather_parquet_files": len(parquet_files),
        "raw_weather_parquet_size_MB": round(raw_parquet_bytes / 1e6, 3),
        "raw_data_dir_total_size_MB": round(raw_total_bytes / 1e6, 3),
        "processed_merged_parquet_size_MB": round(processed_bytes / 1e6, 3),
        "in_memory_pandas_size_MB": round(csv_equiv_bytes / 1e6, 3),
        "avg_bytes_per_row_parquet": round(raw_parquet_bytes / n_rows, 2),
    }
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(TABLES / "task5_scale_profile.csv", index=False)
    print("=== DATASET SCALE PROFILE ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    obs_per_loc_year = df.groupby(["location_id", "year"]).size().rename("n_obs").reset_index()
    obs_per_loc_year.to_csv(TABLES / "task5_obs_per_location_year.csv", index=False)

    print("\n=== HONEST 'IS THIS BIG DATA?' ASSESSMENT ===")
    print(f"""
Raw volume: {stats['raw_data_dir_total_size_MB']:.1f} MB across {stats['n_weather_parquet_files']} Parquet
files, {n_rows:,} rows. This is NOT big data by the classic 3V definition
(volume/velocity/variety) - it comfortably fits in memory on a laptop
(~{stats['in_memory_pandas_size_MB']:.0f} MB as a pandas DataFrame) and every operation in this
project runs in seconds with plain pandas/PyArrow. Claiming otherwise would
be dishonest.

What IS genuinely "big-data-shaped" about this project:
  1. Partitioning strategy: the raw data is partitioned by location_id then
     year (Hive-style directory partitioning), which is exactly the pattern
     used at production scale (e.g. a weather platform ingesting thousands
     of stations) - it allows partition pruning (reading only the
     location/year you need) and safe, parallel, idempotent appends per
     partition without touching existing files.
  2. Columnar Parquet format: ~{stats['avg_bytes_per_row_parquet']:.1f} bytes/row compressed, with
     predicate/column pushdown - efficient for the wide (48-variable) but
     mostly-numeric schema used here, and directly readable by both pandas
     and Spark with no conversion step.
  3. Scale trajectory: study-plan.json shows the *original* ingestion plan
     covered 100 locations x 6 years x hourly = 5,260,800 rows (~830x more
     than what was actually downloaded before the ingestion quota paused it
     at 35/600 chunks). At that full scale (~100 locations) or if extended
     to sub-hourly resolution, multiple variables per grid cell, or many
     more countries, this would cross into a regime (10s-100s of millions of
     rows, multi-GB) where a single machine's RAM becomes the bottleneck and
     distributed processing (PySpark) is genuinely justified - not for this
     6-location snapshot, but for the pipeline this project is a prototype
     of. pyspark_demo.py demonstrates the same operations running on Spark
     against this same partitioned Parquet layout, so the code path already
     scales without a rewrite.
""")


if __name__ == "__main__":
    main()
