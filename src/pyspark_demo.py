"""Task 21: PySpark demonstration.

Runs the same core operations (load partitioned Parquet, filter, group by
location/year, aggregate, join weather with location metadata) using Spark
against the SAME on-disk partitioned layout used by the pandas pipeline, to
show the pipeline already scales without a rewrite (see big_data_profile.py
for the honest volume assessment - this is a demonstration of the pattern,
not a claim that Spark is *necessary* at the current ~300K-row scale).

Run with the project's venv that has pyspark installed:
    /tmp/spark_venv/bin/python3 src/pyspark_demo.py
(a plain pyspark pip install failed in the main environment due to a
setuptools/distutils incompatibility in this sandbox - documented in
README.md - so PySpark lives in a separate virtualenv here.)
"""
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from config import RAW, WEATHER_ROOT, LOCATIONS_FILE, TABLES

# Spark's RPC URL parser rejects underscores in hostnames. Windows machine
# names commonly contain one (e.g. "KAUMINDI_HERATH"), which crashes
# SparkContext init with "Invalid Spark URL" before any code here runs.
# Forcing the driver to bind/advertise on loopback sidesteps that - local[*]
# mode never needs to be reachable from another machine anyway.
os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")


def main():
    spark = (
        SparkSession.builder
        .appName("GeospatialWeatherElevation")
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    print("=== 1. LOAD PARTITIONED PARQUET (Hive-style location_id=/year= partitioning) ===")
    t0 = time.time()
    weather = spark.read.parquet(str(WEATHER_ROOT))
    # location_id and year are recovered automatically from the partition
    # directory names by Spark's partition discovery - no manual parsing needed,
    # unlike the pandas path in build_analytical_dataset.py.
    n = weather.count()
    print(f"Loaded {n:,} rows in {time.time()-t0:.2f}s. Partition columns auto-discovered: "
          f"{[c for c in weather.columns if c in ('location_id', 'year')]}")
    weather.printSchema()

    print("\n=== 2. FILTER (partition pruning demo: only 2023, only Colombo) ===")
    t0 = time.time()
    filtered = weather.filter((F.col("year") == 2023) & (F.col("location_id") == "lk_colombo"))
    print(f"Filtered rows: {filtered.count():,} in {time.time()-t0:.2f}s "
          "(Spark only reads the location_id=lk_colombo/year=2023 partition directory, "
          "not the full dataset - this is the payoff of Hive-style partitioning at scale)")

    print("\n=== 3. GROUP BY location_id, year -> AGGREGATE weather statistics ===")
    agg = (
        weather.groupBy("location_id", "year")
        .agg(
            F.avg("temperature_2m").alias("mean_temp"),
            F.avg("precipitation").alias("mean_precip"),
            F.avg("wind_speed_10m").alias("mean_wind"),
            F.count("*").alias("n_obs"),
        )
        .orderBy("location_id", "year")
    )
    agg_pd = agg.toPandas()
    print(agg_pd.to_string(index=False))
    agg_pd.to_csv(TABLES / "task21_pyspark_location_year_agg.csv", index=False)

    print("\n=== 4. JOIN weather with location metadata ===")
    locations = spark.read.parquet(str(LOCATIONS_FILE))
    joined = weather.join(
        locations.select("location_id", "name", "latitude", "longitude", "elevation_m"),
        on="location_id", how="left",
    )
    rows_before = weather.count()
    rows_after = joined.count()
    unmatched = joined.filter(F.col("latitude").isNull()).count()
    print(f"Rows before join: {rows_before:,} | after join: {rows_after:,} | unmatched: {unmatched:,}")

    print("\n=== 5. PARTITION-AWARE AGGREGATION: mean temperature by elevation-ordered location ===")
    elev_summary = (
        joined.groupBy("location_id", "name", "elevation_m")
        .agg(F.avg("temperature_2m").alias("mean_temp"), F.count("*").alias("n_obs"))
        .orderBy("elevation_m")
    )
    elev_pd = elev_summary.toPandas()
    print(elev_pd.to_string(index=False))
    elev_pd.to_csv(TABLES / "task21_pyspark_elevation_summary.csv", index=False)

    print("""
=== PANDAS vs PYSPARK: conceptual comparison for this project ===
  - At the CURRENT scale (~300K rows, ~20MB), pandas/PyArrow is strictly
    better: sub-second operations, no JVM startup cost (Spark session init
    alone took longer here than the pandas pipeline's entire runtime), full
    eager evaluation, simpler debugging, and every plotting/stats/ML library
    used in this project (seaborn, scipy, statsmodels, sklearn, shap) expects
    pandas/numpy input directly.
  - Spark's structural advantages (this demo) - partition discovery/pruning,
    lazy evaluation with query optimization (Catalyst), distributed
    shuffle/aggregation, and a SQL/DataFrame API that scales from a laptop to
    a cluster with the same code - only start paying off once data exceeds
    single-machine memory (roughly: tens of millions of rows / multi-GB+),
    or once ingestion needs to run continuously against many concurrent
    partitions. The original study-plan.json (100 locations x 6 years
    hourly x 48 variables = 5.26M rows) is still comfortably pandas-scale;
    a genuinely Spark-justified version of this project would be one
    tracking thousands of stations globally, or storing sub-hourly/
    multi-model ensemble data.
  - This script exists to prove the partitioned-Parquet design choice is not
    wasted effort: the exact same directory layout that pandas concatenates
    with a Python for-loop (build_analytical_dataset.py) is read natively by
    Spark's partition discovery with zero changes to the data.
""")

    spark.stop()


if __name__ == "__main__":
    main()
