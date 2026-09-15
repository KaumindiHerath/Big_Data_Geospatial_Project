"""Runs the full reproducible pipeline in order, from raw data to figures/tables.
PySpark demo is separate (needs the /tmp/spark_venv interpreter - see README.md).

Usage: python3 src/run_all.py  (from the project root)
"""
import subprocess
import sys

STEPS = [
    "inspect_dataset.py",
    "build_analytical_dataset.py",
    "data_quality.py",
    "big_data_profile.py",
    "eda.py",
    "geospatial.py",
    "elevation_analysis.py",
    "correlation.py",
    "stat_tests.py",
    "regression.py",
    "ml_pipeline.py",
]


def main():
    for step in STEPS:
        print(f"\n{'#'*80}\n# RUNNING: {step}\n{'#'*80}")
        result = subprocess.run([sys.executable, f"src/{step}"])
        if result.returncode != 0:
            print(f"FAILED at {step}, stopping.")
            sys.exit(1)
    print("\nAll steps completed. Run PySpark demo separately: "
          "PYTHONPATH=src /tmp/spark_venv/bin/python3 src/pyspark_demo.py")


if __name__ == "__main__":
    main()
