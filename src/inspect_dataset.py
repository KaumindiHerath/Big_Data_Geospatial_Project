"""Task 1 & 2: Full dataset inspection.

Inspects every parquet file under data/raw, reports schemas, dtypes, row counts,
date ranges, missing values, duplicates, and cross-checks schema consistency
across locations and years. Also inspects locations.parquet and _source.json files.

Outputs go to reports/tables/ as CSV so downstream scripts and the report can
reuse exact, measured numbers (nothing here is invented).
"""
import json
import glob
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

RAW = Path("data/raw/regional-2020-2025/raw")
OUT = Path("reports/tables")
OUT.mkdir(parents=True, exist_ok=True)


def list_weather_files():
    return sorted(glob.glob(str(RAW / "weather" / "location_id=*" / "year=*" / "weather.parquet")))


def parse_partition(path):
    parts = Path(path).parts
    loc = [p for p in parts if p.startswith("location_id=")][0].split("=", 1)[1]
    year = [p for p in parts if p.startswith("year=")][0].split("=", 1)[1]
    return loc, int(year)


def main():
    files = list_weather_files()
    print(f"Found {len(files)} weather.parquet files")

    # ---- schema inspection per file ----
    schema_rows = []
    file_rows = []
    all_schemas = {}
    for f in files:
        loc, year = parse_partition(f)
        pf = pq.ParquetFile(f)
        schema = pf.schema_arrow
        col_types = tuple(sorted((name, str(schema.field(name).type)) for name in schema.names))
        all_schemas[(loc, year)] = col_types
        nrows = pf.metadata.num_rows
        fsize = os.path.getsize(f)
        file_rows.append({
            "location_id": loc, "year": year, "n_rows": nrows,
            "n_cols": len(schema.names), "file_size_bytes": fsize,
        })
        for name in schema.names:
            schema_rows.append({"location_id": loc, "year": year, "column": name,
                                 "dtype": str(schema.field(name).type)})

    files_df = pd.DataFrame(file_rows).sort_values(["location_id", "year"])
    files_df.to_csv(OUT / "task1_file_inventory.csv", index=False)
    print("\n=== FILE INVENTORY (rows, cols, size per location/year) ===")
    print(files_df.to_string(index=False))

    # ---- schema consistency check ----
    unique_schemas = set(all_schemas.values())
    print(f"\n=== SCHEMA CONSISTENCY: {len(unique_schemas)} unique schema(s) across {len(all_schemas)} files ===")
    schema_id_map = {s: i for i, s in enumerate(sorted(unique_schemas, key=lambda x: -len(x)))}
    consistency_rows = []
    for (loc, year), cols in all_schemas.items():
        consistency_rows.append({"location_id": loc, "year": year, "schema_id": schema_id_map[cols],
                                  "n_columns": len(cols)})
    consistency_df = pd.DataFrame(consistency_rows).sort_values(["location_id", "year"])
    consistency_df.to_csv(OUT / "task1_schema_consistency.csv", index=False)
    print(consistency_df.to_string(index=False))

    if len(unique_schemas) > 1:
        print("\n--- Schema differences detected. Diffing against the most common schema ---")
        from collections import Counter
        cnt = Counter(all_schemas.values())
        reference = cnt.most_common(1)[0][0]
        ref_cols = dict(reference)
        diff_rows = []
        for (loc, year), cols in all_schemas.items():
            cols_d = dict(cols)
            missing = sorted(set(ref_cols) - set(cols_d))
            extra = sorted(set(cols_d) - set(ref_cols))
            dtype_mismatch = sorted(c for c in set(ref_cols) & set(cols_d) if ref_cols[c] != cols_d[c])
            if missing or extra or dtype_mismatch:
                diff_rows.append({"location_id": loc, "year": year,
                                   "missing_cols": ";".join(missing),
                                   "extra_cols": ";".join(extra),
                                   "dtype_mismatch": ";".join(dtype_mismatch)})
        diff_df = pd.DataFrame(diff_rows)
        diff_df.to_csv(OUT / "task1_schema_diffs.csv", index=False)
        print(diff_df.to_string(index=False) if not diff_df.empty else "No column diffs (identical column sets).")
    else:
        print("All weather.parquet files share an identical schema (same columns & dtypes).")

    # ---- full column list (reference schema) ----
    ref_pf = pq.ParquetFile(files[0])
    cols_info = pd.DataFrame({
        "column": ref_pf.schema_arrow.names,
        "dtype": [str(ref_pf.schema_arrow.field(n).type) for n in ref_pf.schema_arrow.names],
    })
    cols_info.to_csv(OUT / "task1_weather_columns.csv", index=False)
    print("\n=== WEATHER COLUMNS (reference file) ===")
    print(cols_info.to_string(index=False))

    # ---- concatenate all weather data for deeper checks (small enough: ~20MB) ----
    dfs = []
    for f in files:
        loc, year = parse_partition(f)
        d = pd.read_parquet(f)
        d["location_id"] = loc
        d["year_partition"] = year
        dfs.append(d)
    weather = pd.concat(dfs, ignore_index=True)
    weather.to_parquet("data/processed/weather_raw_concat.parquet", index=False)
    print(f"\n=== CONCATENATED WEATHER SHAPE: {weather.shape} ===")

    # observations per location / year / location-year
    per_loc = weather.groupby("location_id").size().rename("n_obs").reset_index()
    per_year = weather.groupby("year_partition").size().rename("n_obs").reset_index()
    per_loc_year = weather.groupby(["location_id", "year_partition"]).size().rename("n_obs").reset_index()
    per_loc.to_csv(OUT / "task1_obs_per_location.csv", index=False)
    per_year.to_csv(OUT / "task1_obs_per_year.csv", index=False)
    per_loc_year.to_csv(OUT / "task1_obs_per_location_year.csv", index=False)
    print("\n--- Observations per location ---")
    print(per_loc.to_string(index=False))
    print("\n--- Observations per year ---")
    print(per_year.to_string(index=False))

    # date/time resolution & range
    time_col = None
    for cand in ["time", "date", "datetime", "timestamp"]:
        if cand in weather.columns:
            time_col = cand
            break
    print(f"\nDetected time column: {time_col}")
    if time_col:
        weather[time_col] = pd.to_datetime(weather[time_col])
        diffs = weather.sort_values([ "location_id", time_col])[time_col].diff()
        print("Most common time deltas (resolution):")
        print(diffs.value_counts().head(5))
        print(f"Date range: {weather[time_col].min()} -> {weather[time_col].max()}")

    # missing values overall
    missing = weather.isna().sum().rename("missing_count").reset_index().rename(columns={"index": "column"})
    missing["missing_pct"] = (missing["missing_count"] / len(weather) * 100).round(3)
    missing = missing.sort_values("missing_pct", ascending=False)
    missing.to_csv(OUT / "task4_missing_overall.csv", index=False)
    print("\n=== MISSING VALUES (top 15) ===")
    print(missing.head(15).to_string(index=False))

    # duplicates
    dup_exact = weather.duplicated().sum()
    key_cols = [c for c in ["location_id", time_col] if c]
    dup_key = weather.duplicated(subset=key_cols).sum() if key_cols else None
    print(f"\nExact duplicate rows: {dup_exact}")
    print(f"Duplicate (location_id, {time_col}) rows: {dup_key}")

    # ---- locations.parquet ----
    loc_path = RAW / "locations" / "locations.parquet"
    loc_df = pd.read_parquet(loc_path)
    loc_df.to_csv(OUT / "task2_locations_raw.csv", index=False)
    print(f"\n=== LOCATIONS.PARQUET: shape={loc_df.shape} ===")
    print("Columns:", list(loc_df.columns))
    print(loc_df.dtypes)
    print(loc_df.to_string(index=False))

    # ---- _source.json inspection (sample + diff across all) ----
    source_files = sorted(glob.glob(str(RAW / "weather" / "location_id=*" / "year=*" / "_source.json")))
    print(f"\nFound {len(source_files)} _source.json files")
    src_summaries = []
    for sf in source_files:
        loc, year = parse_partition(sf.replace("_source.json", "weather.parquet"))
        with open(sf) as fh:
            d = json.load(fh)
        src_summaries.append({"location_id": loc, "year": year, **{k: json.dumps(v) if isinstance(v, (list, dict)) else v
                                                                      for k, v in d.items()}})
    src_df = pd.DataFrame(src_summaries)
    src_df.to_csv(OUT / "task1_source_json_summary.csv", index=False)
    print(src_df.head(3).to_string(index=False))

    print("\nDone. All tables written to reports/tables/")


if __name__ == "__main__":
    main()
