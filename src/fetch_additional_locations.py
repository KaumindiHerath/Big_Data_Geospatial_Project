"""Optional data-extension script (NOT run in this session - sandbox has no
outbound access to archive-api.open-meteo.com).

Run this from a machine with normal internet access to pull additional
Sri Lankan locations that already exist in locations.parquet / study-plan.json
but were never downloaded (ingestion was paused after 35/600 chunks). This
gives the elevation analysis real variation: the original 6 locations are all
coastal lowland (3-10m). These additions span 90m -> 2130m.

Output is written in EXACTLY the same partition layout as the existing raw
data (location_id=.../year=.../weather.parquet + _source.json), so it drops
straight into data/raw/regional-2020-2025/raw/weather/ alongside the original
files with no changes needed elsewhere.

Usage:
    python3 fetch_additional_locations.py
"""
import json
import time
from pathlib import Path

import requests
import pandas as pd

OUT_ROOT = Path("data/raw/regional-2020-2025/raw/weather")

# Additional Sri Lankan locations from locations.parquet, chosen to span the
# elevation gradient from low foothills to the island's highest terrain,
# complementing the 6 original coastal cities (3-10m).
ADDITIONAL_LOCATIONS = [
    {"location_id": "lk_anuradhapura", "name": "Anuradhapura", "latitude": 8.3114, "longitude": 80.4037, "elevation_m": 90.0},
    {"location_id": "lk_kurunegala", "name": "Kurunegala", "latitude": 7.4863, "longitude": 80.3647, "elevation_m": 124.0},
    {"location_id": "lk_kandy", "name": "Kandy", "latitude": 7.2906, "longitude": 80.6337, "elevation_m": 499.0},
    {"location_id": "lk_badulla", "name": "Badulla", "latitude": 6.9934, "longitude": 81.0550, "elevation_m": 661.0},
    {"location_id": "lk_nuwara_eliya", "name": "Nuwara Eliya", "latitude": 6.9497, "longitude": 80.7891, "elevation_m": 1888.0},
    {"location_id": "lk_horton_plains", "name": "Horton Plains", "latitude": 6.8021, "longitude": 80.8070, "elevation_m": 2130.0},
]

VARIABLES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "precipitation", "rain", "snowfall", "snow_depth", "weather_code",
    "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low",
    "cloud_cover_mid", "cloud_cover_high", "et0_fao_evapotranspiration",
    "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_100m",
    "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m",
    "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm",
    "boundary_layer_height", "wet_bulb_temperature_2m",
    "total_column_integrated_water_vapour", "is_day", "sunshine_duration",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "direct_normal_irradiance", "global_tilted_irradiance", "terrestrial_radiation",
    "shortwave_radiation_instant", "direct_radiation_instant",
    "diffuse_radiation_instant", "direct_normal_irradiance_instant",
    "global_tilted_irradiance_instant", "terrestrial_radiation_instant",
]

YEARS = range(2020, 2026)
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_year(loc, year):
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    params = {
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(VARIABLES),
        "timezone": "UTC",
        "models": "era5_seamless",
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
    }
    resp = requests.get(BASE_URL, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.rename(columns={"time": "timestamp"})
    df["source_year"] = year
    df["ingested_at"] = pd.Timestamp.utcnow()

    out_dir = OUT_ROOT / f"location_id={loc['location_id']}" / f"year={year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "weather.parquet", index=False)

    source_meta = {
        "request": {"location_id": loc["location_id"], "latitude": loc["latitude"],
                     "longitude": loc["longitude"], "year": year, "start": start,
                     "end": end, "hours": len(df)},
        "model": "era5_seamless",
        "hourly_units": data.get("hourly_units", {}),
        "grid": {"latitude": data.get("latitude"), "longitude": data.get("longitude"),
                 "elevation": data.get("elevation"), "timezone": data.get("timezone", "UTC")},
        "null_counts": {c: int(df[c].isna().sum()) for c in VARIABLES if c in df.columns},
    }
    with open(out_dir / "_source.json", "w") as fh:
        json.dump(source_meta, fh, indent=2)

    print(f"  {loc['location_id']} {year}: {len(df)} rows -> {out_dir}")


def main():
    for loc in ADDITIONAL_LOCATIONS:
        print(f"Fetching {loc['name']} ({loc['location_id']}, elevation {loc['elevation_m']}m)...")
        for year in YEARS:
            fetch_year(loc, year)
            time.sleep(1)  # be polite to the free API
    print("\nDone. New partitions written under data/raw/regional-2020-2025/raw/weather/.")
    print("Re-run the analysis pipeline (src/run_all.py) to pick them up automatically.")


if __name__ == "__main__":
    main()
