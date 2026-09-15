"""Shared paths and constants for the pipeline.

STUDY_LOCATION_IDS is auto-detected from whatever weather partitions actually
exist on disk, so if fetch_additional_locations.py is run later to add more
Sri Lankan highland sites, every downstream script picks them up automatically
without code changes.
"""
import glob
from pathlib import Path

RAW = Path("data/raw/regional-2020-2025/raw")
WEATHER_ROOT = RAW / "weather"
LOCATIONS_FILE = RAW / "locations" / "locations.parquet"
PROCESSED = Path("data/processed")
REPORTS = Path("reports")
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"
MAPS = REPORTS / "maps"

for d in [PROCESSED, TABLES, FIGURES, MAPS]:
    d.mkdir(parents=True, exist_ok=True)


def discover_location_ids():
    dirs = glob.glob(str(WEATHER_ROOT / "location_id=*"))
    return sorted(d.split("location_id=")[-1] for d in dirs)


STUDY_LOCATION_IDS = discover_location_ids()

# The 6 locations originally specified in the project brief.
ORIGINAL_6 = ["lk_jaffna", "lk_batticaloa", "lk_trincomalee", "lk_colombo", "lk_matara", "lk_galle"]

RANDOM_STATE = 42
