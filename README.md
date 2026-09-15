# Geospatial Weather and Elevation Analysis (Option 5)

Big Data Analytics group project: how geographical location and elevation
influence weather conditions across Sri Lankan locations, using historical
Open-Meteo (ERA5 reanalysis) data, 2020-2025.

## Repository layout

```
data/
  raw/                        # untouched, as-provided raw data (never edit)
    regional-2020-2025/raw/
      locations/locations.parquet
      weather/location_id=.../year=.../{weather.parquet,_source.json}
  processed/                  # generated analytical datasets (safe to regenerate)
    analytical_dataset.parquet
    weather_raw_concat.parquet
src/                          # one script per analysis stage (see below)
reports/
  tables/                     # every numeric result, as CSV (task-numbered)
  figures/                    # every chart, as PNG (task-numbered)
  maps/                       # interactive Folium HTML map
  REPORT.md                   # full structured write-up (20 sections)
requirements.txt
```

## Reproducing the analysis

```bash
pip install -r requirements.txt
python3 src/run_all.py        # runs every stage, from raw inspection to ML
```

Each `src/*.py` script can also be run individually (from the project root,
so relative paths resolve) and writes its outputs to `reports/tables/` and
`reports/figures/`, prefixed with the task number it corresponds to.

### PySpark setup

`pip install pyspark` failed directly in this sandbox's system Python due to
a `setuptools`/`distutils` `install_layout` incompatibility (Debian-patched
setuptools vs. pyspark's legacy `setup.py`). Fix used here: a clean
virtualenv.

```bash
python3 -m venv /tmp/spark_venv
/tmp/spark_venv/bin/pip install pyspark pandas pyarrow
PYTHONPATH=src /tmp/spark_venv/bin/python3 src/pyspark_demo.py
```

## Extending the dataset (optional)

The 6 original locations (Jaffna, Batticaloa, Trincomalee, Colombo, Matara,
Galle) are all coastal lowland, elevation 3-10m - not enough range to
meaningfully test an elevation/lapse-rate effect (see REPORT.md, Limitations).
`locations.parquet` already lists 94 other South/Southeast Asian locations
that were never downloaded (the original ingestion was paused after 35/600
planned chunks - see `data/raw/regional-2020-2025/ingestion-status.json`).

`src/fetch_additional_locations.py` fetches 6 more Sri Lankan locations
spanning 90m-2130m elevation, in the exact same partitioned-Parquet format,
from the free/keyless Open-Meteo archive API. It could not be run inside this
session (outbound network access to `archive-api.open-meteo.com` is blocked
by this sandbox's proxy policy). Run it from a machine with normal internet
access, then re-run `src/run_all.py` - every downstream script auto-discovers
whatever locations are present in `data/raw/.../weather/` and the elevation
analysis will automatically use the wider range.

## Data provenance note

`data/raw/regional-2020-2025/study-plan.json` and `ingestion-status.json`
show the raw data was collected by an ingestion job originally planned for
100 locations across Sri Lanka, India, Nepal, Bhutan, Bangladesh, Myanmar,
Thailand, Vietnam, Laos, Cambodia, Malaysia, Indonesia and Singapore, but
paused after downloading only 35 of 600 planned location-year chunks - all of
them the 6 Sri Lankan coastal cities this project uses (`lk_matara/year=2025`
is also missing). This is documented in full in `reports/REPORT.md`,
section 17 (Limitations) and section 1 (Dataset Overview).
