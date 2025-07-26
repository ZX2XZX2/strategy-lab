import os

from pathlib import Path

ROOT_DIR = Path(os.getenv("STRATEGY_LAB_DATA_DIR", "/tmp/strategy_lab_data"))
EOD_DIR = ROOT_DIR / "eod"
INTRADAY_DIR = ROOT_DIR / "intraday"
SPLITS_DIR = ROOT_DIR / "splits"
INDICATORS_DIR = ROOT_DIR / "indicators"
METADATA_DIR = ROOT_DIR / "metadata"
CALENDAR_PATH = ROOT_DIR / "calendar.parquet"

DB_CONFIG = {
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "port": os.getenv("POSTGRES_PORT"),
    "database": os.getenv("POSTGRES_DB"),
    "host": os.getenv("POSTGRES_HOST")
}

# Ensure the directory structure exists
if not ROOT_DIR.exists():
    os.makedirs(ROOT_DIR)
if not EOD_DIR.exists():
    os.makedirs(EOD_DIR)
if not INTRADAY_DIR.exists():
    os.makedirs(INTRADAY_DIR)
if not SPLITS_DIR.exists():
    os.makedirs(SPLITS_DIR)
if not INDICATORS_DIR.exists():
    os.makedirs(INDICATORS_DIR)
if not METADATA_DIR.exists():
    os.makedirs(METADATA_DIR)
