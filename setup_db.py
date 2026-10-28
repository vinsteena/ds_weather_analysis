"""
WeatherWise Database Initialization and Data Seeding Utility
=============================================================
Automatically provisions the local SQLite database (weatherwise.db)
and seeds 45 years (1980-2025) of multi-city daily meteorological data
from the 5 attached historical CSV datasets into the indexed SQLite engine.
"""

import os
import sys
import time
import sqlite3
import unicodedata
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "weatherwise.db")
ARCHIVE_DIR = os.path.join(BASE_DIR, "data", "historical_archives")

def normalize_city_name(name: str) -> str:
    """Removes diacritics/accents to match clean ASCII names."""
    if not isinstance(name, str):
        return str(name)
    return unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8').strip()

def init_database(force_reseed: bool = False):
    """Initializes tables and seeds historical weather data if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1. Investigations Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS investigations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            dataset_name TEXT,
            date_range TEXT,
            variables TEXT,
            methods TEXT,
            findings TEXT,
            created_at TEXT,
            scenario_details TEXT
        )
    """)

    # 2. API Settings Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS api_settings (
            key_name TEXT PRIMARY KEY,
            key_value TEXT,
            provider TEXT,
            updated_at TEXT
        )
    """)

    # 3. Historical Weather Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS historical_weather (
            city_name TEXT,
            clean_city_name TEXT,
            date TEXT,
            temperature REAL,
            temperature_max REAL,
            temperature_min REAL,
            rainfall REAL,
            wind_speed REAL,
            weather_code INTEGER
        )
    """)
    conn.commit()

    # Check existing row count
    cur.execute("SELECT COUNT(*) FROM historical_weather")
    count = cur.fetchone()[0]

    if count > 500000 and not force_reseed:
        print(f"[WeatherWise DB] Database is healthy and pre-seeded ({count:,} historical weather rows).")
        conn.close()
        return

    print("[WeatherWise DB] Seeding historical archives into SQLite database...")
    t0 = time.time()

    if force_reseed:
        cur.execute("DELETE FROM historical_weather")
        conn.commit()

    files = [
        os.path.join(ARCHIVE_DIR, "daily_data_combined_1980_to_1989.csv"),
        os.path.join(ARCHIVE_DIR, "daily_data_combined_1990_to_1999.csv"),
        os.path.join(ARCHIVE_DIR, "daily_data_combined_2000_to_2009.csv"),
        os.path.join(ARCHIVE_DIR, "daily_data_combined_2010_to_2019.csv"),
        os.path.join(ARCHIVE_DIR, "daily_data.csv")
    ]

    cols_mapping = {
        'city_name': 'city_name',
        'datetime': 'date',
        'temperature_2m_mean': 'temperature',
        'temperature_2m_max': 'temperature_max',
        'temperature_2m_min': 'temperature_min',
        'precipitation_sum': 'rainfall',
        'wind_speed_10m_max': 'wind_speed',
        'weather_code': 'weather_code'
    }

    total_imported = 0
    for f in files:
        if not os.path.exists(f):
            print(f"[Warning] Archive file not found: {f}")
            continue

        try:
            df = pd.read_csv(f, usecols=list(cols_mapping.keys()))
            df = df.rename(columns=cols_mapping)
            # Clean null temperatures
            df = df.dropna(subset=['temperature'])
            df['clean_city_name'] = df['city_name'].apply(normalize_city_name)

            df.to_sql('historical_weather', conn, if_exists='append', index=False)
            total_imported += len(df)
            print(f"  -> Ingested {os.path.basename(f)} ({len(df):,} records)")
        except Exception as e:
            print(f"[Error importing {f}]: {e}")

    # Build optimal indexes for fast sub-millisecond retrieval
    print("[WeatherWise DB] Building high-performance B-Tree indexes...")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_clean_city_date ON historical_weather(clean_city_name, date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_city_date ON historical_weather(city_name, date)")
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM historical_weather")
    final_count = cur.fetchone()[0]
    conn.close()

    elapsed = time.time() - t0
    print(f"[WeatherWise DB] Database successfully built: {final_count:,} records in {elapsed:.2f}s.")

if __name__ == "__main__":
    force = "--force" in sys.argv
    init_database(force_reseed=force)
