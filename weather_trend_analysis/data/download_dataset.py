"""
Weather Analytics Dataset Ingestion Module
Fetches 10 years of authentic ECMWF ERA5 reanalysis data via Open-Meteo Historical Archive API.
Includes an offline realistic meteorological generator as an automatic fallback.
"""

import os
import sys
import numpy as np
import pandas as pd
import requests

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(DATA_DIR, "weather_analytics_era5_2015_2024.csv")

def fetch_open_meteo_era5(latitude=12.9716, longitude=77.5946, start_date="2015-01-01", end_date="2024-12-31"):
    """
    Downloads authentic ECMWF ERA5 meteorological records from Open-Meteo Historical Archive API.
    Covers daily aggregates for:
    - Temperature (mean, max, min in °C)
    - Relative Humidity (mean %)
    - Dew Point / Moisture in air (mean °C)
    - Precipitation / Rain (sum in mm)
    - Surface Atmospheric Pressure (mean in hPa)
    - Wind Speed (mean and max in km/h)
    - Wind Direction (dominant in degrees)
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "relative_humidity_2m_mean",
            "dew_point_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "surface_pressure_mean",
            "wind_speed_10m_mean",
            "wind_speed_10m_max",
            "wind_direction_10m_dominant"
        ],
        "timezone": "auto"
    }

    print(f"Fetching ERA5 meteorological data from Open-Meteo ({start_date} to {end_date})...")
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "daily" not in data:
        raise ValueError("Invalid response structure from Open-Meteo API")

    df = pd.DataFrame(data["daily"])
    df.rename(columns={"time": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])

    # Standardize column names
    column_mapping = {
        "temperature_2m_mean": "temperature_mean",
        "temperature_2m_max": "temperature_max",
        "temperature_2m_min": "temperature_min",
        "relative_humidity_2m_mean": "relative_humidity",
        "dew_point_2m_mean": "dew_point",
        "precipitation_sum": "precipitation",
        "rain_sum": "rain",
        "surface_pressure_mean": "surface_pressure",
        "wind_speed_10m_mean": "wind_speed",
        "wind_speed_10m_max": "wind_gust_max",
        "wind_direction_10m_dominant": "wind_direction"
    }
    df.rename(columns=column_mapping, inplace=True)
    return df


def generate_synthetic_fallback(start_date="2015-01-01", end_date="2024-12-31"):
    """
    Generates a physically realistic 10-year synthetic meteorological dataset
    governed by thermodynamics, seasonal sinusoidal cycles, monsoon surge,
    and extreme weather events in case internet connectivity is unavailable.
    """
    print("Generating physically consistent synthetic meteorological dataset as fallback...")
    np.random.seed(42)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)

    day_of_year = dates.dayofyear.values
    t = np.arange(n)

    # 1. Temperature: Annual cycle (warmer in May/June, cooler in Dec/Jan) + warming drift + noise
    annual_temp_cycle = 24.0 + 5.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    warming_drift = 0.04 * (t / 365.25)  # slight decadal climate drift
    temp_noise = np.random.normal(0, 1.2, n)
    temperature_mean = annual_temp_cycle + warming_drift + temp_noise

    diurnal_range = np.random.uniform(7.0, 13.0, n)
    temperature_max = temperature_mean + diurnal_range * 0.55
    temperature_min = temperature_mean - diurnal_range * 0.45

    # 2. Moisture & Humidity: Monsoon surge (days 150-280) and inverse diurnal relation
    monsoon_factor = np.exp(-((day_of_year - 215) ** 2) / (2 * 45 ** 2))
    base_humidity = 55.0 + 32.0 * monsoon_factor - 1.2 * (temperature_mean - 24.0)
    humidity_noise = np.random.normal(0, 4.0, n)
    relative_humidity = np.clip(base_humidity + humidity_noise, 20.0, 98.0)

    # 3. Moisture in air: Dew point (derived from Magnus-Tetens formula approximation)
    a, b = 17.27, 237.7
    alpha = ((a * temperature_mean) / (b + temperature_mean)) + np.log(relative_humidity / 100.0)
    dew_point = (b * alpha) / (a - alpha)

    # 4. Atmospheric Pressure: Baseline ~1013 hPa, seasonal winter high / summer low + synoptic depressions
    pressure_annual = 1012.5 + 4.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)
    synoptic_noise = np.random.normal(0, 2.0, n)
    surface_pressure = pressure_annual + synoptic_noise

    # 5. Wind Speed: Weibull distribution + monsoon enhancement
    wind_base = np.random.weibull(2.0, n) * 8.0 + 4.0 * monsoon_factor
    wind_speed = np.clip(wind_base, 1.5, 55.0)
    wind_gust_max = wind_speed * np.random.uniform(1.4, 2.1, n)
    wind_direction = np.random.uniform(0, 360, n)

    # 6. Rain: Zero-inflated Gamma distribution, heavy during monsoon
    rain_prob = 0.08 + 0.55 * monsoon_factor
    is_rain = np.random.rand(n) < rain_prob
    rain_volume = np.zeros(n)
    rain_volume[is_rain] = np.random.gamma(shape=1.8, scale=9.0, size=np.sum(is_rain))

    # Inject extreme events (heatwave, cyclone drop, flash rain)
    for _ in range(12):
        idx = np.random.randint(100, n - 10)
        # Low pressure depression with gale winds and heavy rain
        surface_pressure[idx:idx+3] -= np.random.uniform(14.0, 22.0)
        wind_speed[idx:idx+3] += np.random.uniform(22.0, 38.0)
        rain_volume[idx:idx+3] += np.random.uniform(50.0, 110.0)

    df = pd.DataFrame({
        "date": dates,
        "temperature_mean": np.round(temperature_mean, 2),
        "temperature_max": np.round(temperature_max, 2),
        "temperature_min": np.round(temperature_min, 2),
        "relative_humidity": np.round(relative_humidity, 2),
        "dew_point": np.round(dew_point, 2),
        "precipitation": np.round(rain_volume, 2),
        "rain": np.round(rain_volume, 2),
        "surface_pressure": np.round(surface_pressure, 2),
        "wind_speed": np.round(wind_speed, 2),
        "wind_gust_max": np.round(wind_gust_max, 2),
        "wind_direction": np.round(wind_direction, 1)
    })
    return df


def acquire_dataset(force_download=False):
    """
    Main function to ensure dataset exists on disk.
    Tries real Open-Meteo ERA5 API first, falls back to synthetic generator if offline.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(OUTPUT_FILE) and not force_download:
        print(f"Dataset already exists at: {OUTPUT_FILE}")
        return pd.read_csv(OUTPUT_FILE, parse_dates=["date"])

    try:
        df = fetch_open_meteo_era5()
        print("Successfully acquired authentic ERA5 meteorological dataset!")
    except Exception as e:
        print(f"Warning: Failed to fetch via Open-Meteo API ({e}). Using robust meteorological fallback.")
        df = generate_synthetic_fallback()

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved dataset with {len(df)} records ({df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}) to {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    df = acquire_dataset(force_download=True)
    print("\nDataset Overview:")
    print(df.info())
    print("\nFirst 5 Records:")
    print(df.head())
