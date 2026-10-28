"""
WeatherWise Sample Data Generator
==================================
Generates a 2-year synthetic daily weather dataset with:
- Realistic seasonal patterns
- ~3% missing values
- ~6 deliberate statistical outliers
- Clear SYNTHETIC label

Output: data/sample_weather.csv
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import random

random.seed(42)
np.random.seed(42)

# ── Date range ──────────────────────────────────────────────────────────────
START = date(2022, 1, 1)
END = date(2023, 12, 31)
LOCATION = "Cityville, Demo Region"

dates = []
d = START
while d <= END:
    dates.append(d)
    d += timedelta(days=1)

n = len(dates)
day_of_year = np.array([(d - date(d.year, 1, 1)).days for d in dates])

# ── Temperature (°C): seasonal sine wave + noise ─────────────────────────────
#   - Northern hemisphere style: hot in summer (days 170-220), cold in winter
TEMP_MEAN = 20.0
TEMP_AMP = 12.0
temp_signal = TEMP_MEAN - TEMP_AMP * np.cos(2 * np.pi * day_of_year / 365)
temperature = temp_signal + np.random.normal(0, 2.5, n)

# ── Humidity (%): inversely related to temperature + some noise ───────────────
humidity_base = 75 - 0.8 * (temperature - TEMP_MEAN)
humidity = np.clip(humidity_base + np.random.normal(0, 7, n), 20, 100)

# ── Rainfall (mm): concentrated in winter/spring months ──────────────────────
# Probability of rain is higher when temp is lower
rain_prob = 0.1 + 0.25 * (1 - (temperature - temperature.min()) / (temperature.max() - temperature.min()))
rainfall = np.where(
    np.random.rand(n) < rain_prob,
    np.abs(np.random.exponential(8, n)),
    0.0
)
rainfall = np.round(rainfall, 1)

# ── Wind speed (km/h): higher variability in colder months ───────────────────
wind_base = 14.0 + 5 * np.cos(2 * np.pi * day_of_year / 365 + np.pi)  # peaks in winter
wind_speed = np.abs(wind_base + np.random.normal(0, 4, n))
wind_speed = np.round(wind_speed, 1)

# ── Pressure (hPa): inversely related to rainfall, slight seasonal variation ─
pressure_base = 1015 + 5 * np.cos(2 * np.pi * day_of_year / 365)
pressure = pressure_base - 0.3 * rainfall + np.random.normal(0, 2, n)
pressure = np.round(np.clip(pressure, 980, 1040), 1)

temperature = np.round(temperature, 1)
humidity = np.round(humidity, 1)

# ── Build DataFrame ──────────────────────────────────────────────────────────
df = pd.DataFrame({
    "date": [str(d) for d in dates],
    "location": LOCATION,
    "temperature": temperature,
    "humidity": humidity,
    "rainfall": rainfall,
    "wind_speed": wind_speed,
    "pressure": pressure,
})

# ── Insert deliberate outliers (labeled for demonstration) ───────────────────
outlier_indices = [45, 190, 310, 490, 600, 680]
outlier_values = [
    ("temperature", 52.8),   # extreme heat
    ("temperature", -14.5),  # extreme cold
    ("rainfall", 285.0),     # extreme rainfall
    ("temperature", 49.1),   # heat spike
    ("wind_speed", 142.0),   # extreme wind
    ("humidity", 99.8),      # humidity spike with rain
]

for idx, (col, val) in zip(outlier_indices, outlier_values):
    df.at[idx, col] = val

# ── Insert missing values (~3% across all measurement columns) ───────────────
measurement_cols = ["temperature", "humidity", "rainfall", "wind_speed", "pressure"]
target_missing = int(n * len(measurement_cols) * 0.03)
missing_cells = [(random.randint(0, n-1), random.choice(measurement_cols))
                 for _ in range(target_missing)]
# Avoid marking outlier rows as missing
outlier_set = set(outlier_indices)
for row, col in missing_cells:
    if row not in outlier_set:
        df.at[row, col] = np.nan

# ── Insert a few missing date records (simulate date gaps) ───────────────────
# Remove rows at indices 120, 121 to create a 2-day gap
df = df.drop(index=[120, 121]).reset_index(drop=True)

# ── Save ─────────────────────────────────────────────────────────────────────
output_path = "data/sample_weather.csv"
df.to_csv(output_path, index=False)

print("Sample dataset generated: " + output_path)
print("  Rows: " + str(len(df)))
print("  Date range: " + str(df['date'].iloc[0]) + " to " + str(df['date'].iloc[-1]))
print("  Outliers inserted at rows: " + str(outlier_indices))
missing_count = int(df[measurement_cols].isna().sum().sum())
total_cells = len(df) * len(measurement_cols)
print("  Missing values: " + str(missing_count) + " (" + str(round(100*missing_count/total_cells, 1)) + "%)")
print("  NOTE: This is SYNTHETIC data for demonstration purposes only.")
