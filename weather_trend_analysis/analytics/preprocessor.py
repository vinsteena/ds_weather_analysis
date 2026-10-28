"""
Meteorological Preprocessing and Physical Feature Derivation Module (CO3).
Cleans data, validates physical ranges, and computes meteorological physics variables:
- Vapor Pressure (direct measurement of moisture in the air)
- Saturation Vapor Pressure
- Dew Point Depression (saturation deficit)
- Diurnal Temperature Range
- Barometric Pressure Tendency (24h delta)
- Heat Index
- Rain Occurrence Flag
"""

import numpy as np
import pandas as pd


def compute_vapor_pressure(dew_point_c):
    """
    Computes actual vapor pressure (e in hPa) using the August-Roche-Magnus formula.
    Represents the exact physical moisture content present in the air.
    """
    return 6.112 * np.exp((17.67 * dew_point_c) / (dew_point_c + 243.5))


def compute_saturation_vapor_pressure(temperature_c):
    """
    Computes saturation vapor pressure (e_s in hPa) using Magnus formula.
    Represents maximum moisture capacity of air at given temperature.
    """
    return 6.112 * np.exp((17.67 * temperature_c) / (temperature_c + 243.5))


def compute_heat_index(temp_c, relative_humidity):
    """
    Computes NOAA Heat Index / Apparent Temperature in Celsius.
    Uses the Rothfusz regression equation (vectorized for scalars, arrays, and Series).
    """
    t_f = np.asarray(temp_c, dtype=float) * 9.0 / 5.0 + 32.0
    rh = np.asarray(relative_humidity, dtype=float)

    simple_hi_f = 0.5 * (t_f + 61.0 + ((t_f - 68.0) * 1.2) + (rh * 0.094))
    full_hi_f = (-42.379 + 2.04901523 * t_f + 10.14333127 * rh
                 - 0.22475541 * t_f * rh - 0.00683783 * (t_f ** 2)
                 - 0.05481717 * (rh ** 2) + 0.00122874 * (t_f ** 2) * rh
                 + 0.00085282 * t_f * (rh ** 2) - 0.00000199 * (t_f ** 2) * (rh ** 2))

    hi_f = np.where(simple_hi_f >= 80.0, full_hi_f, simple_hi_f)
    hi_c = (hi_f - 32.0) * 5.0 / 9.0

    if isinstance(temp_c, pd.Series):
        return pd.Series(np.round(hi_c, 2), index=temp_c.index)
    elif np.ndim(temp_c) == 0:
        return float(np.round(hi_c, 2))
    return np.round(hi_c, 2)


def preprocess_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates, cleans, and enriches historical weather data.
    """
    data = df.copy()

    # 1. Date processing and continuous indexing
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
        data.sort_values("date", inplace=True)
        data.set_index("date", inplace=True)

    # 2. Enforce physical constraints / bounds
    if "relative_humidity" in data.columns:
        data["relative_humidity"] = data["relative_humidity"].clip(lower=0.0, upper=100.0)
    if "surface_pressure" in data.columns:
        data["surface_pressure"] = data["surface_pressure"].clip(lower=850.0, upper=1085.0)
    if "wind_speed" in data.columns:
        data["wind_speed"] = data["wind_speed"].clip(lower=0.0, upper=150.0)
    if "precipitation" in data.columns:
        data["precipitation"] = data["precipitation"].clip(lower=0.0, upper=500.0)
    if "rain" in data.columns:
        data["rain"] = data["rain"].clip(lower=0.0, upper=500.0)

    # 3. Handle any potential missing values via time interpolation
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].interpolate(method="time").bfill().ffill()

    # 4. Compute physical moisture variables
    if "dew_point" in data.columns:
        data["vapor_pressure"] = np.round(compute_vapor_pressure(data["dew_point"]), 2)
    if "temperature_mean" in data.columns:
        data["saturation_vapor_pressure"] = np.round(compute_saturation_vapor_pressure(data["temperature_mean"]), 2)
        if "dew_point" in data.columns:
            data["dew_point_depression"] = np.round(data["temperature_mean"] - data["dew_point"], 2)

    # 5. Diurnal Temperature Range
    if "temperature_max" in data.columns and "temperature_min" in data.columns:
        data["diurnal_temp_range"] = np.round(data["temperature_max"] - data["temperature_min"], 2)

    # 6. Barometric Pressure Tendency (24h change)
    if "surface_pressure" in data.columns:
        data["pressure_tendency_24h"] = np.round(data["surface_pressure"].diff(1), 2)
        data["pressure_tendency_24h"] = data["pressure_tendency_24h"].fillna(0.0)

    # 7. Heat Index
    if "temperature_mean" in data.columns and "relative_humidity" in data.columns:
        data["heat_index"] = compute_heat_index(data["temperature_mean"], data["relative_humidity"])

    # 8. Rain occurrence indicator (binary for classification)
    rain_col = "rain" if "rain" in data.columns else "precipitation"
    if rain_col in data.columns:
        data["rain_occurrence"] = (data[rain_col] >= 0.1).astype(int)

    # 9. Calendar / temporal features
    data["year"] = data.index.year
    data["month"] = data.index.month
    data["day_of_year"] = data.index.dayofyear
    data["day_of_week"] = data.index.dayofweek

    # Assign meteorological seasons (Northern Hemisphere standard)
    # Winter: Dec, Jan, Feb; Spring: Mar, Apr, May; Summer/Monsoon: Jun, Jul, Aug; Autumn: Sep, Oct, Nov
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Pre-Monsoon / Summer", 4: "Pre-Monsoon / Summer", 5: "Pre-Monsoon / Summer",
        6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
        10: "Post-Monsoon", 11: "Post-Monsoon"
    }
    data["season"] = data["month"].map(season_map)

    data.reset_index(inplace=True)
    return data


if __name__ == "__main__":
    import os
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "weather_analytics_era5_2015_2024.csv")
    raw_df = pd.read_csv(data_path)
    clean_df = preprocess_weather_data(raw_df)
    print("Preprocessed Data Shape:", clean_df.shape)
    print("Columns:", clean_df.columns.tolist())
    print("\nSample processed records:")
    print(clean_df[["date", "temperature_mean", "relative_humidity", "dew_point", "vapor_pressure", "dew_point_depression", "pressure_tendency_24h", "rain_occurrence"]].head())
