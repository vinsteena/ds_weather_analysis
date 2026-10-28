"""
Time-Series Feature Engineering Module for Weather Prediction (CO4).
Builds lag features, rolling aggregations, cyclical temporal encodings,
and prevents lookahead leakage for next-day multi-target forecasting.
"""

import numpy as np
import pandas as pd


CORE_METRIC_COLS = [
    "temperature_mean",
    "relative_humidity",
    "dew_point",
    "vapor_pressure",
    "surface_pressure",
    "wind_speed",
    "rain"
]


def create_time_series_features(df: pd.DataFrame, lags=(1, 2, 3, 7, 14), windows=(3, 7, 14, 30)) -> pd.DataFrame:
    """
    Transforms historical preprocessed weather DataFrame into an ML feature matrix.
    All features for predicting target at time t are strictly computed using data
    at or before time t-1, ensuring zero data leakage.
    """
    data = df.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
        data.sort_values("date", inplace=True)
        data.set_index("date", inplace=True)

    feature_dfs = [data]

    # 1. Cyclical time encodings
    doy = data.index.dayofyear.values
    month = data.index.month.values
    cyclical_df = pd.DataFrame(index=data.index)
    cyclical_df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    cyclical_df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    cyclical_df["sin_month"] = np.sin(2 * np.pi * month / 12.0)
    cyclical_df["cos_month"] = np.cos(2 * np.pi * month / 12.0)
    feature_dfs.append(cyclical_df)

    # 2. Lag features (t-1, t-2, ...)
    lag_dict = {}
    for col in CORE_METRIC_COLS:
        if col in data.columns:
            for lag in lags:
                lag_dict[f"{col}_lag_{lag}"] = data[col].shift(lag)
    feature_dfs.append(pd.DataFrame(lag_dict, index=data.index))

    # 3. Rolling window features (computed on shift(1) to avoid including current day t)
    rolling_dict = {}
    for col in CORE_METRIC_COLS:
        if col in data.columns:
            shifted = data[col].shift(1)
            for w in windows:
                rolling_dict[f"{col}_roll_mean_{w}d"] = shifted.rolling(window=w, min_periods=w).mean()
                rolling_dict[f"{col}_roll_std_{w}d"] = shifted.rolling(window=w, min_periods=w).std()
                if col in ["rain", "wind_speed"]:
                    rolling_dict[f"{col}_roll_max_{w}d"] = shifted.rolling(window=w, min_periods=w).max()
    feature_dfs.append(pd.DataFrame(rolling_dict, index=data.index))

    # 4. Atmospheric dynamics and physical interactions at t-1
    dynamics_dict = {}
    if "surface_pressure" in data.columns:
        # Pressure tendency: 24-hour barometric drop or surge
        dynamics_dict["pressure_tendency_1d"] = data["surface_pressure"].shift(1) - data["surface_pressure"].shift(2)
        dynamics_dict["pressure_tendency_3d"] = data["surface_pressure"].shift(1) - data["surface_pressure"].shift(4)

    if "temperature_mean" in data.columns and "dew_point" in data.columns:
        # Dew point depression at t-1 (air saturation deficit)
        dynamics_dict["dew_point_depression_lag_1"] = data["temperature_mean"].shift(1) - data["dew_point"].shift(1)

    if "temperature_max" in data.columns and "temperature_min" in data.columns:
        dynamics_dict["diurnal_range_lag_1"] = data["temperature_max"].shift(1) - data["temperature_min"].shift(1)

    feature_dfs.append(pd.DataFrame(dynamics_dict, index=data.index))

    full_features_df = pd.concat(feature_dfs, axis=1)

    # Drop early warmup rows containing NaN from max lag / max rolling window
    max_warmup = max(max(lags), max(windows))
    clean_features_df = full_features_df.iloc[max_warmup:].copy()

    clean_features_df.reset_index(inplace=True)
    return clean_features_df


def split_train_val_test(df: pd.DataFrame, train_end_year=2022, val_year=2023):
    """
    Splits feature matrix chronologically:
    - Train: start through train_end_year (e.g., 2015-2022)
    - Validation: val_year (e.g., 2023)
    - Test: remaining years (e.g., 2024)
    """
    df_sorted = df.sort_values("date").copy()
    years = df_sorted["date"].dt.year

    train_mask = years <= train_end_year
    val_mask = years == val_year
    test_mask = years > val_year

    train_df = df_sorted[train_mask].copy()
    val_df = df_sorted[val_mask].copy()
    test_df = df_sorted[test_mask].copy()

    return train_df, val_df, test_df


if __name__ == "__main__":
    import os
    from weather_trend_analysis.analytics.preprocessor import preprocess_weather_data

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "weather_analytics_era5_2015_2024.csv")
    raw_df = pd.read_csv(data_path)
    clean_df = preprocess_weather_data(raw_df)
    features_df = create_time_series_features(clean_df)

    print("Engineered Features Matrix Shape:", features_df.shape)
    train_df, val_df, test_df = split_train_val_test(features_df)
    print(f"Train samples: {len(train_df)} ({train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')})")
    print(f"Val samples:   {len(val_df)} ({val_df['date'].min().strftime('%Y-%m-%d')} to {val_df['date'].max().strftime('%Y-%m-%d')})")
    print(f"Test samples:  {len(test_df)} ({test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')})")
