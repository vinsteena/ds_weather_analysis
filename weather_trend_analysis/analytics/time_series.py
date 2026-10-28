"""
Time-Series Analysis and Decomposition Module (CO4).
Performs:
- Additive and Multiplicative seasonal decomposition (Trend, Season, Residuals)
- Annual and monthly cycle profiling
- Autocorrelation Function (ACF) calculation
- Moving average smoothing and volatility bands
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf


def decompose_weather_series(df: pd.DataFrame, target_col="temperature_mean", model="additive", period=365):
    """
    Performs classical time-series decomposition into Trend, Seasonal, and Residual components.
    model: 'additive' (Y = T + S + R) or 'multiplicative' (Y = T * S * R)
    period: 365 for daily data (annual cycle) or 12 for monthly data
    """
    data = df.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
        data.sort_values("date", inplace=True)
        data.set_index("date", inplace=True)

    series = data[target_col].dropna()
    # Interpolate if needed to guarantee uninterrupted daily frequency
    series = series.asfreq("D").interpolate(method="time")

    if model == "multiplicative" and (series <= 0).any():
        # Multiplicative decomposition requires strictly positive values
        shift_val = abs(series.min()) + 1.0
        series = series + shift_val

    decomp = seasonal_decompose(series, model=model, period=period, extrapolate_trend="period")

    result_df = pd.DataFrame({
        "observed": decomp.observed,
        "trend": decomp.trend,
        "seasonal": decomp.seasonal,
        "residual": decomp.resid
    })
    return result_df


def compute_autocorrelation(df: pd.DataFrame, target_col="temperature_mean", nlags=30):
    """
    Computes Autocorrelation Function (ACF) for lags 0 through nlags.
    Reveals persistence and cyclical memory in the weather time-series.
    """
    data = df.copy()
    if "date" in data.columns:
        data.sort_values("date", inplace=True)
    series = data[target_col].dropna().values
    acf_vals = acf(series, nlags=nlags, fft=True)

    acf_df = pd.DataFrame({
        "lag": np.arange(nlags + 1),
        "autocorrelation": np.round(acf_vals, 4)
    })
    return acf_df


def aggregate_temporal_patterns(df: pd.DataFrame, target_cols=None):
    """
    Aggregates multi-year weather observations into:
    1. Daily climatological normal (mean & std across day of year 1..366)
    2. Monthly climatological normal (mean & std across months 1..12)
    3. Annual historical trends (mean, min, max, total rain per year)
    """
    if target_cols is None:
        target_cols = ["temperature_mean", "relative_humidity", "dew_point", "surface_pressure", "wind_speed", "rain"]

    data = df.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
        data["day_of_year"] = data["date"].dt.dayofyear
        data["month"] = data["date"].dt.month
        data["year"] = data["date"].dt.year

    valid_cols = [c for c in target_cols if c in data.columns]

    daily_normals = data.groupby("day_of_year")[valid_cols].agg(["mean", "std"]).round(2)
    monthly_normals = data.groupby("month")[valid_cols].agg(["mean", "std"]).round(2)

    annual_summary = data.groupby("year").agg({
        "temperature_mean": "mean",
        "temperature_max": "max",
        "temperature_min": "min",
        "relative_humidity": "mean",
        "dew_point": "mean",
        "surface_pressure": "mean",
        "wind_speed": "mean",
        "rain": "sum"
    }).round(2)
    annual_summary.rename(columns={"rain": "total_annual_rain_mm"}, inplace=True)

    return daily_normals, monthly_normals, annual_summary


if __name__ == "__main__":
    import os
    from weather_trend_analysis.analytics.preprocessor import preprocess_weather_data

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "weather_analytics_era5_2015_2024.csv")
    raw_df = pd.read_csv(data_path)
    clean_df = preprocess_weather_data(raw_df)

    decomp_df = decompose_weather_series(clean_df, target_col="temperature_mean", model="additive")
    print("Decomposition Head:")
    print(decomp_df.head())

    acf_df = compute_autocorrelation(clean_df, target_col="temperature_mean", nlags=7)
    print("\nTemperature ACF (Lags 0-7):")
    print(acf_df)
