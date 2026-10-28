"""
Descriptive Statistical Analysis and Correlation Module (CO3).
Calculates parametric and non-parametric statistical metrics:
- Mean, Median, Min, Max, Standard Deviation, Variance
- Skewness, Kurtosis, Interquartile Range (IQR)
- Percentiles (5th, 25th, 50th, 75th, 95th, 99th)
- Pearson & Spearman correlation matrices
- Seasonal and monthly distribution breakdowns
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_descriptive_statistics(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Computes a comprehensive statistical profile for specified meteorological metrics.
    """
    if columns is None:
        columns = [
            "temperature_mean",
            "temperature_max",
            "temperature_min",
            "relative_humidity",
            "dew_point",
            "vapor_pressure",
            "surface_pressure",
            "wind_speed",
            "rain"
        ]

    # Filter to existing columns in dataframe
    valid_cols = [c for c in columns if c in df.columns]

    stat_rows = []
    for col in valid_cols:
        series = df[col].dropna()
        n = len(series)
        mean_val = series.mean()
        std_val = series.std(ddof=1)
        var_val = series.var(ddof=1)
        median_val = series.median()
        min_val = series.min()
        max_val = series.max()
        range_val = max_val - min_val

        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr_val = q75 - q25

        p5 = series.quantile(0.05)
        p95 = series.quantile(0.95)
        p99 = series.quantile(0.99)

        skew_val = float(stats.skew(series, bias=False)) if n > 2 else 0.0
        kurt_val = float(stats.kurtosis(series, bias=False)) if n > 3 else 0.0

        stat_rows.append({
            "parameter": col,
            "count": int(n),
            "mean": round(mean_val, 2),
            "std_dev": round(std_val, 2),
            "variance": round(var_val, 2),
            "median": round(median_val, 2),
            "min": round(min_val, 2),
            "max": round(max_val, 2),
            "range": round(range_val, 2),
            "iqr": round(iqr_val, 2),
            "p5": round(p5, 2),
            "q25": round(q25, 2),
            "q75": round(q75, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "skewness": round(skew_val, 2),
            "kurtosis": round(kurt_val, 2)
        })

    return pd.DataFrame(stat_rows).set_index("parameter")


def compute_correlation_matrices(df: pd.DataFrame, columns=None):
    """
    Computes both Pearson linear correlation and Spearman non-parametric rank correlation.
    """
    if columns is None:
        columns = [
            "temperature_mean",
            "relative_humidity",
            "dew_point",
            "vapor_pressure",
            "surface_pressure",
            "wind_speed",
            "rain"
        ]
    valid_cols = [c for c in columns if c in df.columns]
    sub_df = df[valid_cols].dropna()

    pearson_corr = sub_df.corr(method="pearson").round(3)
    spearman_corr = sub_df.corr(method="spearman").round(3)

    return pearson_corr, spearman_corr


def compute_seasonal_breakdown(df: pd.DataFrame, target_col="temperature_mean") -> pd.DataFrame:
    """
    Computes statistical distribution per meteorological season.
    """
    if "season" not in df.columns or target_col not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby("season")[target_col].agg(
        mean="mean",
        std="std",
        median="median",
        min="min",
        max="max",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    ).round(2)
    return grouped


def compute_monthly_breakdown(df: pd.DataFrame, target_col="temperature_mean") -> pd.DataFrame:
    """
    Computes mean and std across the 12 calendar months.
    """
    if "month" not in df.columns or target_col not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby("month")[target_col].agg(
        mean="mean",
        std="std",
        median="median",
        min="min",
        max="max"
    ).round(2)
    return grouped


if __name__ == "__main__":
    import os
    from weather_trend_analysis.analytics.preprocessor import preprocess_weather_data

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "weather_analytics_era5_2015_2024.csv")
    raw_df = pd.read_csv(data_path)
    clean_df = preprocess_weather_data(raw_df)

    stats_table = compute_descriptive_statistics(clean_df)
    print("=== Descriptive Statistics Summary ===")
    print(stats_table[["mean", "std_dev", "median", "min", "max", "iqr", "skewness"]])

    p_corr, _ = compute_correlation_matrices(clean_df)
    print("\n=== Pearson Correlation Matrix ===")
    print(p_corr)
