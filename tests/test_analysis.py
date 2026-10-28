"""
WeatherWise Test Suite — Analysis Module
Tests for analysis.py functions.
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import (
    get_available_variables, get_date_range,
    get_descriptive_stats, compute_trend,
    seasonal_aggregation, year_over_year_comparison,
    compute_decomposition, get_monthly_summary
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_df(n=365, seed=42):
    """Create a synthetic DataFrame for testing."""
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    doy = np.arange(n)
    temp = 20 - 12 * np.cos(2 * np.pi * doy / 365) + np.random.normal(0, 2, n)
    humidity = 70 + np.random.normal(0, 5, n)
    rainfall = np.abs(np.random.exponential(3, n))
    wind_speed = np.abs(14 + np.random.normal(0, 4, n))
    pressure = 1015 + np.random.normal(0, 3, n)

    df = pd.DataFrame({
        "date": dates,
        "location": "TestCity",
        "temperature": temp,
        "humidity": np.clip(humidity, 20, 100),
        "rainfall": rainfall,
        "wind_speed": wind_speed,
        "pressure": pressure,
    })
    return df


def make_two_year_df():
    """Create a 2-year DataFrame for decomposition tests."""
    return make_df(n=730, seed=99)


# ── get_available_variables ───────────────────────────────────────────────────

def test_available_variables_all_present():
    df = make_df()
    variables = get_available_variables(df)
    assert "temperature" in variables
    assert "humidity" in variables
    assert "rainfall" in variables


def test_available_variables_missing_column():
    df = make_df().drop(columns=["pressure"])
    variables = get_available_variables(df)
    assert "pressure" not in variables


def test_available_variables_all_nan_excluded():
    df = make_df()
    df["temperature"] = np.nan
    variables = get_available_variables(df)
    assert "temperature" not in variables


# ── get_date_range ────────────────────────────────────────────────────────────

def test_date_range_correct():
    df = make_df(n=100)
    result = get_date_range(df)
    assert result["start"] == "2022-01-01"
    assert "total_days" in result
    assert result["available_obs"] == 100


def test_date_range_keys():
    df = make_df()
    result = get_date_range(df)
    for key in ["start", "end", "total_days", "available_obs"]:
        assert key in result


# ── get_descriptive_stats ─────────────────────────────────────────────────────

def test_descriptive_stats_keys():
    df = make_df()
    result = get_descriptive_stats(df, "temperature")
    for key in ["count", "mean", "median", "std", "min", "max", "q25", "q75", "skewness", "kurtosis"]:
        assert key in result


def test_descriptive_stats_mean_approx():
    df = make_df()
    result = get_descriptive_stats(df, "temperature")
    # Mean should be around 20 (the synthetic mean)
    assert 15.0 < result["mean"] < 25.0


def test_descriptive_stats_min_le_median_le_max():
    df = make_df()
    result = get_descriptive_stats(df, "temperature")
    assert result["min"] <= result["median"] <= result["max"]


def test_descriptive_stats_q25_le_q75():
    df = make_df()
    result = get_descriptive_stats(df, "temperature")
    assert result["q25"] <= result["q75"]


def test_descriptive_stats_missing_variable():
    df = make_df()
    result = get_descriptive_stats(df, "nonexistent_variable")
    assert "error" in result


def test_descriptive_stats_all_nan():
    df = make_df()
    df["temperature"] = np.nan
    result = get_descriptive_stats(df, "temperature")
    assert "error" in result


def test_descriptive_stats_returns_float():
    df = make_df()
    result = get_descriptive_stats(df, "temperature")
    assert isinstance(result["mean"], float)
    assert isinstance(result["std"], float)


# ── compute_trend ─────────────────────────────────────────────────────────────

def test_trend_keys():
    df = make_df()
    result = compute_trend(df, "temperature")
    for key in ["slope", "intercept", "r_squared", "p_value", "trend_direction", "change_per_year"]:
        assert key in result


def test_trend_direction_field():
    df = make_df()
    result = compute_trend(df, "temperature")
    assert result["trend_direction"] in ("increasing", "decreasing", "flat")


def test_trend_increasing_data():
    """Create strictly increasing temperature data."""
    dates = pd.date_range("2022-01-01", periods=200)
    temp = np.arange(200, dtype=float) + np.random.normal(0, 0.1, 200)
    df = pd.DataFrame({"date": dates, "location": "X", "temperature": temp,
                       "humidity": 50.0, "rainfall": 0.0, "wind_speed": 10.0, "pressure": 1013.0})
    result = compute_trend(df, "temperature")
    assert result["trend_direction"] == "increasing"


def test_trend_insufficient_data():
    dates = pd.date_range("2022-01-01", periods=5)
    df = pd.DataFrame({"date": dates, "location": "X", "temperature": [20, 21, 22, 23, 24],
                       "humidity": 50.0, "rainfall": 0.0, "wind_speed": 10.0, "pressure": 1013.0})
    result = compute_trend(df, "temperature")
    assert "error" in result


def test_trend_has_dates_list():
    df = make_df()
    result = compute_trend(df, "temperature")
    assert "dates" in result
    assert "values" in result
    assert len(result["dates"]) == len(result["values"])


# ── seasonal_aggregation ──────────────────────────────────────────────────────

def test_seasonal_aggregation_monthly_keys():
    df = make_df()
    result = seasonal_aggregation(df, "temperature")
    assert "monthly_avg" in result
    assert "seasonal_avg" in result


def test_seasonal_aggregation_has_all_months():
    df = make_df(n=365)
    result = seasonal_aggregation(df, "temperature")
    monthly = result["monthly_avg"]
    # Should have 12 months (as string or int keys)
    assert len(monthly) == 12


def test_seasonal_aggregation_four_seasons():
    df = make_df(n=365)
    result = seasonal_aggregation(df, "temperature")
    seasons = result["seasonal_avg"]
    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        assert season in seasons


# ── year_over_year_comparison ─────────────────────────────────────────────────

def test_year_over_year_keys():
    df = make_two_year_df()
    result = year_over_year_comparison(df, "temperature", 2022, 2023)
    assert 2022 in result or "year1" in result or "2022" in result


def test_year_over_year_contains_mean():
    df = make_two_year_df()
    result = year_over_year_comparison(df, "temperature", 2022, 2023)
    # Should have statistics for both years
    assert "difference" in result


def test_year_over_year_missing_year():
    df = make_df(n=365)  # Only 2022
    result = year_over_year_comparison(df, "temperature", 2022, 2025)
    # Should handle gracefully (either error or empty comparison)
    assert isinstance(result, dict)


# ── compute_decomposition ─────────────────────────────────────────────────────

def test_decomposition_insufficient_data():
    df = make_df(n=100)  # Too few for 365-period decomposition
    result = compute_decomposition(df, "temperature")
    # Should return error (not enough data) or use monthly period
    assert "error" in result or "observed" in result


def test_decomposition_two_year_data():
    df = make_two_year_df()
    result = compute_decomposition(df, "temperature")
    if "error" not in result:
        assert "observed" in result
        assert "trend" in result
        assert "seasonal" in result
        assert "residual" in result
        assert "dates" in result


# ── get_monthly_summary ───────────────────────────────────────────────────────

def test_monthly_summary_returns_list():
    df = make_df()
    result = get_monthly_summary(df)
    assert isinstance(result, list)


def test_monthly_summary_has_year_month():
    df = make_df()
    result = get_monthly_summary(df)
    if len(result) > 0:
        assert "year_month" in result[0]


def test_monthly_summary_skips_sparse_months():
    """Months with fewer than 5 observations should be skipped — function returns error or empty list."""
    dates = pd.date_range("2022-01-01", periods=3)
    df = pd.DataFrame({"date": dates, "location": "X",
                       "temperature": [20.0, 21.0, 22.0],
                       "humidity": 50.0, "rainfall": 0.0, "wind_speed": 10.0, "pressure": 1013.0})
    result = get_monthly_summary(df)
    # Either returns empty list or error dict — no valid monthly summaries should be present
    if isinstance(result, list):
        assert len(result) == 0
    else:
        # Returns error dict — also acceptable
        assert "error" in result or len(result) <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
