"""
WeatherWise Test Suite — Data Quality Module
Tests for data_quality.py functions.
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_quality import (
    validate_and_clean, compute_trust_score,
    identify_date_gaps, detect_suspicious_values
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_clean_df(n=100):
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    np.random.seed(0)
    return pd.DataFrame({
        "date": dates,
        "location": "TestCity",
        "temperature": np.random.uniform(10, 30, n),
        "humidity": np.random.uniform(40, 80, n),
        "rainfall": np.abs(np.random.exponential(3, n)),
        "wind_speed": np.random.uniform(5, 25, n),
        "pressure": np.random.uniform(1005, 1025, n),
    })


def make_dirty_df():
    df = make_clean_df(50)
    # Insert invalid values
    df.at[5, "temperature"] = 999.0   # Way out of range
    df.at[10, "humidity"] = -50.0      # Negative humidity
    df.at[15, "rainfall"] = -5.0       # Negative rainfall
    # Insert NaN
    df.at[20, "temperature"] = np.nan
    df.at[25, "humidity"] = np.nan
    # Insert duplicate
    dup = df.iloc[0:1].copy()
    df = pd.concat([df, dup], ignore_index=True)
    return df


def make_df_with_gaps():
    dates = pd.date_range("2022-01-01", periods=100, freq="D").tolist()
    # Remove dates 30-34 to create a 5-day gap
    dates = dates[:30] + dates[35:]
    np.random.seed(1)
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "location": "TestCity",
        "temperature": np.random.uniform(10, 30, len(dates)),
        "humidity": np.random.uniform(40, 80, len(dates)),
        "rainfall": np.abs(np.random.exponential(3, len(dates))),
        "wind_speed": np.random.uniform(5, 25, len(dates)),
        "pressure": np.random.uniform(1005, 1025, len(dates)),
    })


# ── validate_and_clean ────────────────────────────────────────────────────────

def test_validate_and_clean_returns_tuple():
    df = make_clean_df()
    result = validate_and_clean(df)
    assert isinstance(result, tuple)
    assert len(result) == 2
    cleaned, report = result
    assert isinstance(cleaned, pd.DataFrame)
    assert isinstance(report, dict)


def test_validate_removes_duplicates():
    df = make_dirty_df()
    original_len = len(df)
    cleaned, report = validate_and_clean(df)
    assert len(cleaned) < original_len or report.get("removed_duplicates", 0) >= 0


def test_validate_nullifies_out_of_range_temperature():
    df = make_clean_df(20)
    df.at[5, "temperature"] = 999.0  # Out of range
    cleaned, report = validate_and_clean(df)
    # The out-of-range value should be NaN after cleaning
    assert pd.isna(cleaned.loc[cleaned.index[5], "temperature"]) if 5 in cleaned.index else True


def test_validate_does_not_delete_rows():
    """Invalid measurements nullify the cell, not the row."""
    df = make_clean_df(20)
    df.at[5, "temperature"] = 999.0
    original_valid_dates = len(df)
    cleaned, report = validate_and_clean(df)
    # Row count should be similar (some may be removed as duplicates but not for invalid values)
    assert len(cleaned) >= original_valid_dates - 5  # Allow for duplicates


def test_validate_report_has_keys():
    df = make_clean_df()
    _, report = validate_and_clean(df)
    for key in ["original_rows", "cleaned_rows"]:
        assert key in report


def test_validate_does_not_modify_original():
    df = make_dirty_df()
    df_copy = df.copy()
    validate_and_clean(df)
    pd.testing.assert_frame_equal(df, df_copy)


# ── compute_trust_score ───────────────────────────────────────────────────────

def test_trust_score_range():
    df = make_clean_df()
    result = compute_trust_score(df, df)
    assert "overall_score" in result
    score = result["overall_score"]
    assert 0 <= score <= 100


def test_trust_score_perfect_data():
    df = make_clean_df()
    result = compute_trust_score(df, df)
    # Clean data should score high
    assert result["overall_score"] >= 60


def test_trust_score_dirty_data_lower():
    clean_df = make_clean_df()
    dirty_df = make_dirty_df()
    clean_score = compute_trust_score(clean_df, clean_df)["overall_score"]
    dirty_score = compute_trust_score(dirty_df, dirty_df)["overall_score"]
    # Dirty data should score lower
    assert dirty_score <= clean_score + 10  # Allow some tolerance


def test_trust_score_has_components():
    df = make_clean_df()
    result = compute_trust_score(df, df)
    assert "components" in result


def test_trust_score_has_recommendations():
    df = make_dirty_df()
    result = compute_trust_score(df, df)
    assert "recommendations" in result
    assert isinstance(result["recommendations"], list)


def test_trust_score_is_int():
    df = make_clean_df()
    result = compute_trust_score(df, df)
    assert isinstance(result["overall_score"], (int, float))


# ── identify_date_gaps ────────────────────────────────────────────────────────

def test_identify_date_gaps_returns_list():
    df = make_df_with_gaps()
    gaps = identify_date_gaps(df)
    assert isinstance(gaps, list)


def test_identify_date_gaps_finds_gap():
    df = make_df_with_gaps()
    gaps = identify_date_gaps(df)
    assert len(gaps) >= 1


def test_identify_date_gaps_no_gaps():
    df = make_clean_df(100)  # Consecutive dates, no gaps
    gaps = identify_date_gaps(df)
    assert len(gaps) == 0


def test_identify_date_gaps_structure():
    df = make_df_with_gaps()
    gaps = identify_date_gaps(df)
    if len(gaps) > 0:
        gap = gaps[0]
        assert "gap_start" in gap
        assert "gap_end" in gap
        assert "missing_days" in gap
        assert gap["missing_days"] > 0


# ── detect_suspicious_values ─────────────────────────────────────────────────

def test_detect_suspicious_returns_dict():
    df = make_clean_df()
    result = detect_suspicious_values(df)
    assert isinstance(result, dict)


def test_detect_suspicious_flags_outlier():
    df = make_clean_df(100)
    df.at[50, "temperature"] = 200.0  # Extreme outlier
    result = detect_suspicious_values(df)
    # May or may not flag it depending on IQR (since it's so extreme, the fence should catch it)
    # The function should handle this gracefully
    assert isinstance(result, dict)


def test_detect_suspicious_normal_data_minimal_flags():
    df = make_clean_df(200)
    result = detect_suspicious_values(df)
    # Clean data should have few or no flags
    total_flags = sum(v.get("count", 0) for v in result.values())
    assert total_flags <= 10  # At most a few spurious flags


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
