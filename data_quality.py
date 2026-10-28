"""
WeatherWise Data Quality Module
===============================
Provides data validation, cleaning, quality reporting, trust score calculation,
temporal gap identification, and suspicious value detection.
"""

import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

VALID_RANGES: Dict[str, Tuple[float, float]] = {
    "temperature": (-60.0, 60.0),
    "humidity": (0.0, 100.0),
    "rainfall": (0.0, 1000.0),
    "wind_speed": (0.0, 200.0),
    "pressure": (870.0, 1085.0),
}

TRUST_WEIGHTS: Dict[str, float] = {
    "missing_values": 0.30,
    "duplicates": 0.20,
    "invalid_measurements": 0.25,
    "date_continuity": 0.25,
}


def validate_and_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate and clean weather dataset.
    - Parses date column to datetime (marking invalid as NaT)
    - Coerces measurement columns to numeric
    - Nullifies out-of-range values according to VALID_RANGES
    - Removes exact duplicate records (matching date and location)
    - Chronologically sorts dataset
    Returns (cleaned_df, report_dict).
    """
    if df is None or df.empty:
        empty_report: Dict[str, Any] = {
            "original_rows": 0,
            "cleaned_rows": 0,
            "removed_duplicates": 0,
            "invalid_dates": 0,
            "nullified_values": {col: 0 for col in VALID_RANGES},
            "operations_log": ["Empty or None dataframe received."],
        }
        return pd.DataFrame(), empty_report

    cleaned = df.copy(deep=True)
    operations_log: List[str] = []
    original_rows = int(len(cleaned))

    # 1. Parse date column
    invalid_dates_count = 0
    if "date" in cleaned.columns:
        parsed_dates = pd.to_datetime(cleaned["date"], errors="coerce")
        invalid_dates_count = int((cleaned["date"].notna() & parsed_dates.isna()).sum())
        cleaned["date"] = parsed_dates
        operations_log.append(f"Parsed 'date' column ({invalid_dates_count} unparseable values converted to NaT).")
    else:
        operations_log.append("Warning: 'date' column not found in dataframe.")

    # 2. Coerce numeric columns
    for col in VALID_RANGES:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    operations_log.append("Coerced measurement columns to numeric floating-point values.")

    # 3. Clip out-of-range values to NaN
    nullified_values: Dict[str, int] = {}
    total_nullified = 0
    for col, (min_v, max_v) in VALID_RANGES.items():
        if col in cleaned.columns:
            out_mask = (cleaned[col] < min_v) | (cleaned[col] > max_v)
            cnt = int(out_mask.sum())
            nullified_values[col] = cnt
            total_nullified += cnt
            if cnt > 0:
                cleaned.loc[out_mask, col] = np.nan
        else:
            nullified_values[col] = 0
    operations_log.append(
        f"Nullified {total_nullified} out-of-range measurements outside valid physical limits."
    )

    # 4. Remove exact duplicate rows (same date + same location)
    subset_cols = []
    if "date" in cleaned.columns:
        subset_cols.append("date")
    if "location" in cleaned.columns:
        subset_cols.append("location")

    before_dup_count = len(cleaned)
    if subset_cols:
        cleaned = cleaned.drop_duplicates(subset=subset_cols, keep="first")
    else:
        cleaned = cleaned.drop_duplicates(keep="first")
    removed_duplicates = int(before_dup_count - len(cleaned))
    operations_log.append(f"Removed {removed_duplicates} duplicate records based on {subset_cols or 'all columns'}.")

    # 5. Sort by date
    if "date" in cleaned.columns:
        cleaned = cleaned.sort_values("date").reset_index(drop=True)
        operations_log.append("Sorted dataset chronologically by date.")

    cleaned_rows = int(len(cleaned))

    report = {
        "original_rows": original_rows,
        "cleaned_rows": cleaned_rows,
        "removed_duplicates": removed_duplicates,
        "invalid_dates": invalid_dates_count,
        "nullified_values": nullified_values,
        "operations_log": operations_log,
    }

    return cleaned, report


def identify_date_gaps(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Sort df by date and find gaps > 1 day between consecutive dates.
    Returns list of {'gap_start': str, 'gap_end': str, 'missing_days': int}.
    """
    if df is None or df.empty or "date" not in df.columns:
        return []

    valid_dates = pd.to_datetime(df["date"], errors="coerce").dropna().drop_duplicates().sort_values()
    if len(valid_dates) < 2:
        return []

    gaps: List[Dict[str, Any]] = []
    prev_d = valid_dates.iloc[0]

    for curr_d in valid_dates.iloc[1:]:
        day_diff = int((curr_d.date() - prev_d.date()).days)
        if day_diff > 1:
            missing_days = day_diff - 1
            gap_start = (prev_d + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            gap_end = (curr_d - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            gaps.append({
                "gap_start": gap_start,
                "gap_end": gap_end,
                "missing_days": missing_days,
            })
        prev_d = curr_d

    return gaps


def compute_trust_score(original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute a composite data trust score (0 - 100) based on:
    - Missing value percentage (weight: 0.30)
    - Duplicate row percentage (weight: 0.20)
    - Invalid physical measurements percentage (weight: 0.25)
    - Temporal continuity / date gaps (weight: 0.25)
    """
    if original_df is None or original_df.empty:
        return {
            "overall_score": 0,
            "components": {
                "missing_values": 0.0,
                "duplicates": 0.0,
                "invalid_measurements": 0.0,
                "date_continuity": 0.0,
            },
            "missing_pct": 100.0,
            "dup_pct": 0.0,
            "invalid_pct": 0.0,
            "gap_count": 0,
            "recommendations": ["Dataset is empty. Provide historical weather records."],
            "per_variable_missing": {},
        }

    measurement_cols = [c for c in VALID_RANGES if c in original_df.columns]

    # 1. Missing values
    if measurement_cols:
        total_measurement_cells = max(1, original_df[measurement_cols].size)
        total_missing = float(original_df[measurement_cols].isna().sum().sum())
        missing_pct = 100.0 * total_missing / total_measurement_cells
        per_variable_missing = {
            col: round(100.0 * float(original_df[col].isna().sum()) / max(1, len(original_df)), 2)
            for col in measurement_cols
        }
    else:
        missing_pct = 0.0
        per_variable_missing = {}

    missing_score = max(0.0, 100.0 - missing_pct * 4.0)

    # 2. Duplicates
    orig_len = max(1, len(original_df))
    dup_count = len(original_df) - len(original_df.drop_duplicates())
    dup_pct = 100.0 * float(dup_count) / orig_len
    dup_score = max(0.0, 100.0 - dup_pct * 10.0)

    # 3. Invalid out-of-range values
    invalid_count = 0
    valid_count = 0
    for col in measurement_cols:
        min_v, max_v = VALID_RANGES[col]
        numeric_series = pd.to_numeric(original_df[col], errors="coerce").dropna()
        valid_count += len(numeric_series)
        invalid_count += int(((numeric_series < min_v) | (numeric_series > max_v)).sum())

    total_valid_obs = max(1, valid_count)
    invalid_pct = 100.0 * float(invalid_count) / total_valid_obs
    invalid_score = max(0.0, 100.0 - invalid_pct * 5.0)

    # 4. Temporal continuity / gaps
    gaps = identify_date_gaps(cleaned_df if cleaned_df is not None and not cleaned_df.empty else original_df)
    gap_count = len(gaps)
    gap_score = max(0.0, 100.0 - float(gap_count) * 5.0)

    # Total score calculation
    total = (
        missing_score * TRUST_WEIGHTS["missing_values"]
        + dup_score * TRUST_WEIGHTS["duplicates"]
        + invalid_score * TRUST_WEIGHTS["invalid_measurements"]
        + gap_score * TRUST_WEIGHTS["date_continuity"]
    )
    overall_score = int(round(total))

    # Recommendations
    recommendations: List[str] = []
    if missing_pct > 10.0:
        recommendations.append(f"High missing data rate ({missing_pct:.1f}%). Impute missing observations before running deep models.")
    elif missing_pct > 2.0:
        recommendations.append("Moderate missing values found; linear or spline interpolation recommended.")

    if dup_pct > 0.0:
        recommendations.append(f"Found {dup_count} duplicate row(s). Deduplication applied.")

    if invalid_pct > 0.0:
        recommendations.append(f"Found {invalid_count} measurement(s) outside physical thresholds. Check sensor calibration.")

    if gap_count > 0:
        total_gap_days = sum(g["missing_days"] for g in gaps)
        recommendations.append(f"Detected {gap_count} date gap(s) spanning {total_gap_days} missing day(s).")

    if overall_score >= 90:
        recommendations.append("Data quality is robust and well-suited for high-confidence meteorological analytics.")
    elif overall_score < 70:
        recommendations.append("Caution advised: overall dataset trust score is degraded.")

    return {
        "overall_score": overall_score,
        "components": {
            "missing_values": round(float(missing_score), 1),
            "duplicates": round(float(dup_score), 1),
            "invalid_measurements": round(float(invalid_score), 1),
            "date_continuity": round(float(gap_score), 1),
        },
        "missing_pct": round(float(missing_pct), 2),
        "dup_pct": round(float(dup_pct), 2),
        "invalid_pct": round(float(invalid_pct), 2),
        "gap_count": int(gap_count),
        "recommendations": recommendations,
        "per_variable_missing": per_variable_missing,
    }


def detect_suspicious_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    For each measurement column, flag extreme values outside 3 IQR fences
    (Q1 - 3*IQR, Q3 + 3*IQR). Returns up to 5 example anomalies per column.
    """
    if df is None or df.empty:
        return {}

    measurement_cols = [c for c in VALID_RANGES if c in df.columns]
    results: Dict[str, Any] = {}

    for col in measurement_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        valid_idx = s.dropna().index
        if len(valid_idx) < 4:
            continue

        q1 = float(s.loc[valid_idx].quantile(0.25))
        q3 = float(s.loc[valid_idx].quantile(0.75))
        iqr = q3 - q1
        lower_fence = q1 - 3.0 * iqr
        upper_fence = q3 + 3.0 * iqr

        suspicious_mask = (df.index.isin(valid_idx)) & ((df[col] < lower_fence) | (df[col] > upper_fence))
        count = int(suspicious_mask.sum())

        if count > 0:
            examples = []
            suspicious_sub = df[suspicious_mask]
            for _, row in suspicious_sub.head(5).iterrows():
                d_val = row.get("date")
                if pd.notna(d_val):
                    date_str = pd.to_datetime(d_val).strftime("%Y-%m-%d")
                else:
                    date_str = "Unknown"

                examples.append({
                    "date": date_str,
                    "value": round(float(row[col]), 3),
                })

            results[col] = {
                "count": count,
                "examples": examples,
            }

    return results
