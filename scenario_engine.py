"""
WeatherWise Scenario Engine Module
==================================
Allows what-if weather simulations by adjusting meteorological variables
(additive, multiplicative, or set-value overrides) across specified date ranges,
comparing before/after impacts on statistical distribution, anomalies, and trends,
and exporting scenarios to CSV bytes.
"""

import io
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from analysis import compute_trend, get_descriptive_stats
from anomaly_detection import detect_iqr_anomalies


def apply_scenario(
    df: pd.DataFrame,
    variable: str,
    adjustment: float,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    adjustment_type: str = "add",
) -> pd.DataFrame:
    """
    Apply a what-if adjustment to a specified variable in the dataset.
    ALWAYS operates on and returns a deep copy without mutating the original dataframe.

    Parameters:
    - df: input weather dataframe
    - variable: name of column to modify
    - adjustment: numeric delta or multiplier
    - date_start: optional starting date filter (inclusive)
    - date_end: optional ending date filter (inclusive)
    - adjustment_type: 'add' (default), 'multiply', or 'set'
    """
    if df is None:
        return pd.DataFrame()

    modified = df.copy(deep=True)
    if variable not in modified.columns or modified.empty:
        return modified

    modified[variable] = pd.to_numeric(modified[variable], errors="coerce")

    # Build date mask
    if "date" in modified.columns and (date_start is not None or date_end is not None):
        parsed_dates = pd.to_datetime(modified["date"], errors="coerce")
        mask = pd.Series(True, index=modified.index)
        if date_start is not None:
            mask = mask & (parsed_dates >= pd.to_datetime(date_start))
        if date_end is not None:
            mask = mask & (parsed_dates <= pd.to_datetime(date_end))
    else:
        mask = pd.Series(True, index=modified.index)

    adj_val = float(adjustment)
    adj_type = adjustment_type.lower()

    if adj_type == "multiply":
        modified.loc[mask, variable] = modified.loc[mask, variable] * adj_val
    elif adj_type == "set":
        modified.loc[mask, variable] = adj_val
    else:
        # Default: 'add'
        modified.loc[mask, variable] = modified.loc[mask, variable] + adj_val

    return modified


def compare_scenarios(
    original: pd.DataFrame, modified: pd.DataFrame, variable: str
) -> Dict[str, Any]:
    """
    Compare original and scenario-modified datasets across key descriptive metrics,
    IQR anomaly counts, and trend regression slopes.
    """
    if (
        original is None
        or modified is None
        or original.empty
        or modified.empty
        or variable not in original.columns
        or variable not in modified.columns
    ):
        return {"error": f"Insufficient data for comparing scenarios for variable: {variable}"}

    stat_orig = get_descriptive_stats(original, variable)
    stat_mod = get_descriptive_stats(modified, variable)

    if "error" in stat_orig or "error" in stat_mod:
        return {"error": f"Failed to compute descriptive statistics for variable {variable}"}

    trend_orig = compute_trend(original, variable)
    trend_mod = compute_trend(modified, variable)

    anom_orig = detect_iqr_anomalies(original, variable)
    anom_mod = detect_iqr_anomalies(modified, variable)

    orig_stats: Dict[str, Any] = {
        "mean": stat_orig.get("mean", 0.0),
        "median": stat_orig.get("median", 0.0),
        "std": stat_orig.get("std", 0.0),
        "min": stat_orig.get("min", 0.0),
        "max": stat_orig.get("max", 0.0),
        "anomaly_count": int(anom_orig.get("anomaly_count", 0)),
        "trend_slope": trend_orig.get("slope", 0.0),
    }

    mod_stats: Dict[str, Any] = {
        "mean": stat_mod.get("mean", 0.0),
        "median": stat_mod.get("median", 0.0),
        "std": stat_mod.get("std", 0.0),
        "min": stat_mod.get("min", 0.0),
        "max": stat_mod.get("max", 0.0),
        "anomaly_count": int(anom_mod.get("anomaly_count", 0)),
        "trend_slope": trend_mod.get("slope", 0.0),
    }

    changes: Dict[str, Any] = {
        "mean_diff": round(float(mod_stats["mean"] - orig_stats["mean"]), 3),
        "max_diff": round(float(mod_stats["max"] - orig_stats["max"]), 3),
        "min_diff": round(float(mod_stats["min"] - orig_stats["min"]), 3),
        "anomaly_count_diff": int(mod_stats["anomaly_count"] - orig_stats["anomaly_count"]),
        "trend_slope_diff": round(float(mod_stats["trend_slope"] - orig_stats["trend_slope"]), 4),
    }

    adj_description = (
        f"Scenario simulation for {variable}: mean shifted by {changes['mean_diff']:+.2f}, "
        f"extremes shifted min={changes['min_diff']:+.2f} / max={changes['max_diff']:+.2f}, "
        f"IQR anomalies changed by {changes['anomaly_count_diff']:+d}, "
        f"trend slope adjusted by {changes['trend_slope_diff']:+.4f}."
    )

    return {
        "variable": variable,
        "original": orig_stats,
        "modified": mod_stats,
        "changes": changes,
        "adjustment_description": adj_description,
    }


def export_scenario_csv(df: pd.DataFrame) -> bytes:
    """
    Export the dataframe as UTF-8 encoded CSV bytes for download or streaming.
    """
    if df is None:
        return b""
    return df.to_csv(index=False).encode("utf-8")
