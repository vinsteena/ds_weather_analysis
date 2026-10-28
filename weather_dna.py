"""
WeatherWise Weather DNA Module
==============================
Computes multi-dimensional meteorological climate fingerprints ("Weather DNA"),
normalizes metrics against typical global climate envelopes, calculates
period-to-period similarity scores, and isolates key drivers of climatic divergence.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

FEATURES: List[str] = [
    "mean_temperature",
    "temp_variability",
    "total_rainfall",
    "mean_humidity",
    "mean_wind_speed",
    "pressure_stability",
]

GLOBAL_RANGES: Dict[str, tuple] = {
    "mean_temperature": (-10.0, 45.0),
    "temp_variability": (0.0, 20.0),
    "total_rainfall": (0.0, 3000.0),
    "mean_humidity": (0.0, 100.0),
    "mean_wind_speed": (0.0, 60.0),
    "pressure_stability": (0.0, 100.0),
}

FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "mean_temperature": "Mean Temperature",
    "temp_variability": "Temperature Variability",
    "total_rainfall": "Annual Rainfall",
    "mean_humidity": "Mean Humidity",
    "mean_wind_speed": "Mean Wind Speed",
    "pressure_stability": "Pressure Stability",
}

FEATURE_UNITS: Dict[str, str] = {
    "mean_temperature": "°C",
    "temp_variability": "°C",
    "total_rainfall": "mm",
    "mean_humidity": "%",
    "mean_wind_speed": "km/h",
    "pressure_stability": "",
}


def compute_fingerprint(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    label: str = "Period",
) -> Dict[str, Any]:
    """
    Compute normalized 6-dimensional Weather DNA fingerprint for a given timeframe.
    Requires at least 30 observations.
    """
    if df is None or df.empty:
        return {"error": "Dataframe is empty or None."}

    sub = df.copy(deep=True)
    if "date" in sub.columns:
        sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
        if start_date is not None:
            sub = sub[sub["date"] >= pd.to_datetime(start_date)]
        if end_date is not None:
            sub = sub[sub["date"] <= pd.to_datetime(end_date)]
        sub = sub.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    n_observations = int(len(sub))
    if n_observations < 30:
        return {
            "error": f"Insufficient observations for fingerprint (minimum 30 required, got {n_observations})"
        }

    raw_features: Dict[str, float] = {}

    # 1. mean_temperature
    if "temperature" in sub.columns and sub["temperature"].notna().sum() > 0:
        t_clean = pd.to_numeric(sub["temperature"], errors="coerce").dropna()
        t_mean = float(t_clean.mean())
        t_std = float(t_clean.std(ddof=1)) if len(t_clean) > 1 else 0.0
    else:
        t_mean = 17.5
        t_std = 5.0

    raw_features["mean_temperature"] = round(t_mean, 3)
    raw_features["temp_variability"] = round(t_std, 3)

    # 2. total_rainfall (annualized: sum / count * 365)
    if "rainfall" in sub.columns and sub["rainfall"].notna().sum() > 0:
        r_clean = pd.to_numeric(sub["rainfall"], errors="coerce").dropna()
        r_annualized = float((r_clean.sum() / max(1, len(r_clean))) * 365.0)
    else:
        r_annualized = 0.0
    raw_features["total_rainfall"] = round(r_annualized, 3)

    # 3. mean_humidity
    if "humidity" in sub.columns and sub["humidity"].notna().sum() > 0:
        h_clean = pd.to_numeric(sub["humidity"], errors="coerce").dropna()
        h_mean = float(h_clean.mean())
    else:
        h_mean = 50.0
    raw_features["mean_humidity"] = round(h_mean, 3)

    # 4. mean_wind_speed
    if "wind_speed" in sub.columns and sub["wind_speed"].notna().sum() > 0:
        w_clean = pd.to_numeric(sub["wind_speed"], errors="coerce").dropna()
        w_mean = float(w_clean.mean())
    else:
        w_mean = 10.0
    raw_features["mean_wind_speed"] = round(w_mean, 3)

    # 5. pressure_stability (1 / std(pressure) capped at 100)
    if "pressure" in sub.columns and sub["pressure"].notna().sum() > 1:
        p_clean = pd.to_numeric(sub["pressure"], errors="coerce").dropna()
        p_std = float(p_clean.std(ddof=1)) if len(p_clean) > 1 else 0.0
        p_stab = (1.0 / p_std) if p_std > 0 else 100.0
        p_stab = min(100.0, max(0.0, p_stab))
    else:
        p_stab = 100.0
    raw_features["pressure_stability"] = round(p_stab, 3)

    # Normalized features (0 to 1 based on GLOBAL_RANGES)
    normalized_features: Dict[str, float] = {}
    for feat in FEATURES:
        val = raw_features[feat]
        min_r, max_r = GLOBAL_RANGES[feat]
        norm = (val - min_r) / (max_r - min_r)
        norm = max(0.0, min(1.0, norm))
        normalized_features[feat] = round(float(norm), 3)

    available_variables = [
        c
        for c in ["temperature", "humidity", "rainfall", "wind_speed", "pressure"]
        if c in sub.columns and pd.to_numeric(sub[c], errors="coerce").notna().sum() > 0
    ]

    actual_start = (
        sub["date"].iloc[0].strftime("%Y-%m-%d")
        if "date" in sub.columns and len(sub) > 0 and pd.notna(sub["date"].iloc[0])
        else start_date
    )
    actual_end = (
        sub["date"].iloc[-1].strftime("%Y-%m-%d")
        if "date" in sub.columns and len(sub) > 0 and pd.notna(sub["date"].iloc[-1])
        else end_date
    )

    return {
        "label": label,
        "start_date": actual_start,
        "end_date": actual_end,
        "n_observations": n_observations,
        "raw_features": raw_features,
        "normalized_features": normalized_features,
        "available_variables": available_variables,
    }


def compare_fingerprints(fp1: Dict[str, Any], fp2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two Weather DNA fingerprints and return overall similarity score (0 to 1)
    along with detailed per-feature absolute divergences and directions.
    """
    if "error" in fp1 or "error" in fp2:
        return {"error": "Invalid fingerprints provided for comparison"}

    feature_differences: Dict[str, Any] = {}
    diff_values: List[float] = []

    norm1 = fp1.get("normalized_features", {})
    norm2 = fp2.get("normalized_features", {})

    for feat in FEATURES:
        val1 = float(norm1.get(feat, 0.0))
        val2 = float(norm2.get(feat, 0.0))
        abs_diff = round(abs(val2 - val1), 3)
        diff_values.append(abs_diff)

        if val2 > val1:
            direction = "higher"
        elif val2 < val1:
            direction = "lower"
        else:
            direction = "equal"

        feature_differences[feat] = {
            "fp1_val": round(val1, 3),
            "fp2_val": round(val2, 3),
            "abs_diff": abs_diff,
            "direction": direction,
        }

    similarity = max(0.0, 1.0 - float(np.mean(diff_values))) if diff_values else 1.0

    return {
        "period1": fp1.get("label", "Period 1"),
        "period2": fp2.get("label", "Period 2"),
        "similarity_score": round(similarity, 3),
        "feature_differences": feature_differences,
        "n_obs_1": int(fp1.get("n_observations", 0)),
        "n_obs_2": int(fp2.get("n_observations", 0)),
    }


def biggest_differences(fp1: Dict[str, Any], fp2: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Sort Weather DNA features by normalized absolute divergence descending,
    providing human-readable explanations of each meteorological divergence.
    """
    if "error" in fp1 or "error" in fp2:
        return []

    norm1 = fp1.get("normalized_features", {})
    norm2 = fp2.get("normalized_features", {})
    raw1 = fp1.get("raw_features", {})
    raw2 = fp2.get("raw_features", {})

    p1_label = fp1.get("label", "Period 1")
    p2_label = fp2.get("label", "Period 2")

    items: List[Dict[str, Any]] = []

    for feat in FEATURES:
        n1 = float(norm1.get(feat, 0.0))
        n2 = float(norm2.get(feat, 0.0))
        r1 = float(raw1.get(feat, 0.0))
        r2 = float(raw2.get(feat, 0.0))
        abs_diff = round(abs(n2 - n1), 3)

        if n2 > n1:
            direction = "higher"
        elif n2 < n1:
            direction = "lower"
        else:
            direction = "equal"

        display_name = FEATURE_DISPLAY_NAMES.get(feat, feat)
        unit = FEATURE_UNITS.get(feat, "")

        if direction != "equal":
            raw_delta = abs(r2 - r1)
            desc = (
                f"{display_name} is {raw_delta:.1f}{unit} {direction} in {p2_label} "
                f"({n2:.2f} vs {n1:.2f} normalized)."
            )
        else:
            desc = (
                f"{display_name} is identical between {p1_label} and {p2_label} "
                f"({n1:.2f} normalized)."
            )

        items.append({
            "feature": feat,
            "fp1_value_raw": round(r1, 3),
            "fp2_value_raw": round(r2, 3),
            "fp1_normalized": round(n1, 3),
            "fp2_normalized": round(n2, 3),
            "abs_diff": abs_diff,
            "direction": direction,
            "description": desc,
        })

    items.sort(key=lambda x: x["abs_diff"], reverse=True)
    return items
