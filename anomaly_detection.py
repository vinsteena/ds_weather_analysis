"""
WeatherWise Anomaly Detection Module
====================================
Statistical anomaly detection methods:
- Interquartile Range (IQR) fences
- Z-score deviation from sample mean
- Rolling window local moving z-score / deviation
- Anomaly investigation report generation with meteorological context
"""

import calendar
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

UNITS: Dict[str, str] = {
    "temperature": "°C",
    "humidity": "%",
    "rainfall": "mm",
    "wind_speed": "km/h",
    "pressure": "hPa",
}


def _safe_float(val: Any, default: Optional[float] = None, round_dp: Optional[int] = 3) -> Optional[float]:
    """Safely convert a scalar to a Python float, handling NaN/inf."""
    if val is None or pd.isna(val) or np.isnan(val) or np.isinf(val):
        return default
    f = float(val)
    return round(f, round_dp) if round_dp is not None else f


def detect_iqr_anomalies(
    df: pd.DataFrame, variable: str, multiplier: float = 1.5
) -> Dict[str, Any]:
    """
    Detect anomalies using Interquartile Range (IQR) fences.
    Lower fence = Q1 - multiplier * IQR
    Upper fence = Q3 + multiplier * IQR
    """
    if df is None or df.empty or variable not in df.columns or "date" not in df.columns:
        return {"error": f"Insufficient data for variable: {variable}"}

    sub = df[["date", variable]].copy()
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub[variable] = pd.to_numeric(sub[variable], errors="coerce")
    sub = sub.dropna().sort_values("date").reset_index(drop=True)

    if len(sub) < 4:
        return {"error": f"Insufficient data for IQR anomaly detection (minimum 4 observations required, got {len(sub)})"}

    vals = sub[variable].values.astype(float)
    q1 = float(np.percentile(vals, 25))
    q3 = float(np.percentile(vals, 75))
    iqr = q3 - q1
    lower_fence = q1 - float(multiplier) * iqr
    upper_fence = q3 + float(multiplier) * iqr

    anomalies: List[Dict[str, Any]] = []
    all_dates: List[str] = []
    all_values: List[float] = []
    is_anomaly: List[bool] = []

    for _, row in sub.iterrows():
        d_str = row["date"].strftime("%Y-%m-%d")
        v = float(row[variable])
        all_dates.append(d_str)
        all_values.append(round(v, 3))

        flag = bool(v < lower_fence or v > upper_fence)
        is_anomaly.append(flag)

        if flag:
            direction = "high" if v > upper_fence else "low"
            anomalies.append({
                "date": d_str,
                "value": round(v, 3),
                "direction": direction,
            })

    return {
        "method": "iqr",
        "variable": variable,
        "q1": round(q1, 3),
        "q3": round(q3, 3),
        "iqr": round(iqr, 3),
        "lower_fence": round(lower_fence, 3),
        "upper_fence": round(upper_fence, 3),
        "anomaly_count": int(len(anomalies)),
        "anomalies": anomalies,
        "all_dates": all_dates,
        "all_values": all_values,
        "is_anomaly": is_anomaly,
    }


def detect_zscore_anomalies(
    df: pd.DataFrame, variable: str, threshold: float = 3.0
) -> Dict[str, Any]:
    """
    Detect anomalies using Standard Z-Score.
    Flag when |z| > threshold.
    """
    if df is None or df.empty or variable not in df.columns or "date" not in df.columns:
        return {"error": f"Insufficient data for variable: {variable}"}

    sub = df[["date", variable]].copy()
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub[variable] = pd.to_numeric(sub[variable], errors="coerce")
    sub = sub.dropna().sort_values("date").reset_index(drop=True)

    if len(sub) < 4:
        return {"error": f"Insufficient data for Z-score anomaly detection (minimum 4 observations required, got {len(sub)})"}

    vals = sub[variable].values.astype(float)
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    anomalies: List[Dict[str, Any]] = []
    all_dates: List[str] = []
    all_values: List[float] = []
    is_anomaly: List[bool] = []

    for _, row in sub.iterrows():
        d_str = row["date"].strftime("%Y-%m-%d")
        v = float(row[variable])
        all_dates.append(d_str)
        all_values.append(round(v, 3))

        if std_val > 0.0:
            z = float((v - mean_val) / std_val)
        else:
            z = 0.0

        flag = bool(abs(z) > threshold)
        is_anomaly.append(flag)

        if flag:
            direction = "high" if z > 0 else "low"
            anomalies.append({
                "date": d_str,
                "value": round(v, 3),
                "z_score": round(z, 3),
                "direction": direction,
            })

    return {
        "method": "zscore",
        "variable": variable,
        "threshold": float(threshold),
        "mean": round(mean_val, 3),
        "std": round(std_val, 3),
        "anomaly_count": int(len(anomalies)),
        "anomalies": anomalies,
        "all_dates": all_dates,
        "all_values": all_values,
        "is_anomaly": is_anomaly,
    }


def detect_rolling_anomalies(
    df: pd.DataFrame, variable: str, window: int = 30, sigma: float = 2.0
) -> Dict[str, Any]:
    """
    Detect local anomalies using rolling moving window mean and standard deviation.
    Flag when |value - rolling_mean| > sigma * rolling_std.
    """
    if df is None or df.empty or variable not in df.columns or "date" not in df.columns:
        return {"error": f"Insufficient data for variable: {variable}"}

    sub = df[["date", variable]].copy()
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub[variable] = pd.to_numeric(sub[variable], errors="coerce")
    sub = sub.dropna().sort_values("date").reset_index(drop=True)

    min_p = max(5, int(window) // 3)
    if len(sub) < min_p:
        return {"error": f"Insufficient data for rolling anomaly detection (minimum {min_p} observations required, got {len(sub)})"}

    roll = sub[variable].rolling(window=int(window), min_periods=min_p)
    r_mean = roll.mean()
    r_std = roll.std(ddof=1)

    anomalies: List[Dict[str, Any]] = []
    all_dates: List[str] = []
    all_values: List[float] = []
    rolling_means: List[Optional[float]] = []
    is_anomaly: List[bool] = []

    for i in range(len(sub)):
        d_str = sub["date"].iloc[i].strftime("%Y-%m-%d")
        v = float(sub[variable].iloc[i])
        m = r_mean.iloc[i]
        s = r_std.iloc[i]

        all_dates.append(d_str)
        all_values.append(round(v, 3))

        if pd.isna(m) or pd.isna(s) or s == 0.0:
            rolling_means.append(_safe_float(m, None, 3))
            is_anomaly.append(False)
            continue

        m_val = float(m)
        s_val = float(s)
        rolling_means.append(round(m_val, 3))

        dev = v - m_val
        flag = bool(abs(dev) > float(sigma) * s_val)
        is_anomaly.append(flag)

        if flag:
            anomalies.append({
                "date": d_str,
                "value": round(v, 3),
                "rolling_mean": round(m_val, 3),
                "deviation": round(dev, 3),
            })

    return {
        "method": "rolling",
        "variable": variable,
        "window": int(window),
        "sigma": float(sigma),
        "anomaly_count": int(len(anomalies)),
        "anomalies": anomalies,
        "all_dates": all_dates,
        "all_values": all_values,
        "rolling_means": rolling_means,
        "is_anomaly": is_anomaly,
    }


def get_investigation_report(
    df: pd.DataFrame, date_str: str, variable: str, method: str = "iqr"
) -> Dict[str, Any]:
    """
    Generate an in-depth meteorological investigation report for a specific date and variable.
    Compares the observed value against calendar month normals, evaluates multiple detection
    methods, gathers temporal context (adjacent days), and synthesizes a plain-English explanation.
    """
    if df is None or df.empty or variable not in df.columns or "date" not in df.columns:
        return {
            "error": "Dataframe is empty or required columns are missing.",
            "data_available": False,
            "date": str(date_str),
            "variable": variable,
        }

    sub = df.copy()
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub[variable] = pd.to_numeric(sub[variable], errors="coerce")
    sub = sub.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    target_dt = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(target_dt):
        return {
            "error": f"Invalid date string format: {date_str}",
            "data_available": False,
            "date": str(date_str),
            "variable": variable,
        }

    target_date_norm = target_dt.strftime("%Y-%m-%d")
    date_mask = sub["date"].dt.strftime("%Y-%m-%d") == target_date_norm
    if not date_mask.any():
        return {
            "error": f"No observation found for {variable} on date {target_date_norm}.",
            "data_available": False,
            "date": target_date_norm,
            "variable": variable,
        }

    target_row = sub[date_mask].iloc[0]
    target_idx = sub[date_mask].index[0]
    observed_val = target_row[variable]

    if pd.isna(observed_val):
        return {
            "error": f"Observation for {variable} on {target_date_norm} is missing (NaN).",
            "data_available": False,
            "date": target_date_norm,
            "variable": variable,
        }

    observed_val = float(observed_val)
    unit = UNITS.get(variable, "")

    # Monthly comparison
    target_month = target_row["date"].month
    month_data = sub[sub["date"].dt.month == target_month][variable].dropna().astype(float)
    month_name = calendar.month_name[target_month]
    month_avg = float(month_data.mean()) if len(month_data) > 0 else observed_val
    month_std = float(month_data.std(ddof=1)) if len(month_data) > 1 else 0.0
    difference = observed_val - month_avg
    pct_difference = (difference / abs(month_avg) * 100.0) if month_avg != 0 else 0.0

    # Cross-method detection checks
    iqr_res = detect_iqr_anomalies(df, variable)
    zscore_res = detect_zscore_anomalies(df, variable)
    rolling_res = detect_rolling_anomalies(df, variable)

    is_iqr = False
    is_zscore = False
    is_rolling = False

    if "all_dates" in iqr_res and target_date_norm in iqr_res["all_dates"]:
        idx = iqr_res["all_dates"].index(target_date_norm)
        is_iqr = bool(iqr_res["is_anomaly"][idx])

    if "all_dates" in zscore_res and target_date_norm in zscore_res["all_dates"]:
        idx = zscore_res["all_dates"].index(target_date_norm)
        is_zscore = bool(zscore_res["is_anomaly"][idx])

    if "all_dates" in rolling_res and target_date_norm in rolling_res["all_dates"]:
        idx = rolling_res["all_dates"].index(target_date_norm)
        is_rolling = bool(rolling_res["is_anomaly"][idx])

    flagged_by = {
        "iqr": is_iqr,
        "zscore": is_zscore,
        "rolling": is_rolling,
    }

    # Anomaly details for the requested primary method
    primary_method = method.lower()
    anomaly_details: Dict[str, Any] = {}
    if primary_method == "zscore" and "all_dates" in zscore_res:
        z_val = round((observed_val - zscore_res["mean"]) / max(1e-6, zscore_res["std"]), 3) if zscore_res["std"] > 0 else 0.0
        anomaly_details = {
            "zscore": {
                "z_score": z_val,
                "threshold": zscore_res["threshold"],
                "flagged": is_zscore,
            }
        }
    elif primary_method == "rolling" and "all_dates" in rolling_res:
        idx = rolling_res["all_dates"].index(target_date_norm)
        r_mean = rolling_res["rolling_means"][idx]
        anomaly_details = {
            "rolling": {
                "rolling_mean": r_mean,
                "deviation": round(observed_val - r_mean, 3) if r_mean is not None else None,
                "sigma": rolling_res["sigma"],
                "flagged": is_rolling,
            }
        }
    else:
        # Default to IQR
        anomaly_details = {
            "iqr": {
                "lower_fence": iqr_res.get("lower_fence"),
                "upper_fence": iqr_res.get("upper_fence"),
                "iqr": iqr_res.get("iqr"),
                "flagged": is_iqr,
            }
        }

    # Context: previous and next days
    prev_day_dict: Dict[str, Any] = {"date": None, "value": None}
    if target_idx > 0:
        prev_row = sub.iloc[target_idx - 1]
        prev_day_dict = {
            "date": prev_row["date"].strftime("%Y-%m-%d"),
            "value": _safe_float(prev_row[variable], None, 3),
        }

    next_day_dict: Dict[str, Any] = {"date": None, "value": None}
    if target_idx < len(sub) - 1:
        next_row = sub.iloc[target_idx + 1]
        next_day_dict = {
            "date": next_row["date"].strftime("%Y-%m-%d"),
            "value": _safe_float(next_row[variable], None, 3),
        }

    # Other variables on the same date
    other_vars: Dict[str, Optional[float]] = {}
    for col in UNITS:
        if col != variable and col in target_row:
            other_vars[col] = _safe_float(target_row[col], None, 3)

    # Explanation generation
    methods_flagged = [m.upper() for m, flagged in flagged_by.items() if flagged]
    direction_word = "higher" if difference > 0 else "lower"
    abs_diff = abs(difference)

    if methods_flagged:
        flag_summary = f"Flagged as anomalous by {', '.join(methods_flagged)}."
    else:
        flag_summary = "Not flagged as anomalous by standard statistical thresholds."

    explanation = (
        f"On {target_date_norm}, {variable} was {observed_val:.1f}{unit}, which is "
        f"{abs_diff:.1f}{unit} {direction_word} than the {month_name} seasonal average of "
        f"{month_avg:.1f}{unit} ({pct_difference:+.1f}% deviation). {flag_summary}"
    )

    return {
        "date": target_date_norm,
        "variable": variable,
        "observed_value": round(observed_val, 3),
        "unit": unit,
        "monthly_comparison": {
            "month_name": month_name,
            "month_avg": round(month_avg, 3),
            "month_std": round(month_std, 3),
            "difference": round(difference, 3),
            "pct_difference": round(pct_difference, 2),
        },
        "flagged_by": flagged_by,
        "anomaly_details": anomaly_details,
        "context": {
            "prev_day": prev_day_dict,
            "next_day": next_day_dict,
            "other_variables": other_vars,
        },
        "explanation": explanation,
        "data_available": True,
    }


def get_all_anomalies(
    df: pd.DataFrame, variable: str, method: str = "iqr"
) -> Dict[str, Any]:
    """
    Run selected anomaly detection method ('iqr', 'zscore', or 'rolling')
    and compute overall anomaly percentage.
    """
    m = method.lower()
    if m == "zscore":
        res = detect_zscore_anomalies(df, variable)
    elif m == "rolling":
        res = detect_rolling_anomalies(df, variable)
    else:
        res = detect_iqr_anomalies(df, variable)

    if "error" in res:
        return res

    total_obs = len(res.get("all_values", []))
    anom_cnt = res.get("anomaly_count", 0)
    anom_pct = (100.0 * anom_cnt / max(1, total_obs)) if total_obs > 0 else 0.0

    res["anomaly_percentage"] = round(float(anom_pct), 2)
    return res
