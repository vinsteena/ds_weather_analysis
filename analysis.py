"""
WeatherWise Analysis Module
===========================
Core statistical analysis, trend estimation, seasonal aggregation,
year-over-year comparison, time-series decomposition, and monthly summaries.
"""

import calendar
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, linregress, skew
from statsmodels.tsa.seasonal import seasonal_decompose


def _to_float(val: Any, default: Optional[float] = None, round_digits: Optional[int] = None) -> Optional[float]:
    """Safely convert a scalar value to a Python float, handling NaN/inf."""
    if val is None or pd.isna(val) or np.isnan(val) or np.isinf(val):
        return default
    f = float(val)
    return round(f, round_digits) if round_digits is not None else f


class MonthlySummaryList(list):
    """
    List of monthly summary records that also supports dict-style access
    (e.g., .get('error')) and transparent JSON serialization.
    """

    def __init__(self, items: Optional[list] = None, error: Optional[str] = None):
        super().__init__(items or [])
        self._error = error

    def get(self, key: str, default: Any = None) -> Any:
        if key == "error":
            return self._error
        if key in ("summary", "data", "months", "monthly_summary"):
            return list(self)
        return default

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key == "error":
                if self._error is not None:
                    return self._error
                raise KeyError(key)
            if key in ("summary", "data", "months", "monthly_summary"):
                return list(self)
            raise KeyError(key)
        return super().__getitem__(key)


def load_dataframe(path: str) -> pd.DataFrame:
    """
    Read CSV from path, parse 'date' column to datetime, sort by date,
    and return DataFrame.
    """
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
    return df


def get_available_variables(df: pd.DataFrame) -> List[str]:
    """
    Return list of numeric columns present in df that have at least
    some non-NaN values from:
    ['temperature', 'humidity', 'rainfall', 'wind_speed', 'pressure']
    """
    candidate_cols = ["temperature", "humidity", "rainfall", "wind_speed", "pressure"]
    available = []
    for col in candidate_cols:
        if col in df.columns:
            numeric_s = pd.to_numeric(df[col], errors="coerce")
            if numeric_s.notna().sum() > 0:
                available.append(col)
    return available


def get_date_range(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return date range information:
    {'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD', 'total_days': int, 'available_obs': int}
    """
    if df is None or df.empty or "date" not in df.columns:
        return {"error": "No valid dates found in dataframe"}

    valid_dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if len(valid_dates) == 0:
        return {"error": "No valid dates found in dataframe"}

    min_date = valid_dates.min()
    max_date = valid_dates.max()
    total_days = int((max_date.date() - min_date.date()).days) + 1
    available_obs = int(len(valid_dates))

    return {
        "start": min_date.strftime("%Y-%m-%d"),
        "end": max_date.strftime("%Y-%m-%d"),
        "total_days": total_days,
        "available_obs": available_obs,
    }


def get_descriptive_stats(df: pd.DataFrame, variable: str) -> Dict[str, Any]:
    """
    Return descriptive statistics for a specified variable:
    {variable, count, mean, median, std, min, max, q25, q75, skewness, kurtosis}
    All numeric values as Python float, rounded to 3 decimal places.
    """
    if df is None or df.empty or variable not in df.columns:
        return {"error": f"Insufficient data for variable: {variable}"}

    s = pd.to_numeric(df[variable], errors="coerce").dropna()
    cnt = int(len(s))
    if cnt == 0:
        return {"error": f"Insufficient data for variable: {variable}"}

    mean_val = _to_float(s.mean(), 0.0, 3)
    median_val = _to_float(s.median(), 0.0, 3)
    std_val = _to_float(s.std(ddof=1), 0.0, 3) if cnt > 1 else 0.0
    min_val = _to_float(s.min(), 0.0, 3)
    max_val = _to_float(s.max(), 0.0, 3)
    q25_val = _to_float(s.quantile(0.25), 0.0, 3)
    q75_val = _to_float(s.quantile(0.75), 0.0, 3)

    if cnt >= 3 and std_val > 0.0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sk = skew(s.values, bias=False, nan_policy="omit")
            kt = kurtosis(s.values, bias=False, nan_policy="omit")
            skew_val = _to_float(sk, 0.0, 3)
            kurt_val = _to_float(kt, 0.0, 3)
    else:
        skew_val = 0.0
        kurt_val = 0.0

    return {
        "variable": variable,
        "count": cnt,
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "q25": q25_val,
        "q75": q75_val,
        "skewness": skew_val,
        "kurtosis": kurt_val,
    }


def compute_trend(df: pd.DataFrame, variable: str) -> Dict[str, Any]:
    """
    Compute linear regression of variable vs. day index.
    Requires at least 10 observations.
    """
    if df is None or df.empty or variable not in df.columns or "date" not in df.columns:
        return {"error": f"Insufficient data for variable: {variable}"}

    sub = df[["date", variable]].copy()
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub[variable] = pd.to_numeric(sub[variable], errors="coerce")
    sub = sub.dropna().sort_values("date").reset_index(drop=True)

    if len(sub) < 10:
        return {
            "error": f"Insufficient data for trend computation (minimum 10 observations required, got {len(sub)})"
        }

    # Day index relative to the earliest observation date
    min_date = sub["date"].iloc[0]
    days = (sub["date"] - min_date).dt.total_seconds() / 86400.0

    if days.max() == days.min():
        days = np.arange(len(sub), dtype=float)
    else:
        days = days.values.astype(float)

    reg = linregress(days, sub[variable].values.astype(float))
    slope = _to_float(reg.slope, 0.0, 4)
    intercept = _to_float(reg.intercept, 0.0, 4)
    r_squared = _to_float(reg.rvalue**2, 0.0, 4)
    p_value = _to_float(reg.pvalue, 1.0, 4)
    change_per_year = _to_float(slope * 365.0, 0.0, 3)

    if p_value < 0.05 and abs(slope) > 1e-5:
        trend_direction = "increasing" if slope > 0 else "decreasing"
    else:
        trend_direction = "flat"

    significance = "significant" if p_value < 0.05 else "not significant"

    trend_vals = reg.slope * days + reg.intercept
    dates = [d.strftime("%Y-%m-%d") for d in sub["date"]]
    values = [_to_float(v, 0.0, 3) for v in sub[variable].values]
    trend_line = [_to_float(v, 0.0, 3) for v in trend_vals]

    return {
        "variable": variable,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "p_value": p_value,
        "trend_direction": trend_direction,
        "change_per_year": change_per_year,
        "significance": significance,
        "dates": dates,
        "values": values,
        "trend_line": trend_line,
    }


def seasonal_aggregation(df: pd.DataFrame, variable: str) -> Dict[str, Any]:
    """
    Aggregate variable by month and season.
    Returns:
    - monthly_avg: dict of month number (1-12) -> avg
    - monthly_total: dict of month name -> sum for rainfall, mean for others
    - seasonal_avg: dict of season name -> avg (Winter, Spring, Summer, Autumn)
    - monthly_by_year: list of {year, month, value}
    - hottest/coldest or wettest/driest month names when applicable
    """
    if df is None or df.empty or variable not in df.columns or "date" not in df.columns:
        return {"error": f"Insufficient data for variable: {variable}"}

    sub = df[["date", variable]].copy()
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub[variable] = pd.to_numeric(sub[variable], errors="coerce")
    sub = sub.dropna().reset_index(drop=True)

    if len(sub) == 0:
        return {"error": f"Insufficient data for variable: {variable}"}

    sub["year"] = sub["date"].dt.year
    sub["month"] = sub["date"].dt.month

    # monthly_avg: keyed by month number
    monthly_grp = sub.groupby("month")[variable].mean()
    monthly_avg = {int(m): _to_float(v, 0.0, 3) for m, v in monthly_grp.items()}

    # monthly_total: sum for rainfall, mean for others, keyed by month name
    is_rainfall = "rain" in variable.lower()
    if is_rainfall:
        tot_grp = sub.groupby("month")[variable].sum()
    else:
        tot_grp = sub.groupby("month")[variable].mean()
    monthly_total = {calendar.month_name[int(m)]: _to_float(v, 0.0, 3) for m, v in tot_grp.items()}

    # seasonal_avg: Winter (12, 1, 2), Spring (3, 4, 5), Summer (6, 7, 8), Autumn (9, 10, 11)
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn",
    }
    sub["season"] = sub["month"].map(season_map)
    season_grp = sub.groupby("season")[variable].mean()
    seasonal_avg = {
        s: _to_float(season_grp[s], 0.0, 3) if s in season_grp else 0.0
        for s in ["Winter", "Spring", "Summer", "Autumn"]
    }

    # monthly_by_year: heatmap data
    mby = sub.groupby(["year", "month"])[variable].mean().reset_index()
    monthly_by_year = [
        {
            "year": int(r["year"]),
            "month": int(r["month"]),
            "value": _to_float(r[variable], 0.0, 3),
        }
        for _, r in mby.iterrows()
    ]

    result: Dict[str, Any] = {
        "variable": variable,
        "monthly_avg": monthly_avg,
        "monthly_total": monthly_total,
        "seasonal_avg": seasonal_avg,
        "monthly_by_year": monthly_by_year,
    }

    if monthly_avg:
        max_month = max(monthly_avg, key=lambda m: monthly_avg[m] if monthly_avg[m] is not None else float("-inf"))
        min_month = min(monthly_avg, key=lambda m: monthly_avg[m] if monthly_avg[m] is not None else float("inf"))
        max_month_name = calendar.month_name[max_month]
        min_month_name = calendar.month_name[min_month]

        if "temp" in variable.lower():
            result["hottest_month"] = max_month_name
            result["coldest_month"] = min_month_name
        elif "rain" in variable.lower():
            result["wettest_month"] = max_month_name
            result["driest_month"] = min_month_name
        else:
            result["highest_month"] = max_month_name
            result["lowest_month"] = min_month_name

    return result


def year_over_year_comparison(df: pd.DataFrame, variable: str, year1: int, year2: int) -> Dict[str, Any]:
    """
    Compare statistical metrics between year1 and year2 for a variable.
    """
    if df is None or df.empty or variable not in df.columns or "date" not in df.columns:
        return {"error": f"Insufficient data for comparison of years {year1} and {year2}"}

    sub = df[["date", variable]].copy()
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub[variable] = pd.to_numeric(sub[variable], errors="coerce")
    sub = sub.dropna().reset_index(drop=True)
    sub["year"] = sub["date"].dt.year
    sub["month"] = sub["date"].dt.month

    df1 = sub[sub["year"] == int(year1)]
    df2 = sub[sub["year"] == int(year2)]

    if len(df1) == 0 or len(df2) == 0:
        return {"error": f"Insufficient data for year {year1} or {year2}"}

    def calc_stats(sub_df: pd.DataFrame) -> Dict[str, Any]:
        s = sub_df[variable]
        cnt = int(len(s))
        return {
            "mean": _to_float(s.mean(), 0.0, 3),
            "median": _to_float(s.median(), 0.0, 3),
            "std": _to_float(s.std(ddof=1), 0.0, 3) if cnt > 1 else 0.0,
            "min": _to_float(s.min(), 0.0, 3),
            "max": _to_float(s.max(), 0.0, 3),
            "total": _to_float(s.sum(), 0.0, 3),
            "count": cnt,
        }

    stats1 = calc_stats(df1)
    stats2 = calc_stats(df2)

    mean_diff = _to_float(stats2["mean"] - stats1["mean"], 0.0, 3)
    if stats1["mean"] != 0:
        pct_diff = _to_float(((stats2["mean"] - stats1["mean"]) / abs(stats1["mean"])) * 100.0, 0.0, 2)
    else:
        pct_diff = 0.0

    shared_months = sorted(list(set(df1["month"].unique()).intersection(set(df2["month"].unique()))))

    diff_dict = {
        "mean_difference": mean_diff,
        "percentage_difference": pct_diff,
        "total_difference": _to_float(stats2["total"] - stats1["total"], 0.0, 3),
    }

    # Return dictionary with both integer keys, string keys, and 'year1'/'year2' aliases
    return {
        year1: stats1,
        year2: stats2,
        str(year1): stats1,
        str(year2): stats2,
        "difference": diff_dict,
        "shared_months": [int(m) for m in shared_months],
    }


def compute_decomposition(df: pd.DataFrame, variable: str) -> Dict[str, Any]:
    """
    Decompose time series into trend, seasonal, and residual components.
    Requires at least 730 observations with period=365,
    or at least 24 months with period=12.
    """
    if df is None or df.empty or variable not in df.columns or "date" not in df.columns:
        return {"error": "Need at least 2 years of data for seasonal decomposition"}

    sub = df[["date", variable]].copy()
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub[variable] = pd.to_numeric(sub[variable], errors="coerce")
    sub = sub.dropna().sort_values("date").reset_index(drop=True)

    def clean_series(series_vals: Any) -> List[Optional[float]]:
        out = []
        for v in series_vals:
            if v is None or pd.isna(v) or np.isnan(v) or np.isinf(v):
                out.append(None)
            else:
                out.append(_to_float(v, 0.0, 3))
        return out

    # Strategy 1: Daily data with period=365 if >= 730 observations
    if len(sub) >= 730:
        period_used = 365
        dates = [d.strftime("%Y-%m-%d") for d in sub["date"]]
        values_series = sub[variable].interpolate().bfill().ffill()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                decomp = seasonal_decompose(
                    values_series.values,
                    model="additive",
                    period=period_used,
                    extrapolate_trend="freq",
                )
            return {
                "observed": clean_series(decomp.observed),
                "trend": clean_series(decomp.trend),
                "seasonal": clean_series(decomp.seasonal),
                "residual": clean_series(decomp.resid),
                "dates": dates,
                "period_used": period_used,
            }
        except Exception:
            pass

    # Strategy 2: Monthly aggregation with period=12 if >= 24 months
    sub_monthly = (
        sub.set_index("date")
        .resample("ME")[variable]
        .mean()
        .interpolate()
        .bfill()
        .ffill()
    )
    if len(sub_monthly) >= 24:
        period_used = 12
        dates = [d.strftime("%Y-%m") for d in sub_monthly.index]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                decomp = seasonal_decompose(
                    sub_monthly.values,
                    model="additive",
                    period=period_used,
                    extrapolate_trend="freq",
                )
            return {
                "observed": clean_series(decomp.observed),
                "trend": clean_series(decomp.trend),
                "seasonal": clean_series(decomp.seasonal),
                "residual": clean_series(decomp.resid),
                "dates": dates,
                "period_used": period_used,
            }
        except Exception:
            pass

    return {"error": "Need at least 2 years of data for seasonal decomposition"}


def get_monthly_summary(df: pd.DataFrame) -> Union[MonthlySummaryList, Dict[str, Any]]:
    """
    Group by year-month, compute mean temperature, total rainfall,
    mean humidity, mean wind_speed. Skip months with < 5 observations.
    Returns list of dicts:
    [{'year_month': 'YYYY-MM', 'temperature': float, 'rainfall': float, 'humidity': float, 'wind_speed': float}]
    """
    if df is None or df.empty or "date" not in df.columns:
        return {"error": "Insufficient data for monthly summary"}

    sub = df.copy()
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub = sub.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if len(sub) == 0:
        return {"error": "Insufficient data for monthly summary"}

    sub["year_month"] = sub["date"].dt.strftime("%Y-%m")
    groups = sub.groupby("year_month")

    results = []
    for ym, grp in groups:
        if len(grp) < 5:
            continue

        temp_val = _to_float(grp["temperature"].mean(), None, 2) if "temperature" in grp else None
        rain_val = _to_float(grp["rainfall"].sum(), None, 2) if "rainfall" in grp else None
        hum_val = _to_float(grp["humidity"].mean(), None, 2) if "humidity" in grp else None
        wind_val = _to_float(grp["wind_speed"].mean(), None, 2) if "wind_speed" in grp else None

        results.append({
            "year_month": str(ym),
            "temperature": temp_val,
            "rainfall": rain_val,
            "humidity": hum_val,
            "wind_speed": wind_val,
        })

    if not results:
        return {"error": "Insufficient data for monthly summary (no months with >= 5 observations)"}

    return MonthlySummaryList(results)
