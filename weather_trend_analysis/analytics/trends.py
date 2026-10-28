"""
Weather Trend Identification and Significance Module (CO4).
Implements:
- Mann-Kendall Non-Parametric Monotonic Trend Test
- Sen's Slope Estimator
- Ordinary Least Squares (OLS) Linear Trend Slope
- Trend Classification: Increasing, Decreasing, or Stable
"""

import numpy as np
import pandas as pd
from scipy import stats


def mann_kendall_test(series: np.ndarray, alpha=0.05):
    """
    Computes the Mann-Kendall monotonic trend test.
    Returns:
    - trend: 'Increasing', 'Decreasing', or 'Stable'
    - p_value: two-tailed p-value
    - z_score: standard normal test statistic
    - tau: Kendall's rank correlation coefficient
    - s_score: Mann-Kendall score S
    - var_s: Variance of S
    """
    x = np.asarray(series).copy()
    n = len(x)
    if n < 4:
        return {"trend": "Stable", "p_value": 1.0, "z_score": 0.0, "tau": 0.0, "s_score": 0, "slope_per_year": 0.0}

    # Vectorized computation of S
    # S = sum(sgn(x_j - x_i)) for j > i
    diff = x[:, None] - x[None, :]
    # Upper triangle has j < i; we want j > i, which is strictly lower triangle or opposite sign
    s = np.sum(np.sign(np.tril(diff, -1)))

    # Unique values and tie correction
    unique_vals, tie_counts = np.unique(x, return_counts=True)
    ties = tie_counts[tie_counts > 1]

    # Variance of S
    var_s = (n * (n - 1) * (2 * n + 5) - np.sum(ties * (ties - 1) * (2 * ties + 5))) / 18.0

    if var_s == 0:
        z = 0.0
    elif s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))

    # Calculate Sen's Slope
    # pairwise slopes for j > i
    i_idx, j_idx = np.tril_indices(n, -1)
    slopes = (x[i_idx] - x[j_idx]) / (i_idx - j_idx)
    sens_slope = float(np.median(slopes))

    # Trend determination
    if p_value < alpha:
        if s > 0:
            trend = "Increasing"
        else:
            trend = "Decreasing"
    else:
        trend = "Stable"

    return {
        "trend": trend,
        "p_value": round(float(p_value), 5),
        "z_score": round(float(z), 3),
        "tau": round(float(tau), 4),
        "s_score": int(s),
        "sens_slope_daily": sens_slope,
        "sens_slope_annual": round(sens_slope * 365.25, 4)
    }


def compute_linear_trend(dates: pd.Series, values: pd.Series):
    """
    Computes standard OLS linear regression trend slope per year.
    """
    clean_mask = (~dates.isna()) & (~values.isna())
    clean_dates = dates[clean_mask]
    clean_vals = values[clean_mask]

    # Convert dates to fractional years
    origin = clean_dates.min()
    days_elapsed = (clean_dates - origin).dt.days.values
    years_elapsed = days_elapsed / 365.25

    slope, intercept, r_value, p_value, std_err = stats.linregress(years_elapsed, clean_vals.values)

    return {
        "slope_per_year": round(float(slope), 4),
        "slope_per_decade": round(float(slope * 10.0), 3),
        "r_squared": round(float(r_value ** 2), 4),
        "p_value": round(float(p_value), 5),
        "std_err": round(float(std_err), 4)
    }


def analyze_all_trends(df: pd.DataFrame, target_columns=None) -> pd.DataFrame:
    """
    Performs comprehensive trend tests (Mann-Kendall + OLS) across all key meteorological metrics.
    """
    if target_columns is None:
        target_columns = [
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

    data = df.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
        data.sort_values("date", inplace=True)

    results = []
    for col in target_columns:
        if col in data.columns:
            series = data[col].dropna().values
            mk = mann_kendall_test(series)
            ols = compute_linear_trend(data["date"], data[col])

            results.append({
                "parameter": col,
                "trend_status": mk["trend"],
                "mk_p_value": mk["p_value"],
                "mk_tau": mk["tau"],
                "sens_slope_annual": mk["sens_slope_annual"],
                "ols_slope_annual": ols["slope_per_year"],
                "ols_slope_decade": ols["slope_per_decade"],
                "ols_r2": ols["r_squared"],
                "ols_p_value": ols["p_value"]
            })

    return pd.DataFrame(results).set_index("parameter")


if __name__ == "__main__":
    import os
    from weather_trend_analysis.analytics.preprocessor import preprocess_weather_data

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "weather_analytics_era5_2015_2024.csv")
    raw_df = pd.read_csv(data_path)
    clean_df = preprocess_weather_data(raw_df)

    trends_summary = analyze_all_trends(clean_df)
    print("=== Multi-Year Weather Trend Analysis ===")
    print(trends_summary[["trend_status", "sens_slope_annual", "ols_slope_decade", "mk_p_value"]])
