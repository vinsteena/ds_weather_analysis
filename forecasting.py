"""
WeatherWise Forecasting Module
==============================
Provides chronological time-series splitting, seasonal naive baseline forecasting,
Holt-Winters Exponential Smoothing forecasting, model evaluation metrics (MAE, RMSE),
multi-model comparative evaluation, and future out-of-sample forecasting.
"""

import datetime
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def _to_float(val: Any, default: Optional[float] = None, round_dp: Optional[int] = 3) -> Optional[float]:
    """Safely convert a scalar to a Python float, handling NaN/inf."""
    if val is None or pd.isna(val) or np.isnan(val) or np.isinf(val):
        return default
    f = float(val)
    return round(f, round_dp) if round_dp is not None else f


def chronological_split(
    df: pd.DataFrame, variable: str, test_ratio: float = 0.2
) -> Dict[str, Any]:
    """
    Split time series chronologically into train and test sets.
    The last test_ratio fraction forms the test set without shuffling.
    Requires at least 30 observations.
    """
    if df is None or df.empty or variable not in df.columns or "date" not in df.columns:
        return {"error": f"Insufficient data for variable: {variable}"}

    sub = df[["date", variable]].copy()
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub[variable] = pd.to_numeric(sub[variable], errors="coerce")
    sub = sub.dropna().sort_values("date").reset_index(drop=True)

    total_n = len(sub)
    if total_n < 30:
        return {
            "error": f"Insufficient data for split (minimum 30 observations required, got {total_n})"
        }

    test_size = max(1, int(round(total_n * float(test_ratio))))
    train_size = total_n - test_size

    if train_size < 10 or test_size < 1:
        return {"error": f"Split ratio {test_ratio} leaves invalid train/test sizes ({train_size}/{test_size})"}

    train_df = sub.iloc[:train_size]
    test_df = sub.iloc[train_size:]

    split_date = test_df["date"].iloc[0].strftime("%Y-%m-%d")

    return {
        "train_dates": [d.strftime("%Y-%m-%d") for d in train_df["date"]],
        "train_values": [_to_float(v, 0.0, 3) for v in train_df[variable]],
        "test_dates": [d.strftime("%Y-%m-%d") for d in test_df["date"]],
        "test_values": [_to_float(v, 0.0, 3) for v in test_df[variable]],
        "split_date": split_date,
        "train_size": int(train_size),
        "test_size": int(test_size),
    }


def baseline_forecast(
    train_dates: List[str], train_values: List[float], test_dates: List[str]
) -> Dict[str, Any]:
    """
    Seasonal naive baseline forecast.
    For each test date, looks up the same day-of-year from historical training data,
    falling back to the most recent training observation.
    """
    if not train_values or not test_dates:
        return {
            "method": "Seasonal Naive",
            "predictions": [],
            "dates": [str(d) for d in test_dates],
        }

    # Build day-of-year lookup from training data (keeps latest observation per day of year)
    doy_map: Dict[int, float] = {}
    for d_str, v in zip(train_dates, train_values):
        dt = pd.to_datetime(d_str, errors="coerce")
        if pd.notna(dt):
            doy_map[dt.dayofyear] = float(v)

    fallback_val = float(train_values[-1])
    predictions: List[float] = []
    formatted_dates: List[str] = []

    for td in test_dates:
        td_dt = pd.to_datetime(td, errors="coerce")
        if pd.notna(td_dt):
            val = doy_map.get(td_dt.dayofyear, fallback_val)
            formatted_dates.append(td_dt.strftime("%Y-%m-%d"))
        else:
            val = fallback_val
            formatted_dates.append(str(td))
        predictions.append(round(float(val), 3))

    return {
        "method": "Seasonal Naive",
        "predictions": predictions,
        "dates": formatted_dates,
    }


def exponential_smoothing_forecast(
    train_dates: List[str],
    train_values: List[float],
    test_dates: List[str],
    forecast_horizon: int = 0,
) -> Dict[str, Any]:
    """
    Fit Holt-Winters Exponential Smoothing on train_values.
    Forecasts for len(test_dates) + forecast_horizon steps.
    Computes confidence intervals using residual standard error (±1.96 * RMSE).
    """
    if len(train_values) < 10:
        return {"error": "Insufficient training observations for exponential smoothing (minimum 10 required)"}

    train_arr = np.array(train_values, dtype=float)
    fit_res = None
    model = None

    # Strategy 1: Additive seasonality if >= 730 observations (2 full years)
    if len(train_arr) >= 730:
        for damped in [True, False]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = ExponentialSmoothing(
                        train_arr,
                        trend="add",
                        seasonal="add",
                        seasonal_periods=365,
                        damped_trend=damped,
                    )
                    fit_res = m.fit()
                    model = m
                    break
            except Exception:
                pass

    # Strategy 2: Additive trend without seasonal component
    if fit_res is None:
        for damped in [True, False]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = ExponentialSmoothing(
                        train_arr,
                        trend="add",
                        seasonal=None,
                        damped_trend=damped,
                    )
                    fit_res = m.fit()
                    model = m
                    break
            except Exception:
                pass

    # Strategy 3: Simple exponential smoothing (level only)
    if fit_res is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = ExponentialSmoothing(train_arr, trend=None, seasonal=None)
                fit_res = m.fit()
                model = m
        except Exception as e:
            return {"error": f"Exponential smoothing fit failed: {str(e)}"}

    total_forecast_steps = len(test_dates) + max(0, int(forecast_horizon))
    if total_forecast_steps == 0:
        total_forecast_steps = 1

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast_all = fit_res.forecast(total_forecast_steps)
    except Exception as e:
        return {"error": f"Forecasting step failed: {str(e)}"}

    test_len = len(test_dates)
    test_preds = [round(float(p), 3) for p in forecast_all[:test_len]]
    future_preds = [round(float(p), 3) for p in forecast_all[test_len:]]

    # Calculate residuals and confidence interval (±1.96 * RMSE)
    fitted_vals = fit_res.fittedvalues
    residuals = train_arr - fitted_vals
    rmse = float(np.sqrt(np.mean(residuals**2))) if len(residuals) > 0 else 1.0
    ci_margin = 1.96 * rmse

    conf_lower = [round(float(p - ci_margin), 3) for p in test_preds]
    conf_upper = [round(float(p + ci_margin), 3) for p in test_preds]

    # Model parameters dict
    model_params: Dict[str, Any] = {}
    if hasattr(fit_res, "params"):
        for k, v in fit_res.params.items():
            if isinstance(v, (int, float, np.integer, np.floating)) and not np.isnan(v):
                model_params[str(k)] = round(float(v), 4)

    # Future dates generation
    future_dates: List[str] = []
    if forecast_horizon > 0:
        base_date_str = test_dates[-1] if test_dates else (train_dates[-1] if train_dates else "2023-01-01")
        base_dt = pd.to_datetime(base_date_str, errors="coerce")
        if pd.isna(base_dt):
            base_dt = pd.Timestamp.now()
        for i in range(1, int(forecast_horizon) + 1):
            next_dt = base_dt + datetime.timedelta(days=i)
            future_dates.append(next_dt.strftime("%Y-%m-%d"))

    return {
        "method": "Exponential Smoothing",
        "predictions": test_preds,
        "dates": [str(d) for d in test_dates],
        "model_params": model_params,
        "future_dates": future_dates,
        "future_predictions": future_preds,
        "confidence_lower": conf_lower,
        "confidence_upper": conf_upper,
    }


def evaluate_forecast(actual: List[float], predicted: List[float]) -> Dict[str, Any]:
    """
    Calculate forecast accuracy metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - Residuals: list of actual - predicted
    """
    n = min(len(actual), len(predicted))
    if n == 0:
        return {"error": "No overlapping observations to evaluate"}

    act_arr = np.array(actual[:n], dtype=float)
    pred_arr = np.array(predicted[:n], dtype=float)

    diff = act_arr - pred_arr
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "n_observations": int(n),
        "residuals": [round(float(d), 3) for d in diff],
    }


def compare_models(
    df: pd.DataFrame, variable: str, test_ratio: float = 0.2
) -> Dict[str, Any]:
    """
    Perform head-to-head comparison between Seasonal Naive and Holt-Winters
    Exponential Smoothing on the identical train/test split.
    """
    split_info = chronological_split(df, variable, test_ratio=test_ratio)
    if "error" in split_info:
        return split_info

    train_dates = split_info["train_dates"]
    train_values = split_info["train_values"]
    test_dates = split_info["test_dates"]
    test_values = split_info["test_values"]

    # 1. Baseline forecast
    base_res = baseline_forecast(train_dates, train_values, test_dates)
    base_metrics = evaluate_forecast(test_values, base_res["predictions"])

    # 2. Exponential smoothing forecast
    es_res = exponential_smoothing_forecast(train_dates, train_values, test_dates, forecast_horizon=0)
    if "error" in es_res:
        return {"error": f"Exponential smoothing failed during model comparison: {es_res['error']}"}

    es_metrics = evaluate_forecast(test_values, es_res["predictions"])

    # Determine winner based on RMSE
    es_rmse = es_metrics.get("rmse", float("inf"))
    base_rmse = base_metrics.get("rmse", float("inf"))
    winner = "exponential_smoothing" if es_rmse <= base_rmse else "baseline"

    return {
        "split_info": split_info,
        "baseline": {
            "predictions": base_res["predictions"],
            "metrics": base_metrics,
        },
        "es": {
            "predictions": es_res["predictions"],
            "metrics": es_metrics,
            "confidence_lower": es_res.get("confidence_lower", []),
            "confidence_upper": es_res.get("confidence_upper", []),
        },
        "winner": winner,
        "test_dates": test_dates,
        "actual_values": test_values,
        "train_dates": train_dates,
        "train_values": train_values,
    }


def generate_future_forecast(
    df: pd.DataFrame, variable: str, model_type: str, horizon: int = 30
) -> Dict[str, Any]:
    """
    Train model on ALL available data (no test split) and forecast future horizon steps.
    Supports model_type: 'baseline' or 'exponential_smoothing'.
    """
    if df is None or df.empty or variable not in df.columns or "date" not in df.columns:
        return {"error": f"Insufficient data for variable: {variable}"}

    sub = df[["date", variable]].copy()
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub[variable] = pd.to_numeric(sub[variable], errors="coerce")
    sub = sub.dropna().sort_values("date").reset_index(drop=True)

    if len(sub) < 10:
        return {"error": f"Insufficient observations for future forecast (minimum 10 required, got {len(sub)})"}

    historical_dates = [d.strftime("%Y-%m-%d") for d in sub["date"]]
    historical_values = [_to_float(v, 0.0, 3) for v in sub[variable]]
    last_dt = sub["date"].iloc[-1]
    last_observed_date = last_dt.strftime("%Y-%m-%d")

    # Generate future dates
    forecast_dates: List[str] = []
    for i in range(1, int(horizon) + 1):
        f_dt = last_dt + datetime.timedelta(days=i)
        forecast_dates.append(f_dt.strftime("%Y-%m-%d"))

    m_type = model_type.lower()
    if m_type == "baseline":
        doy_map = {}
        for d, v in zip(sub["date"], sub[variable]):
            doy_map[d.dayofyear] = float(v)

        last_val = float(sub[variable].iloc[-1])
        forecast_values: List[float] = []
        for fd_str in forecast_dates:
            dt = pd.to_datetime(fd_str)
            val = doy_map.get(dt.dayofyear, last_val)
            forecast_values.append(round(float(val), 3))

        std_dev = float(np.std(sub[variable].values, ddof=1)) if len(sub) > 1 else 1.0
        ci_margin = 1.96 * std_dev
        conf_lower = [round(float(v - ci_margin), 3) for v in forecast_values]
        conf_upper = [round(float(v + ci_margin), 3) for v in forecast_values]

    else:
        # Default / Exponential Smoothing
        m_type = "exponential_smoothing"
        es_res = exponential_smoothing_forecast(
            train_dates=historical_dates,
            train_values=historical_values,
            test_dates=[],
            forecast_horizon=int(horizon),
        )

        if "error" in es_res:
            return es_res

        forecast_values = es_res.get("future_predictions", [])
        fitted_len = len(forecast_values)

        # Confidence intervals for future predictions
        if es_res.get("confidence_lower") and len(es_res["confidence_lower"]) == fitted_len:
            conf_lower = es_res["confidence_lower"]
            conf_upper = es_res["confidence_upper"]
        else:
            # Derive from training residuals
            train_arr = sub[variable].values.astype(float)
            std_resid = float(np.std(train_arr, ddof=1)) if len(train_arr) > 1 else 1.0
            ci_margin = 1.96 * std_resid
            conf_lower = [round(float(v - ci_margin), 3) for v in forecast_values]
            conf_upper = [round(float(v + ci_margin), 3) for v in forecast_values]

    return {
        "model_type": m_type,
        "variable": variable,
        "horizon": int(horizon),
        "historical_dates": historical_dates,
        "historical_values": historical_values,
        "forecast_dates": forecast_dates,
        "forecast_values": forecast_values,
        "confidence_lower": conf_lower,
        "confidence_upper": conf_upper,
        "last_observed_date": last_observed_date,
    }
