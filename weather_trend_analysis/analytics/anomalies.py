"""
Weather Anomaly and Extreme Condition Detection Module (CO5).
Implements:
1. Rolling Seasonal Z-Score Anomaly Detection
2. Seasonal Interquartile Range (IQR) Outlier Boundaries
3. Multivariate Isolation Forest Compound Anomaly Detector
4. Weather Broadcast Hazard Classifier (Heatwaves, Cold Snaps, Storms, Barometric Drops)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_seasonal_zscore_anomalies(df: pd.DataFrame, target_col="temperature_mean", threshold=2.5, window=15):
    """
    Computes climatological anomaly by comparing each observation to its
    day-of-year normal (computed using a window of +- window days across all historical years).
    """
    data = df.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
        data["day_of_year"] = data["date"].dt.dayofyear

    # Compute climatological daily mean and std
    daily_stats = data.groupby("day_of_year")[target_col].agg(["mean", "std"]).reset_index()

    # Smooth the 366-day cycle using rolling circular smoothing
    smooth_mean = pd.concat([daily_stats["mean"], daily_stats["mean"], daily_stats["mean"]]).rolling(window, center=True).mean()
    smooth_std = pd.concat([daily_stats["std"], daily_stats["std"], daily_stats["std"]]).rolling(window, center=True).mean()

    n = len(daily_stats)
    daily_stats["clim_mean"] = smooth_mean.iloc[n:2*n].values
    daily_stats["clim_std"] = smooth_std.iloc[n:2*n].values

    # Merge back to original data
    merged = pd.merge(data, daily_stats[["day_of_year", "clim_mean", "clim_std"]], on="day_of_year", how="left")
    merged["clim_std"] = merged["clim_std"].replace(0, 1e-6)

    merged[f"{target_col}_anomaly_value"] = merged[target_col] - merged["clim_mean"]
    merged[f"{target_col}_zscore"] = merged[f"{target_col}_anomaly_value"] / merged["clim_std"]
    merged[f"{target_col}_is_anomaly"] = merged[f"{target_col}_zscore"].abs() >= threshold

    return merged


def detect_multivariate_anomalies(df: pd.DataFrame, contamination=0.03, random_state=42):
    """
    Detects compound extreme weather anomalies using Isolation Forest across:
    Temperature, Humidity, Moisture (Dew Point), Pressure, Wind, and Rain.
    """
    features = [
        "temperature_mean",
        "relative_humidity",
        "dew_point",
        "surface_pressure",
        "wind_speed",
        "rain"
    ]
    valid_features = [f for f in features if f in df.columns]

    data = df.copy()
    X = data[valid_features].ffill().bfill()

    iso = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=100)
    preds = iso.fit_predict(X)  # -1 for anomaly, 1 for normal
    scores = iso.decision_function(X)

    data["multivariate_anomaly"] = (preds == -1).astype(int)
    data["anomaly_score"] = np.round(-scores, 4)  # higher = more anomalous

    return data


def classify_severe_weather_hazards(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifies specific meteorological hazards and anomalies for broadcast alerts:
    - Heatwave
    - Cold Snap
    - Flash Rain / Cloudburst
    - High Wind / Gale Alert
    - Barometric Depression (Low Pressure Storm)
    """
    data = df.copy()
    hazards = []

    for _, row in data.iterrows():
        active_alerts = []

        # Temperature extremes
        t_mean = row.get("temperature_mean", np.nan)
        t_max = row.get("temperature_max", t_mean)
        t_min = row.get("temperature_min", t_mean)
        t_z = row.get("temperature_mean_zscore", 0.0)

        if (t_max >= 37.0 and t_z >= 2.0) or t_z >= 3.0:
            active_alerts.append("Heatwave Warning")
        elif (t_min <= 13.0 and t_z <= -2.0) or t_z <= -3.0:
            active_alerts.append("Cold Snap Advisory")

        # Heavy Precipitation
        rain_val = row.get("rain", row.get("precipitation", 0.0))
        if rain_val >= 50.0:
            active_alerts.append("Torrential Rain / Flood Advisory")
        elif rain_val >= 25.0:
            active_alerts.append("Heavy Rainfall Alert")

        # Wind Speed
        wind_val = row.get("wind_gust_max", row.get("wind_speed", 0.0))
        if wind_val >= 40.0:
            active_alerts.append("Gale / Strong Wind Warning")

        # Barometric Plunge (Synoptic Storm / Cyclone)
        p_tend = row.get("pressure_tendency_24h", row.get("pressure_tendency_1d", 0.0))
        p_surf = row.get("surface_pressure", np.nan)
        mean_p = data["surface_pressure"].mean() if "surface_pressure" in data.columns else 1013.0
        if p_tend <= -4.0 or (not np.isnan(p_surf) and p_surf <= mean_p - 5.5):
            active_alerts.append("Severe Low Pressure Depression")

        hazards.append("; ".join(active_alerts) if active_alerts else "Normal Conditions")

    data["hazard_status"] = hazards
    data["has_hazard"] = (data["hazard_status"] != "Normal Conditions").astype(int)
    return data


def run_full_anomaly_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs univariate Z-score, multivariate Isolation Forest, and hazard classification.
    """
    res = detect_seasonal_zscore_anomalies(df, target_col="temperature_mean", threshold=2.5)
    res = detect_multivariate_anomalies(res, contamination=0.03)
    res = classify_severe_weather_hazards(res)
    return res


if __name__ == "__main__":
    import os
    from weather_trend_analysis.analytics.preprocessor import preprocess_weather_data

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "weather_analytics_era5_2015_2024.csv")
    raw_df = pd.read_csv(data_path)
    clean_df = preprocess_weather_data(raw_df)

    anomaly_df = run_full_anomaly_pipeline(clean_df)
    hazard_events = anomaly_df[anomaly_df["has_hazard"] == 1]
    print(f"Total anomaly hazard events identified: {len(hazard_events)} / {len(anomaly_df)}")
    print("\nSample Identified Hazards:")
    print(hazard_events[["date", "temperature_mean", "rain", "wind_speed", "surface_pressure", "hazard_status"]].head(10))
