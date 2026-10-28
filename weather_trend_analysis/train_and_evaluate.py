"""
Master Pipeline Runner for Weather Trend Analysis & Machine Learning Forecasting.
Executes end-to-end:
1. Dataset Acquisition (10-Year ECMWF ERA5)
2. Preprocessing & Meteorological Feature Derivation (CO3)
3. Descriptive Statistics & Correlation Analysis (CO3)
4. Time Series Seasonal Decomposition & ACF (CO4)
5. Mann-Kendall Trend Identification (CO4)
6. Extreme Anomaly & Hazard Detection (CO5)
7. Chronological ML Model Training & Test Evaluation (CO4-CO5)
8. Visualizations Generation (Matplotlib/Seaborn)
9. Broadcast Bulletin Script & Operational Advisories
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure root directory is on python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from data.download_dataset import acquire_dataset
from analytics.preprocessor import preprocess_weather_data
from analytics.feature_engineering import create_time_series_features, split_train_val_test
from analytics.statistics import compute_descriptive_statistics, compute_correlation_matrices, compute_seasonal_breakdown
from analytics.time_series import decompose_weather_series, compute_autocorrelation, aggregate_temporal_patterns
from analytics.trends import analyze_all_trends
from analytics.anomalies import run_full_anomaly_pipeline
from analytics.insights import generate_broadcast_script, generate_sector_advisory
from models.weather_predictor import WeatherPredictorEngine
from visualization.plotter import (
    plot_multi_parameter_trends,
    plot_seasonal_decomposition,
    plot_monthly_boxplots,
    plot_correlation_heatmap,
    plot_anomalies_and_hazards,
    plot_predictions_vs_actual
)

REPORTS_DIR = os.path.join(CURRENT_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def run_pipeline():
    print("=" * 80)
    print("      WEATHER TREND ANALYSIS & PREDICTIVE MODEL TRAINING SYSTEM")
    print("=" * 80)

    # 1. Dataset Ingestion
    print("\n[STEP 1] Ingesting 10-Year Meteorological Dataset (2015–2024)...")
    raw_df = acquire_dataset(force_download=False)
    print(f"Loaded {len(raw_df)} daily records from {raw_df['date'].min()} to {raw_df['date'].max()}")

    # 2. Preprocessing & Physical Feature Calculation (CO3)
    print("\n[STEP 2] Preprocessing & Deriving Atmospheric Physical Metrics...")
    clean_df = preprocess_weather_data(raw_df)
    print(f"Enriched dataset shape: {clean_df.shape}")
    print("Derived: Vapor Pressure (Moisture), Saturation Vapor Pressure, Dew Point Depression, Pressure Tendency, Diurnal Range, Rain Occurrence")

    # 3. Descriptive Statistical Analysis (CO3)
    print("\n[STEP 3] Performing Descriptive Statistical Analysis...")
    stats_df = compute_descriptive_statistics(clean_df)
    print(stats_df[["mean", "std_dev", "median", "min", "max", "iqr", "skewness"]])
    stats_df.to_csv(os.path.join(REPORTS_DIR, "descriptive_statistics.csv"))

    pearson_corr, spearman_corr = compute_correlation_matrices(clean_df)
    pearson_corr.to_csv(os.path.join(REPORTS_DIR, "pearson_correlation.csv"))
    print("\nTop Correlations with Rain:")
    if "rain" in pearson_corr.columns:
        print(pearson_corr["rain"].sort_values(ascending=False))

    # 4. Time Series Seasonal Decomposition (CO4)
    print("\n[STEP 4] Executing Time Series Seasonal Decomposition...")
    decomp_temp = decompose_weather_series(clean_df, target_col="temperature_mean", model="additive")
    decomp_temp.to_csv(os.path.join(REPORTS_DIR, "temperature_decomposition.csv"))

    daily_normals, monthly_normals, annual_summary = aggregate_temporal_patterns(clean_df)
    annual_summary.to_csv(os.path.join(REPORTS_DIR, "annual_climate_summary.csv"))
    print("\nAnnual Climate Summary (Last 5 Years):")
    print(annual_summary.tail(5))

    # 5. Trend Analysis (Mann-Kendall & Sen's Slope) (CO4)
    print("\n[STEP 5] Computing Monotonic Trends & Statistical Significance (Mann-Kendall)...")
    trends_df = analyze_all_trends(clean_df)
    print(trends_df[["trend_status", "sens_slope_annual", "ols_slope_decade", "mk_p_value"]])
    trends_df.to_csv(os.path.join(REPORTS_DIR, "trend_analysis_summary.csv"))

    # 6. Anomaly & Severe Weather Hazard Detection (CO5)
    print("\n[STEP 6] Detecting Seasonal Climatological Anomalies & Extreme Hazards...")
    anomaly_df = run_full_anomaly_pipeline(clean_df)
    total_hazards = (anomaly_df["has_hazard"] == 1).sum()
    print(f"Detected {total_hazards} severe weather hazard events across 10 years.")
    hazard_subset = anomaly_df[anomaly_df["has_hazard"] == 1]
    hazard_subset[["date", "temperature_mean", "rain", "wind_speed", "surface_pressure", "hazard_status"]].tail(5).to_csv(
        os.path.join(REPORTS_DIR, "recent_hazards.csv"), index=False
    )

    # 7. Time-Series Feature Engineering for Machine Learning (CO4)
    print("\n[STEP 7] Constructing Predictive Time-Series Features (Lags, Rolling, Cyclical)...")
    feature_df = create_time_series_features(clean_df)
    print(f"Feature matrix constructed: {feature_df.shape[0]} rows, {feature_df.shape[1]} columns")

    train_df, val_df, test_df = split_train_val_test(feature_df, train_end_year=2022, val_year=2023)
    print(f"Chronological Splits -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # 8. Train Machine Learning Models (CO4 - CO5)
    print("\n[STEP 8] Training Machine Learning Predictors (Humidity, Moisture, Pressure, Wind, Rain)...")
    predictor = WeatherPredictorEngine(random_state=42)
    predictor.train_models(train_df, val_df)
    predictor.save()

    # 9. Out-of-Sample Evaluation on 2024 Test Set
    print("\n[STEP 9] Evaluating Models on Unseen 2024 Test Set...")
    eval_metrics = predictor.evaluate(test_df)
    with open(os.path.join(REPORTS_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2)

    print("\n=== Model Performance Benchmark (Test Year 2024) ===")
    for target, stats in eval_metrics.items():
        if target == "rain_classification":
            print(f"Rain Classifier: Acc={stats['accuracy']:.3f}, F1={stats['f1_score']:.3f}, ROC-AUC={stats['roc_auc']:.3f}")
        elif target == "rain_combined":
            print(f"Rain Volume (Combined): RMSE={stats['rmse']:.2f} mm, MAE={stats['mae']:.2f} mm")
        else:
            print(f"{target.upper()} ({stats['unit']}): R² = {stats['r2_score']:.4f} | RMSE = {stats['rmse']:.2f} (Baseline RMSE: {stats['baseline_rmse']}) | Improvement = +{stats['rmse_improvement_pct']}%")

    # 10. Generate Visualizations (Matplotlib / Seaborn)
    print("\n[STEP 10] Generating High-Resolution Matplotlib/Seaborn Graphics...")
    plot_multi_parameter_trends(clean_df)
    plot_seasonal_decomposition(decomp_temp, target_name="Temperature (°C)")
    plot_monthly_boxplots(clean_df)
    plot_correlation_heatmap(pearson_corr)
    plot_anomalies_and_hazards(anomaly_df)

    # Generate 2024 predicted test dictionary for visual comparison
    X_test = test_df[predictor.feature_cols]
    pred_dict = {
        "temperature_mean": predictor.models["temperature_mean"].predict(X_test),
        "relative_humidity": predictor.models["relative_humidity"].predict(X_test),
        "dew_point": predictor.models["dew_point"].predict(X_test),
        "surface_pressure": predictor.models["surface_pressure"].predict(X_test),
        "wind_speed": predictor.models["wind_speed"].predict(X_test),
        "rain": predictor.models["rain_occurrence"].predict_proba(X_test)[:, 1] * np.clip(predictor.models["rain_volume"].predict(X_test), 0, None)
    }
    plot_predictions_vs_actual(test_df, pred_dict)

    # 11. Next-Day Live Forecast & Broadcast Script Generation (CO5)
    print("\n[STEP 11] Generating Live Next-Day Forecast & Broadcast Script...")
    latest_row = test_df.iloc[[-1]]
    next_day_forecast = predictor.predict_next_day(latest_row)
    latest_hazard = anomaly_df.iloc[-1].get("hazard_status", "Normal Conditions")

    broadcast_script = generate_broadcast_script(next_day_forecast, active_hazards=latest_hazard)
    sector_advisory = generate_sector_advisory(next_day_forecast)

    with open(os.path.join(REPORTS_DIR, "broadcast_script.txt"), "w", encoding="utf-8") as f:
        f.write(broadcast_script)

    print("\n" + broadcast_script)
    print("\nSector Advisories for Decision-Makers:")
    for sector, advisory in sector_advisory.items():
        print(f"  [{sector.upper()}]: {advisory}")

    # 12. Save Comprehensive Markdown Summary Report
    report_md_path = os.path.join(REPORTS_DIR, "WEATHER_ANALYTICS_REPORT.md")
    generate_markdown_report(report_md_path, stats_df, trends_df, eval_metrics, next_day_forecast, total_hazards)
    print(f"\nComprehensive Project Report generated at: {report_md_path}")
    print("\nPipeline execution complete successfully!")


def generate_markdown_report(report_path, stats_df, trends_df, eval_metrics, forecast, total_hazards):
    """Writes an academic and industrial project summary report."""
    md_content = f"""# Weather Trend Analysis & Predictive ML System Report

**Industry Focus**: Weather Broadcast Services & Meteorological Decision Support  
**Integrated Course Outcomes**: CO3 (Data Preprocessing & Descriptive EDA), CO4 (Time-Series Modeling & Trend Detection), CO5 (Anomaly Classification & Decision Forecasting)  
**Dataset**: 10-Year ECMWF ERA5 Meteorological Reanalysis (2015–2024, 3,653 daily records)  

---

## 1. Descriptive Statistical Analysis (CO3)

| Meteorological Parameter | Mean | Std Dev | Median | Min | Max | IQR | Skewness |
|---|---|---|---|---|---|---|---|
"""
    for param, row in stats_df.iterrows():
        md_content += f"| `{param}` | {row['mean']} | {row['std_dev']} | {row['median']} | {row['min']} | {row['max']} | {row['iqr']} | {row['skewness']} |\n"

    md_content += f"""
---

## 2. Decadal Trend Identification (CO4: Mann-Kendall & Sen's Slope)

| Parameter | Trend Classification | Annual Sen's Slope | Decadal Slope (OLS) | Mann-Kendall p-value | Significance |
|---|---|---|---|---|---|
"""
    for param, row in trends_df.iterrows():
        sig = "Significant (p < 0.05)" if row["mk_p_value"] < 0.05 else "Not Significant"
        md_content += f"| `{param}` | **{row['trend_status']}** | {row['sens_slope_annual']} / yr | {row['ols_slope_decade']} / decade | {row['mk_p_value']} | {sig} |\n"

    md_content += f"""
---

## 3. Machine Learning Predictive Model Evaluation (2024 Out-of-Sample Test Set)

| Target Variable | Physical Unit | R² Score | ML RMSE | Persistence Baseline RMSE | Error Reduction (%) |
|---|---|---|---|---|---|
| Temperature (`temperature_mean`) | °C | {eval_metrics['temperature_mean']['r2_score']} | {eval_metrics['temperature_mean']['rmse']} | {eval_metrics['temperature_mean']['baseline_rmse']} | +{eval_metrics['temperature_mean']['rmse_improvement_pct']}% |
| Relative Humidity (`relative_humidity`) | % | {eval_metrics['relative_humidity']['r2_score']} | {eval_metrics['relative_humidity']['rmse']} | {eval_metrics['relative_humidity']['baseline_rmse']} | +{eval_metrics['relative_humidity']['rmse_improvement_pct']}% |
| Moisture / Dew Point (`dew_point`) | °C | {eval_metrics['dew_point']['r2_score']} | {eval_metrics['dew_point']['rmse']} | {eval_metrics['dew_point']['baseline_rmse']} | +{eval_metrics['dew_point']['rmse_improvement_pct']}% |
| Moisture / Vapor Pressure (`vapor_pressure`) | hPa | {eval_metrics['vapor_pressure']['r2_score']} | {eval_metrics['vapor_pressure']['rmse']} | {eval_metrics['vapor_pressure']['baseline_rmse']} | +{eval_metrics['vapor_pressure']['rmse_improvement_pct']}% |
| Atmospheric Pressure (`surface_pressure`) | hPa | {eval_metrics['surface_pressure']['r2_score']} | {eval_metrics['surface_pressure']['rmse']} | {eval_metrics['surface_pressure']['baseline_rmse']} | +{eval_metrics['surface_pressure']['rmse_improvement_pct']}% |
| Wind Speed (`wind_speed`) | km/h | {eval_metrics['wind_speed']['r2_score']} | {eval_metrics['wind_speed']['rmse']} | {eval_metrics['wind_speed']['baseline_rmse']} | +{eval_metrics['wind_speed']['rmse_improvement_pct']}% |

### Rainfall Predictive Modeling (Two-Stage Architecture)
* **Stage 1 (Occurrence Classification)**:
  * Accuracy: **{eval_metrics['rain_classification']['accuracy'] * 100:.2f}%**
  * ROC-AUC: **{eval_metrics['rain_classification']['roc_auc']:.3f}**
  * F1-Score: **{eval_metrics['rain_classification']['f1_score']:.3f}**
* **Stage 2 (Rainfall Volume Combined)**:
  * MAE: **{eval_metrics['rain_combined']['mae']:.2f} mm**
  * RMSE: **{eval_metrics['rain_combined']['rmse']:.2f} mm**

---

## 4. Anomaly Detection & Hazard Alerts (CO5)
* Total Severe Weather Hazards Identified: **{total_hazards} events** across 10 years.
* Hazard Categories: Heatwave Warnings, Cold Snap Advisories, Torrential Rain Alerts, Gale Wind Warnings, and Synoptic Pressure Depressions.

---

## 5. Live Forecast & Teleprompter Bulletin (Next Day Outlook)
* **Predicted Temperature**: {forecast['temperature_mean_c']} °C
* **Predicted Humidity**: {forecast['relative_humidity_pct']} %
* **Predicted Moisture (Dew Point)**: {forecast['moisture_dew_point_c']} °C ({forecast['moisture_vapor_pressure_hpa']} hPa vapor pressure)
* **Predicted Surface Pressure**: {forecast['surface_pressure_hpa']} hPa
* **Predicted Wind Speed**: {forecast['wind_speed_kmh']} km/h
* **Rain Probability**: {forecast['rain_probability_pct']} % (Expected Volume: {forecast['expected_rain_mm']} mm)
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)


if __name__ == "__main__":
    run_pipeline()
