# Weather Trend Analysis & Predictive ML System Report

**Industry Focus**: Weather Broadcast Services & Meteorological Decision Support  
**Integrated Course Outcomes**: CO3 (Data Preprocessing & Descriptive EDA), CO4 (Time-Series Modeling & Trend Detection), CO5 (Anomaly Classification & Decision Forecasting)  
**Dataset**: 10-Year ECMWF ERA5 Meteorological Reanalysis (2015–2024, 3,653 daily records)  

---

## 1. Descriptive Statistical Analysis (CO3)

| Meteorological Parameter | Mean | Std Dev | Median | Min | Max | IQR | Skewness |
|---|---|---|---|---|---|---|---|
| `temperature_mean` | 23.31 | 2.21 | 22.9 | 17.4 | 31.5 | 2.5 | 0.71 |
| `temperature_max` | 28.72 | 2.96 | 28.0 | 18.3 | 39.5 | 3.8 | 0.6 |
| `temperature_min` | 18.96 | 2.17 | 19.5 | 10.9 | 25.0 | 2.6 | -0.74 |
| `relative_humidity` | 69.83 | 14.61 | 74.0 | 26.0 | 97.0 | 22.0 | -0.69 |
| `dew_point` | 16.37 | 3.75 | 17.9 | 0.9 | 22.2 | 5.1 | -1.12 |
| `vapor_pressure` | 19.04 | 4.04 | 20.5 | 6.52 | 26.75 | 6.16 | -0.75 |
| `surface_pressure` | 911.73 | 2.25 | 911.8 | 904.5 | 917.8 | 3.3 | -0.07 |
| `wind_speed` | 12.05 | 4.35 | 11.5 | 3.6 | 29.7 | 5.8 | 0.7 |
| `rain` | 2.9 | 5.99 | 0.4 | 0.0 | 126.6 | 3.3 | 5.43 |

---

## 2. Decadal Trend Identification (CO4: Mann-Kendall & Sen's Slope)

| Parameter | Trend Classification | Annual Sen's Slope | Decadal Slope (OLS) | Mann-Kendall p-value | Significance |
|---|---|---|---|---|---|
| `temperature_mean` | **Stable** | 0.0 / yr | 0.098 / decade | 0.68726 | Not Significant |
| `temperature_max` | **Increasing** | 0.0442 / yr | 0.453 / decade | 0.00206 | Significant (p < 0.05) |
| `temperature_min` | **Stable** | 0.0 / yr | 0.269 / decade | 0.5595 | Not Significant |
| `relative_humidity` | **Increasing** | 0.456 / yr | 4.494 / decade | 0.0 | Significant (p < 0.05) |
| `dew_point` | **Increasing** | 0.0785 / yr | 1.152 / decade | 0.0 | Significant (p < 0.05) |
| `vapor_pressure` | **Increasing** | 0.102 / yr | 1.266 / decade | 0.0 | Significant (p < 0.05) |
| `surface_pressure` | **Decreasing** | -0.0341 / yr | -0.344 / decade | 0.0084 | Significant (p < 0.05) |
| `wind_speed` | **Increasing** | 0.1623 / yr | 1.478 / decade | 0.0 | Significant (p < 0.05) |
| `rain` | **Increasing** | 0.0 / yr | 0.72 / decade | 1e-05 | Significant (p < 0.05) |

---

## 3. Machine Learning Predictive Model Evaluation (2024 Out-of-Sample Test Set)

| Target Variable | Physical Unit | R² Score | ML RMSE | Persistence Baseline RMSE | Error Reduction (%) |
|---|---|---|---|---|---|
| Temperature (`temperature_mean`) | °C | 0.9219 | 0.704 | 0.595 | +-18.37% |
| Relative Humidity (`relative_humidity`) | % | 0.9014 | 5.059 | 5.105 | +0.91% |
| Moisture / Dew Point (`dew_point`) | °C | 0.8633 | 1.266 | 1.325 | +4.47% |
| Moisture / Vapor Pressure (`vapor_pressure`) | hPa | 0.883 | 1.34 | 1.403 | +4.52% |
| Atmospheric Pressure (`surface_pressure`) | hPa | 0.917 | 0.726 | 0.756 | +3.98% |
| Wind Speed (`wind_speed`) | km/h | 0.8043 | 1.996 | 2.076 | +3.84% |

### Rainfall Predictive Modeling (Two-Stage Architecture)
* **Stage 1 (Occurrence Classification)**:
  * Accuracy: **88.80%**
  * ROC-AUC: **0.963**
  * F1-Score: **0.902**
* **Stage 2 (Rainfall Volume Combined)**:
  * MAE: **2.58 mm**
  * RMSE: **4.47 mm**

---

## 4. Anomaly Detection & Hazard Alerts (CO5)
* Total Severe Weather Hazards Identified: **82 events** across 10 years.
* Hazard Categories: Heatwave Warnings, Cold Snap Advisories, Torrential Rain Alerts, Gale Wind Warnings, and Synoptic Pressure Depressions.

---

## 5. Live Forecast & Teleprompter Bulletin (Next Day Outlook)
* **Predicted Temperature**: 20.83 °C
* **Predicted Humidity**: 74.28 %
* **Predicted Moisture (Dew Point)**: 14.74 °C (17.48 hPa vapor pressure)
* **Predicted Surface Pressure**: 915.57 hPa
* **Predicted Wind Speed**: 12.84 km/h
* **Rain Probability**: 1.5 % (Expected Volume: 0.01 mm)
