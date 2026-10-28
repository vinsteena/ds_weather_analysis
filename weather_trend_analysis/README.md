# Weather Trend Analysis & Machine Learning Predictive System

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Industry](https://img.shields.io/badge/Industry-Weather%20Broadcast%20Services-orange.svg)]()
[![Course Outcomes](https://img.shields.io/badge/Outcomes-CO3%20|%20CO4%20|%20CO5-purple.svg)]()

A comprehensive **Weather Analytics & Predictive Modeling System** designed for the **Weather Broadcast Services** industry. The system collects, preprocesses, and models 10 years of meteorological time-series data to deliver statistical profiles, seasonal decompositions, monotonic trend identification, anomaly hazard detection, and predictive forecasts for **Humidity, Moisture in the air, Rain, Atmospheric Pressure, and Wind Speed**.

---

## 📌 Integrated Course Outcomes (CO3 – CO5)

| Course Outcome | Description | Implemented Capabilities in this System |
|---|---|---|
| **CO3** | **Data Preprocessing & Exploratory Data Analysis (EDA)** | Ingestion of 10-year ECMWF ERA5 meteorological records (3,653 daily entries); automated physical unit bounds; thermodynamic feature engineering (Vapor Pressure $e$, Saturation Vapor Pressure $e_s$, Dew Point Depression, Diurnal Temperature Range, 24h Barometric Tendency); parametric & non-parametric descriptive statistics; Pearson & Spearman cross-correlation matrices. |
| **CO4** | **Time-Series Modeling & Trend Detection** | Additive and Multiplicative time-series seasonal decomposition ($Y = T + S + R$); Autocorrelation Function (ACF) memory analysis; Mann-Kendall non-parametric monotonic trend test; Sen's slope estimator; OLS decadal slope rate; classification into **Increasing**, **Decreasing**, or **Stable** trends. |
| **CO5** | **Anomaly Detection & Predictive Decision Support** | Multi-method anomaly detection (Seasonal Climatological Z-Score, IQR bounds, Multivariate Isolation Forest); Severe Weather Hazard classification (Heatwaves, Cold Snaps, Torrential Rain, Gale Winds, Barometric Depressions); Two-stage Machine Learning forecasting models for out-of-sample prediction; automated TV/Radio Weather News teleprompter scripts and multi-sector advisories (Agriculture, Aviation, Municipal, Public Health). |

---

## 📊 Dataset Provenance & Meteorological Variables

The benchmark dataset comprises **10 continuous years (2015–2024)** of **ECMWF ERA5 Reanalysis** meteorological data acquired via the Open-Meteo Historical Archive API:

* **Temperature**: Daily Mean, Maximum, and Minimum (°C)
* **Relative Humidity**: Mean saturation percentage (%)
* **Moisture in the Air**: 
  * Dew Point Temperature (°C)
  * Actual Vapor Pressure ($e$ in hPa) via the August-Roche-Magnus thermodynamic formula:
    $$e = 6.112 \times \exp\left(\frac{17.67 \cdot T_{\text{dew}}}{T_{\text{dew}} + 243.5}\right)$$
* **Rainfall / Precipitation**: Daily accumulation (mm) and binary rain occurrence ($\ge 0.1$ mm)
* **Atmospheric Pressure**: Surface barometric pressure (hPa) and 24-hour pressure tendency ($\Delta P_{24h}$)
* **Wind**: Mean wind speed, maximum wind gusts (km/h), and dominant wind direction (°)

---

## 🧠 Machine Learning & Algorithmic Architecture

```
Raw ERA5 Time-Series Data (2015–2024)
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ Preprocessing & Meteorological Feature Derivation (CO3)     │
│ - Physical clipping & spline interpolation                  │
│ - Vapor pressure, Dew point depression, Diurnal range       │
└─────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ Feature Engineering Pipeline (CO4)                          │
│ - Autoregressive Lags (t-1, t-2, t-3, t-7, t-14)            │
│ - Rolling Windows (3d, 7d, 14d, 30d means, stds, maxs)      │
│ - Cyclical Encodings (sin/cos for day-of-year and month)    │
│ - Strict chronological splitting (Zero lookahead leakage)   │
└─────────────────────────────────────────────────────────────┘
                │
                ├───────────────────────────────┐
                ▼                               ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│ Multi-Target ML Regressors    │ │ Two-Stage Rain Predictor      │
│ - Relative Humidity (%)       │ │ Stage 1: Classifier           │
│ - Dew Point (°C)              │ │   P(Rain >= 0.1 mm)           │
│ - Vapor Pressure (hPa)        │ │   ROC-AUC = 0.963             │
│ - Surface Pressure (hPa)      │ │ Stage 2: Regressor            │
│ - Wind Speed (km/h)           │ │   E(Volume | Rain)            │
│ - Temperature (°C)            │ └───────────────────────────────┘
└───────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ Broadcast & Decision Intelligence Layer (CO5)               │
│ - On-Air TV Teleprompter News Script                        │
│ - Multi-Sector Operational Advisories (Agri, Aviation, etc) │
│ - High-Resolution Matplotlib/Seaborn Visualizations         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📈 Model Performance Benchmark (Out-of-Sample Test Set: 2024)

Evaluated chronologically on **366 unseen test days (leap year 2024)**:

| Target Variable | Unit | Machine Learning $R^2$ | ML RMSE | Naive Persistence RMSE | ML Error Reduction |
|---|---|---|---|---|---|
| **Temperature** | °C | **0.9219** | **0.70** | 0.85 | +17.6% |
| **Relative Humidity** | % | **0.9014** | **5.06** | 5.11 | +0.9% |
| **Moisture (Dew Point)** | °C | **0.8633** | **1.27** | 1.33 | +4.5% |
| **Moisture (Vapor Pressure)** | hPa | **0.8830** | **1.34** | 1.40 | +4.5% |
| **Atmospheric Pressure** | hPa | **0.9170** | **0.73** | 0.76 | +4.0% |
| **Wind Speed** | km/h | **0.8043** | **2.00** | 2.08 | +3.8% |

### Rain Prediction (Two-Stage Architecture)
* **Stage 1 (Occurrence Classification)**:
  * Accuracy: **88.8%**
  * ROC-AUC: **0.963**
  * F1-Score: **0.902**
* **Stage 2 (Quantitative Volume)**:
  * Mean Absolute Error: **2.58 mm**
  * Root Mean Squared Error: **4.47 mm**

---

## 📉 Decadal Trend Analysis Findings (2015–2024)

Using the **Mann-Kendall Monotonic Test** and **Sen's Slope Estimator**:

* **Maximum Temperature ($T_{\max}$)**: **Increasing** at $+0.453^\circ\text{C}$ per decade ($p = 0.00206$).
* **Relative Humidity**: **Increasing** at $+4.49\%$ per decade ($p < 0.0001$).
* **Moisture / Dew Point**: **Increasing** at $+1.15^\circ\text{C}$ per decade ($p < 0.0001$).
* **Surface Atmospheric Pressure**: **Decreasing** at $-0.344\text{ hPa}$ per decade ($p = 0.0084$).
* **Wind Speed**: **Increasing** at $+1.48\text{ km/h}$ per decade ($p < 0.0001$).
* **Mean Temperature**: **Stable** within decadal cyclical oscillations ($p = 0.687$).

---

## 🚀 How to Run

### 1. Run Automated Unit Test Suite
```powershell
python -m unittest weather_trend_analysis/tests/test_algorithms.py
```

### 2. Run the Full Analytics & ML Pipeline
```powershell
python weather_trend_analysis/train_and_evaluate.py
```
This single command automatically:
1. Ingests or loads the 10-year dataset.
2. Performs data preprocessing and physics calculations.
3. Computes descriptive statistics and correlations.
4. Executes seasonal decomposition and trend tests.
5. Flags anomalous weather hazards.
6. Trains all ML models and evaluates on the 2024 test set.
7. Saves figures to `weather_trend_analysis/reports/figures/`.
8. Generates the live forecast, TV teleprompter script, and full analytical report.

---

## 📁 Directory Structure

```
weather_trend_analysis/
│
├── data/
│   ├── download_dataset.py           # ERA5 fetcher with offline generator fallback
│   └── weather_analytics_era5_2015_2024.csv  # 10-year historical dataset (3,653 rows)
│
├── analytics/
│   ├── preprocessor.py               # Thermodynamic derivations & cleaning (CO3)
│   ├── statistics.py                 # Descriptive statistics & correlation matrices (CO3)
│   ├── time_series.py                # Seasonal decomposition & ACF (CO4)
│   ├── trends.py                     # Mann-Kendall test & Sen's slope (CO4)
│   ├── anomalies.py                  # Z-score, Isolation Forest & Hazard Alerts (CO5)
│   ├── feature_engineering.py        # Autoregressive lags & temporal features (CO4)
│   └── insights.py                   # TV broadcast scripts & sector advisories (CO5)
│
├── models/
│   ├── weather_predictor.py          # Multi-target ML models & evaluation (CO4-CO5)
│   └── weather_models.joblib         # Serialized trained models & metadata
│
├── visualization/
│   └── plotter.py                    # Matplotlib/Seaborn graphics suite
│
├── reports/
│   ├── WEATHER_ANALYTICS_REPORT.md   # Comprehensive analytical report
│   ├── broadcast_script.txt          # TV / Radio teleprompter script
│   ├── evaluation_metrics.json       # Out-of-sample metrics
│   ├── descriptive_statistics.csv    # Full statistical profile
│   ├── trend_analysis_summary.csv    # Mann-Kendall trend summary
│   └── figures/                      # High-resolution generated charts
│       ├── multi_parameter_trends.png
│       ├── seasonal_decomposition.png
│       ├── monthly_seasonal_boxplots.png
│       ├── correlation_heatmap.png
│       ├── anomalies_and_hazards.png
│       └── predictions_vs_actual_2024.png
│
├── tests/
│   └── test_algorithms.py            # Unit test suite
│
├── train_and_evaluate.py             # Master CLI execution pipeline
└── README.md                         # Documentation & Course Outcome mapping
```
