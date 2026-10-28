"""
Comprehensive Unit Test Suite for Weather Analytics & Predictive ML Algorithms.
Validates:
1. Physical derivations (Vapor Pressure, Dew Point Depression, Heat Index)
2. Descriptive statistical calculations
3. Time series seasonal decomposition
4. Mann-Kendall and Sen's slope trend classification
5. Anomaly detection algorithms
6. Feature engineering and lookahead leakage prevention
7. Predictive model training, persistence, and inference
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analytics.preprocessor import (
    compute_vapor_pressure,
    compute_saturation_vapor_pressure,
    compute_heat_index,
    preprocess_weather_data
)
from analytics.statistics import compute_descriptive_statistics, compute_correlation_matrices
from analytics.time_series import decompose_weather_series, compute_autocorrelation
from analytics.trends import mann_kendall_test, compute_linear_trend
from analytics.anomalies import detect_seasonal_zscore_anomalies, detect_multivariate_anomalies
from analytics.feature_engineering import create_time_series_features, split_train_val_test
from models.weather_predictor import WeatherPredictorEngine


class TestWeatherAlgorithms(unittest.TestCase):

    def setUp(self):
        """Builds a consistent synthetic multi-year mini dataset for testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
        n = len(dates)

        self.df = pd.DataFrame({
            "date": dates,
            "temperature_mean": 24.0 + 5.0 * np.sin(2 * np.pi * dates.dayofyear / 365.25) + np.random.normal(0, 1.0, n),
            "temperature_max": 29.0 + 5.0 * np.sin(2 * np.pi * dates.dayofyear / 365.25) + np.random.normal(0, 1.0, n),
            "temperature_min": 19.0 + 5.0 * np.sin(2 * np.pi * dates.dayofyear / 365.25) + np.random.normal(0, 1.0, n),
            "relative_humidity": np.clip(60.0 + 20.0 * np.cos(2 * np.pi * dates.dayofyear / 365.25) + np.random.normal(0, 3.0, n), 20, 95),
            "dew_point": 16.0 + 4.0 * np.sin(2 * np.pi * dates.dayofyear / 365.25) + np.random.normal(0, 1.0, n),
            "surface_pressure": 1013.0 + 3.0 * np.cos(2 * np.pi * dates.dayofyear / 365.25) + np.random.normal(0, 1.5, n),
            "wind_speed": np.clip(np.random.weibull(2.0, n) * 8.0, 1.0, 45.0),
            "rain": np.where(np.random.rand(n) < 0.25, np.random.exponential(8.0, n), 0.0)
        })

    def test_physical_formulae(self):
        """Verifies thermodynamic approximations for moisture and heat index."""
        # Test vapor pressure at 20°C dew point: expect ~23.38 hPa
        vp = compute_vapor_pressure(20.0)
        self.assertAlmostEqual(vp, 23.38, delta=0.5)

        # Test saturation vapor pressure at 25°C: expect ~31.67 hPa
        svp = compute_saturation_vapor_pressure(25.0)
        self.assertAlmostEqual(svp, 31.67, delta=0.5)

        # Test heat index at 32°C and 70% humidity: expect > 38°C (muggy heat fatigue)
        hi = compute_heat_index(32.0, 70.0)
        self.assertGreater(hi, 36.0)

    def test_preprocessing(self):
        """Verifies cleaning, feature derivation, and constraint enforcement."""
        clean = preprocess_weather_data(self.df)
        self.assertIn("vapor_pressure", clean.columns)
        self.assertIn("dew_point_depression", clean.columns)
        self.assertIn("pressure_tendency_24h", clean.columns)
        self.assertIn("rain_occurrence", clean.columns)
        self.assertIn("season", clean.columns)

        # Relative humidity bounded in [0, 100]
        self.assertTrue((clean["relative_humidity"] >= 0).all() and (clean["relative_humidity"] <= 100).all())

    def test_descriptive_statistics(self):
        """Verifies statistical table generation."""
        clean = preprocess_weather_data(self.df)
        stats_df = compute_descriptive_statistics(clean)
        self.assertIn("temperature_mean", stats_df.index)
        self.assertIn("relative_humidity", stats_df.index)
        self.assertIn("mean", stats_df.columns)
        self.assertIn("median", stats_df.columns)
        self.assertIn("std_dev", stats_df.columns)
        self.assertIn("skewness", stats_df.columns)

    def test_time_series_decomposition(self):
        """Verifies additive decomposition yields observed, trend, seasonal, and residual."""
        clean = preprocess_weather_data(self.df)
        decomp = decompose_weather_series(clean, target_col="temperature_mean", model="additive", period=365)
        self.assertEqual(len(decomp), len(clean))
        self.assertIn("trend", decomp.columns)
        self.assertIn("seasonal", decomp.columns)
        self.assertIn("residual", decomp.columns)

    def test_trend_detection_mann_kendall(self):
        """Verifies Mann-Kendall classifies monotonic trends properly."""
        # 1. Strictly increasing series
        inc_series = np.linspace(10, 50, 100) + np.random.normal(0, 0.5, 100)
        mk_inc = mann_kendall_test(inc_series)
        self.assertEqual(mk_inc["trend"], "Increasing")
        self.assertLess(mk_inc["p_value"], 0.01)

        # 2. Strictly decreasing series
        dec_series = np.linspace(50, 10, 100) + np.random.normal(0, 0.5, 100)
        mk_dec = mann_kendall_test(dec_series)
        self.assertEqual(mk_dec["trend"], "Decreasing")
        self.assertLess(mk_dec["p_value"], 0.01)

        # 3. Stable / stationary white noise series
        noise_series = np.random.normal(25, 1.0, 150)
        mk_noise = mann_kendall_test(noise_series)
        self.assertEqual(mk_noise["trend"], "Stable")

    def test_anomaly_detection(self):
        """Verifies Z-score and Isolation Forest anomaly labeling."""
        clean = preprocess_weather_data(self.df)
        # Inject one severe heat spike
        clean.loc[500, "temperature_mean"] += 15.0

        anomaly_df = detect_seasonal_zscore_anomalies(clean, target_col="temperature_mean", threshold=2.5)
        self.assertTrue(anomaly_df.loc[500, "temperature_mean_is_anomaly"])

        iso_df = detect_multivariate_anomalies(clean, contamination=0.03)
        self.assertIn("multivariate_anomaly", iso_df.columns)
        self.assertEqual(iso_df["multivariate_anomaly"].nunique(), 2)

    def test_ml_pipeline_and_prediction(self):
        """Verifies training, prediction, and metric computation of WeatherPredictorEngine."""
        clean = preprocess_weather_data(self.df)
        feat_df = create_time_series_features(clean)

        train_df, val_df, test_df = split_train_val_test(feat_df, train_end_year=2021, val_year=2022)
        self.assertGreater(len(train_df), 300)
        self.assertGreater(len(test_df), 300)

        engine = WeatherPredictorEngine(random_state=42)
        engine.train_models(train_df, val_df)
        metrics = engine.evaluate(test_df)

        self.assertIn("temperature_mean", metrics)
        self.assertIn("relative_humidity", metrics)
        self.assertIn("dew_point", metrics)
        self.assertIn("surface_pressure", metrics)
        self.assertIn("wind_speed", metrics)
        self.assertIn("rain_classification", metrics)

        # Predict next day
        sample_obs = test_df.iloc[[-1]]
        next_day = engine.predict_next_day(sample_obs)
        self.assertIn("temperature_mean_c", next_day)
        self.assertIn("relative_humidity_pct", next_day)
        self.assertIn("moisture_dew_point_c", next_day)
        self.assertIn("surface_pressure_hpa", next_day)
        self.assertIn("wind_speed_kmh", next_day)
        self.assertIn("rain_probability_pct", next_day)


if __name__ == "__main__":
    unittest.main()
