"""
Comprehensive Unit Tests for WeatherWise Analysis Modules
=========================================================
Tests all 6 modules:
1. analysis.py
2. data_quality.py
3. anomaly_detection.py
4. forecasting.py
5. scenario_engine.py
6. weather_dna.py
"""

import json
import unittest
import numpy as np
import pandas as pd

import analysis
import anomaly_detection
import data_quality
import forecasting
import scenario_engine
import weather_dna


class TestWeatherWiseModules(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_df = analysis.load_dataframe("data/sample_weather.csv")
        cls.empty_df = pd.DataFrame()
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        cls.small_df = pd.DataFrame({
            "date": dates,
            "location": "Testville",
            "temperature": [10.0, 12.0, 11.5, 13.0, 14.2],
            "humidity": [80.0, 75.0, 78.0, 70.0, 68.0],
            "rainfall": [0.0, 2.5, 0.0, 1.2, 0.0],
            "wind_speed": [12.0, 15.0, 14.0, 10.0, 11.0],
            "pressure": [1013.0, 1012.0, 1014.0, 1015.0, 1013.5],
        })

    # =========================================================================
    # FILE 1: analysis.py tests
    # =========================================================================

    def test_load_dataframe(self):
        self.assertIsInstance(self.sample_df, pd.DataFrame)
        self.assertFalse(self.sample_df.empty)
        self.assertIn("date", self.sample_df.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.sample_df["date"]))

    def test_get_available_variables(self):
        vars_available = analysis.get_available_variables(self.sample_df)
        self.assertIsInstance(vars_available, list)
        self.assertIn("temperature", vars_available)
        self.assertIn("humidity", vars_available)

        empty_vars = analysis.get_available_variables(self.empty_df)
        self.assertEqual(empty_vars, [])

    def test_get_date_range(self):
        dr = analysis.get_date_range(self.sample_df)
        self.assertIn("start", dr)
        self.assertIn("end", dr)
        self.assertGreaterEqual(dr["total_days"], dr["available_obs"])
        json.dumps(dr)

        err = analysis.get_date_range(self.empty_df)
        self.assertIn("error", err)
        json.dumps(err)

    def test_get_descriptive_stats(self):
        stats = analysis.get_descriptive_stats(self.sample_df, "temperature")
        self.assertIn("count", stats)
        self.assertIn("mean", stats)
        self.assertIn("median", stats)
        self.assertIn("std", stats)
        self.assertIn("skewness", stats)
        self.assertIn("kurtosis", stats)
        self.assertIsInstance(stats["mean"], float)
        json.dumps(stats)

        err = analysis.get_descriptive_stats(self.sample_df, "nonexistent")
        self.assertIn("error", err)
        json.dumps(err)

        err_empty = analysis.get_descriptive_stats(self.empty_df, "temperature")
        self.assertIn("error", err_empty)

    def test_compute_trend(self):
        trend = analysis.compute_trend(self.sample_df, "temperature")
        self.assertIn("slope", trend)
        self.assertIn("trend_direction", trend)
        self.assertIn(trend["trend_direction"], ["increasing", "decreasing", "flat"])
        self.assertIn("change_per_year", trend)
        self.assertIn("dates", trend)
        self.assertIn("values", trend)
        self.assertIn("trend_line", trend)
        json.dumps(trend)

        err = analysis.compute_trend(self.small_df, "temperature")
        self.assertIn("error", err)
        json.dumps(err)

    def test_seasonal_aggregation(self):
        res = analysis.seasonal_aggregation(self.sample_df, "rainfall")
        self.assertIn("monthly_avg", res)
        self.assertIn("monthly_total", res)
        self.assertIn("seasonal_avg", res)
        self.assertIn("Winter", res["seasonal_avg"])
        self.assertIn("wettest_month", res)
        json.dumps(res)

    def test_year_over_year_comparison(self):
        yoy = analysis.year_over_year_comparison(self.sample_df, "temperature", 2022, 2023)
        self.assertIn("difference", yoy)
        self.assertIn("mean_difference", yoy["difference"])
        self.assertIn("shared_months", yoy)
        json.dumps(yoy)

        err = analysis.year_over_year_comparison(self.sample_df, "temperature", 1990, 2023)
        self.assertIn("error", err)

    def test_compute_decomposition(self):
        decomp = analysis.compute_decomposition(self.sample_df, "temperature")
        self.assertIn("observed", decomp)
        self.assertIn("trend", decomp)
        self.assertIn("seasonal", decomp)
        self.assertIn("residual", decomp)
        self.assertIn("period_used", decomp)
        json.dumps(decomp)

    def test_get_monthly_summary(self):
        summary = analysis.get_monthly_summary(self.sample_df)
        self.assertGreater(len(summary), 0)
        first = summary[0]
        self.assertIn("year_month", first)
        self.assertIn("temperature", first)
        json.dumps(summary)

        # Check dict-like behavior of MonthlySummaryList
        self.assertIsNone(summary.get("error"))
        self.assertIsInstance(summary, list)

        err = analysis.get_monthly_summary(pd.DataFrame({"date": [pd.to_datetime("2023-01-01")]}))
        self.assertIn("error", err)

    # =========================================================================
    # FILE 2: data_quality.py tests
    # =========================================================================

    def test_validate_and_clean(self):
        cleaned, report = data_quality.validate_and_clean(self.sample_df)
        self.assertIsInstance(cleaned, pd.DataFrame)
        self.assertIn("original_rows", report)
        self.assertIn("cleaned_rows", report)
        self.assertIn("operations_log", report)
        json.dumps(report)

    def test_compute_trust_score(self):
        cleaned, _ = data_quality.validate_and_clean(self.sample_df)
        score = data_quality.compute_trust_score(self.sample_df, cleaned)
        self.assertTrue(0 <= score["overall_score"] <= 100)
        self.assertIn("components", score)
        self.assertIn("recommendations", score)
        json.dumps(score)

    def test_identify_date_gaps(self):
        cleaned, _ = data_quality.validate_and_clean(self.sample_df)
        gaps = data_quality.identify_date_gaps(cleaned)
        self.assertIsInstance(gaps, list)
        if gaps:
            self.assertIn("gap_start", gaps[0])
            self.assertIn("missing_days", gaps[0])
        json.dumps(gaps)

    def test_detect_suspicious_values(self):
        suspicious = data_quality.detect_suspicious_values(self.sample_df)
        self.assertIsInstance(suspicious, dict)
        json.dumps(suspicious)

    # =========================================================================
    # FILE 3: anomaly_detection.py tests
    # =========================================================================

    def test_detect_iqr_anomalies(self):
        res = anomaly_detection.detect_iqr_anomalies(self.sample_df, "temperature")
        self.assertEqual(res["method"], "iqr")
        self.assertIn("q1", res)
        self.assertIn("q3", res)
        self.assertIn("iqr", res)
        self.assertIn("anomalies", res)
        self.assertEqual(len(res["all_dates"]), len(res["all_values"]))
        self.assertEqual(len(res["all_values"]), len(res["is_anomaly"]))
        json.dumps(res)

    def test_detect_zscore_anomalies(self):
        res = anomaly_detection.detect_zscore_anomalies(self.sample_df, "temperature")
        self.assertEqual(res["method"], "zscore")
        self.assertIn("mean", res)
        self.assertIn("std", res)
        self.assertIn("anomalies", res)
        json.dumps(res)

    def test_detect_rolling_anomalies(self):
        res = anomaly_detection.detect_rolling_anomalies(self.sample_df, "temperature", window=14, sigma=2.0)
        self.assertEqual(res["method"], "rolling")
        self.assertIn("rolling_means", res)
        json.dumps(res)

    def test_get_investigation_report(self):
        first_date = self.sample_df["date"].dropna().iloc[10].strftime("%Y-%m-%d")
        rep = anomaly_detection.get_investigation_report(self.sample_df, first_date, "temperature")
        self.assertTrue(rep["data_available"])
        self.assertIn("monthly_comparison", rep)
        self.assertIn("flagged_by", rep)
        self.assertIn("explanation", rep)
        json.dumps(rep)

        rep_nonexistent = anomaly_detection.get_investigation_report(self.sample_df, "1980-01-01", "temperature")
        self.assertFalse(rep_nonexistent["data_available"])
        self.assertIn("error", rep_nonexistent)
        json.dumps(rep_nonexistent)

    def test_get_all_anomalies(self):
        res = anomaly_detection.get_all_anomalies(self.sample_df, "temperature", method="iqr")
        self.assertIn("anomaly_percentage", res)
        json.dumps(res)

    # =========================================================================
    # FILE 4: forecasting.py tests
    # =========================================================================

    def test_chronological_split(self):
        split = forecasting.chronological_split(self.sample_df, "temperature", test_ratio=0.2)
        self.assertIn("train_dates", split)
        self.assertIn("test_dates", split)
        self.assertEqual(len(split["train_dates"]), split["train_size"])
        self.assertEqual(len(split["test_dates"]), split["test_size"])
        json.dumps(split)

        err = forecasting.chronological_split(self.small_df, "temperature")
        self.assertIn("error", err)

    def test_baseline_forecast(self):
        split = forecasting.chronological_split(self.sample_df, "temperature", test_ratio=0.1)
        base = forecasting.baseline_forecast(split["train_dates"], split["train_values"], split["test_dates"])
        self.assertEqual(len(base["predictions"]), len(split["test_dates"]))
        json.dumps(base)

    def test_exponential_smoothing_forecast(self):
        split = forecasting.chronological_split(self.sample_df, "temperature", test_ratio=0.1)
        es = forecasting.exponential_smoothing_forecast(
            split["train_dates"], split["train_values"], split["test_dates"], forecast_horizon=7
        )
        self.assertEqual(len(es["predictions"]), len(split["test_dates"]))
        self.assertEqual(len(es["future_predictions"]), 7)
        self.assertEqual(len(es["future_dates"]), 7)
        json.dumps(es)

    def test_evaluate_forecast(self):
        actual = [10.0, 12.0, 15.0]
        predicted = [9.5, 12.5, 14.0]
        metrics = forecasting.evaluate_forecast(actual, predicted)
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertEqual(metrics["n_observations"], 3)
        json.dumps(metrics)

    def test_compare_models(self):
        comp = forecasting.compare_models(self.sample_df, "temperature", test_ratio=0.15)
        self.assertIn("winner", comp)
        self.assertIn(comp["winner"], ["baseline", "exponential_smoothing"])
        self.assertIn("baseline", comp)
        self.assertIn("es", comp)
        json.dumps(comp)

    def test_generate_future_forecast(self):
        fut_es = forecasting.generate_future_forecast(self.sample_df, "temperature", "exponential_smoothing", horizon=14)
        self.assertEqual(len(fut_es["forecast_values"]), 14)
        self.assertEqual(len(fut_es["forecast_dates"]), 14)
        json.dumps(fut_es)

        fut_base = forecasting.generate_future_forecast(self.sample_df, "temperature", "baseline", horizon=14)
        self.assertEqual(len(fut_base["forecast_values"]), 14)
        json.dumps(fut_base)

    # =========================================================================
    # FILE 5: scenario_engine.py tests
    # =========================================================================

    def test_apply_scenario(self):
        orig_sum = self.sample_df["temperature"].sum()
        mod_add = scenario_engine.apply_scenario(self.sample_df, "temperature", 2.0, adjustment_type="add")
        self.assertEqual(self.sample_df["temperature"].sum(), orig_sum)
        self.assertGreater(mod_add["temperature"].sum(), orig_sum)

        mod_mult = scenario_engine.apply_scenario(self.sample_df, "temperature", 1.1, adjustment_type="multiply")
        self.assertNotEqual(mod_mult["temperature"].sum(), orig_sum)

        mod_set = scenario_engine.apply_scenario(
            self.sample_df, "temperature", 25.0, date_start="2022-06-01", date_end="2022-06-10", adjustment_type="set"
        )
        sub_check = mod_set[mod_set["date"] == "2022-06-05"]
        self.assertTrue((sub_check["temperature"] == 25.0).all())

    def test_compare_scenarios(self):
        mod = scenario_engine.apply_scenario(self.sample_df, "temperature", 3.0, adjustment_type="add")
        comp = scenario_engine.compare_scenarios(self.sample_df, mod, "temperature")
        self.assertIn("changes", comp)
        self.assertGreater(comp["changes"]["mean_diff"], 0)
        self.assertIn("adjustment_description", comp)
        json.dumps(comp)

    def test_export_scenario_csv(self):
        csv_bytes = scenario_engine.export_scenario_csv(self.sample_df)
        self.assertIsInstance(csv_bytes, bytes)
        self.assertGreater(len(csv_bytes), 0)

    # =========================================================================
    # FILE 6: weather_dna.py tests
    # =========================================================================

    def test_compute_fingerprint(self):
        fp = weather_dna.compute_fingerprint(self.sample_df, label="TestPeriod")
        self.assertIn("raw_features", fp)
        self.assertIn("normalized_features", fp)
        self.assertEqual(fp["label"], "TestPeriod")
        for feat in weather_dna.FEATURES:
            self.assertIn(feat, fp["normalized_features"])
            self.assertTrue(0.0 <= fp["normalized_features"][feat] <= 1.0)
        json.dumps(fp)

        err = weather_dna.compute_fingerprint(self.small_df)
        self.assertIn("error", err)
        json.dumps(err)

    def test_compare_fingerprints(self):
        fp1 = weather_dna.compute_fingerprint(self.sample_df, start_date="2022-01-01", end_date="2022-12-31", label="2022")
        fp2 = weather_dna.compute_fingerprint(self.sample_df, start_date="2023-01-01", end_date="2023-12-31", label="2023")
        comp = weather_dna.compare_fingerprints(fp1, fp2)
        self.assertTrue(0.0 <= comp["similarity_score"] <= 1.0)
        self.assertIn("feature_differences", comp)
        json.dumps(comp)

    def test_biggest_differences(self):
        fp1 = weather_dna.compute_fingerprint(self.sample_df, start_date="2022-01-01", end_date="2022-12-31", label="2022")
        fp2 = weather_dna.compute_fingerprint(self.sample_df, start_date="2023-01-01", end_date="2023-12-31", label="2023")
        diffs = weather_dna.biggest_differences(fp1, fp2)
        self.assertEqual(len(diffs), len(weather_dna.FEATURES))
        self.assertGreaterEqual(diffs[0]["abs_diff"], diffs[-1]["abs_diff"])
        self.assertIn("description", diffs[0])
        json.dumps(diffs)


if __name__ == "__main__":
    unittest.main()
