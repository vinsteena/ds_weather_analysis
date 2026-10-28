"""
WeatherWise Test Suite — Anomaly Detection, Forecasting & Scenario Engine
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_detection import (
    detect_iqr_anomalies, detect_zscore_anomalies,
    detect_rolling_anomalies, get_investigation_report
)
from forecasting import (
    chronological_split, baseline_forecast,
    evaluate_forecast, compare_models, generate_future_forecast
)
from scenario_engine import apply_scenario, compare_scenarios, export_scenario_csv


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_df(n=200, seed=7):
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    doy = np.arange(n)
    temp = 20 - 10 * np.cos(2 * np.pi * doy / 365) + np.random.normal(0, 2, n)
    return pd.DataFrame({
        "date": dates,
        "location": "TestCity",
        "temperature": temp,
        "humidity": np.clip(70 + np.random.normal(0, 5, n), 20, 100),
        "rainfall": np.abs(np.random.exponential(2, n)),
        "wind_speed": np.abs(12 + np.random.normal(0, 3, n)),
        "pressure": 1015 + np.random.normal(0, 2, n),
    })


def make_df_with_outlier():
    df = make_df()
    df.at[100, "temperature"] = 99.0  # Extreme outlier
    return df


def make_two_year_df():
    return make_df(n=730, seed=42)


# ══════════════════════════════════════════════════════════════════════════════
# ANOMALY DETECTION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestIQRAnomalies:
    def test_returns_dict(self):
        df = make_df()
        result = detect_iqr_anomalies(df, "temperature")
        assert isinstance(result, dict)

    def test_method_field(self):
        df = make_df()
        result = detect_iqr_anomalies(df, "temperature")
        assert result.get("method") == "iqr"

    def test_has_required_keys(self):
        df = make_df()
        result = detect_iqr_anomalies(df, "temperature")
        for key in ["q1", "q3", "iqr", "lower_fence", "upper_fence",
                    "anomaly_count", "anomalies", "all_dates", "all_values", "is_anomaly"]:
            assert key in result, f"Missing key: {key}"

    def test_flags_obvious_outlier(self):
        df = make_df_with_outlier()
        result = detect_iqr_anomalies(df, "temperature")
        assert result["anomaly_count"] >= 1
        dates_flagged = [a["date"] for a in result["anomalies"]]
        assert any("2022-04-11" in d or "2022-04" in d for d in dates_flagged)

    def test_no_false_flags_on_normal_data(self):
        """Normal data should have minimal IQR anomalies."""
        np.random.seed(0)
        dates = pd.date_range("2022-01-01", periods=200)
        temp = np.random.normal(20, 2, 200)  # Tight normal distribution
        df = pd.DataFrame({"date": dates, "location": "X", "temperature": temp,
                           "humidity": 50.0, "rainfall": 0.0, "wind_speed": 10.0, "pressure": 1013.0})
        result = detect_iqr_anomalies(df, "temperature")
        # With normal distribution, IQR should flag very few (<5%)
        assert result["anomaly_count"] / len(temp) < 0.05

    def test_upper_fence_gt_lower_fence(self):
        df = make_df()
        result = detect_iqr_anomalies(df, "temperature")
        assert result["upper_fence"] > result["lower_fence"]

    def test_is_anomaly_list_length_matches(self):
        df = make_df()
        result = detect_iqr_anomalies(df, "temperature")
        assert len(result["is_anomaly"]) == len(result["all_dates"])
        assert len(result["all_values"]) == len(result["all_dates"])


class TestZScoreAnomalies:
    def test_returns_dict(self):
        df = make_df()
        result = detect_zscore_anomalies(df, "temperature")
        assert isinstance(result, dict)

    def test_method_field(self):
        df = make_df()
        result = detect_zscore_anomalies(df, "temperature")
        assert result.get("method") == "zscore"

    def test_flags_extreme_outlier(self):
        df = make_df_with_outlier()
        result = detect_zscore_anomalies(df, "temperature", threshold=3.0)
        assert result["anomaly_count"] >= 1

    def test_threshold_affects_count(self):
        df = make_df()
        strict = detect_zscore_anomalies(df, "temperature", threshold=2.0)
        lenient = detect_zscore_anomalies(df, "temperature", threshold=4.0)
        assert strict["anomaly_count"] >= lenient["anomaly_count"]

    def test_z_score_in_anomaly_report(self):
        df = make_df_with_outlier()
        result = detect_zscore_anomalies(df, "temperature")
        if len(result["anomalies"]) > 0:
            assert "z_score" in result["anomalies"][0]


class TestRollingAnomalies:
    def test_returns_dict(self):
        df = make_df()
        result = detect_rolling_anomalies(df, "temperature")
        assert isinstance(result, dict)

    def test_method_field(self):
        df = make_df()
        result = detect_rolling_anomalies(df, "temperature")
        assert result.get("method") == "rolling"

    def test_flags_spike(self):
        df = make_df()
        # Insert a local spike
        df.at[50, "temperature"] = df["temperature"].mean() + 5 * df["temperature"].std()
        result = detect_rolling_anomalies(df, "temperature")
        assert result["anomaly_count"] >= 1


class TestInvestigationReport:
    def test_returns_dict(self):
        df = make_df_with_outlier()
        result = get_investigation_report(df, "2022-04-11", "temperature", "iqr")
        assert isinstance(result, dict)

    def test_has_explanation(self):
        df = make_df_with_outlier()
        result = get_investigation_report(df, "2022-04-11", "temperature", "iqr")
        assert "explanation" in result
        assert isinstance(result.get("explanation"), str)
        assert len(result.get("explanation", "")) > 10

    def test_has_flagged_by(self):
        df = make_df_with_outlier()
        result = get_investigation_report(df, "2022-04-11", "temperature", "iqr")
        if "error" not in result:
            assert "flagged_by" in result

    def test_invalid_date_handled(self):
        df = make_df()
        result = get_investigation_report(df, "1900-01-01", "temperature", "iqr")
        # Should return error or empty result, not crash
        assert isinstance(result, dict)


# ══════════════════════════════════════════════════════════════════════════════
# FORECASTING TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestChronologicalSplit:
    def test_returns_dict(self):
        df = make_df()
        result = chronological_split(df, "temperature")
        assert isinstance(result, dict)

    def test_split_is_chronological(self):
        """Train dates must all precede test dates."""
        df = make_df(n=100)
        result = chronological_split(df, "temperature", test_ratio=0.2)
        if "error" not in result:
            train_dates = pd.to_datetime(result["train_dates"])
            test_dates = pd.to_datetime(result["test_dates"])
            assert train_dates.max() <= test_dates.min()

    def test_no_future_data_in_training(self):
        """Critical: test set must never appear in training set."""
        df = make_df(n=100)
        result = chronological_split(df, "temperature", test_ratio=0.2)
        if "error" not in result:
            train_set = set(result["train_dates"])
            test_set = set(result["test_dates"])
            # No overlap
            assert len(train_set.intersection(test_set)) == 0

    def test_sizes_sum_to_total(self):
        df = make_df(n=100)
        result = chronological_split(df, "temperature", test_ratio=0.2)
        if "error" not in result:
            total = result["train_size"] + result["test_size"]
            # Should sum to original data length (minus NaN rows)
            df_clean = df.dropna(subset=["temperature"])
            assert total == len(df_clean)

    def test_insufficient_data(self):
        dates = pd.date_range("2022-01-01", periods=10)
        df = pd.DataFrame({"date": dates, "location": "X",
                           "temperature": [20.0] * 10,
                           "humidity": 50.0, "rainfall": 0.0, "wind_speed": 10.0, "pressure": 1013.0})
        result = chronological_split(df, "temperature", test_ratio=0.2)
        assert "error" in result


class TestBaselineForecast:
    def test_returns_dict(self):
        df = make_df(n=100)
        split = chronological_split(df, "temperature", test_ratio=0.2)
        if "error" not in split:
            result = baseline_forecast(split["train_dates"], split["train_values"], split["test_dates"])
            assert isinstance(result, dict)

    def test_prediction_count_matches_test(self):
        df = make_df(n=100)
        split = chronological_split(df, "temperature", test_ratio=0.2)
        if "error" not in split:
            result = baseline_forecast(split["train_dates"], split["train_values"], split["test_dates"])
            assert len(result["predictions"]) == len(split["test_dates"])


class TestEvaluateForecast:
    def test_mae_non_negative(self):
        actual = [10.0, 15.0, 20.0, 25.0, 30.0]
        predicted = [11.0, 14.0, 22.0, 24.0, 31.0]
        result = evaluate_forecast(actual, predicted)
        assert result["mae"] >= 0

    def test_rmse_non_negative(self):
        actual = [10.0, 15.0, 20.0]
        predicted = [10.0, 15.0, 20.0]  # Perfect prediction
        result = evaluate_forecast(actual, predicted)
        assert result["rmse"] >= 0

    def test_perfect_forecast_zero_error(self):
        actual = [10.0, 20.0, 30.0]
        predicted = [10.0, 20.0, 30.0]
        result = evaluate_forecast(actual, predicted)
        assert abs(result["mae"]) < 1e-9
        assert abs(result["rmse"]) < 1e-9

    def test_mae_le_rmse_for_diverse_errors(self):
        """RMSE penalizes large errors more, so RMSE >= MAE."""
        actual = [10.0, 10.0, 10.0, 10.0, 10.0]
        predicted = [20.0, 10.5, 10.5, 10.5, 10.5]  # One large error
        result = evaluate_forecast(actual, predicted)
        assert result["rmse"] >= result["mae"]

    def test_n_observations(self):
        actual = [1.0, 2.0, 3.0, 4.0]
        predicted = [1.1, 2.1, 3.1, 4.1]
        result = evaluate_forecast(actual, predicted)
        assert result["n_observations"] == 4


class TestCompareModels:
    def test_returns_dict(self):
        df = make_df(n=100)
        result = compare_models(df, "temperature")
        assert isinstance(result, dict)

    def test_has_winner(self):
        df = make_df(n=100)
        result = compare_models(df, "temperature")
        if "error" not in result:
            assert "winner" in result
            assert result["winner"] in ("baseline", "exponential_smoothing")

    def test_metrics_present(self):
        df = make_df(n=100)
        result = compare_models(df, "temperature")
        if "error" not in result:
            assert "baseline" in result
            assert "es" in result


class TestGenerateFutureForecast:
    def test_returns_dict(self):
        df = make_df(n=100)
        result = generate_future_forecast(df, "temperature", "baseline", horizon=14)
        assert isinstance(result, dict)

    def test_forecast_horizon_length(self):
        df = make_df(n=100)
        result = generate_future_forecast(df, "temperature", "baseline", horizon=14)
        if "error" not in result:
            assert len(result["forecast_dates"]) == 14
            assert len(result["forecast_values"]) == 14

    def test_forecast_dates_after_historical(self):
        df = make_df(n=100)
        result = generate_future_forecast(df, "temperature", "baseline", horizon=7)
        if "error" not in result:
            hist_dates = pd.to_datetime(result["historical_dates"])
            forecast_dates = pd.to_datetime(result["forecast_dates"])
            assert forecast_dates.min() > hist_dates.max()


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO ENGINE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestApplyScenario:
    def test_does_not_modify_original(self):
        """Critical: apply_scenario must never modify the original DataFrame."""
        df = make_df(n=50)
        original_temp = df["temperature"].copy()
        _ = apply_scenario(df, "temperature", 5.0)
        pd.testing.assert_series_equal(df["temperature"], original_temp)

    def test_add_adjustment(self):
        df = make_df(n=50)
        original_mean = df["temperature"].mean()
        modified = apply_scenario(df, "temperature", 2.0, adjustment_type="add")
        # Mean should increase by ~2
        assert abs(modified["temperature"].mean() - (original_mean + 2.0)) < 0.01

    def test_multiply_adjustment(self):
        df = make_df(n=50)
        original_mean = df["temperature"].mean()
        modified = apply_scenario(df, "temperature", 1.1, adjustment_type="multiply")
        assert abs(modified["temperature"].mean() - original_mean * 1.1) < 0.01

    def test_date_range_restriction(self):
        df = make_df(n=100)
        original_first_value = df.iloc[0]["temperature"]
        # Adjust only dates after 2022-04-01
        modified = apply_scenario(df, "temperature", 5.0,
                                  date_start="2022-04-01", date_end="2022-12-31",
                                  adjustment_type="add")
        # First value (Jan) should be unchanged
        assert abs(modified.iloc[0]["temperature"] - original_first_value) < 0.01

    def test_returns_dataframe(self):
        df = make_df(n=50)
        result = apply_scenario(df, "temperature", 1.0)
        assert isinstance(result, pd.DataFrame)

    def test_returns_copy_not_view(self):
        df = make_df(n=50)
        modified = apply_scenario(df, "temperature", 5.0)
        # Modifying modified should not affect df
        modified.at[modified.index[0], "temperature"] = -9999.0
        assert df["temperature"].iloc[0] != -9999.0


class TestCompareScenarios:
    def test_returns_dict(self):
        df = make_df(n=50)
        modified = apply_scenario(df, "temperature", 2.0)
        result = compare_scenarios(df, modified, "temperature")
        assert isinstance(result, dict)

    def test_has_original_and_modified(self):
        df = make_df(n=50)
        modified = apply_scenario(df, "temperature", 2.0)
        result = compare_scenarios(df, modified, "temperature")
        assert "original" in result
        assert "modified" in result

    def test_mean_difference_correct(self):
        df = make_df(n=50)
        adj = 3.0
        modified = apply_scenario(df, "temperature", adj, adjustment_type="add")
        result = compare_scenarios(df, modified, "temperature")
        if "changes" in result and "mean_diff" in result["changes"]:
            assert abs(result["changes"]["mean_diff"] - adj) < 0.01


class TestExportScenarioCSV:
    def test_returns_bytes(self):
        df = make_df(n=20)
        result = export_scenario_csv(df)
        assert isinstance(result, bytes)

    def test_csv_parseable(self):
        df = make_df(n=20)
        csv_bytes = export_scenario_csv(df)
        import io
        parsed = pd.read_csv(io.BytesIO(csv_bytes))
        assert len(parsed) == len(df)

    def test_csv_has_temperature_column(self):
        df = make_df(n=20)
        csv_bytes = export_scenario_csv(df)
        import io
        parsed = pd.read_csv(io.BytesIO(csv_bytes))
        assert "temperature" in parsed.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
