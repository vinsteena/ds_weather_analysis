"""
Multi-Target Weather Prediction Machine Learning Architecture (CO4 - CO5).
Trains and evaluates predictive algorithms for:
1. Humidity (Relative Humidity %)
2. Moisture in the Air (Dew Point °C and Vapor Pressure hPa)
3. Atmospheric Pressure (Surface Pressure hPa)
4. Wind Speed (km/h)
5. Rain (Two-Stage Model: Rain Occurrence Classifier + Rain Volume Regressor)
6. Temperature (2m Mean Temperature °C)

Includes strict chronological train/val/test splits, persistence baselines,
metric reporting (R2, RMSE, MAE, F1, ROC-AUC), and joblib serialization.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODEL_FILE = os.path.join(MODEL_DIR, "weather_models.joblib")


class WeatherPredictorEngine:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.feature_cols = []
        self.models = {}
        self.metrics = {}
        self.baselines = {}

    def _prepare_feature_matrix(self, df: pd.DataFrame):
        """
        Extracts all lag, rolling, cyclical, and dynamic interaction features.
        Excludes target columns from the current day t.
        """
        exclude_cols = [
            "date", "year", "month", "day_of_year", "day_of_week", "season",
            "temperature_mean", "temperature_max", "temperature_min",
            "relative_humidity", "dew_point", "vapor_pressure", "saturation_vapor_pressure",
            "dew_point_depression", "diurnal_temp_range", "surface_pressure",
            "pressure_tendency_24h", "wind_speed", "wind_gust_max", "wind_direction",
            "precipitation", "rain", "rain_occurrence", "heat_index"
        ]
        feat_cols = [c for c in df.columns if c not in exclude_cols]
        return feat_cols

    def train_models(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """
        Trains specialized algorithms for each meteorological variable.
        """
        self.feature_cols = self._prepare_feature_matrix(train_df)
        X_train = train_df[self.feature_cols]

        print(f"Training Weather Predictor on {len(X_train)} samples with {len(self.feature_cols)} features...")

        # 1. Temperature Regressor
        print("  - Training 2m Temperature Predictor...")
        y_temp = train_df["temperature_mean"]
        model_temp = HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=31, random_state=self.random_state
        )
        model_temp.fit(X_train, y_temp)
        self.models["temperature_mean"] = model_temp

        # 2. Humidity Regressor (Relative Humidity %)
        print("  - Training Relative Humidity Predictor...")
        y_humid = train_df["relative_humidity"]
        model_humid = HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=31, random_state=self.random_state
        )
        model_humid.fit(X_train, y_humid)
        self.models["relative_humidity"] = model_humid

        # 3. Moisture in the Air: Dew Point Temperature (°C)
        print("  - Training Moisture (Dew Point) Predictor...")
        y_dew = train_df["dew_point"]
        model_dew = HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=31, random_state=self.random_state
        )
        model_dew.fit(X_train, y_dew)
        self.models["dew_point"] = model_dew

        # 4. Moisture in the Air: Vapor Pressure (hPa)
        print("  - Training Moisture (Vapor Pressure) Predictor...")
        y_vp = train_df["vapor_pressure"]
        model_vp = HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=31, random_state=self.random_state
        )
        model_vp.fit(X_train, y_vp)
        self.models["vapor_pressure"] = model_vp

        # 5. Atmospheric Pressure (Surface Pressure hPa)
        print("  - Training Atmospheric Pressure Predictor...")
        y_press = train_df["surface_pressure"]
        model_press = HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=31, random_state=self.random_state
        )
        model_press.fit(X_train, y_press)
        self.models["surface_pressure"] = model_press

        # 6. Wind Speed (km/h)
        print("  - Training Wind Speed Predictor...")
        y_wind = train_df["wind_speed"]
        model_wind = HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.06, max_leaf_nodes=31, random_state=self.random_state
        )
        model_wind.fit(X_train, y_wind)
        self.models["wind_speed"] = model_wind

        # 7. Rain: Two-Stage Model
        # Stage 1: Rain Occurrence Classifier (P(Rain >= 0.1 mm))
        print("  - Training Rain Occurrence Classifier (Stage 1)...")
        y_rain_class = train_df["rain_occurrence"]
        model_rain_clf = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, class_weight="balanced", random_state=self.random_state
        )
        model_rain_clf.fit(X_train, y_rain_class)
        self.models["rain_occurrence"] = model_rain_clf

        # Stage 2: Rain Volume Regressor (conditioned on rain > 0.1 mm)
        print("  - Training Rain Volume Regressor (Stage 2)...")
        rain_mask = train_df["rain"] >= 0.1
        if rain_mask.sum() > 30:
            X_rain_wet = X_train[rain_mask]
            y_rain_vol = train_df.loc[rain_mask, "rain"]
            model_rain_reg = HistGradientBoostingRegressor(
                max_iter=100, learning_rate=0.05, max_leaf_nodes=21, random_state=self.random_state
            )
            model_rain_reg.fit(X_rain_wet, y_rain_vol)
        else:
            model_rain_reg = HistGradientBoostingRegressor(random_state=self.random_state)
            model_rain_reg.fit(X_train, train_df["rain"])
        self.models["rain_volume"] = model_rain_reg

        print("All models successfully trained!")
        return self

    def evaluate(self, test_df: pd.DataFrame):
        """
        Evaluates models on out-of-sample test set (e.g., year 2024).
        Computes R2, RMSE, MAE for continuous targets, and Accuracy, Precision, Recall, F1, ROC-AUC for Rain classification.
        Also computes persistence baseline (yesterday's value) for rigorous comparison.
        """
        X_test = test_df[self.feature_cols]
        eval_results = {}

        # 1. Continuous Regression Metrics
        continuous_targets = [
            ("temperature_mean", "°C"),
            ("relative_humidity", "%"),
            ("dew_point", "°C"),
            ("vapor_pressure", "hPa"),
            ("surface_pressure", "hPa"),
            ("wind_speed", "km/h")
        ]

        for target, unit in continuous_targets:
            y_true = test_df[target].values
            y_pred = self.models[target].predict(X_test)

            # Persistence baseline: lag_1
            lag1_col = f"{target}_lag_1"
            if lag1_col in test_df.columns:
                y_baseline = test_df[lag1_col].values
                base_r2 = r2_score(y_true, y_baseline)
                base_rmse = np.sqrt(mean_squared_error(y_true, y_baseline))
                base_mae = mean_absolute_error(y_true, y_baseline)
            else:
                base_r2, base_rmse, base_mae = np.nan, np.nan, np.nan

            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)

            eval_results[target] = {
                "unit": unit,
                "r2_score": round(float(r2), 4),
                "rmse": round(float(rmse), 3),
                "mae": round(float(mae), 3),
                "baseline_r2": round(float(base_r2), 4) if not np.isnan(base_r2) else None,
                "baseline_rmse": round(float(base_rmse), 3) if not np.isnan(base_rmse) else None,
                "baseline_mae": round(float(base_mae), 3) if not np.isnan(base_mae) else None,
                "rmse_improvement_pct": round(float((base_rmse - rmse) / base_rmse * 100), 2) if not np.isnan(base_rmse) and base_rmse > 0 else 0.0
            }

        # 2. Rain Occurrence Classifier Evaluation
        y_rain_true = test_df["rain_occurrence"].values
        rain_proba = self.models["rain_occurrence"].predict_proba(X_test)[:, 1]
        y_rain_pred = (rain_proba >= 0.50).astype(int)

        eval_results["rain_classification"] = {
            "accuracy": round(float(accuracy_score(y_rain_true, y_rain_pred)), 4),
            "precision": round(float(precision_score(y_rain_true, y_rain_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_rain_true, y_rain_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_rain_true, y_rain_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_rain_true, rain_proba)), 4)
        }

        # 3. Overall Rain Amount (Two-Stage Combined)
        y_rain_vol_true = test_df["rain"].values
        raw_wet_pred = self.models["rain_volume"].predict(X_test)
        # Two-stage expected rainfall: P(rain) * E(volume|rain)
        two_stage_rain = rain_proba * np.clip(raw_wet_pred, 0, None)
        rain_mae = mean_absolute_error(y_rain_vol_true, two_stage_rain)
        rain_rmse = np.sqrt(mean_squared_error(y_rain_vol_true, two_stage_rain))

        eval_results["rain_combined"] = {
            "unit": "mm",
            "rmse": round(float(rain_rmse), 3),
            "mae": round(float(rain_mae), 3),
            "rain_days_detected": int(y_rain_pred.sum()),
            "rain_days_actual": int(y_rain_true.sum())
        }

        self.metrics = eval_results
        return eval_results

    def predict_next_day(self, latest_feature_row: pd.DataFrame):
        """
        Generates next-day predictions across all variables from the latest observation row.
        """
        X = latest_feature_row[self.feature_cols]

        temp_pred = float(self.models["temperature_mean"].predict(X)[0])
        humid_pred = float(np.clip(self.models["relative_humidity"].predict(X)[0], 0, 100))
        dew_pred = float(self.models["dew_point"].predict(X)[0])
        vp_pred = float(np.clip(self.models["vapor_pressure"].predict(X)[0], 0, None))
        press_pred = float(self.models["surface_pressure"].predict(X)[0])
        wind_pred = float(np.clip(self.models["wind_speed"].predict(X)[0], 0, None))

        rain_prob = float(self.models["rain_occurrence"].predict_proba(X)[:, 1][0])
        will_rain = rain_prob >= 0.50
        raw_vol = float(np.clip(self.models["rain_volume"].predict(X)[0], 0, None))
        expected_rain_mm = round(rain_prob * raw_vol, 2) if will_rain else round(rain_prob * raw_vol * 0.4, 2)

        return {
            "temperature_mean_c": round(temp_pred, 2),
            "relative_humidity_pct": round(humid_pred, 2),
            "moisture_dew_point_c": round(dew_pred, 2),
            "moisture_vapor_pressure_hpa": round(vp_pred, 2),
            "surface_pressure_hpa": round(press_pred, 2),
            "wind_speed_kmh": round(wind_pred, 2),
            "rain_probability_pct": round(rain_prob * 100.0, 1),
            "will_rain": will_rain,
            "expected_rain_mm": expected_rain_mm
        }

    def save(self, file_path=SAVED_MODEL_FILE):
        """Saves trained models, feature columns, and evaluation metrics."""
        payload = {
            "models": self.models,
            "feature_cols": self.feature_cols,
            "metrics": self.metrics
        }
        joblib.dump(payload, file_path)
        print(f"Weather predictor package saved to: {file_path}")

    @classmethod
    def load(cls, file_path=SAVED_MODEL_FILE):
        """Loads serialized models."""
        payload = joblib.load(file_path)
        engine = cls()
        engine.models = payload["models"]
        engine.feature_cols = payload["feature_cols"]
        engine.metrics = payload.get("metrics", {})
        return engine


if __name__ == "__main__":
    import os
    from weather_trend_analysis.analytics.preprocessor import preprocess_weather_data
    from weather_trend_analysis.analytics.feature_engineering import create_time_series_features, split_train_val_test

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "weather_analytics_era5_2015_2024.csv")
    raw_df = pd.read_csv(data_path)
    clean_df = preprocess_weather_data(raw_df)
    feat_df = create_time_series_features(clean_df)

    train_df, val_df, test_df = split_train_val_test(feat_df)

    engine = WeatherPredictorEngine()
    engine.train_models(train_df, val_df)
    results = engine.evaluate(test_df)

    print("\n=== Model Evaluation on 2024 Test Set ===")
    for target, stats_dict in results.items():
        print(f"Target: {target}")
        for k, v in stats_dict.items():
            print(f"  {k}: {v}")
