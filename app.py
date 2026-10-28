"""
WeatherWise - Professional Weather Intelligence & Trend Analysis Platform
==========================================================================
Main Flask application routing, Open-Meteo live API integration,
real historical datasets, monthly comparisons, seasonal pattern analysis,
outlier anomaly detection, and forecasting accuracy metrics.
"""

import os
import io
import uuid
import json
import sqlite3
import tempfile
import warnings
from datetime import datetime, timedelta
from functools import wraps

import numpy as np
import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    send_file,
    redirect,
    url_for,
)

from weather_engine import (
    CITIES,
    get_city_list,
    fetch_live_forecast,
    fetch_weekly_forecast,
    fetch_historical_dataset,
    get_time_period_analysis,
    get_monthly_comparisons,
    get_seasonal_patterns,
    get_unusual_weather,
    get_forecasting_accuracy,
    get_climate_prediction,
    get_preparedness_guide,
    get_api_key_setting,
    save_api_key_setting,
)

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'weatherwise-pro-secret-2026')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

DB_PATH = os.path.join(os.path.dirname(__file__), 'weatherwise.db')

def init_db():
    try:
        from setup_db import init_database
        init_database(force_reseed=False)
    except Exception as e:
        print(f"DB init error: {e}")

init_db()

def safe_json(obj):
    """Recursively converts NumPy and Pandas types to standard Python primitives."""
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(float(obj), 4)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return [safe_json(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [safe_json(x) for x in obj]
    return obj

# ============================================================================
# PRIMARY USER PAGES
# ============================================================================

@app.route('/')
def index():
    """Main executive WeatherWise dashboard."""
    return render_template('index.html')

# Legacy route aliases redirect to unified dashboard
@app.route('/workspace')
@app.route('/detective')
@app.route('/time-machine')
@app.route('/seasons')
@app.route('/what-if')
@app.route('/forecast')
@app.route('/weather-dna')
@app.route('/climate-connect')
@app.route('/data-trust')
@app.route('/reports')
@app.route('/methodology')
@app.route('/history')
def route_aliases():
    return redirect(url_for('index'))

# ============================================================================
# LIVE OPEN-METEO & HISTORICAL WEATHER APIS
# ============================================================================

@app.route('/api/cities', methods=['GET'])
def api_cities():
    """Returns supported locations."""
    try:
        cities = get_city_list()
        return jsonify({"cities": cities})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/weather/current', methods=['GET'])
def api_current_weather():
    """Returns live weather from Open-Meteo and 24h hourly forecast."""
    try:
        city = request.args.get('city', 'London')
        mode = request.args.get('mode', 'live')
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        force_demo = (mode == 'demo')
        data = fetch_live_forecast(city, lat=lat, lon=lon, force_demo=force_demo, force_refresh=force_refresh)
        return jsonify(safe_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/weather/weekly', methods=['GET'])
def api_weekly_forecast():
    """Returns 7-day comprehensive weekly forecast from Open-Meteo."""
    try:
        city = request.args.get('city', 'London')
        mode = request.args.get('mode', 'live')
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        force_demo = (mode == 'demo')
        data = fetch_weekly_forecast(city, lat=lat, lon=lon, force_demo=force_demo, force_refresh=force_refresh)
        return jsonify(safe_json({"city": city, "forecast": data}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/reverse-geocode', methods=['GET'])
def api_reverse_geocode():
    """Resolves GPS coordinates to city name and country."""
    try:
        import requests as req
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        if lat is None or lon is None:
            return jsonify({"error": "Latitude and longitude required"}), 400

        url = f"https://api.bigdatacloud.net/data/reverse-geocode-client?latitude={lat}&longitude={lon}&localityLanguage=en"
        resp = req.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            city = data.get("city") or data.get("locality") or data.get("principalSubdivision") or "Current Location"
            country = data.get("countryName") or ""
            return jsonify({"city": city, "country": country, "lat": lat, "lon": lon})
    except Exception:
        pass
    return jsonify({"city": "Current Location", "country": "", "lat": lat, "lon": lon})

@app.route('/api/search-city', methods=['GET'])
def api_search_city():
    """Searches for any city across the world using Open-Meteo geocoding."""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({"results": []})
    try:
        import requests as req
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={req.utils.quote(query)}&count=6&language=en&format=json"
        resp = req.get(url, timeout=5)
        if resp.status_code == 200:
            raw = resp.json().get("results", [])
            results = []
            for r in raw:
                results.append({
                    "name": r.get("name"),
                    "country": r.get("country", ""),
                    "admin1": r.get("admin1", ""),
                    "lat": float(r.get("latitude")),
                    "lon": float(r.get("longitude"))
                })
            return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"results": []})

@app.route('/api/weather/trends', methods=['GET'])
def api_weather_trends():
    """Returns real historical time series, moving averages, and trendline."""
    try:
        city = request.args.get('city', 'London')
        period = request.args.get('period', '30d')
        variable = request.args.get('variable', 'temperature')
        mode = request.args.get('mode', 'live')
        force_demo = (mode == 'demo')
        data = get_time_period_analysis(city, period=period, variable=variable, force_demo=force_demo)
        return jsonify(safe_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/weather/monthly', methods=['GET'])
def api_monthly_comparison():
    """Returns monthly comparisons and Year-over-Year aggregations."""
    try:
        city = request.args.get('city', 'London')
        mode = request.args.get('mode', 'live')
        force_demo = (mode == 'demo')
        data = get_monthly_comparisons(city, force_demo=force_demo)
        return jsonify(safe_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/weather/seasons', methods=['GET'])
def api_seasonal_patterns():
    """Returns seasonal patterns (Winter, Spring, Summer, Autumn) with hemisphere awareness."""
    try:
        city = request.args.get('city', 'London')
        mode = request.args.get('mode', 'live')
        force_demo = (mode == 'demo')
        data = get_seasonal_patterns(city, force_demo=force_demo)
        return jsonify(safe_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/weather/anomalies', methods=['GET'])
def api_weather_anomalies():
    """Detects unusual weather events and statistical anomalies using IQR and Z-Score."""
    try:
        city = request.args.get('city', 'London')
        variable = request.args.get('variable', 'temperature')
        method = request.args.get('method', 'iqr')
        mode = request.args.get('mode', 'live')
        force_demo = (mode == 'demo')
        data = get_unusual_weather(city, variable=variable, method=method, force_demo=force_demo)
        return jsonify(safe_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/weather/forecast-accuracy', methods=['GET'])
def api_forecast_accuracy():
    """Evaluates forecasting model performance on historical observations with MAE and RMSE."""
    try:
        city = request.args.get('city', 'London')
        variable = request.args.get('variable', 'temperature')
        mode = request.args.get('mode', 'live')
        force_demo = (mode == 'demo')
        data = get_forecasting_accuracy(city, variable=variable, force_demo=force_demo)
        return jsonify(safe_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/weather/climate', methods=['GET'])
def api_climate_prediction():
    """Returns 12-month future climate projection and 95% confidence intervals."""
    try:
        city = request.args.get('city', 'London')
        mode = request.args.get('mode', 'live')
        force_demo = (mode == 'demo')
        data = get_climate_prediction(city, force_demo=force_demo)
        return jsonify(safe_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/weather/preparedness', methods=['GET'])
def api_weather_preparedness():
    """Returns intelligent preparedness advice, gear checklist, and commute alerts."""
    try:
        city = request.args.get('city', 'London')
        mode = request.args.get('mode', 'live')
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        force_demo = (mode == 'demo')
        data = get_preparedness_guide(city, lat=lat, lon=lon, force_demo=force_demo)
        return jsonify(safe_json(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/settings/api-key', methods=['GET', 'POST'])
def api_key_settings():
    """Retrieves or configures the Live Weather API Key and Provider."""
    try:
        if request.method == 'POST':
            payload = request.get_json() or {}
            provider = payload.get('provider', 'openmeteo')
            key = payload.get('api_key', '')
            res = save_api_key_setting(provider, key)
            return jsonify(res)
        else:
            return jsonify(get_api_key_setting())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("[WeatherWise] Starting server on http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
