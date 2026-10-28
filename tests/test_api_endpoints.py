"""
Unit tests for the new WeatherWise professional API endpoints.
"""

import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"WeatherWise" in response.data

def test_api_cities(client):
    response = client.get('/api/cities')
    assert response.status_code == 200
    json_data = response.get_json()
    assert "cities" in json_data
    assert len(json_data["cities"]) >= 6

def test_api_current_weather(client):
    response = client.get('/api/weather/current?city=London')
    assert response.status_code == 200
    data = response.get_json()
    assert data["city"] == "London"
    assert "temperature" in data
    assert "hourly" in data
    assert len(data["hourly"]) == 24

def test_api_weekly_forecast(client):
    response = client.get('/api/weather/weekly?city=Tokyo')
    assert response.status_code == 200
    data = response.get_json()
    assert "forecast" in data
    assert len(data["forecast"]) == 7

def test_api_weather_trends(client):
    response = client.get('/api/weather/trends?city=Paris&period=30d&variable=temperature')
    assert response.status_code == 200
    data = response.get_json()
    assert data["period"] == "30d"
    assert len(data["dates"]) == 30
    assert len(data["values"]) == 30
    assert len(data["moving_average"]) == 30

def test_api_climate(client):
    response = client.get('/api/weather/climate?city=New%20York')
    assert response.status_code == 200
    data = response.get_json()
    assert "projected_temps" in data
    assert len(data["future_months"]) == 12

def test_api_preparedness(client):
    response = client.get('/api/weather/preparedness?city=Mumbai')
    assert response.status_code == 200
    data = response.get_json()
    assert "gear" in data
    assert "attire" in data
    assert "activity_score" in data

def test_api_monthly(client):
    response = client.get('/api/weather/monthly?city=London')
    assert response.status_code == 200
    data = response.get_json()
    assert "monthly_summary" in data
    assert len(data["monthly_summary"]) == 12

def test_api_seasons(client):
    response = client.get('/api/weather/seasons?city=Sydney')
    assert response.status_code == 200
    data = response.get_json()
    assert "seasons" in data
    assert len(data["seasons"]) == 4
    assert data["is_southern_hemisphere"] is True

def test_api_anomalies(client):
    response = client.get('/api/weather/anomalies?city=Paris&variable=temperature&method=iqr')
    assert response.status_code == 200
    data = response.get_json()
    assert "anomalies" in data
    assert "anomaly_percentage" in data

def test_api_forecast_accuracy(client):
    response = client.get('/api/weather/forecast-accuracy?city=Tokyo&variable=temperature')
    assert response.status_code == 200
    data = response.get_json()
    assert "baseline" in data
    assert "exponential_smoothing" in data
    assert "mae" in data["exponential_smoothing"]
    assert "rmse" in data["exponential_smoothing"]

def test_api_key_settings(client):
    response = client.get('/api/settings/api-key')
    assert response.status_code == 200
    data = response.get_json()
    assert "provider" in data
    assert "available_providers" in data

    post_resp = client.post('/api/settings/api-key', json={"provider": "openmeteo", "api_key": ""})
    assert post_resp.status_code == 200
    post_data = post_resp.get_json()
    assert post_data["success"] is True


