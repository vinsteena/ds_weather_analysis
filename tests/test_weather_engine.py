"""
Tests for the WeatherWise core weather engine.
"""

import pytest
from weather_engine import (
    CITIES,
    WEATHER_TYPES,
    get_city_list,
    generate_city_timeseries,
    get_current_weather,
    get_weekly_forecast,
    get_time_period_analysis,
    get_climate_prediction,
    get_preparedness_guide
)

def test_get_city_list():
    cities = get_city_list()
    assert len(cities) >= 6
    names = [c["name"] for c in cities]
    assert "London" in names
    assert "New York" in names
    assert "Tokyo" in names

def test_generate_city_timeseries():
    df = generate_city_timeseries("London", days=100)
    assert len(df) == 100
    assert "temperature" in df.columns
    assert "humidity" in df.columns
    assert "rainfall" in df.columns
    assert "wind_speed" in df.columns

def test_get_current_weather():
    curr = get_current_weather("Paris")
    assert curr["city"] == "Paris"
    assert "temperature" in curr
    assert "feels_like" in curr
    assert "hourly" in curr
    assert len(curr["hourly"]) == 24
    assert curr["condition_vibe"] in WEATHER_TYPES

def test_get_weekly_forecast():
    weekly = get_weekly_forecast("Tokyo")
    assert len(weekly) == 7
    assert weekly[0]["day_name"] == "Today"
    assert "high" in weekly[0]
    assert "low" in weekly[0]
    assert "rain_prob" in weekly[0]

def test_get_time_period_analysis():
    analysis = get_time_period_analysis("New York", period="30d", variable="temperature")
    assert analysis["city"] == "New York"
    assert len(analysis["dates"]) == 30
    assert len(analysis["values"]) == 30
    assert len(analysis["moving_average"]) == 30
    assert "mean" in analysis
    assert "slope" in analysis

def test_get_climate_prediction():
    climate = get_climate_prediction("Sydney")
    assert climate["city"] == "Sydney"
    assert len(climate["future_months"]) == 12
    assert len(climate["projected_temps"]) == 12
    assert len(climate["upper_bounds"]) == 12
    assert len(climate["lower_bounds"]) == 12

def test_get_preparedness_guide():
    prep = get_preparedness_guide("Mumbai", force_weather="rainy")
    assert prep["city"] == "Mumbai"
    assert prep["vibe"] == "rainy"
    assert len(prep["gear"]) >= 2
    assert "attire" in prep
    assert "hydration_target_litres" in prep
    assert 0 <= prep["activity_score"] <= 100
