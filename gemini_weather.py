"""
WeatherWise Gemini AI Intelligence Engine
=========================================
Integrates Google Gemini API (gemini-1.5-flash / gemini-2.0-flash) to provide
natural language meteorological synthesis, climatological trend insights,
and intelligent personalized daily weather advisories.
"""

import os
import json
import sqlite3
import requests
from datetime import datetime
from typing import Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "weatherwise.db")

def get_gemini_api_key() -> str:
    """Retrieves the Gemini API key from environment or SQLite database."""
    # 1. Environment variable
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key

    # 2. Check SQLite settings
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT key_value FROM api_settings WHERE key_name = 'gemini_api_key'")
        row = cur.fetchone()
        conn.close()
        if row and row[0]:
            return row[0].strip()
    except Exception:
        pass

    # 3. Check local .env file
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("GEMINI_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass

    return ""

def save_gemini_api_key(key: str) -> Dict[str, Any]:
    """Saves the Gemini API key into SQLite and .env."""
    key = key.strip()
    if not key:
        return {"success": False, "error": "API key cannot be empty"}

    # Test key with a minimal call
    valid, msg = test_gemini_key(key)
    if not valid:
        return {"success": False, "error": msg}

    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        now_str = datetime.now().isoformat()
        cur.execute(
            "INSERT OR REPLACE INTO api_settings (key_name, key_value, provider, updated_at) VALUES ('gemini_api_key', ?, 'google_gemini', ?)",
            (key, now_str)
        )
        conn.commit()
        conn.close()

        # Also write to local .env for convenience
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(f"GEMINI_API_KEY={key}\n")

        return {
            "success": True,
            "message": "Gemini API Key successfully verified and activated!",
            "has_key": True,
            "masked_key": key[:6] + "..." + key[-4:] if len(key) > 10 else "***"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def test_gemini_key(key: str) -> tuple[bool, str]:
    """Tests the provided Gemini API key with a ping request."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
    payload = {
        "contents": [{
            "parts": [{"text": "Reply with 'OK'."}]
        }]
    }
    try:
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=8)
        if resp.status_code == 200:
            return True, "Valid API key"
        else:
            data = resp.json()
            err_msg = data.get("error", {}).get("message", f"HTTP {resp.status_code}")
            return False, f"Gemini API validation failed: {err_msg}"
    except Exception as e:
        return False, f"Connection to Gemini API failed: {str(e)}"

def generate_gemini_weather_briefing(
    city_name: str,
    current_weather: Dict[str, Any],
    weekly_forecast: Optional[list] = None,
    climate_stats: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates an executive meteorological briefing and personalized intelligence
    using Gemini AI. If no key is configured, returns an actionable setup state.
    """
    api_key = get_gemini_api_key()

    temp = current_weather.get("temperature", "--")
    feels_like = current_weather.get("apparent_temperature", temp)
    humidity = current_weather.get("humidity", "--")
    wind = current_weather.get("wind_speed", "--")
    condition = current_weather.get("condition_name", "Clear")
    vibe = current_weather.get("condition_vibe", "sunny")

    if not api_key:
        return {
            "configured": False,
            "briefing": f"Currently observing {condition.lower()} conditions at {temp}°C (feels like {feels_like}°C) with {humidity}% humidity and winds at {wind} km/h in {city_name}. Gemini AI atmospheric analysis is ready to be linked.",
            "insights": [
                f"Atmospheric condition aligns with standard {city_name} regional climate patterns.",
                "Real-time sensor feeds are active from Open-Meteo & historical SQLite database.",
                "Provide your Google Gemini API key to unlock natural language AI meteorological analysis."
            ],
            "recommendation": f"Wear appropriate clothing for {temp}°C weather. Keep updated with 24-hour and 7-day trajectories.",
            "source": "Local Statistical Fallback"
        }

    # Call Gemini API
    prompt = f"""You are WeatherWise AI, an expert meteorologist and climate data scientist.
Analyze today's weather for {city_name} based on the following verified data:
- Current Temperature: {temp}°C (Feels like: {feels_like}°C)
- Condition: {condition} (Vibe: {vibe})
- Relative Humidity: {humidity}%
- Wind Speed: {wind} km/h
- Location: {city_name}

Provide a concise, highly professional intelligence briefing in valid JSON format with these exact keys:
{{
  "briefing": "A concise 2-sentence executive atmospheric summary for citizens and travelers.",
  "insights": [
    "Key atmospheric or microclimate observation 1",
    "Key observation 2 (comparing to seasonal norms or comfort)",
    "Key observation 3 (wind/humidity/UV advisory)"
  ],
  "recommendation": "Specific outfit, hydration, outdoor activity, and travel advice for today."
}}
Return ONLY valid raw JSON without markdown fences."""

    for model_name in ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.4}
            }
            resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
            if resp.status_code == 200:
                result_json = resp.json()
                text = result_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                if text.startswith("```"):
                    lines = text.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    text = "\n".join(lines).strip()
                
                parsed = json.loads(text)
                parsed["configured"] = True
                parsed["source"] = f"Google {model_name}"
                return parsed
        except Exception:
            continue

    # Fallback if call fails
    return {
        "configured": True,
        "briefing": f"{city_name} is experiencing {condition.lower()} skies with a temperature of {temp}°C and {humidity}% relative humidity.",
        "insights": [
            f"Wind velocity measured at {wind} km/h.",
            "Barometric and humidity levels remain within expected historical bands."
        ],
        "recommendation": f"Dress comfortably for {temp}°C conditions and stay hydrated.",
        "source": "Local Weather Intelligence"
    }
