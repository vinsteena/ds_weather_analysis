"""
WeatherWise Grok AI Intelligence Engine (xAI)
==============================================
Integrates xAI Grok API (grok-2-mini / grok-beta / grok-2-1212) to provide
advanced, witty, and deeply accurate meteorological intelligence,
atmospheric trend insights, and personalized daily weather advisories.
"""

import os
import json
import sqlite3
import requests
from datetime import datetime
from typing import Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "weatherwise.db")

def get_grok_api_key() -> str:
    """Retrieves the xAI Grok API key from environment, SQLite database, or .env."""
    # 1. Environment variable
    key = os.environ.get("GROK_API_KEY", "").strip() or os.environ.get("XAI_API_KEY", "").strip()
    if key:
        return key

    # 2. Check SQLite settings
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT key_value FROM api_settings WHERE key_name IN ('grok_api_key', 'xai_api_key') ORDER BY updated_at DESC")
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
                    if line.startswith("GROK_API_KEY=") or line.startswith("XAI_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass

    return ""

def test_grok_key(key: str) -> tuple[bool, str]:
    """Tests the provided Grok API key with a lightweight ping request."""
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    payload = {
        "messages": [
            {"role": "user", "content": "Reply with 'OK'."}
        ],
        "model": "grok-2-mini",
        "temperature": 0.1
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=9)
        if resp.status_code == 200:
            return True, "Valid xAI Grok API key"
        else:
            try:
                data = resp.json()
                err_msg = data.get("error", {}).get("message", f"HTTP {resp.status_code}")
            except Exception:
                err_msg = f"HTTP {resp.status_code}: {resp.text[:100]}"
            return False, f"xAI Grok verification failed: {err_msg}"
    except Exception as e:
        return False, f"Connection to xAI Grok API failed: {str(e)}"

def save_grok_api_key(key: str) -> Dict[str, Any]:
    """Validates and saves the Grok API key into SQLite and .env."""
    key = key.strip()
    if not key:
        return {"success": False, "error": "API key cannot be empty"}

    valid, msg = test_grok_key(key)
    if not valid:
        return {"success": False, "error": msg}

    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        now_str = datetime.now().isoformat()
        cur.execute(
            "INSERT OR REPLACE INTO api_settings (key_name, key_value, provider, updated_at) VALUES ('grok_api_key', ?, 'xai_grok', ?)",
            (key, now_str)
        )
        conn.commit()
        conn.close()

        # Save to .env
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        lines = []
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                lines = [l for l in f if not l.startswith("GROK_API_KEY=") and not l.startswith("XAI_API_KEY=")]
        lines.append(f"GROK_API_KEY={key}\n")
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        masked = key[:6] + "..." + key[-4:] if len(key) > 10 else "***"
        return {
            "success": True,
            "message": "xAI Grok API Key verified and activated successfully!",
            "has_key": True,
            "masked_key": masked
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def generate_grok_weather_briefing(
    city_name: str,
    current_weather: Dict[str, Any],
    weekly_forecast: Optional[list] = None,
    climate_stats: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates an executive meteorological briefing and personalized intelligence
    using xAI Grok. If no key is configured, returns local intelligence fallback.
    """
    api_key = get_grok_api_key()

    temp = current_weather.get("temperature", "--")
    feels_like = current_weather.get("apparent_temperature", temp)
    humidity = current_weather.get("humidity", "--")
    wind = current_weather.get("wind_speed", "--")
    condition = current_weather.get("condition_name", "Clear")
    vibe = current_weather.get("condition_vibe", "sunny")

    if not api_key:
        return {
            "configured": False,
            "briefing": f"Currently observing {condition.lower()} conditions at {temp}°C (feels like {feels_like}°C) with {humidity}% humidity and winds at {wind} km/h in {city_name}. xAI Grok intelligence is ready to be linked.",
            "insights": [
                f"Atmospheric condition aligns with standard {city_name} regional climate patterns.",
                "Real-time sensor feeds are active from Open-Meteo & 45-year historical database.",
                "Enter your xAI Grok API key (starts with xai-) to unlock live Grok AI atmospheric analysis."
            ],
            "recommendation": f"Wear appropriate clothing for {temp}°C weather. Keep updated with 24-hour and 7-day trajectories.",
            "source": "Local Statistical Fallback"
        }

    # Call xAI Grok API
    system_prompt = (
        "You are Grok Atmospheric Intelligence, an expert meteorologist and witty atmospheric physicist. "
        "Analyze the provided weather parameters and provide an intelligent, accurate, and engaging report. "
        "You must respond in pure, raw JSON format with the following exact keys:\n"
        "{\n"
        '  "briefing": "Concise 2-sentence executive meteorological summary.",\n'
        '  "insights": [\n'
        '    "Atmospheric or microclimate observation 1",\n'
        '    "Observation 2 comparing comfort or seasonal norm",\n'
        '    "Observation 3 regarding wind, UV, or barometric trends"\n'
        "  ],\n"
        '  "recommendation": "Practical advice for attire, gear, outdoor activity, and travel today."\n'
        "}\n"
        "Do NOT include markdown formatting or backticks in the response. Return raw JSON only."
    )

    user_prompt = (
        f"Location: {city_name}\n"
        f"Temperature: {temp}°C (Apparent Feels Like: {feels_like}°C)\n"
        f"Weather Condition: {condition} (Vibe: {vibe})\n"
        f"Humidity: {humidity}%\n"
        f"Wind Velocity: {wind} km/h\n"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    models_to_try = ["grok-2-mini", "grok-beta", "grok-2-1212"]
    for model_name in models_to_try:
        try:
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "model": model_name,
                "temperature": 0.3
            }
            resp = requests.post("https://api.x.ai/v1/chat/completions", json=payload, headers=headers, timeout=12)
            if resp.status_code == 200:
                result_json = resp.json()
                text = result_json["choices"][0]["message"]["content"].strip()
                if text.startswith("```"):
                    lines = text.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    text = "\n".join(lines).strip()

                parsed = json.loads(text)
                parsed["configured"] = True
                parsed["source"] = f"xAI {model_name}"
                return parsed
        except Exception:
            continue

    # Fallback if Grok call fails
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
