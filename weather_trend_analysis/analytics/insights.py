"""
Weather Broadcast Script & Decision-Support Insights Module (CO5).
Translates quantitative analytical results, trends, and ML predictions into:
1. Broadcast-Ready TV/Radio Weather News Scripts
2. Sector-Specific Operational Advisories (Agriculture, Aviation, Municipal, Public Health)
3. Decadal Climate Trend Bulletins
"""

from typing import Dict, Any


def generate_broadcast_script(predictions: Dict[str, Any], trend_summary: Dict[str, Any] = None, active_hazards: str = None) -> str:
    """
    Generates a natural-language teleprompter script for meteorologists and TV/Radio broadcasters.
    """
    temp = predictions.get("temperature_mean_c", 25.0)
    humidity = predictions.get("relative_humidity_pct", 60.0)
    dew_point = predictions.get("moisture_dew_point_c", 16.0)
    pressure = predictions.get("surface_pressure_hpa", 1013.0)
    wind = predictions.get("wind_speed_kmh", 12.0)
    rain_prob = predictions.get("rain_probability_pct", 20.0)
    expected_rain = predictions.get("expected_rain_mm", 0.0)
    will_rain = predictions.get("will_rain", False)

    # Moisture perception
    if dew_point >= 22.0:
        moisture_desc = "oppressive atmospheric moisture and heavy mugginess"
    elif dew_point >= 18.0:
        moisture_desc = "noticeable moisture in the air with moderate humidity"
    elif dew_point >= 12.0:
        moisture_desc = "comfortable moisture levels with mild air"
    else:
        moisture_desc = "crisp, dry air with low atmospheric moisture"

    # Barometric pressure interpretation (handles both sea-level and elevated station baselines)
    ref_pressure = 1013.25 if pressure > 980.0 else 912.0
    pressure_diff = pressure - ref_pressure
    if pressure_diff < -4.5:
        pressure_desc = f"a low-pressure trough at {pressure:.1f} hPa, facilitating cloud buildup"
    elif pressure_diff > 4.0:
        pressure_desc = f"a high-pressure ridge at {pressure:.1f} hPa, bringing stable, settled skies"
    else:
        pressure_desc = f"steady atmospheric pressure near {pressure:.1f} hPa"

    # Wind interpretation
    if wind >= 35.0:
        wind_desc = f"gusty and brisk winds up to {wind:.1f} km/h"
    elif wind >= 20.0:
        wind_desc = f"a moderate breeze averaging {wind:.1f} km/h"
    else:
        wind_desc = f"gentle, light winds around {wind:.1f} km/h"

    # Rain interpretation
    if will_rain or rain_prob >= 50.0:
        rain_desc = f"Scattered showers and localized precipitation are expected, with a {rain_prob:.0f}% chance of rain accumulating approximately {expected_rain:.1f} mm."
    elif rain_prob >= 25.0:
        rain_desc = f"There is an isolated {rain_prob:.0f}% chance of passing light showers, but most areas will remain predominantly dry."
    else:
        rain_desc = f"Dry conditions will prevail through tomorrow, with rain probability suppressed at just {rain_prob:.0f}%."

    script = f"""================================================================================
                    METEOROLOGICAL BROADCAST BULLETIN
================================================================================
[ANCHOR INTRO]
"Good day and welcome to your regional weather briefing. Here is your comprehensive
meteorological outlook for the next 24 hours, based on our time-series predictive model."

[ON-AIR SUMMARY]
"Expect a mean daily temperature of {temp:.1f}°C tomorrow with relative humidity
holding near {humidity:.1f}%. The dew point is tracking at {dew_point:.1f}°C, delivering
{moisture_desc}."

[SYNOPTIC SYSTEM & WINDS]
"On the barometric front, we are monitoring {pressure_desc}.
Surface winds will behave as {wind_desc}."

[PRECIPITATION OUTLOOK]
"{rain_desc}"
"""

    if active_hazards and active_hazards != "Normal Conditions":
        script += f"""
[SEVERE WEATHER ALERT / HAZARD DESK]
"SPECIAL ADVISORY: {active_hazards}. Residents and outdoor operators are advised to stay alert."
"""
    else:
        script += """
[SAFETY & HAZARD STATUS]
"All meteorological parameters remain within normal seasonal bounds. No severe weather watches active."
"""

    script += """================================================================================"""
    return script


def generate_sector_advisory(predictions: Dict[str, Any]) -> Dict[str, str]:
    """
    Translates meteorological predictions into operational guidance for key decision-makers.
    """
    rain_prob = predictions.get("rain_probability_pct", 20.0)
    expected_rain = predictions.get("expected_rain_mm", 0.0)
    wind = predictions.get("wind_speed_kmh", 12.0)
    dew_point = predictions.get("moisture_dew_point_c", 16.0)
    humidity = predictions.get("relative_humidity_pct", 60.0)
    temp = predictions.get("temperature_mean_c", 25.0)

    # Agriculture
    if rain_prob >= 60.0 and expected_rain >= 15.0:
        agri = "Heavy rain anticipated. Postpone field spraying and fertilizer applications; ensure field drainage."
    elif rain_prob <= 20.0 and temp >= 30.0:
        agri = "Warm and dry conditions. Elevated soil moisture evaporation; schedule standard crop irrigation."
    else:
        agri = f"Moderate moisture conditions (Dew point {dew_point:.1f}°C, RH {humidity:.1f}%). Normal agronomic operations recommended."

    # Aviation & Transport
    if wind >= 35.0:
        aviation = f"Elevated surface wind speeds ({wind:.1f} km/h). Expect low-altitude turbulence and crosswind gusts."
    else:
        aviation = f"Favorable surface wind conditions ({wind:.1f} km/h). Standard visual and instrument flight operations."

    # Municipal & Urban Disaster Management
    if expected_rain >= 30.0 or rain_prob >= 75.0:
        municipal = "High rainfall accumulation probability. Alert urban drainage maintenance teams for localized stormwater ponding."
    else:
        municipal = "Low hydrological risk. Standard stormwater infrastructure capacity."

    # Public Health
    if temp >= 35.0 and humidity >= 70.0:
        health = "High Heat Index caution: muggy conditions increase heat fatigue risk during peak afternoon hours."
    else:
        health = "Comfortable bioclimatic range. Normal outdoor activity conditions."

    return {
        "agriculture": agri,
        "aviation_transport": aviation,
        "municipal_infrastructure": municipal,
        "public_health": health
    }
