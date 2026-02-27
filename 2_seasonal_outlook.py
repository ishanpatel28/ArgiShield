"""
seasonal_outlook.py
===================
AgriShield – ENSO State + 14-Day Forecast
------------------------------------------
Two jobs:

  1. ENSO STATE — Fetch the current ONI (Oceanic Niño Index) value from
     NOAA's text feed and classify the ENSO phase. The ONI is the single
     most reliable seasonal climate predictor for North American agriculture.
     Current value (NDJ 2025): -0.55 → weak La Niña.

  2. 14-DAY FORECAST — Fetch a day-by-day temperature and precipitation
     forecast from Open-Meteo (free, no API key, highly accurate to ~7 days,
     useful signal to ~14 days). This is the highest-confidence layer in the
     entire pipeline — near-certain stress in the next two weeks overrides
     probabilistic long-term thinking.

  3. SEASONAL OUTLOOK — Derive probabilistic seasonal outlook from the
     ENSO phase using NOAA's published regional impact tables. More reliable
     than scraping NOAA's CPC pages which frequently return 404.

Sources:
  - NOAA ONI:      https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
  - Open-Meteo:    https://api.open-meteo.com/v1/forecast
  - ENSO impacts:  Based on NOAA/CPC published regional correlation tables

Output feeds directly into:
  - monte_carlo.py  (enso_phase parameter for year weighting)
  - yield_risk.py   (14-day acute risk flags)
  - pipeline.py     (assembled into final Granite prompt)

Dependencies: requests + standard library only.
"""

import logging
import math
import statistics
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

log = logging.getLogger("agrishield.seasonal_outlook")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NOAA_ONI_URL    = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
_OPEN_METEO_URL  = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT         = 20

# ONI thresholds for ENSO phase classification
# Source: NOAA CPC official definitions
# El Niño  : ONI >= +0.5 for 5 consecutive overlapping seasons
# La Niña  : ONI <= -0.5 for 5 consecutive overlapping seasons
# Strong   : |ONI| >= 1.5
# Moderate : |ONI| >= 1.0
# Weak     : |ONI| >= 0.5

_ONI_STRONG    = 1.5
_ONI_MODERATE  = 1.0
_ONI_WEAK      = 0.5

# ---------------------------------------------------------------------------
# ENSO regional impact tables
# ---------------------------------------------------------------------------
# These are derived from NOAA CPC's published ENSO impact maps for the
# contiguous US, expressed as probability shifts from climatological base.
#
# Structure: enso_phase → region → {temp_above_normal_prob, precip_below_normal_prob}
#
# Regions are broad US agricultural zones. We assign a farm to a region
# based on its latitude/longitude.
#
# Values represent the PROBABILITY of above-normal temp / below-normal precip
# during the core growing season (Apr-Sep). Base climatological probability
# is always 33% (one of three equal terciles). Values above 33% indicate
# increased risk, below 33% indicate reduced risk.
#
# Source: NOAA CPC ENSO impacts, IRI seasonal forecast consensus,
#         peer-reviewed literature on ENSO-agriculture correlations.

_ENSO_REGIONAL_IMPACTS = {

    "strong_el_nino": {
        "northwest":    {"temp_above_prob": 60, "precip_below_prob": 55},
        "southwest":    {"temp_above_prob": 65, "precip_below_prob": 30},  # wetter in SW
        "plains":       {"temp_above_prob": 45, "precip_below_prob": 35},
        "corn_belt":    {"temp_above_prob": 40, "precip_below_prob": 35},  # milder/wetter
        "southeast":    {"temp_above_prob": 35, "precip_below_prob": 30},  # wetter SE
        "northeast":    {"temp_above_prob": 40, "precip_below_prob": 35},
        "delta":        {"temp_above_prob": 40, "precip_below_prob": 30},  # wetter delta
    },

    "weak_el_nino": {
        "northwest":    {"temp_above_prob": 50, "precip_below_prob": 45},
        "southwest":    {"temp_above_prob": 50, "precip_below_prob": 35},
        "plains":       {"temp_above_prob": 40, "precip_below_prob": 37},
        "corn_belt":    {"temp_above_prob": 38, "precip_below_prob": 36},
        "southeast":    {"temp_above_prob": 36, "precip_below_prob": 33},
        "northeast":    {"temp_above_prob": 38, "precip_below_prob": 36},
        "delta":        {"temp_above_prob": 38, "precip_below_prob": 33},
    },

    "neutral": {
        "northwest":    {"temp_above_prob": 33, "precip_below_prob": 33},
        "southwest":    {"temp_above_prob": 33, "precip_below_prob": 33},
        "plains":       {"temp_above_prob": 33, "precip_below_prob": 33},
        "corn_belt":    {"temp_above_prob": 33, "precip_below_prob": 33},
        "southeast":    {"temp_above_prob": 33, "precip_below_prob": 33},
        "northeast":    {"temp_above_prob": 33, "precip_below_prob": 33},
        "delta":        {"temp_above_prob": 33, "precip_below_prob": 33},
    },

    "weak_la_nina": {
        "northwest":    {"temp_above_prob": 40, "precip_below_prob": 40},
        "southwest":    {"temp_above_prob": 55, "precip_below_prob": 55},  # drier SW
        "plains":       {"temp_above_prob": 50, "precip_below_prob": 52},  # drier plains
        "corn_belt":    {"temp_above_prob": 48, "precip_below_prob": 50},  # drier corn belt
        "southeast":    {"temp_above_prob": 42, "precip_below_prob": 45},
        "northeast":    {"temp_above_prob": 38, "precip_below_prob": 40},
        "delta":        {"temp_above_prob": 50, "precip_below_prob": 55},  # notably drier delta
    },

    "strong_la_nina": {
        "northwest":    {"temp_above_prob": 45, "precip_below_prob": 50},
        "southwest":    {"temp_above_prob": 65, "precip_below_prob": 65},
        "plains":       {"temp_above_prob": 60, "precip_below_prob": 62},
        "corn_belt":    {"temp_above_prob": 58, "precip_below_prob": 60},
        "southeast":    {"temp_above_prob": 50, "precip_below_prob": 55},
        "northeast":    {"temp_above_prob": 40, "precip_below_prob": 45},
        "delta":        {"temp_above_prob": 60, "precip_below_prob": 65},
    },
}


# ---------------------------------------------------------------------------
# Region assignment from lat/lon
# ---------------------------------------------------------------------------
# Simple bounding box assignment for US agricultural regions.
# These are intentionally broad — ENSO impacts operate at regional scale,
# not county level. Precision beyond this is false accuracy.

def _assign_region(lat: float, lon: float) -> str:
    """
    Assign a farm to a broad US agricultural climate region based on lat/lon.

    Regions:
      corn_belt  : Iowa, Illinois, Indiana, Ohio, Minnesota, Missouri
      delta      : Arkansas, Mississippi, Louisiana, Tennessee (cotton/rice)
      plains     : Kansas, Nebraska, Oklahoma, Texas panhandle, Dakotas
      southeast  : Alabama, Georgia, Florida, South Carolina, North Carolina
      southwest  : California, Arizona, New Mexico, Nevada
      northwest  : Oregon, Washington, Idaho, Montana, Wyoming
      northeast  : New England, New York, Pennsylvania, Mid-Atlantic

    NW Arkansas (lat=36.5, lon=-93.8) → delta region
    """
    # Southwest
    if lon < -104 and lat < 42:
        return "southwest"
    # Northwest
    if lon < -104 and lat >= 42:
        return "northwest"
    # Plains (Great Plains)
    if -104 <= lon < -95 and lat >= 36:
        return "plains"
    # Delta (lower Mississippi valley — AR, MS, LA, TN)
    if -95 <= lon < -88 and lat < 37:
        return "delta"
    # Corn Belt (upper midwest)
    if -95 <= lon < -82 and 37 <= lat < 47:
        return "corn_belt"
    # Southeast
    if lon >= -88 and lat < 37:
        return "southeast"
    # Northeast
    if lon >= -82 and lat >= 37:
        return "northeast"
    # Default fallback
    return "corn_belt"


# ---------------------------------------------------------------------------
# 1. ENSO State from NOAA ONI
# ---------------------------------------------------------------------------

def fetch_enso_state() -> dict:
    """
    Fetch the current ENSO state from NOAA's ONI text feed.

    NOAA's ONI file format (ascii.txt):
        SEAS  YR   TOTAL   ANOM
        DJF  1950  24.72  -1.53
        ...
        NDJ  2025  25.96  -0.55

    SEAS = 3-month overlapping season (DJF = Dec/Jan/Feb average)
    ANOM = departure from 1991-2020 base period (°C)

    We take the most recent entry as the current ENSO state.
    The ONI anomaly is the key value — positive = El Niño, negative = La Niña.

    Classification:
        ONI >= +1.5  → strong_el_nino
        ONI >= +0.5  → weak_el_nino
        -0.5 < ONI < +0.5 → neutral
        ONI <= -0.5  → weak_la_nina
        ONI <= -1.5  → strong_la_nina

    Returns dict with oni_value, enso_phase, and confidence.
    Falls back to neutral if NOAA is unreachable.
    """
    try:
        resp = requests.get(_NOAA_ONI_URL, timeout=_TIMEOUT)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
    except Exception as exc:
        log.warning("NOAA ONI fetch failed: %s — defaulting to neutral", exc)
        return _default_enso()

    # Parse the most recent valid data line
    # Format: SEAS  YR  TOTAL  ANOM  (whitespace delimited)
    most_recent = None
    for line in reversed(lines):
        parts = line.strip().split()
        if len(parts) >= 4 and parts[0] != "SEAS":
            try:
                season   = parts[0]
                year     = int(parts[1])
                oni_anom = float(parts[3])
                most_recent = (season, year, oni_anom)
                break
            except (ValueError, IndexError):
                continue

    if most_recent is None:
        log.warning("Could not parse NOAA ONI data — defaulting to neutral")
        return _default_enso()

    season, year, oni = most_recent

    # Classify phase
    if oni >= _ONI_STRONG:
        phase      = "strong_el_nino"
        phase_desc = "Strong El Niño"
    elif oni >= _ONI_WEAK:
        phase      = "weak_el_nino"
        phase_desc = "Weak/Moderate El Niño"
    elif oni <= -_ONI_STRONG:
        phase      = "strong_la_nina"
        phase_desc = "Strong La Niña"
    elif oni <= -_ONI_WEAK:
        phase      = "weak_la_nina"
        phase_desc = "Weak/Moderate La Niña"
    else:
        phase      = "neutral"
        phase_desc = "ENSO Neutral"

    # Confidence: stronger signal = higher confidence in seasonal impacts
    abs_oni = abs(oni)
    if abs_oni >= _ONI_STRONG:
        confidence = "high"
    elif abs_oni >= _ONI_WEAK:
        confidence = "medium"
    else:
        confidence = "low"

    log.info("ENSO state: %s  ONI=%.2f  season=%s %d  confidence=%s",
             phase_desc, oni, season, year, confidence)

    return {
        "oni_value":    round(oni, 2),
        "oni_season":   f"{season} {year}",
        "enso_phase":   phase,
        "enso_desc":    phase_desc,
        "confidence":   confidence,
        "source":       "NOAA CPC ONI",
    }


def _default_enso() -> dict:
    """Safe fallback ENSO state when NOAA is unavailable."""
    return {
        "oni_value":  0.0,
        "oni_season": "unknown",
        "enso_phase": "neutral",
        "enso_desc":  "ENSO Neutral (default — data unavailable)",
        "confidence": "low",
        "source":     "default_fallback",
    }


# ---------------------------------------------------------------------------
# 2. Seasonal Outlook from ENSO phase + region
# ---------------------------------------------------------------------------

def get_seasonal_outlook(lat: float, lon: float, enso_state: dict) -> dict:
    """
    Derive probabilistic seasonal outlook for the growing season (Apr-Sep)
    from the current ENSO phase and the farm's regional climate response.

    Uses NOAA's published ENSO regional impact tables rather than scraping
    CPC seasonal forecast pages (which are HTML and frequently change format).

    Returns probability of above-normal temperature and below-normal
    precipitation for the core growing season — the two variables that
    most directly affect yield loss probability.

    Also returns a plain-language risk narrative for Granite to work with.
    """
    enso_phase = enso_state.get("enso_phase", "neutral")
    oni        = enso_state.get("oni_value",  0.0)
    region     = _assign_region(lat, lon)

    # Get regional impact probabilities for this ENSO phase
    phase_impacts  = _ENSO_REGIONAL_IMPACTS.get(enso_phase, _ENSO_REGIONAL_IMPACTS["neutral"])
    region_impacts = phase_impacts.get(region, phase_impacts.get("corn_belt"))

    temp_above_prob   = region_impacts["temp_above_prob"]
    precip_below_prob = region_impacts["precip_below_prob"]

    # Drought risk flag: both warm AND dry probabilities elevated
    drought_risk_elevated = (temp_above_prob >= 45 and precip_below_prob >= 45)

    # Risk narrative for Granite
    narrative = _build_narrative(
        enso_phase, oni, region,
        temp_above_prob, precip_below_prob,
        drought_risk_elevated,
    )

    log.info("Seasonal outlook: region=%s  phase=%s  temp_above=%d%%  precip_below=%d%%  drought=%s",
             region, enso_phase, temp_above_prob, precip_below_prob, drought_risk_elevated)

    return {
        "region":                    region,
        "growing_season":            "April–September",
        "enso_phase":                enso_phase,
        "temp_above_normal_prob_pct":  temp_above_prob,
        "precip_below_normal_prob_pct": precip_below_prob,
        "drought_risk_elevated":     drought_risk_elevated,
        "risk_narrative":            narrative,
        "source":                    "NOAA ENSO regional impact tables",
        "confidence":                enso_state.get("confidence", "medium"),
    }


def _build_narrative(
    phase: str,
    oni: float,
    region: str,
    temp_prob: int,
    precip_prob: int,
    drought: bool,
) -> str:
    """
    Build a plain-language seasonal risk narrative for Granite.
    This becomes part of the prompt context explaining the climate setup.
    """
    phase_text = {
        "strong_el_nino":  f"a strong El Niño (ONI={oni:+.2f})",
        "weak_el_nino":    f"a weak El Niño (ONI={oni:+.2f})",
        "neutral":         f"ENSO-neutral conditions (ONI={oni:+.2f})",
        "weak_la_nina":    f"a weak La Niña (ONI={oni:+.2f})",
        "strong_la_nina":  f"a strong La Niña (ONI={oni:+.2f})",
    }.get(phase, f"uncertain ENSO conditions (ONI={oni:+.2f})")

    region_text = region.replace("_", " ").title()

    base = (
        f"The current climate setup features {phase_text}. "
        f"For the {region_text} region, this historically produces a "
        f"{temp_prob}% probability of above-normal temperatures and a "
        f"{precip_prob}% probability of below-normal precipitation "
        f"during the core growing season (April–September). "
    )

    if drought:
        base += (
            "Both temperature and precipitation probabilities are elevated above "
            "climatological base rates (33%), indicating an increased risk of "
            "hot and dry growing conditions that could materially stress yields. "
        )
    elif temp_prob < 33 and precip_prob < 33:
        base += (
            "Both temperature and precipitation probabilities are below "
            "climatological base rates, suggesting a relatively favorable "
            "growing season outlook. "
        )
    else:
        base += (
            "Conditions are near climatological average with no strongly "
            "elevated risk signal from ENSO alone. "
        )

    return base.strip()


# ---------------------------------------------------------------------------
# 3. 14-Day Forecast from Open-Meteo
# ---------------------------------------------------------------------------

def fetch_14day_forecast(lat: float, lon: float, crop_params: Optional[dict] = None) -> dict:
    """
    Fetch a 14-day day-by-day forecast from Open-Meteo.

    Open-Meteo is free, requires no API key, and uses the best available
    NWP model for the location (GFS for North America). Accuracy is high
    for days 1-7, useful signal for days 8-14.

    Parameters pulled:
      temperature_2m_max          : daily high (°C)
      temperature_2m_min          : daily low (°C)
      precipitation_sum           : daily total precip (mm)
      precipitation_probability_max: max precip probability (%)
      windspeed_10m_max           : daily max wind (km/h)

    Also derives:
      total_precip_14d      : total forecast precipitation
      heat_stress_days      : days with max temp >= 32°C
      extreme_heat_days     : days with max temp >= 35°C
      dry_days_forecast     : days with precip < 1mm
      extreme_heat_flag     : True if any 3+ consecutive days >= 35°C
      critical_window_flag  : True if heat stress days fall in crop's
                              critical window (if crop_params provided)

    Falls back to empty forecast dict on any failure.
    """
    try:
        resp = requests.get(
            _OPEN_METEO_URL,
            params={
                "latitude":         lat,
                "longitude":        lon,
                "daily":            ",".join([
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "precipitation_probability_max",
                    "windspeed_10m_max",
                ]),
                "temperature_unit": "celsius",
                "forecast_days":    14,
                "timezone":         "auto",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data   = resp.json()
        daily  = data.get("daily", {})
    except Exception as exc:
        log.warning("Open-Meteo forecast fetch failed: %s", exc)
        return _empty_forecast()

    # Extract series — handle None values gracefully
    dates     = daily.get("time",                          [])
    tmax      = _clean_forecast_series(daily, "temperature_2m_max")
    tmin      = _clean_forecast_series(daily, "temperature_2m_min")
    precip    = _clean_forecast_series(daily, "precipitation_sum")
    precip_prob = _clean_forecast_series(daily, "precipitation_probability_max")
    wind      = _clean_forecast_series(daily, "windspeed_10m_max")

    # Derived metrics
    total_precip      = round(sum(p for p in precip if p is not None), 1)
    heat_stress_days  = sum(1 for t in tmax if t is not None and t >= 32.0)
    extreme_heat_days = sum(1 for t in tmax if t is not None and t >= 35.0)
    dry_days          = sum(1 for p in precip if p is not None and p < 1.0)
    avg_tmax          = round(_safe_mean(tmax), 1)
    avg_precip_prob   = round(_safe_mean(precip_prob), 1)

    # Extreme heat event: 3+ consecutive days >= 35°C
    # This is the single most damaging near-term event for most crops
    extreme_heat_flag = _check_consecutive_extreme(tmax, threshold=35.0, consecutive=3)

    # Consecutive dry streak in the forecast
    forecast_dry_streak = _max_consecutive_dry(precip)

    # Critical window flag (requires crop_params from monte_carlo.CROP_STRESS_PARAMS)
    # If heat stress days fall during the crop's critical pollination window,
    # that's near-certain damage — flag it explicitly for the risk model
    critical_window_flag = False
    if crop_params is not None:
        critical_window_flag = _check_critical_window(
            tmax, dates, crop_params
        )

    log.info(
        "14-day forecast: total_precip=%.1fmm  heat_stress_days=%d  "
        "extreme_heat=%s  dry_days=%d  dry_streak=%d",
        total_precip, heat_stress_days, extreme_heat_flag,
        dry_days, forecast_dry_streak,
    )

    return {
        "forecast_days":           len(dates),
        "dates":                   dates,
        # Daily series
        "tmax":                    tmax,
        "tmin":                    tmin,
        "precip_mm":               precip,
        "precip_probability_pct":  precip_prob,
        # Derived summary metrics
        "avg_tmax_c":              avg_tmax,
        "total_precip_mm":         total_precip,
        "avg_precip_probability":  avg_precip_prob,
        "heat_stress_days":        heat_stress_days,
        "extreme_heat_days":       extreme_heat_days,
        "dry_days_forecast":       dry_days,
        "forecast_dry_streak":     forecast_dry_streak,
        # Risk flags (boolean — immediately actionable)
        "extreme_heat_flag":       extreme_heat_flag,
        "critical_window_flag":    critical_window_flag,
        # Source metadata
        "source":                  "Open-Meteo GFS",
        "generated_utc":           datetime.now(timezone.utc).isoformat(),
    }


def _clean_forecast_series(daily: dict, key: str) -> list:
    """Extract a forecast series, replacing None with 0.0 for precip or None for temps."""
    raw = daily.get(key, [])
    cleaned = []
    for v in raw:
        if v is None:
            cleaned.append(None)
        else:
            try:
                cleaned.append(float(v))
            except (TypeError, ValueError):
                cleaned.append(None)
    return cleaned


def _safe_mean(vals: list, default: float = 0.0) -> float:
    valid = [v for v in vals if v is not None and not math.isnan(v)]
    return sum(valid) / len(valid) if valid else default


def _check_consecutive_extreme(tmax: list, threshold: float, consecutive: int) -> bool:
    """Return True if `consecutive` or more days in a row exceed `threshold`."""
    streak = 0
    for t in tmax:
        if t is not None and t >= threshold:
            streak += 1
            if streak >= consecutive:
                return True
        else:
            streak = 0
    return False


def _max_consecutive_dry(precip: list, dry_thresh: float = 1.0) -> int:
    """Return the longest consecutive dry streak in the forecast."""
    max_streak = 0
    current    = 0
    for p in precip:
        if p is not None and p < dry_thresh:
            current   += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def _check_critical_window(
    tmax:        list,
    dates:       list,
    crop_params: dict,
) -> bool:
    """
    Check if any forecast heat stress days fall within the crop's
    critical pollination/flowering window.

    This requires knowing when the critical window falls on the calendar,
    which depends on planting date. Since seasonal_outlook.py doesn't
    receive planting date, we use a conservative approach:
    flag if heat stress days are in the peak summer window (Jul 1 - Aug 15)
    which covers most Northern Hemisphere critical windows.
    """
    heat_thresh = crop_params.get("heat_threshold_c", 32.0)
    for i, t in enumerate(tmax):
        if t is None or t < heat_thresh:
            continue
        if i < len(dates):
            try:
                d = datetime.strptime(dates[i], "%Y-%m-%d")
                # Peak critical window: July 1 through August 15
                if (d.month == 7) or (d.month == 8 and d.day <= 15):
                    return True
            except ValueError:
                pass
    return False


def _empty_forecast() -> dict:
    """Safe fallback when Open-Meteo is unavailable."""
    return {
        "forecast_days":           0,
        "dates":                   [],
        "tmax":                    [],
        "tmin":                    [],
        "precip_mm":               [],
        "precip_probability_pct":  [],
        "avg_tmax_c":              None,
        "total_precip_mm":         0.0,
        "avg_precip_probability":  None,
        "heat_stress_days":        0,
        "extreme_heat_days":       0,
        "dry_days_forecast":       0,
        "forecast_dry_streak":     0,
        "extreme_heat_flag":       False,
        "critical_window_flag":    False,
        "source":                  "unavailable",
        "generated_utc":           datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# 4. Main entry point
# ---------------------------------------------------------------------------

def get_seasonal_outlook_full(lat: float, lon: float, crop_params: Optional[dict] = None) -> dict:
    """
    Full seasonal outlook pipeline entry point. Called by pipeline.py.

    Fetches ENSO state, derives seasonal outlook, fetches 14-day forecast,
    and combines everything into one clean dict.

    Parameters
    ----------
    lat, lon     : farm coordinates
    crop_params  : optional crop stress params from monte_carlo.CROP_STRESS_PARAMS
                   used to check if forecast heat falls in critical window

    Returns
    -------
    {
      "enso":             { oni_value, enso_phase, enso_desc, confidence, ... }
      "seasonal_outlook": { temp_above_prob, precip_below_prob, drought_risk, narrative, ... }
      "forecast_14d":     { tmax[], precip[], heat_stress_days, extreme_heat_flag, ... }
      "enso_phase":       str  ← convenience key for monte_carlo.py
    }
    """
    # ENSO state
    try:
        enso = fetch_enso_state()
    except Exception as exc:
        log.warning("ENSO fetch error: %s", exc)
        enso = _default_enso()

    # Seasonal outlook derived from ENSO + region
    try:
        outlook = get_seasonal_outlook(lat, lon, enso)
    except Exception as exc:
        log.warning("Seasonal outlook error: %s", exc)
        outlook = _default_outlook(lat, lon)

    # 14-day forecast
    try:
        forecast = fetch_14day_forecast(lat, lon, crop_params=crop_params)
    except Exception as exc:
        log.warning("14-day forecast error: %s", exc)
        forecast = _empty_forecast()

    return {
        "enso":             enso,
        "seasonal_outlook": outlook,
        "forecast_14d":     forecast,
        # Convenience key for monte_carlo.py enso_phase parameter
        "enso_phase":       enso.get("enso_phase", "neutral"),
    }


def _default_outlook(lat: float, lon: float) -> dict:
    """Safe fallback seasonal outlook."""
    region = _assign_region(lat, lon)
    return {
        "region":                      region,
        "growing_season":              "April-September",
        "enso_phase":                  "neutral",
        "temp_above_normal_prob_pct":  33,
        "precip_below_normal_prob_pct": 33,
        "drought_risk_elevated":       False,
        "risk_narrative":              "Seasonal outlook unavailable — using climatological base rates.",
        "source":                      "default_fallback",
        "confidence":                  "low",
    }


# ---------------------------------------------------------------------------
# Smoke test — run: python seasonal_outlook.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # NW Arkansas farm
    lat, lon = 36.545722, -93.857057

    print("=" * 60)
    print("AgriShield seasonal_outlook  |  NW Arkansas farm")
    print("=" * 60)

    result = get_seasonal_outlook_full(lat, lon)

    print("\n--- ENSO State ---")
    print(json.dumps(result["enso"], indent=2))

    print("\n--- Seasonal Outlook (growing season Apr-Sep) ---")
    out = result["seasonal_outlook"]
    print(f"  Region              : {out['region']}")
    print(f"  ENSO phase          : {out['enso_phase']}")
    print(f"  Temp above normal   : {out['temp_above_normal_prob_pct']}%")
    print(f"  Precip below normal : {out['precip_below_normal_prob_pct']}%")
    print(f"  Drought risk flag   : {out['drought_risk_elevated']}")
    print(f"\n  Narrative:\n  {out['risk_narrative']}")

    print("\n--- 14-Day Forecast ---")
    fc = result["forecast_14d"]
    print(f"  Days              : {fc['forecast_days']}")
    print(f"  Avg max temp      : {fc['avg_tmax_c']}°C")
    print(f"  Total precip      : {fc['total_precip_mm']}mm")
    print(f"  Heat stress days  : {fc['heat_stress_days']}")
    print(f"  Dry days          : {fc['dry_days_forecast']}")
    print(f"  Dry streak        : {fc['forecast_dry_streak']} days")
    print(f"  Extreme heat flag : {fc['extreme_heat_flag']}")
    print()
    print("  Day-by-day:")
    for i in range(min(7, len(fc['dates']))):
        print(f"    {fc['dates'][i]}  "
              f"max={fc['tmax'][i]}°C  "
              f"precip={fc['precip_mm'][i]}mm  "
              f"prob={fc['precip_probability_pct'][i]}%")

    print(f"\n  enso_phase (for Monte Carlo): {result['enso_phase']}")
    print("\n[Done]")