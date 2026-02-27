"""
agri_data.py
============
AgriShield – Soil Data + Current Climate Anomaly Signal
--------------------------------------------------------
Responsible for two things only:

  1. SOIL — Pull soil properties from SoilGrids v2 across multiple depth
     layers (0-5cm, 5-15cm, 15-30cm) and compute a crop-aware, root-zone-
     weighted soil quality score including pH. Soil score modulates how much
     damage a given climate stress event causes. The same drought hits sandy
     low-SOC soil much harder than clay loam.

  2. CURRENT ANOMALY SIGNAL — Pull the last 30 days of actual weather from
     NASA POWER and compare against the 30-year climatology baseline for the
     overlapping calendar period (handles month boundaries). This tells the
     pipeline whether THIS season is trending hot/dry/wet RIGHT NOW, including
     detection of dry streaks that ended recently, not just current streaks.

Updates vs v1:
  - Multi-depth soil averaging (0-5, 5-15, 15-30cm) weighted by root zone
  - pH added to soil score (affects nutrient availability)
  - Anomaly correctly weights two calendar months when 30-day window spans both
  - Dry streak now tracks both current AND recent max (not just current tail)

What this file does NOT do (handled elsewhere):
  - Long-term Monte Carlo simulation        → monte_carlo.py
  - ENSO / seasonal outlook                 → seasonal_outlook.py
  - 14-day forecast                         → seasonal_outlook.py
  - Futures prices                          → market_data.py
  - Final risk scoring                      → yield_risk.py

Dependencies: requests + standard library only.
"""

import logging
import math
from datetime import datetime, date, timedelta, timezone
from typing import Optional

import requests

log = logging.getLogger("agrishield.agri_data")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SOILGRIDS_URL  = "https://rest.isric.org/soilgrids/v2.0/properties/query"
_NASA_DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
_NASA_CLIM_URL  = "https://power.larc.nasa.gov/api/temporal/climatology/point"

_TIMEOUT = 30
_MISSING = -900.0   # NASA POWER missing value sentinel

_DAILY_PARAMS = [
    "T2M",          # avg temperature (°C)
    "T2M_MAX",      # max temperature (°C)
    "PRECTOTCORR",  # bias-corrected precipitation (mm/day)
    "RH2M",         # relative humidity (%)
]

_CLIM_PARAMS = ["T2M", "PRECTOTCORR"]

# NASA POWER climatology month key format
_MONTH_KEYS = ["JAN","FEB","MAR","APR","MAY","JUN",
               "JUL","AUG","SEP","OCT","NOV","DEC"]

# ---------------------------------------------------------------------------
# Soil depth layers and root-zone weights
# ---------------------------------------------------------------------------
# We pull three depth layers and weight them by how much each contributes
# to the root zone water and nutrient supply for typical row crops.
#
#   0-5cm   → 20%  surface layer, most variable, least representative
#   5-15cm  → 35%  upper root zone, most active nutrient uptake
#   15-30cm → 45%  mid root zone, primary water storage reservoir
#
# Deeper layers (30-60cm, 60-100cm) would be even better for corn/soy
# but SoilGrids deeper layers have higher uncertainty. These three give
# a meaningful improvement over surface-only at acceptable data quality.

_SOIL_DEPTHS = ["0-5cm", "5-15cm", "15-30cm"]
_DEPTH_WEIGHTS = {"0-5cm": 0.20, "5-15cm": 0.35, "15-30cm": 0.45}

# ---------------------------------------------------------------------------
# Crop-aware soil scoring
# ---------------------------------------------------------------------------
# Each crop has different ideal soil properties based on agronomic literature.
#
# ideal_clay      : optimal clay % for this crop's texture needs
# min_soc         : minimum SOC (g/kg) for acceptable fertility
# water_tolerance : how well crop handles waterlogged/heavy clay soil
#                   1.0 = thrives in wet/heavy (rice)
#                   0.0 = needs well-drained light soil (wheat, canola)
# ideal_ph_low    : lower bound of optimal pH range
# ideal_ph_high   : upper bound of optimal pH range
#                   Outside this range nutrient availability drops sharply

_CROP_SOIL_PREFS = {
    "corn":     {"ideal_clay": 30.0, "min_soc": 10.0, "water_tolerance": 0.4,
                 "ideal_ph_low": 5.8, "ideal_ph_high": 7.0},
    "soybeans": {"ideal_clay": 28.0, "min_soc": 10.0, "water_tolerance": 0.4,
                 "ideal_ph_low": 6.0, "ideal_ph_high": 7.0},
    "wheat":    {"ideal_clay": 25.0, "min_soc":  8.0, "water_tolerance": 0.2,
                 "ideal_ph_low": 6.0, "ideal_ph_high": 7.5},
    "rice":     {"ideal_clay": 45.0, "min_soc": 12.0, "water_tolerance": 1.0,
                 "ideal_ph_low": 5.5, "ideal_ph_high": 6.5},
    "cotton":   {"ideal_clay": 25.0, "min_soc":  8.0, "water_tolerance": 0.2,
                 "ideal_ph_low": 5.8, "ideal_ph_high": 7.0},
    "oats":     {"ideal_clay": 25.0, "min_soc":  8.0, "water_tolerance": 0.3,
                 "ideal_ph_low": 6.0, "ideal_ph_high": 7.0},
    "canola":   {"ideal_clay": 25.0, "min_soc":  8.0, "water_tolerance": 0.2,
                 "ideal_ph_low": 5.5, "ideal_ph_high": 7.0},
    # Safe default for unknown specialty crops
    "_default": {"ideal_clay": 30.0, "min_soc": 10.0, "water_tolerance": 0.4,
                 "ideal_ph_low": 6.0, "ideal_ph_high": 7.0},
}

_SOC_NORMALIZER = 30.0   # g/kg — SOC values above this get full score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mean(vals: list, default: float = 0.0) -> float:
    valid = [v for v in vals if v is not None and not math.isnan(v)]
    return sum(valid) / len(valid) if valid else default


def _safe_sum(vals: list, default: float = 0.0) -> float:
    valid = [v for v in vals if v is not None and not math.isnan(v)]
    return sum(valid) if valid else default


def _clean_series(raw_dict: dict, key: str) -> list:
    """Extract and clean a NASA POWER parameter series, filtering sentinels."""
    series_dict = raw_dict.get(key, {})
    if not isinstance(series_dict, dict):
        return []
    cleaned = []
    for val in series_dict.values():
        try:
            fval = float(val)
            if fval > _MISSING:
                cleaned.append(fval)
        except (TypeError, ValueError):
            pass
    return cleaned


def _date_str(d: date) -> str:
    return d.strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# 1. Soil Data — multi-depth, pH-aware, crop-specific scoring
# ---------------------------------------------------------------------------

def _fetch_soilgrids(lat: float, lon: float) -> dict:
    """
    Fetch soil properties across three depth layers from SoilGrids v2.

    Pulling multiple depths gives a much better picture of the root zone
    than surface alone. Corn roots go 100-150cm deep — a drought stress
    model based only on 0-5cm surface soil is systematically optimistic.

    Depths: 0-5cm, 5-15cm, 15-30cm (weighted 20/35/45% by root zone).
    Properties: clay, soc, sand, silt, phh2o (pH in water).
    Returns raw JSON or {} on any failure.
    """
    try:
        resp = requests.get(
            _SOILGRIDS_URL,
            params={
                "lat":      lat,
                "lon":      lon,
                "property": ["phh2o", "soc", "clay", "silt", "sand"],
                "depth":    _SOIL_DEPTHS,
                "value":    "mean",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.warning("SoilGrids request failed: %s", exc)
        return {}


def _extract_soil_prop_weighted(raw: dict, prop_name: str) -> Optional[float]:
    """
    Extract a depth-weighted mean soil property from SoilGrids v2 response.

    SoilGrids v2 structure:
      raw["properties"]["layers"][i]["name"]           → property name
      raw["properties"]["layers"][i]["depths"][j]["label"]        → depth label e.g. "0-5cm"
      raw["properties"]["layers"][i]["depths"][j]["values"]["mean"] → value

    Unit conversions applied here:
      clay/silt/sand : g/kg  → divide by 10 for %
      soc            : dg/kg → divide by 10 for g/kg
      phh2o          : pH×10 → divide by 10 for actual pH

    Weights each depth layer by _DEPTH_WEIGHTS and returns weighted mean.
    Falls back to whatever depths are available if some are missing.
    """
    try:
        layers = raw["properties"]["layers"]
        for layer in layers:
            if layer.get("name") != prop_name:
                continue

            weighted_sum   = 0.0
            weight_total   = 0.0

            for depth_entry in layer.get("depths", []):
                depth_label = depth_entry.get("label", "")
                val         = depth_entry.get("values", {}).get("mean")
                weight      = _DEPTH_WEIGHTS.get(depth_label, 0.0)

                if val is not None and weight > 0:
                    weighted_sum  += float(val) * weight
                    weight_total  += weight

            if weight_total > 0:
                return weighted_sum / weight_total

    except (KeyError, TypeError, ValueError):
        pass
    return None


def _compute_ph_score(ph_actual: float, ph_low: float, ph_high: float) -> float:
    """
    Score soil pH for a given crop's optimal range.

    pH outside the optimal range reduces nutrient availability:
      - Too acidic (low pH): aluminum toxicity, P/Ca/Mg lockout
      - Too alkaline (high pH): iron/manganese/zinc lockout

    Score = 1.0 within optimal range, decays linearly outside it.
    A full pH unit outside the range = 0.5 score (significant impact).
    Two units outside = 0.0 (severe nutrient stress likely).
    """
    if ph_low <= ph_actual <= ph_high:
        return 1.0
    elif ph_actual < ph_low:
        deviation = ph_low - ph_actual
    else:
        deviation = ph_actual - ph_high
    return max(0.0, 1.0 - deviation / 2.0)


def get_soil_data(lat: float, lon: float, crop_type: str) -> dict:
    """
    Fetch multi-depth soil properties and compute a crop-aware soil score.

    Soil score [0, 1] has three components (updated from v1):
      1. Clay score  — how close weighted clay % is to crop's ideal texture
                       accounts for water_tolerance (rice needs heavy clay)
      2. SOC score   — higher organic carbon = better water/nutrient retention
                       normalised against _SOC_NORMALIZER (30 g/kg)
      3. pH score    — NEW: how close weighted pH is to crop's optimal range
                       pH outside range causes nutrient lockout regardless of
                       good clay/SOC

    Final score = weighted average: clay 35%, SOC 40%, pH 25%
    pH weighted lowest because it can be amended (lime, sulfur).
    Clay and SOC reflect structural properties that take decades to change.

    Also computes drought_buffer_days — how many days of evapotranspiration
    demand the root-zone soil can meet before crop water stress begins.
    Uses multi-depth weighted values for accuracy.
    """
    raw = _fetch_soilgrids(lat, lon)

    crop_key  = crop_type.lower().strip()
    soil_pref = _CROP_SOIL_PREFS.get(crop_key, _CROP_SOIL_PREFS["_default"])

    if not raw:
        log.warning("SoilGrids unavailable — using neutral soil defaults")
        return _default_soil_result(crop_type)

    # Extract weighted multi-depth values
    clay_raw = _extract_soil_prop_weighted(raw, "clay")
    soc_raw  = _extract_soil_prop_weighted(raw, "soc")
    sand_raw = _extract_soil_prop_weighted(raw, "sand")
    ph_raw   = _extract_soil_prop_weighted(raw, "phh2o")

    if clay_raw is None and soc_raw is None:
        log.warning("Could not parse weighted soil properties — using defaults")
        return _default_soil_result(crop_type)

    components        = []
    component_weights = []

    # --- Clay score (35% of final score) ---
    if clay_raw is not None:
        clay_pct   = clay_raw / 10.0
        ideal_clay = soil_pref["ideal_clay"]
        water_tol  = soil_pref["water_tolerance"]
        deviation  = abs(clay_pct - ideal_clay)
        clay_score = max(0.0, 1.0 - deviation / ideal_clay)

        # Water tolerance adjustment for crops that prefer heavy clay
        if clay_pct > ideal_clay:
            excess      = (clay_pct - ideal_clay) / 100.0
            clay_score  = min(1.0, clay_score * (1.0 + water_tol * excess * 2))

        components.append(clay_score)
        component_weights.append(0.35)
        log.info("Soil clay (weighted depth): %.1f%%  ideal: %.1f%%  score: %.3f",
                 clay_pct, ideal_clay, clay_score)

    # --- SOC score (40% of final score) ---
    if soc_raw is not None:
        soc_gkg   = soc_raw / 10.0
        soc_score = min(1.0, soc_gkg / _SOC_NORMALIZER)
        components.append(soc_score)
        component_weights.append(0.40)
        log.info("Soil SOC (weighted depth): %.2f g/kg  score: %.3f", soc_gkg, soc_score)

    # --- pH score (25% of final score) — NEW ---
    if ph_raw is not None:
        ph_actual = ph_raw / 10.0      # SoilGrids returns pH × 10
        ph_score  = _compute_ph_score(
            ph_actual,
            soil_pref["ideal_ph_low"],
            soil_pref["ideal_ph_high"],
        )
        components.append(ph_score)
        component_weights.append(0.25)
        log.info("Soil pH (weighted depth): %.1f  optimal: %.1f-%.1f  score: %.3f",
                 ph_actual, soil_pref["ideal_ph_low"], soil_pref["ideal_ph_high"], ph_score)

    # Weighted average of available components
    if components:
        total_w    = sum(component_weights)
        soil_score = sum(c * w for c, w in zip(components, component_weights)) / total_w
    else:
        soil_score = 0.5

    # Drought buffer using multi-depth weighted values
    clay_pct_val = (clay_raw / 10.0) if clay_raw else 25.0
    soc_val      = (soc_raw  / 10.0) if soc_raw  else 10.0
    drought_buffer_days = round((clay_pct_val * 0.4 + soc_val * 0.5) / 4.0, 1)
    drought_buffer_days = max(2.0, min(20.0, drought_buffer_days))

    return {
        "soil_score":           round(soil_score, 4),
        "drought_buffer_days":  drought_buffer_days,
        "crop_type":            crop_type,
        "properties": {
            "clay_pct":  round(clay_raw / 10.0, 1) if clay_raw else None,
            "soc_gkg":   round(soc_raw  / 10.0, 2) if soc_raw  else None,
            "sand_pct":  round(sand_raw / 10.0, 1) if sand_raw else None,
            "ph":        round(ph_raw   / 10.0, 1) if ph_raw   else None,
            "depth_weighted": True,    # flag so pipeline knows this is multi-depth
        },
    }


def _default_soil_result(crop_type: str) -> dict:
    """Safe fallback when SoilGrids is unavailable."""
    return {
        "soil_score":          0.5,
        "drought_buffer_days": 8.0,
        "crop_type":           crop_type,
        "properties": {
            "clay_pct": None,
            "soc_gkg":  None,
            "sand_pct": None,
            "ph":       None,
            "depth_weighted": False,
        },
    }


# ---------------------------------------------------------------------------
# 2. Current Climate Anomaly Signal
# ---------------------------------------------------------------------------

def _fetch_recent_daily(lat: float, lon: float, days_back: int = 30) -> dict:
    """
    Pull the last `days_back` days of daily weather from NASA POWER.
    Returns cleaned dict of float lists. Empty lists on failure.
    """
    today      = datetime.now(timezone.utc).date()
    end_date   = today - timedelta(days=2)
    start_date = end_date - timedelta(days=days_back - 1)

    try:
        resp = requests.get(
            _NASA_DAILY_URL,
            params={
                "parameters": ",".join(_DAILY_PARAMS),
                "community":  "AG",
                "longitude":  lon,
                "latitude":   lat,
                "start":      _date_str(start_date),
                "end":        _date_str(end_date),
                "format":     "JSON",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        param_data = resp.json()["properties"]["parameter"]
    except Exception as exc:
        log.warning("NASA POWER daily fetch failed: %s", exc)
        return {p: [] for p in _DAILY_PARAMS}

    return {p: _clean_series(param_data, p) for p in _DAILY_PARAMS}


def _fetch_climatology(lat: float, lon: float) -> dict:
    """
    Pull NASA POWER 30-year monthly climatology normals.
    Returns dict keyed by month number (1-12).
    NASA returns 3-letter month keys: JAN, FEB, ..., DEC.
    """
    try:
        resp = requests.get(
            _NASA_CLIM_URL,
            params={
                "parameters": ",".join(_CLIM_PARAMS),
                "community":  "AG",
                "longitude":  lon,
                "latitude":   lat,
                "format":     "JSON",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        param_data = resp.json()["properties"]["parameter"]
    except Exception as exc:
        log.warning("NASA POWER climatology fetch failed: %s", exc)
        return {}

    monthly = {}
    for m, month_key in enumerate(_MONTH_KEYS, start=1):
        entry = {}
        for p in _CLIM_PARAMS:
            val = param_data.get(p, {}).get(month_key)
            if val is not None:
                try:
                    fval = float(val)
                    entry[p] = fval if fval > _MISSING else None
                except (TypeError, ValueError):
                    entry[p] = None
        monthly[m] = entry

    return monthly


def _get_two_month_climatology_normal(
    clim: dict,
    start_date: date,
    end_date:   date,
) -> dict:
    """
    Compute the climatological normal for a period that may span two months.

    Fix for v1 bug: the 30-day window often crosses a month boundary
    (e.g. Jan 28 – Feb 26). Using only the current month's normal was
    systematically biased in the first/last week of each month.

    We weight each month's normal by how many days of the window fall in it.
    e.g. if 8 days are in January and 22 days are in February:
      normal_temp = (jan_temp * 8 + feb_temp * 22) / 30

    Returns weighted T2M and PRECTOTCORR normals for the period.
    PRECTOTCORR from NASA is mm/day average, so we return mm/day
    and let the caller multiply by days for totals.
    """
    total_days = (end_date - start_date).days + 1
    temp_weighted   = 0.0
    precip_weighted = 0.0
    days_counted    = 0

    current = start_date
    while current <= end_date:
        m        = current.month
        m_data   = clim.get(m, {})
        t2m      = m_data.get("T2M")
        prec     = m_data.get("PRECTOTCORR")

        if t2m is not None:
            temp_weighted   += t2m
        if prec is not None:
            precip_weighted += prec

        days_counted += 1
        current      += timedelta(days=1)

    if days_counted == 0:
        return {"T2M": None, "PRECTOTCORR_daily": None}

    return {
        "T2M":               round(temp_weighted   / days_counted, 2),
        "PRECTOTCORR_daily": round(precip_weighted / days_counted, 3),
    }


def _compute_dry_streaks(precip_series: list) -> dict:
    """
    Compute both current AND recent max dry streaks.

    Fix for v1 bug: v1 only counted backward from the last observation,
    so a significant dry streak that ended a week ago was invisible.
    Both metrics are useful:
      current_dry_streak : how many consecutive dry days right now
      max_dry_streak_30d : longest dry streak anywhere in the 30-day window
                           (the more agronomically relevant stress indicator)
    """
    # Current streak — count backward from most recent day
    current_streak = 0
    for p in reversed(precip_series):
        if p < 1.0:
            current_streak += 1
        else:
            break

    # Max streak — scan entire period
    max_streak     = 0
    running        = 0
    for p in precip_series:
        if p < 1.0:
            running    += 1
            max_streak = max(max_streak, running)
        else:
            running = 0

    return {
        "current_dry_streak":    current_streak,
        "max_dry_streak_30d":    max_streak,
    }


def get_current_anomaly(lat: float, lon: float, days_back: int = 30) -> dict:
    """
    Compute how the current season is tracking versus the 30-year normal.

    Returns anomaly metrics that tell the Monte Carlo whether this year
    is already trending above or below normal before the growing season.

    Improvements over v1:
      - Two-month weighted climatology baseline (handles month boundaries)
      - Both current AND max dry streak tracked
      - pH not applicable here (weather metric only)
    """
    today      = datetime.now(timezone.utc).date()
    end_date   = today - timedelta(days=2)
    start_date = end_date - timedelta(days=days_back - 1)

    # Fetch recent actual weather and climatology baseline
    recent = _fetch_recent_daily(lat, lon, days_back=days_back)
    clim   = _fetch_climatology(lat, lon)

    # Recent observed values
    recent_temp   = _safe_mean(recent.get("T2M",         []), default=None)
    recent_precip = _safe_sum( recent.get("PRECTOTCORR", []), default=None)
    recent_tmax   = recent.get("T2M_MAX", [])

    # Climatological normal for the exact period (two-month weighted)
    period_normal = _get_two_month_climatology_normal(clim, start_date, end_date)
    clim_temp             = period_normal["T2M"]
    clim_precip_daily     = period_normal["PRECTOTCORR_daily"]
    clim_precip_scaled    = (clim_precip_daily * days_back) if clim_precip_daily else None

    # Compute anomalies
    temp_anomaly = None
    if recent_temp is not None and clim_temp is not None:
        temp_anomaly = round(recent_temp - clim_temp, 2)

    precip_anomaly_pct = None
    if recent_precip is not None and clim_precip_scaled and clim_precip_scaled > 0:
        precip_anomaly_pct = round(
            ((recent_precip - clim_precip_scaled) / clim_precip_scaled) * 100.0, 1
        )

    # Heat stress days
    heat_stress_days = sum(1 for t in recent_tmax if t >= 32.0)

    # Dry streaks — current AND max over period
    dry_streaks = _compute_dry_streaks(recent.get("PRECTOTCORR", []))

    # Anomaly signal label for Monte Carlo and Granite
    warm = temp_anomaly is not None and temp_anomaly >  1.0
    cool = temp_anomaly is not None and temp_anomaly < -1.0
    dry  = precip_anomaly_pct is not None and precip_anomaly_pct < -20
    wet  = precip_anomaly_pct is not None and precip_anomaly_pct >  20

    if warm and dry:   signal = "warm_and_dry"
    elif warm and wet: signal = "warm_and_wet"
    elif cool and dry: signal = "cool_and_dry"
    elif cool and wet: signal = "cool_and_wet"
    elif dry:          signal = "dry"
    elif wet:          signal = "wet"
    elif warm:         signal = "warm"
    elif cool:         signal = "cool"
    else:              signal = "normal"

    log.info("Current anomaly: temp=%s°C  precip=%s%%  signal=%s  dry_streak=%sd",
             temp_anomaly, precip_anomaly_pct, signal,
             dry_streaks["current_dry_streak"])

    return {
        "period_start":              str(start_date),
        "period_end":                str(end_date),
        "days_back":                 days_back,
        # Observed
        "recent_avg_temp_c":         round(recent_temp,   2) if recent_temp   is not None else None,
        "recent_total_precip_mm":    round(recent_precip, 1) if recent_precip is not None else None,
        "heat_stress_days":          heat_stress_days,
        # Streaks (v1 only had current, now both)
        "current_dry_streak":        dry_streaks["current_dry_streak"],
        "max_dry_streak_30d":        dry_streaks["max_dry_streak_30d"],
        # Baseline
        "clim_normal_temp_c":        clim_temp,
        "clim_normal_precip_mm":     round(clim_precip_scaled, 1) if clim_precip_scaled else None,
        # Anomalies
        "temp_anomaly_c":            temp_anomaly,
        "precip_anomaly_pct":        precip_anomaly_pct,
        # Signal
        "anomaly_signal":            signal,
    }


# ---------------------------------------------------------------------------
# 3. Main entry point — accepts frontend JSON, handles multiple crops
# ---------------------------------------------------------------------------

def get_farm_soil_and_anomaly(farm_input: dict) -> dict:
    """
    Main entry point called by pipeline.py.

    Accepts the full frontend JSON and returns soil + anomaly data
    for each crop at the farm location.

    Note: soil is fetched once per crop (crop-aware scoring differs).
          anomaly is fetched once for the location (same weather for all crops).
    """
    farm = farm_input.get("farm", {})
    lat  = farm.get("lat")
    lon  = farm.get("lon")

    if lat is None or lon is None:
        log.error("Farm lat/lon missing from input")
        return _empty_farm_result(farm_input)

    log.info("Processing farm at lat=%.6f lon=%.6f", lat, lon)

    # Current anomaly — same for all crops at this location
    try:
        anomaly = get_current_anomaly(lat, lon, days_back=30)
    except Exception as exc:
        log.warning("Anomaly fetch failed: %s", exc)
        anomaly = _empty_anomaly()

    # Soil per crop — scoring differs by crop type
    crops_input  = farm_input.get("crops", [])
    crop_results = {}

    for crop_entry in crops_input:
        crop_name = crop_entry.get("name", "unknown")
        log.info("Fetching soil data for crop: %s", crop_name)
        try:
            soil = get_soil_data(lat, lon, crop_name)
        except Exception as exc:
            log.warning("Soil fetch failed for %s: %s", crop_name, exc)
            soil = _default_soil_result(crop_name)

        crop_results[crop_name] = {
            "acres": crop_entry.get("acres", 0),
            "soil":  soil,
        }

    return {
        "location":        {"lat": lat, "lon": lon},
        "current_anomaly": anomaly,
        "crops":           crop_results,
    }


def _empty_anomaly() -> dict:
    """Safe fallback anomaly dict when APIs are unavailable."""
    return {
        "period_start": None, "period_end": None, "days_back": 30,
        "recent_avg_temp_c": None, "recent_total_precip_mm": None,
        "heat_stress_days": 0, "current_dry_streak": 0, "max_dry_streak_30d": 0,
        "clim_normal_temp_c": None, "clim_normal_precip_mm": None,
        "temp_anomaly_c": None, "precip_anomaly_pct": None,
        "anomaly_signal": "unknown",
    }


def _empty_farm_result(farm_input: dict) -> dict:
    """Safe fallback for the entire farm when coordinates are missing."""
    return {
        "location":        {"lat": None, "lon": None},
        "current_anomaly": _empty_anomaly(),
        "crops": {
            c.get("name", "unknown"): {
                "acres": c.get("acres", 0),
                "soil":  _default_soil_result(c.get("name", "unknown")),
            }
            for c in farm_input.get("crops", [])
        },
    }


# ---------------------------------------------------------------------------
# Smoke test — run: python agri_data.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    farm_input = {
        "farm": {"lat": 36.545722, "lon": -93.857057},
        "crops": [
            {"name": "Rice",     "acres": 50.0},
            {"name": "Wheat",    "acres": 20.0},
            {"name": "Soybeans", "acres": 30.0},
        ],
        "planting_window": {"start": "2026-02-28", "end": "2026-05-28"},
        "preferences": {"risk_aversion_0_100": 65.0, "hedge_budget_pct_revenue": 3.0},
    }

    print("=" * 60)
    print("AgriShield agri_data v2  |  NW Arkansas farm")
    print("=" * 60)

    result = get_farm_soil_and_anomaly(farm_input)

    print("\n--- Current Climate Anomaly Signal ---")
    print(json.dumps(result["current_anomaly"], indent=2))

    print("\n--- Soil Data Per Crop (multi-depth, pH-aware) ---")
    for crop_name, data in result["crops"].items():
        s = data["soil"]
        print(f"\n  {crop_name} ({data['acres']} acres):")
        print(f"    soil_score         : {s['soil_score']}")
        print(f"    drought_buffer_days: {s['drought_buffer_days']}")
        print(f"    properties         : {s['properties']}")

    print("\n[Done]")