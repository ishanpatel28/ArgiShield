"""
monte_carlo.py
==============
AgriShield – Weighted Monte Carlo Yield Simulation Engine
----------------------------------------------------------
How it works:
  1. Pull 30 years of historical daily weather from NASA POWER
     for the farm's exact lat/lon.
  2. Assign each historical year an exponential decay weight —
     recent years count more to reflect climate change reality.
  3. Tilt weights further based on:
       a. ENSO phase (La Nina years upweighted during La Nina)
       b. Current anomaly signal from agri_data.py
          ("dry" → upweight historically dry years even more)
  4. Run N simulations. Each simulation:
       a. Randomly samples a historical year (weighted)
       b. Extracts growing-season stress metrics for that year
       c. Passes metrics through crop stress → yield loss function
       d. Records the yield outcome as % of normal
  5. From 10,000 outcomes compute the full distribution:
       P10, P25, P50, P75, P90, std_dev, skewness
     plus median stress metrics for the evidence block.

Changes vs v1:
  - ENSO yield bias now comes from seasonal_outlook regional impact table
    instead of hardcoded corn-belt-only values
  - Crop-specific planting DOY offsets within the window (wheat early,
    rice late) instead of using the midpoint for everything
  - Current anomaly signal from agri_data.py now tilts year weights
  - Smoke test updated to NW Arkansas farm with correct crops

Output feeds directly into yield_risk.py and then Granite.
No breaking changes to other files — same return dict keys as v1.

Dependencies: requests + standard library only.
"""

import logging
import math
import random
import statistics
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

log = logging.getLogger("agrishield.monte_carlo")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_NASA_DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

_PARAMS = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "PRECTOTCORR",   # bias-corrected precip (replaces deprecated PRECTOT)
    "RH2M",
    "ALLSKY_SFC_SW_DWN",
]

_HISTORY_YEARS   = 30
_N_SIMULATIONS   = 10_000
_WEIGHT_HALF_LIFE = 10.0
_MISSING          = -900.0
_TIMEOUT          = 60

# ---------------------------------------------------------------------------
# Crop stress parameters
# ---------------------------------------------------------------------------
# heat_threshold_c      : daily max temp above which stress accumulates
# extreme_heat_c        : daily max temp causing severe single-day damage
# dry_day_mm            : daily precip below which counts as a dry day
# critical_window_start : days after planting when critical period begins
# critical_window_end   : days after planting when critical period ends
# gdd_base_c            : base temp for growing degree day accumulation
# optimal_season_rain_mm: ideal total growing season precipitation (mm)
# planting_doy_offset   : NEW — days after window midpoint to shift DOY
#                         wheat plants early in window, rice plants late
#                         0 = use midpoint as-is (corn, soybeans)
#                         negative = plant earlier than midpoint (wheat)
#                         positive = plant later than midpoint (rice)

CROP_STRESS_PARAMS = {
    "corn": {
        "heat_threshold_c":       32.0,
        "extreme_heat_c":         35.0,
        "dry_day_mm":             3.0,
        "critical_window_start":  55,
        "critical_window_end":    75,
        "gdd_base_c":             10.0,
        "optimal_season_rain_mm": 500.0,
        "futures_symbol":         "ZC",
        "planting_doy_offset":    0,
    },
    "soybeans": {
        "heat_threshold_c":       34.0,
        "extreme_heat_c":         38.0,
        "dry_day_mm":             3.0,
        "critical_window_start":  60,
        "critical_window_end":    90,
        "gdd_base_c":             10.0,
        "optimal_season_rain_mm": 450.0,
        "futures_symbol":         "ZS",
        "planting_doy_offset":    0,
    },
    "wheat": {
        "heat_threshold_c":       30.0,
        "extreme_heat_c":         34.0,
        "dry_day_mm":             2.0,
        "critical_window_start":  40,
        "critical_window_end":    60,
        "gdd_base_c":             0.0,
        "optimal_season_rain_mm": 350.0,
        "futures_symbol":         "ZW",
        # Wheat plants significantly earlier than the window midpoint
        # A Feb 28 - May 28 window has midpoint ~Apr 13 (DOY 103)
        # Wheat in Arkansas actually plants late Feb / early Mar → DOY ~65
        # So we shift 38 days earlier than the midpoint
        "planting_doy_offset":    -38,
    },
    "rice": {
        "heat_threshold_c":       33.0,   # heading sensitive to heat >33°C
        "extreme_heat_c":         37.0,
        "dry_day_mm":             5.0,
        "critical_window_start":  60,
        "critical_window_end":    80,
        "gdd_base_c":             10.0,
        # Arkansas rice is ~100% irrigated from alluvial aquifer.
        # Precipitation deficit is largely irrelevant — yield stress
        # comes from heat during heading/flowering, not rainfall.
        # optimal_season_rain set to actual AR growing season average
        # (~450mm) so deficit calc is near-zero in typical years.
        "optimal_season_rain_mm": 450.0,
        "irrigated":              True,   # disables precip deficit penalty
        "futures_symbol":         "ZR",
        # Rice plants late in the window — May in Arkansas
        # A Feb 28 - May 28 window midpoint is ~Apr 13 (DOY 103)
        # Rice plants ~May 15 → DOY 135 → shift +32 days
        "planting_doy_offset":    +32,
    },
    "cotton": {
        "heat_threshold_c":       35.0,
        "extreme_heat_c":         38.0,
        "dry_day_mm":             3.0,
        "critical_window_start":  50,
        "critical_window_end":    80,
        "gdd_base_c":             15.6,
        "optimal_season_rain_mm": 600.0,
        "futures_symbol":         "CT",
        "planting_doy_offset":    +15,
    },
    "oats": {
        "heat_threshold_c":       28.0,
        "extreme_heat_c":         32.0,
        "dry_day_mm":             2.0,
        "critical_window_start":  45,
        "critical_window_end":    65,
        "gdd_base_c":             0.0,
        "optimal_season_rain_mm": 380.0,
        "futures_symbol":         "ZO",
        "planting_doy_offset":    -20,
    },
    "canola": {
        "heat_threshold_c":       29.0,
        "extreme_heat_c":         33.0,
        "dry_day_mm":             2.0,
        "critical_window_start":  35,
        "critical_window_end":    55,
        "gdd_base_c":             5.0,
        "optimal_season_rain_mm": 400.0,
        "futures_symbol":         "RS",
        "planting_doy_offset":    -25,
    },
}

# ---------------------------------------------------------------------------
# ENSO phase year classifications
# ---------------------------------------------------------------------------
_EL_NINO_YEARS = {1983,1987,1988,1992,1995,1998,2003,2005,2007,2010,2016,2019,2023}
_LA_NINA_YEARS = {1984,1985,1989,1996,1999,2000,2001,2008,2009,2011,2012,2021,2022}

# ---------------------------------------------------------------------------
# Anomaly signal → additional year weight tilt
# ---------------------------------------------------------------------------
# When agri_data.py reports a current anomaly signal, we further tilt
# the year weights to oversample years that match the current pattern.
#
# This is applied on top of the ENSO match bonus.
# Values are weight multipliers applied to years whose pattern matches.
#
# "dry" → upweight historically dry years (precip_deficit_pct < -15%)
# "warm_and_dry" → upweight years that were both hot and dry
# etc.
#
# We don't hardcode which historical years were dry — instead we check
# the actual precip_deficit metric after computing season_metrics.
# These multipliers are applied post-metrics in a second weighting pass.

_ANOMALY_WEIGHT_BOOST = {
    "warm_and_dry": {"dry_year_boost": 1.8, "hot_year_boost": 1.8},
    "dry":          {"dry_year_boost": 1.6, "hot_year_boost": 1.0},
    "warm":         {"dry_year_boost": 1.0, "hot_year_boost": 1.5},
    "warm_and_wet": {"dry_year_boost": 1.0, "hot_year_boost": 1.3},
    "cool_and_dry": {"dry_year_boost": 1.4, "hot_year_boost": 1.0},
    "cool_and_wet": {"dry_year_boost": 1.0, "hot_year_boost": 1.0},
    "cool":         {"dry_year_boost": 1.0, "hot_year_boost": 1.0},
    "wet":          {"dry_year_boost": 1.0, "hot_year_boost": 1.0},
    "normal":       {"dry_year_boost": 1.0, "hot_year_boost": 1.0},
    "unknown":      {"dry_year_boost": 1.0, "hot_year_boost": 1.0},
}

# Thresholds for classifying a historical year as "dry" or "hot"
_DRY_YEAR_DEFICIT_PCT  = -15.0   # precip_deficit_pct below this = dry year
_HOT_YEAR_HEAT_DAYS    = 8       # heat_stress_days above this = hot year


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mean(vals: list, default: float = 0.0) -> float:
    valid = [v for v in vals if v is not None and not math.isnan(v)]
    return statistics.mean(valid) if valid else default


def _safe_sum(vals: list, default: float = 0.0) -> float:
    valid = [v for v in vals if v is not None and not math.isnan(v)]
    return sum(valid) if valid else default


def _percentile(sorted_vals: list, p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx  = (p / 100.0) * (len(sorted_vals) - 1)
    lo   = int(idx)
    hi   = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _skewness(vals: list) -> float:
    """Pearson moment skewness. Negative = fat left tail = bad years are really bad."""
    if len(vals) < 3:
        return 0.0
    try:
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals)
        if std == 0:
            return 0.0
        n = len(vals)
        return (n / ((n - 1) * (n - 2))) * sum(((v - mean) / std) ** 3 for v in vals)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Step 1: Pull historical weather from NASA POWER
# ---------------------------------------------------------------------------

def _fetch_historical_year(lat: float, lon: float, year: int) -> Optional[dict]:
    """Fetch one calendar year of daily data. Returns None on any failure."""
    try:
        resp = requests.get(
            _NASA_DAILY_URL,
            params={
                "parameters": ",".join(_PARAMS),
                "community":  "AG",
                "longitude":  lon,
                "latitude":   lat,
                "start":      f"{year}0101",
                "end":        f"{year}1231",
                "format":     "JSON",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()["properties"]["parameter"]
    except Exception as exc:
        log.warning("NASA POWER fetch failed for year %d: %s", year, exc)
        return None

    cleaned = {}
    for param in _PARAMS:
        series = []
        for val in raw.get(param, {}).values():
            try:
                fval = float(val)
                if fval > _MISSING:
                    series.append(fval)
            except (TypeError, ValueError):
                pass
        cleaned[param] = series
    return cleaned


def fetch_historical_data(lat: float, lon: float, history_years: int = _HISTORY_YEARS) -> dict:
    """Pull `history_years` of daily weather. Skips years where API fails."""
    current_year = datetime.now(timezone.utc).year
    end_year     = current_year - 1
    start_year   = end_year - history_years + 1

    log.info("Fetching historical weather %d-%d for lat=%.4f lon=%.4f",
             start_year, end_year, lat, lon)

    historical = {}
    for year in range(start_year, end_year + 1):
        data = _fetch_historical_year(lat, lon, year)
        if data:
            historical[year] = data
            log.info("  year %d loaded", year)
        else:
            log.warning("  year %d skipped", year)

    log.info("Historical data: %d of %d years loaded", len(historical), history_years)
    return historical


# ---------------------------------------------------------------------------
# Step 2: Compute year weights
# ---------------------------------------------------------------------------

def compute_year_weights(
    years:          list,
    enso_phase:     str  = "neutral",
    anomaly_signal: str  = "normal",
    metrics_cache:  dict = None,
    half_life:      float = _WEIGHT_HALF_LIFE,
) -> dict:
    """
    Assign each historical year a sampling weight.

    Three components (new in v2 — v1 only had 1 and 2):

    1. Recency weight (exponential decay):
       w = 2^(-(max_year - year) / half_life)
       Most recent year = 1.0. A year `half_life` years ago = 0.5.

    2. ENSO match bonus:
       If the historical year's ENSO phase matches the current phase,
       that year gets a 50% weight boost.

    3. Anomaly signal tilt (NEW):
       If agri_data.py reports a current anomaly (e.g. "dry"), we boost
       historical years whose actual metrics match that pattern.
       Requires metrics_cache to be provided (passed from run_simulation).
       If not provided, this component is skipped gracefully.

    Weights normalised to sum to 1.0.
    """
    if not years:
        return {}

    max_year   = max(years)
    is_el_nino = "el_nino" in enso_phase
    is_la_nina = "la_nina" in enso_phase

    # Get anomaly boost config
    anomaly_boost = _ANOMALY_WEIGHT_BOOST.get(anomaly_signal, _ANOMALY_WEIGHT_BOOST["normal"])
    dry_boost     = anomaly_boost["dry_year_boost"]
    hot_boost     = anomaly_boost["hot_year_boost"]

    raw_weights = {}
    for y in years:
        # 1. Recency decay
        age = max_year - y
        w   = math.pow(2.0, -age / half_life)

        # 2. ENSO phase match bonus
        if is_el_nino and y in _EL_NINO_YEARS:
            w *= 1.5
        elif is_la_nina and y in _LA_NINA_YEARS:
            w *= 1.5
        elif not is_el_nino and not is_la_nina:
            if y not in _EL_NINO_YEARS and y not in _LA_NINA_YEARS:
                w *= 1.2

        # 3. Anomaly signal tilt (only if metrics available)
        if metrics_cache and y in metrics_cache:
            m = metrics_cache[y]
            if dry_boost != 1.0 and m.get("precip_deficit_pct", 0) < _DRY_YEAR_DEFICIT_PCT:
                w *= dry_boost
            if hot_boost != 1.0 and m.get("heat_stress_days", 0) > _HOT_YEAR_HEAT_DAYS:
                w *= hot_boost

        raw_weights[y] = w

    total = sum(raw_weights.values())
    return {y: w / total for y, w in raw_weights.items()}


# ---------------------------------------------------------------------------
# Step 3: Derive crop-specific planting DOY
# ---------------------------------------------------------------------------

def _derive_planting_doy(
    planting_start: str,
    planting_end:   str,
    crop_params:    dict,
) -> int:
    """
    Compute crop-specific planting day-of-year.

    Takes the midpoint of the planting window then applies the crop's
    planting_doy_offset. This ensures wheat (early in window) and rice
    (late in window) get correct critical window placement even when
    the farm uses a single broad planting window for all crops.

    Example: Feb 28 - May 28 window → midpoint Apr 13 (DOY 103)
      Wheat offset = -38  → DOY 65  (early March — correct for AR winter wheat)
      Rice  offset = +32  → DOY 135 (mid May    — correct for AR rice)
      Corn  offset =   0  → DOY 103 (mid April  — correct for early corn)
    """
    try:
        d1  = datetime.strptime(planting_start, "%Y-%m-%d")
        d2  = datetime.strptime(planting_end,   "%Y-%m-%d")
        mid = d1 + (d2 - d1) / 2
        doy = mid.timetuple().tm_yday
    except Exception:
        doy = 103   # fallback: mid-April

    offset      = crop_params.get("planting_doy_offset", 0)
    adjusted    = doy + offset

    # Clamp to valid DOY range
    adjusted = max(1, min(365, adjusted))

    log.info("Planting DOY: base=%d  offset=%+d  final=%d", doy, offset, adjusted)
    return adjusted


# ---------------------------------------------------------------------------
# Step 4: Extract growing-season stress metrics
# ---------------------------------------------------------------------------

def extract_season_metrics(year_data: dict, crop_params: dict, planting_doy: int) -> dict:
    """
    Compute growing-season stress metrics for one historical year.

    Growing season: planting_doy → planting_doy + 150 days
    Critical window: planting_doy + critical_window_start → + critical_window_end
    """
    t2m     = year_data.get("T2M",               [])
    t2m_max = year_data.get("T2M_MAX",           [])
    precip  = year_data.get("PRECTOTCORR",       [])
    solar   = year_data.get("ALLSKY_SFC_SW_DWN", [])

    season_start = planting_doy
    season_end   = min(planting_doy + 150, 365)
    crit_start   = planting_doy + crop_params["critical_window_start"]
    crit_end     = planting_doy + crop_params["critical_window_end"]

    gdd_base     = crop_params["gdd_base_c"]
    heat_thresh  = crop_params["heat_threshold_c"]
    extreme_heat = crop_params["extreme_heat_c"]
    dry_thresh   = crop_params["dry_day_mm"]
    opt_rain     = crop_params["optimal_season_rain_mm"]

    gdd_total          = 0.0
    heat_stress_days   = 0
    extreme_heat_days  = 0
    critical_heat_days = 0
    season_precip      = []
    dry_days           = 0
    avg_temp_vals      = []
    avg_solar_vals     = []
    current_dry        = 0
    max_dry            = 0

    n_days = min(len(t2m), len(t2m_max), len(precip))

    for i in range(n_days):
        doy = i + 1
        if not (season_start <= doy <= season_end):
            continue

        gdd_total += max(0.0, t2m[i] - gdd_base)

        if t2m_max[i] >= heat_thresh:
            heat_stress_days += 1
            if crit_start <= doy <= crit_end:
                critical_heat_days += 1
        if t2m_max[i] >= extreme_heat:
            extreme_heat_days += 1

        p = precip[i] if i < len(precip) else 0.0
        season_precip.append(p)
        if p < dry_thresh:
            dry_days  += 1
            current_dry += 1
            max_dry = max(max_dry, current_dry)
        else:
            current_dry = 0

        avg_temp_vals.append(t2m[i])
        if i < len(solar):
            avg_solar_vals.append(solar[i])

    total_precip       = _safe_sum(season_precip)
    precip_deficit_pct = ((total_precip - opt_rain) / opt_rain * 100.0) if opt_rain > 0 else 0.0

    return {
        "gdd_total":           round(gdd_total, 1),
        "heat_stress_days":    heat_stress_days,
        "extreme_heat_days":   extreme_heat_days,
        "critical_heat_days":  critical_heat_days,
        "total_precip_mm":     round(total_precip, 1),
        "dry_days":            dry_days,
        "consecutive_dry_max": max_dry,
        "precip_deficit_pct":  round(precip_deficit_pct, 1),
        "avg_temp":            round(_safe_mean(avg_temp_vals), 2),
        "avg_solar":           round(_safe_mean(avg_solar_vals), 2),
    }


# ---------------------------------------------------------------------------
# Step 5: Yield loss function
# ---------------------------------------------------------------------------

def compute_yield_loss_pct(
    metrics:    dict,
    crop_params: dict,
    enso_bias:  float = 0.0,
) -> float:
    """
    Translate season stress metrics into yield as % of normal.

    enso_bias is now passed in from seasonal_outlook's regional impact
    table (not hardcoded corn-belt values). Positive = yield boost,
    negative = yield suppression.

    Returns yield_pct_of_normal clamped to [20, 105].
    """
    score = 100.0

    # Critical window heat — biggest single factor
    crit_heat = metrics.get("critical_heat_days", 0)
    score -= crit_heat * 1.5

    # Extreme heat — additional damage on top of threshold stress
    extreme = metrics.get("extreme_heat_days", 0)
    score -= extreme * 1.0

    # General heat outside critical window — cumulative chronic stress
    general_heat = max(0, metrics.get("heat_stress_days", 0) - crit_heat)
    score -= general_heat * 0.4

    # Precipitation deficit — nonlinear drought penalty
    # Skipped entirely for irrigated crops (e.g. Arkansas rice) where
    # yield stress is driven by heat at heading, not water availability.
    irrigated   = crop_params.get("irrigated", False)
    deficit_pct = metrics.get("precip_deficit_pct", 0.0)
    if not irrigated:
        if deficit_pct < 0:
            drought_penalty = abs(deficit_pct) * 0.25
            if abs(deficit_pct) > 30:
                drought_penalty += (abs(deficit_pct) - 30) * 0.15
            score -= drought_penalty
        elif deficit_pct > 20:
            score -= (deficit_pct - 20) * 0.1   # mild waterlogging penalty

        # Consecutive dry streak — sustained drought worse than scattered dry days
        dry_streak = metrics.get("consecutive_dry_max", 0)
        if dry_streak > 10:
            score -= (dry_streak - 10) * 0.5

    # ENSO regional bias — now from seasonal_outlook regional table
    # e.g. delta region weak La Nina → -0.04 → score -= 4 points
    score += enso_bias * 100.0

    return round(max(20.0, min(105.0, score)), 2)


# ---------------------------------------------------------------------------
# Step 6: Run the Monte Carlo simulation
# ---------------------------------------------------------------------------

def _add_noise(metrics: dict) -> dict:
    """Add small Gaussian noise to represent unsampled inter-year variability."""
    noisy = dict(metrics)
    for key in ["critical_heat_days", "heat_stress_days", "consecutive_dry_max"]:
        val = noisy.get(key, 0)
        noisy[key] = max(0, val + random.gauss(0, max(1.0, val * 0.15)))
    noisy["precip_deficit_pct"] = noisy.get("precip_deficit_pct", 0.0) + random.gauss(0, 5.0)
    return noisy


def _weighted_median_metrics(metrics_cache: dict, year_weights: dict) -> dict:
    """Compute weighted median of each stress metric across historical years."""
    keys = [
        "gdd_total", "heat_stress_days", "extreme_heat_days",
        "critical_heat_days", "total_precip_mm", "dry_days",
        "consecutive_dry_max", "precip_deficit_pct", "avg_temp",
    ]
    result = {}
    for key in keys:
        vw = [(metrics_cache[y].get(key, 0), year_weights.get(y, 0)) for y in metrics_cache]
        vw.sort(key=lambda x: x[0])
        cumulative = 0.0
        median_val = vw[0][0] if vw else 0.0
        for val, w in vw:
            cumulative += w
            if cumulative >= 0.5:
                median_val = val
                break
        result[key] = round(median_val, 2)
    return result


def _neutral_distribution() -> dict:
    """Safe fallback when no historical data is available."""
    return {
        "yield_distribution": {
            "p10": 75.0, "p25": 85.0, "p50": 95.0,
            "p75": 100.0, "p90": 103.0,
            "mean": 92.0, "std_dev": 10.0, "skewness": -0.5,
            "unit": "pct_of_normal_yield",
        },
        "simulation_meta": {
            "n_simulations": 0, "historical_years": 0,
            "weighting": "none_data_unavailable",
            "enso_phase": "unknown", "enso_yield_bias_pct": 0.0,
            "anomaly_signal": "unknown",
        },
        "season_stress_medians": {},
    }


def _derive_enso_bias(seasonal_outlook: Optional[dict], enso_phase: str) -> float:
    """
    Extract the ENSO regional yield bias from seasonal_outlook dict.

    seasonal_outlook comes from seasonal_outlook.py and contains
    region-specific probabilities. We convert the precip_below_normal_prob
    into a yield bias using a simple linear mapping:
      33% (climatological base) → 0.0 bias
      55% (weak La Nina delta)  → -0.04 bias
      65% (strong La Nina)      → -0.06 bias

    This replaces the hardcoded corn-belt-only _ENSO_YIELD_BIAS table.
    """
    if seasonal_outlook is None:
        # Fallback to simple hardcoded values if no outlook provided
        fallback = {
            "strong_el_nino": +0.03, "weak_el_nino": +0.01,
            "neutral": 0.00, "weak_la_nina": -0.02, "strong_la_nina": -0.05,
        }
        return fallback.get(enso_phase, 0.0)

    # Use the regional precip probability to derive bias.
    # Calibration: each 10 percentage points above the 33% base rate
    # corresponds to roughly 0.8% yield suppression from precip stress
    # and 0.4% from temperature stress, based on USDA historical data.
    # e.g. delta weak La Nina: precip_prob=55% → (55-33)/10 × 0.008 = -0.018
    #                          temp_prob=50%   → (50-33)/10 × 0.004 = -0.007
    #                          total bias ≈ -0.025 (-2.5%) — realistic
    precip_below_prob = seasonal_outlook.get("precip_below_normal_prob_pct", 33)
    temp_above_prob   = seasonal_outlook.get("temp_above_normal_prob_pct",   33)

    precip_bias = -((precip_below_prob - 33) / 10.0) * 0.008
    temp_bias   = -((temp_above_prob   - 33) / 10.0) * 0.004

    total_bias = round(precip_bias + temp_bias, 4)
    log.info("ENSO regional bias: precip_prob=%d%%  temp_prob=%d%%  bias=%.4f",
             precip_below_prob, temp_above_prob, total_bias)
    return total_bias


def run_simulation(
    historical:       dict,
    year_weights:     dict,
    crop_params:      dict,
    planting_doy:     int,
    enso_phase:       str  = "neutral",
    enso_bias:        float = 0.0,
    anomaly_signal:   str  = "normal",
    n_simulations:    int  = _N_SIMULATIONS,
    seed:             int  = 42,
) -> dict:
    """Core Monte Carlo loop."""
    if not historical or not year_weights:
        return _neutral_distribution()

    random.seed(seed)

    years   = list(year_weights.keys())
    weights = [year_weights[y] for y in years]

    # Pre-compute metrics for all years once
    log.info("Pre-computing stress metrics for %d historical years...", len(years))
    metrics_cache = {
        year: extract_season_metrics(historical[year], crop_params, planting_doy)
        for year in years
    }

    # Re-weight with anomaly signal now that metrics are available
    # This is the second weighting pass that v1 was missing
    final_weights = compute_year_weights(
        years          = years,
        enso_phase     = enso_phase,
        anomaly_signal = anomaly_signal,
        metrics_cache  = metrics_cache,
    )
    weights = [final_weights[y] for y in years]

    log.info("Running %d Monte Carlo simulations (enso_bias=%.4f  anomaly=%s)...",
             n_simulations, enso_bias, anomaly_signal)

    outcomes = []
    for _ in range(n_simulations):
        sampled_year = random.choices(years, weights=weights, k=1)[0]
        noisy        = _add_noise(metrics_cache[sampled_year])
        yield_pct    = compute_yield_loss_pct(noisy, crop_params, enso_bias)
        outcomes.append(yield_pct)

    outcomes.sort()

    p10  = _percentile(outcomes, 10)
    p25  = _percentile(outcomes, 25)
    p50  = _percentile(outcomes, 50)
    p75  = _percentile(outcomes, 75)
    p90  = _percentile(outcomes, 90)
    mean = statistics.mean(outcomes)
    std  = statistics.stdev(outcomes) if len(outcomes) > 1 else 0.0
    skew = _skewness(outcomes)

    log.info("Monte Carlo complete: P10=%.1f  P50=%.1f  P90=%.1f  std=%.1f  skew=%.2f",
             p10, p50, p90, std, skew)

    return {
        "yield_distribution": {
            "p10":      round(p10,  1),
            "p25":      round(p25,  1),
            "p50":      round(p50,  1),
            "p75":      round(p75,  1),
            "p90":      round(p90,  1),
            "mean":     round(mean, 2),
            "std_dev":  round(std,  2),
            "skewness": round(skew, 3),
            "unit":     "pct_of_normal_yield",
        },
        "simulation_meta": {
            "n_simulations":       n_simulations,
            "historical_years":    len(historical),
            "weighting":           "exponential_decay_enso_anomaly",
            "enso_phase":          enso_phase,
            "enso_yield_bias_pct": round(enso_bias * 100, 2),
            "anomaly_signal":      anomaly_signal,
        },
        "season_stress_medians": _weighted_median_metrics(metrics_cache, final_weights),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_monte_carlo(
    lat:                   float,
    lon:                   float,
    crop_type:             str,
    planting_window_start: str,
    planting_window_end:   str,
    enso_phase:            str  = "neutral",
    seasonal_outlook:      Optional[dict] = None,
    anomaly_signal:        str  = "normal",
    n_simulations:         int  = _N_SIMULATIONS,
    history_years:         int  = _HISTORY_YEARS,
) -> dict:
    """
    Full Monte Carlo pipeline entry point. Called by pipeline.py.

    Parameters
    ----------
    lat, lon               : farm coordinates
    crop_type              : "corn", "soybeans", "wheat", "rice", or specialty
    planting_window_start  : "YYYY-MM-DD"
    planting_window_end    : "YYYY-MM-DD"
    enso_phase             : from seasonal_outlook.py e.g. "weak_la_nina"
    seasonal_outlook       : full seasonal outlook dict from seasonal_outlook.py
                             used to derive region-specific ENSO yield bias
    anomaly_signal         : from agri_data.py e.g. "dry", "warm_and_dry"
                             tilts year weights to match current conditions
    n_simulations          : Monte Carlo runs (default 10,000)
    history_years          : years of NASA POWER history (default 30)

    Returns same keys as v1 plus anomaly_signal in simulation_meta.
    No breaking changes.
    """
    crop_key    = crop_type.lower().strip()
    crop_params = CROP_STRESS_PARAMS.get(crop_key)
    crop_tier   = "known" if crop_params else "unknown"

    if crop_tier == "unknown":
        log.warning("Crop '%s' not in known profiles — using corn proxy", crop_type)
        crop_params = {**CROP_STRESS_PARAMS["corn"], "futures_symbol": None,
                       "planting_doy_offset": 0}

    # Crop-specific planting DOY (v2 fix)
    planting_doy = _derive_planting_doy(
        planting_window_start, planting_window_end, crop_params
    )

    # Regional ENSO yield bias from seasonal_outlook (v2 fix)
    enso_bias = _derive_enso_bias(seasonal_outlook, enso_phase)

    # Fetch historical data
    historical = fetch_historical_data(lat, lon, history_years=history_years)

    if not historical:
        log.warning("No historical data — returning neutral distribution")
        result = _neutral_distribution()
        result.update({
            "crop_tier": crop_tier, "crop_params_used": None,
            "planting_doy": planting_doy,
        })
        return result

    # Initial year weights (without anomaly — metrics not yet computed)
    year_weights = compute_year_weights(
        years          = list(historical.keys()),
        enso_phase     = enso_phase,
        anomaly_signal = anomaly_signal,
    )

    # Run simulation (anomaly re-weighting happens inside after metrics computed)
    result = run_simulation(
        historical     = historical,
        year_weights   = year_weights,
        crop_params    = crop_params,
        planting_doy   = planting_doy,
        enso_phase     = enso_phase,
        enso_bias      = enso_bias,
        anomaly_signal = anomaly_signal,
        n_simulations  = n_simulations,
    )

    result.update({
        "crop_tier":        crop_tier,
        "crop_params_used": crop_params,
        "planting_doy":     planting_doy,
    })
    return result


# ---------------------------------------------------------------------------
# Smoke test — run: python monte_carlo.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # NW Arkansas farm — test all three crops
    lat, lon = 36.545722, -93.857057

    # Simulate seasonal_outlook output for delta region / weak La Nina
    mock_outlook = {
        "temp_above_normal_prob_pct":   50,
        "precip_below_normal_prob_pct": 55,
    }

    for crop in ["wheat", "rice", "soybeans"]:
        print(f"\n{'='*60}")
        print(f"AgriShield Monte Carlo  |  NW Arkansas  |  {crop.title()}")
        print(f"{'='*60}")

        result = run_monte_carlo(
            lat                   = lat,
            lon                   = lon,
            crop_type             = crop,
            planting_window_start = "2026-02-28",
            planting_window_end   = "2026-05-28",
            enso_phase            = "weak_la_nina",
            seasonal_outlook      = mock_outlook,
            anomaly_signal        = "dry",
            n_simulations         = 500,    # small for quick test
            history_years         = 30,
        )

        dist = result["yield_distribution"]
        print(f"  Planting DOY : {result['planting_doy']}")
        print(f"  P10 (bad)    : {dist['p10']}%")
        print(f"  P50 (median) : {dist['p50']}%")
        print(f"  P90 (good)   : {dist['p90']}%")
        print(f"  Skewness     : {dist['skewness']}")
        print(f"  ENSO bias    : {result['simulation_meta']['enso_yield_bias_pct']}%")
        print(f"  Anomaly      : {result['simulation_meta']['anomaly_signal']}")

    print("\n[Done]")