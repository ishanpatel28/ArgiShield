"""
pipeline.py
===========
AgriShield – Full Pipeline Orchestrator
----------------------------------------
Chains all modules in order:
  1. agri_data            → soil + current climate anomaly
  2. seasonal_outlook     → ENSO phase + 14-day forecast
  3. monte_carlo          → yield loss distributions (per crop)
  4. market_data          → commodity prices + revenue baselines
  5. yield_risk           → Granite Call 1: structured risk JSON
  6. hedge_optimizer      → Black-Scholes protective put sizing
  7. granite_advisor      → Granite Call 2: plain-language narrative

Returns a single result dict suitable for the FastAPI response.
"""

import importlib.util
import logging
import os
import sys
from datetime import datetime, timezone

log = logging.getLogger("agrishield.pipeline")

# ---------------------------------------------------------------------------
# Module loader — handles numbered filenames (e.g. 1_agri_data.py)
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_FILES = {
    "agri_data":       "1_agri_data.py",
    "seasonal_outlook":"2_seasonal_outlook.py",
    "monte_carlo":     "3_monte_carlo.py",
    "market_data":     "4_market_data.py",
    "yield_risk":      "5_yield_risk.py",
}


def _import(alias: str):
    """Import a numbered Python file by its logical alias."""
    if alias in sys.modules:
        return sys.modules[alias]
    filename = _MODULE_FILES[alias]
    path     = os.path.join(_HERE, filename)
    spec     = importlib.util.spec_from_file_location(alias, path)
    mod      = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def run_pipeline(farm_input: dict) -> dict:
    """
    Run the full AgriShield pipeline for a given farm input payload.

    Parameters
    ----------
    farm_input : dict with keys: farm, crops, planting_window, preferences

    Returns
    -------
    Merged result dict for the frontend.
    """
    t0     = datetime.now(timezone.utc)
    errors: list[str] = []

    lat      = farm_input.get("farm", {}).get("lat")
    lon      = farm_input.get("farm", {}).get("lon")
    crops    = farm_input.get("crops", [])
    planting = farm_input.get("planting_window", {})
    pw_start = planting.get("start", "")
    pw_end   = planting.get("end", "")

    # ----------------------------------------------------------------
    # Step 1: Agri data (soil + anomaly)
    # ----------------------------------------------------------------
    try:
        get_farm_soil_and_anomaly = _import("agri_data").get_farm_soil_and_anomaly
        log.info("[1/7] agri_data")
        soil_anomaly = get_farm_soil_and_anomaly(farm_input)
    except Exception as exc:
        log.warning("agri_data failed: %s", exc)
        errors.append(f"agri_data: {exc}")
        soil_anomaly = {"current_anomaly": {}, "crops": {}}

    # ----------------------------------------------------------------
    # Step 2: Seasonal outlook
    # ----------------------------------------------------------------
    try:
        get_seasonal_outlook_full = _import("seasonal_outlook").get_seasonal_outlook_full
        log.info("[2/7] seasonal_outlook")
        seasonal_outlook = get_seasonal_outlook_full(lat=lat, lon=lon)
    except Exception as exc:
        log.warning("seasonal_outlook failed: %s", exc)
        errors.append(f"seasonal_outlook: {exc}")
        seasonal_outlook = {
            "enso": {}, "enso_phase": "neutral", "enso_desc": "Neutral",
            "seasonal_outlook": {}, "forecast_14d": {},
        }

    # Merge current anomaly into seasonal_outlook for granite_advisor
    seasonal_outlook["current_anomaly"] = soil_anomaly.get("current_anomaly", {})

    enso_phase     = seasonal_outlook.get("enso_phase", "neutral")
    anomaly_signal = soil_anomaly.get("current_anomaly", {}).get("anomaly_signal", "normal")

    # ----------------------------------------------------------------
    # Step 3: Monte Carlo — run per crop
    # ----------------------------------------------------------------
    monte_carlo_all: dict = {}
    try:
        run_monte_carlo = _import("monte_carlo").run_monte_carlo
        log.info("[3/7] monte_carlo")
        for crop_entry in crops:
            name = crop_entry.get("name", "unknown")
            try:
                mc = run_monte_carlo(
                    lat=lat,
                    lon=lon,
                    crop_type=name,
                    planting_window_start=pw_start,
                    planting_window_end=pw_end,
                    enso_phase=enso_phase,
                    seasonal_outlook=seasonal_outlook.get("seasonal_outlook"),
                    anomaly_signal=anomaly_signal,
                )
                monte_carlo_all[name] = mc
            except Exception as crop_exc:
                log.warning("monte_carlo failed for %s: %s", name, crop_exc)
                errors.append(f"monte_carlo[{name}]: {crop_exc}")
    except Exception as exc:
        log.warning("monte_carlo import failed: %s", exc)
        errors.append(f"monte_carlo: {exc}")

    # ----------------------------------------------------------------
    # Step 4: Market data
    # ----------------------------------------------------------------
    try:
        get_market_data_all_crops = _import("market_data").get_market_data_all_crops
        log.info("[4/7] market_data")
        market_data = get_market_data_all_crops(crops=crops)
    except Exception as exc:
        log.warning("market_data failed: %s", exc)
        errors.append(f"market_data: {exc}")
        market_data = {"_farm_summary": {"total_expected_revenue": 0}}

    # ----------------------------------------------------------------
    # Step 5: Yield risk (Granite Call 1)
    # ----------------------------------------------------------------
    try:
        run_risk_assessment = _import("yield_risk").run_risk_assessment
        log.info("[5/7] yield_risk (Granite Call 1)")
        risk_result = run_risk_assessment(
            farm_input=farm_input,
            soil_anomaly=soil_anomaly,
            monte_carlo_all=monte_carlo_all,
            market_data=market_data,
            seasonal_outlook=seasonal_outlook,
        )
    except Exception as exc:
        log.warning("yield_risk failed: %s", exc)
        errors.append(f"yield_risk: {exc}")
        risk_result = {
            "granite_assessment": {"crops": {}, "farm_level": {}},
            "loss_scenarios": {},
            "farm_risk_summary": {
                "total_full_revenue": 0, "total_p50_loss": 0, "total_p10_loss": 0,
                "total_p50_loss_pct": 0, "total_p10_loss_pct": 0,
                "overall_severity": "unknown", "dominant_risk": "unknown",
            },
            "fallback_used": True,
        }

    # ----------------------------------------------------------------
    # Step 6: Hedge optimizer
    # ----------------------------------------------------------------
    try:
        from hedge_optimizer import run_hedge_optimizer
        log.info("[6/7] hedge_optimizer")
        hedge_result = run_hedge_optimizer(
            farm_input=farm_input,
            market_data=market_data,
            risk_result=risk_result,
        )
    except Exception as exc:
        log.warning("hedge_optimizer failed: %s", exc)
        errors.append(f"hedge_optimizer: {exc}")
        hedge_result = {"recommendations": [], "farm_hedge_summary": {}}

    # ----------------------------------------------------------------
    # Step 7: Granite advisor (Call 2)
    # ----------------------------------------------------------------
    try:
        from granite_advisor import run_granite_advisor
        log.info("[7/7] granite_advisor (Granite Call 2)")
        advisory = run_granite_advisor(
            farm_input=farm_input,
            risk_result=risk_result,
            hedge_result=hedge_result,
            seasonal_outlook=seasonal_outlook,
        )
    except Exception as exc:
        log.warning("granite_advisor failed: %s", exc)
        errors.append(f"granite_advisor: {exc}")
        advisory = {
            "short_summary": "Analysis complete. See breakdown below.",
            "detailed_narrative": "",
            "granite_used": False,
        }

    # ----------------------------------------------------------------
    # Assemble final result
    # ----------------------------------------------------------------
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    log.info("Pipeline complete in %.1fs  errors=%d", elapsed, len(errors))

    return {
        "farm_input":       farm_input,
        "soil_anomaly":     soil_anomaly,
        "seasonal_outlook": seasonal_outlook,
        "monte_carlo":      monte_carlo_all,
        "market_data":      market_data,
        "risk_result":      risk_result,
        "hedge_result":     hedge_result,
        "advisory":         advisory,
        "meta": {
            "elapsed_seconds":      round(elapsed, 1),
            "errors":               errors,
            "granite_call1_used":   not risk_result.get("fallback_used", True),
            "granite_call2_used":   advisory.get("granite_used", False),
            "generated_utc":        datetime.now(timezone.utc).isoformat(),
        },
    }
