"""
yield_risk.py
=============
AgriShield – Granite Risk Reasoning Engine (Call 1 of 2)
---------------------------------------------------------
Granite receives ALL raw quantitative data from upstream modules and
reasons over everything to produce a structured JSON risk assessment:
  - risk_coefficients per crop (drought_risk, heat_risk, soil_penalty, enso_impact)
  - overall_risk_score 0-100 per crop
  - primary_risk_driver per crop
  - yield_loss_estimate P10/P50/P90 per crop with reasoning
  - farm-level dominant risk and confidence

Python then takes Granite's yield_loss_pct × market price × acres
for dollar figures — pure arithmetic, no AI needed for multiplication.

Output feeds into hedge_optimizer.py (pure math) and
granite_advisor.py (Call 2 narrative).
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

import requests

log = logging.getLogger("agrishield.yield_risk")

_WATSONX_URL        = "https://us-south.ml.cloud.ibm.com"
_WATSONX_API_KEY    = "b3d5f298-5dd1-4f8d-a868-b3c4b223a517"
_WATSONX_PROJECT_ID = "e4164bef-4b12-4681-933c-d3ad03941cb5"
_GRANITE_MODEL      = "ibm/granite-4-0-tiny-preview"
_IAM_URL            = "https://iam.cloud.ibm.com/identity/token"
_INFER_URL          = f"{_WATSONX_URL}/ml/v1/text/chat?version=2023-05-29"
_TIMEOUT            = 60


# ---------------------------------------------------------------------------
# Watson helpers
# ---------------------------------------------------------------------------

def _get_iam_token() -> Optional[str]:
    """
    FIX: grant_type and apikey must be separate form fields.
    Original code merged them into one string value, causing 400 Bad Request.
    """
    try:
        resp = requests.post(
            url=_IAM_URL,
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": _WATSONX_API_KEY,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        resp.raise_for_status()
        token = resp.json().get("access_token")
        log.info("IAM token obtained")
        return token
    except Exception as exc:
        log.error("IAM token failed: %s", exc)
        return None


def _call_granite(prompt: str, max_tokens: int = 1200) -> Optional[str]:
    """
    FIX: /text/chat uses OpenAI-compatible chat format.
    - Request:  messages: [{role, content}]  (not "input")
    - Response: choices[0].message.content   (not results[0].generated_text)
    """
    token = _get_iam_token()
    if not token:
        return None
    try:
        resp = requests.post(
            _INFER_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={
                "model_id":   _GRANITE_MODEL,
                "project_id": _WATSONX_PROJECT_ID,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "parameters": {
                    "decoding_method": "greedy",
                    "max_new_tokens":  max_tokens,
                    "min_new_tokens":  100,
                    "temperature":     0.1,
                    "top_p":           0.9,
                },
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        token_count = data.get("usage", {}).get("completion_tokens", 0)
        log.info("Granite: %d tokens", token_count)
        return text
    except requests.HTTPError as exc:
        log.error("Granite HTTP %s: %s", exc.response.status_code, exc.response.text[:200])
        return None
    except Exception as exc:
        log.error("Granite failed: %s", exc)
        return None


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    clean = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    try:
        start = clean.index("{")
        end   = clean.rindex("}") + 1
        return json.loads(clean[start:end])
    except (ValueError, json.JSONDecodeError):
        pass
    log.warning("Could not extract JSON from Granite response")
    return None


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_risk_prompt(
    farm_input:       dict,
    soil_anomaly:     dict,
    monte_carlo_all:  dict,
    market_data:      dict,
    seasonal_outlook: dict,
) -> str:
    farm      = farm_input.get("farm", {})
    crops     = farm_input.get("crops", [])
    prefs     = farm_input.get("preferences", {})

    anomaly      = soil_anomaly.get("current_anomaly", {})
    enso         = seasonal_outlook.get("enso", {})
    outlook      = seasonal_outlook.get("seasonal_outlook", {})
    forecast     = seasonal_outlook.get("forecast_14d", {})

    crop_blocks = ""
    for crop_entry in crops:
        name      = crop_entry.get("name")
        acres     = crop_entry.get("acres", 0)
        mc        = monte_carlo_all.get(name, {})
        mkt       = market_data.get(name, {})
        soil      = soil_anomaly.get("crops", {}).get(name, {}).get("soil", {})
        dist      = mc.get("yield_distribution", {})
        stress    = mc.get("season_stress_medians", {})
        props     = soil.get("properties", {})
        irrigated = props.get("irrigated", soil.get("irrigated", False))

        crop_blocks += f"""
CROP: {name} ({acres} acres)
  Price: ${mkt.get('price_usd','N/A')}/{mkt.get('unit','unit')}  |  Full revenue: ${mkt.get('revenue_per_acre',0)*acres:,.0f}
  Soil score: {soil.get('soil_score','N/A')}  |  pH: {props.get('ph','N/A')}  |  Clay: {props.get('clay_pct','N/A')}%  |  SOC: {props.get('soc_gkg','N/A')} g/kg
  Drought buffer: {soil.get('drought_buffer_days','N/A')} days  |  Irrigated: {irrigated}
  Monte Carlo (10,000 sims, 30yr weighted history):
    P10={dist.get('p10','N/A')}%  P25={dist.get('p25','N/A')}%  P50={dist.get('p50','N/A')}%  P75={dist.get('p75','N/A')}%  P90={dist.get('p90','N/A')}%
    StdDev={dist.get('std_dev','N/A')}%  Skewness={dist.get('skewness','N/A')} (negative=fat bad tail)
  Season stress medians: heat_days={stress.get('heat_stress_days','N/A')}  crit_heat={stress.get('critical_heat_days','N/A')}  precip_deficit={stress.get('precip_deficit_pct','N/A')}%  max_dry_streak={stress.get('consecutive_dry_max','N/A')}d
"""

    crop_names = [c.get("name") for c in crops]

    return f"""You are AgriShield's AI risk assessment engine powered by IBM Granite.
Reason over the following agricultural and climate data and produce a structured JSON risk assessment.

=== FARM: lat={farm.get('lat')}, lon={farm.get('lon')} | Crops: {", ".join(crop_names)} ===
Risk aversion: {prefs.get('risk_aversion_0_100',65)}/100

=== CURRENT CONDITIONS (last 30 days vs 30yr normal) ===
Signal: {anomaly.get('anomaly_signal','unknown')} | Temp anomaly: {anomaly.get('temp_anomaly_c',0):+.2f}°C | Precip anomaly: {anomaly.get('precip_anomaly_pct',0):.1f}%
Current dry streak: {anomaly.get('current_dry_streak',0)}d | Max dry streak (30d): {anomaly.get('max_dry_streak_30d',0)}d

=== ENSO + SEASONAL OUTLOOK ===
Phase: {enso.get('enso_desc','unknown')} (ONI={enso.get('oni_value',0)})
Temp above normal prob: {outlook.get('temp_above_normal_prob_pct',33)}% | Precip below normal prob: {outlook.get('precip_below_normal_prob_pct',33)}%
Drought risk elevated: {outlook.get('drought_risk_elevated',False)}
{outlook.get('risk_narrative','')}

=== 14-DAY FORECAST ===
Precip: {forecast.get('total_precip_mm',0):.1f}mm | Heat stress days: {forecast.get('heat_stress_days',0)} | Dry days: {forecast.get('dry_days_forecast',0)} | Extreme heat: {forecast.get('extreme_heat_flag',False)}

=== PER-CROP DATA ==={crop_blocks}
=== TASK ===
Reason carefully about:
- How the current drought/wet signal interacts with each crop's irrigation status and soil
- What the Monte Carlo distribution SHAPE tells you (skewness, std dev, spread between P10 and P90)
- How ENSO historically affects the Delta region for each crop type
- Which crops face compounding risks vs which are more resilient
- What the soil pH of 5.4 means for nutrient availability under stress conditions

Output ONLY valid JSON — no markdown, no explanation, just the JSON object:

{{
  "assessment_summary": "<2-3 sentence overall risk picture>",
  "crops": {{
    "<crop_name>": {{
      "risk_coefficients": {{
        "drought_risk": <0.0-1.0>,
        "heat_risk": <0.0-1.0>,
        "soil_penalty": <0.0-1.0>,
        "enso_impact": <0.0-1.0>,
        "overall_risk_score": <0-100>
      }},
      "primary_risk_driver": "<one sentence>",
      "yield_loss_estimate": {{
        "p10_yield_pct": <float>,
        "p50_yield_pct": <float>,
        "p90_yield_pct": <float>,
        "reasoning": "<why these numbers given the data>"
      }},
      "key_insight": "<one agronomic insight the math alone cannot capture>"
    }}
  }},
  "farm_level": {{
    "dominant_risk": "<biggest single threat to this farm>",
    "risk_interactions": "<how risks across crops interact>",
    "confidence": "<high|medium|low — brief reason>"
  }}
}}"""


# ---------------------------------------------------------------------------
# Dollar loss arithmetic
# ---------------------------------------------------------------------------

def _compute_dollar_losses(granite_crops: dict, market_data: dict, farm_crops: list) -> dict:
    """Granite's yield_pct × market price × acres = dollars. Pure arithmetic."""
    results = {}
    for crop_entry in farm_crops:
        name  = crop_entry.get("name")
        acres = crop_entry.get("acres", 0)
        mkt   = market_data.get(name, {})
        g     = granite_crops.get(name, {})

        rev_per_acre = mkt.get("revenue_per_acre")
        if not rev_per_acre or not g:
            results[name] = {"error": "missing data"}
            continue

        full_rev = round(rev_per_acre * acres, 2)
        ye       = g.get("yield_loss_estimate", {})

        p10_pct = ye.get("p10_yield_pct", 75.0)
        p50_pct = ye.get("p50_yield_pct", 90.0)
        p90_pct = ye.get("p90_yield_pct", 100.0)

        rev_p10 = round(full_rev * p10_pct / 100, 2)
        rev_p50 = round(full_rev * p50_pct / 100, 2)
        rev_p90 = round(full_rev * p90_pct / 100, 2)

        loss_p10 = round(full_rev - rev_p10, 2)
        loss_p50 = round(full_rev - rev_p50, 2)
        loss_p90 = round(full_rev - rev_p90, 2)

        loss_pct_p10 = round(loss_p10 / full_rev * 100, 1) if full_rev > 0 else 0
        loss_pct_p50 = round(loss_p50 / full_rev * 100, 1) if full_rev > 0 else 0

        severity = ("critical" if loss_pct_p10 >= 50 else
                    "high"     if loss_pct_p10 >= 35 else
                    "moderate" if loss_pct_p10 >= 20 else "low")

        log.info("%s: full=$%.0f  P50_loss=$%.0f (%.1f%%)  P10_loss=$%.0f (%.1f%%)  sev=%s",
                 name, full_rev, loss_p50, loss_pct_p50, loss_p10, loss_pct_p10, severity)

        results[name] = {
            "crop": name, "acres": acres, "full_revenue": full_rev,
            "scenarios": {
                "good_year_p90": {"yield_pct": p90_pct, "revenue": rev_p90, "loss": loss_p90},
                "median_p50":    {"yield_pct": p50_pct, "revenue": rev_p50, "loss": loss_p50},
                "bad_year_p10":  {"yield_pct": p10_pct, "revenue": rev_p10, "loss": loss_p10},
            },
            "loss_pct_p50":  loss_pct_p50,
            "loss_pct_p10":  loss_pct_p10,
            "severity":      severity,
            "risk_score":    g.get("risk_coefficients", {}).get("overall_risk_score", 50),
            "primary_risk_driver": g.get("primary_risk_driver", ""),
            "key_insight":   g.get("key_insight", ""),
            "risk_coefficients": g.get("risk_coefficients", {}),
            "yield_reasoning":   ye.get("reasoning", ""),
        }
    return results


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def _fallback_assessment(farm_crops, monte_carlo_all):
    log.warning("Granite unavailable — using Monte Carlo fallback")
    crops_out = {}
    for c in farm_crops:
        name = c.get("name")
        dist = monte_carlo_all.get(name, {}).get("yield_distribution", {})
        crops_out[name] = {
            "risk_coefficients": {"drought_risk": 0.5, "heat_risk": 0.3,
                                  "soil_penalty": 0.3, "enso_impact": 0.3,
                                  "overall_risk_score": 50},
            "primary_risk_driver": "Granite unavailable — using Monte Carlo output directly",
            "yield_loss_estimate": {
                "p10_yield_pct": dist.get("p10", 75.0),
                "p50_yield_pct": dist.get("p50", 90.0),
                "p90_yield_pct": dist.get("p90", 100.0),
                "reasoning": "Monte Carlo fallback — no Granite reasoning",
            },
            "key_insight": "",
        }
    return {
        "assessment_summary": "Risk assessment from Monte Carlo only (Granite unavailable).",
        "crops": crops_out,
        "farm_level": {"dominant_risk": "unknown", "risk_interactions": "unknown",
                       "confidence": "low — Granite unavailable"},
        "_fallback": True,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_risk_assessment(
    farm_input:       dict,
    soil_anomaly:     dict,
    monte_carlo_all:  dict,
    market_data:      dict,
    seasonal_outlook: dict,
) -> dict:
    """
    Granite Call 1: risk reasoning over all upstream data.
    Returns loss_scenarios (dollar figures), granite_assessment (JSON),
    and farm_risk_summary (aggregates). Feeds into hedge_optimizer.py.
    """
    farm_crops = farm_input.get("crops", [])

    prompt   = _build_risk_prompt(farm_input, soil_anomaly, monte_carlo_all,
                                   market_data, seasonal_outlook)
    raw_text = _call_granite(prompt, max_tokens=1200)

    granite_assessment = _extract_json(raw_text) if raw_text else None
    fallback_used      = False

    if not granite_assessment:
        granite_assessment = _fallback_assessment(farm_crops, monte_carlo_all)
        fallback_used      = True

    # Fill in any missing crops
    granite_crops = granite_assessment.get("crops", {})
    for c in farm_crops:
        name = c.get("name")
        if name not in granite_crops:
            dist = monte_carlo_all.get(name, {}).get("yield_distribution", {})
            granite_crops[name] = {
                "risk_coefficients": {"overall_risk_score": 50},
                "primary_risk_driver": "Missing from Granite response",
                "yield_loss_estimate": {"p10_yield_pct": dist.get("p10", 75),
                                        "p50_yield_pct": dist.get("p50", 90),
                                        "p90_yield_pct": dist.get("p90", 100),
                                        "reasoning": "Monte Carlo fallback"},
                "key_insight": "",
            }

    loss_scenarios = _compute_dollar_losses(granite_crops, market_data, farm_crops)

    total_rev  = sum(v.get("full_revenue", 0) for v in loss_scenarios.values() if isinstance(v, dict))
    total_p50  = sum(v.get("scenarios", {}).get("median_p50",   {}).get("loss", 0) for v in loss_scenarios.values() if isinstance(v, dict))
    total_p10  = sum(v.get("scenarios", {}).get("bad_year_p10", {}).get("loss", 0) for v in loss_scenarios.values() if isinstance(v, dict))
    sev_rank   = {"critical": 4, "high": 3, "moderate": 2, "low": 1, "unknown": 0}
    severities = [v.get("severity", "low") for v in loss_scenarios.values() if isinstance(v, dict)]
    overall    = max(severities, key=lambda s: sev_rank.get(s, 0)) if severities else "unknown"

    farm_risk_summary = {
        "total_full_revenue": round(total_rev, 2),
        "total_p50_loss":     round(total_p50, 2),
        "total_p10_loss":     round(total_p10, 2),
        "total_p50_loss_pct": round(total_p50 / total_rev * 100, 1) if total_rev > 0 else 0,
        "total_p10_loss_pct": round(total_p10 / total_rev * 100, 1) if total_rev > 0 else 0,
        "overall_severity":   overall,
        "dominant_risk":      granite_assessment.get("farm_level", {}).get("dominant_risk", "unknown"),
        "assessment_summary": granite_assessment.get("assessment_summary", ""),
        "confidence":         granite_assessment.get("farm_level", {}).get("confidence", "unknown"),
        "enso_phase":         seasonal_outlook.get("enso_phase", "neutral"),
        "drought_risk_elevated": seasonal_outlook.get("seasonal_outlook", {}).get("drought_risk_elevated", False),
    }

    log.info("Risk complete: rev=$%.0f  P50_loss=$%.0f  P10_loss=$%.0f  sev=%s  fallback=%s",
             total_rev, total_p50, total_p10, overall, fallback_used)

    return {
        "granite_assessment": granite_assessment,
        "loss_scenarios":     loss_scenarios,
        "farm_risk_summary":  farm_risk_summary,
        "granite_raw":        raw_text,
        "fallback_used":      fallback_used,
        "generated_utc":      datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json as jsonlib
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

    monte_carlo_all = {
        "Rice":     {"yield_distribution": {"p10": 30.7, "p25": 48.0, "p50": 60.9, "p75": 75.0, "p90": 81.1, "std_dev": 19.6, "skewness": -0.30}, "season_stress_medians": {"heat_stress_days": 8, "critical_heat_days": 2, "precip_deficit_pct": -12.0, "consecutive_dry_max": 9, "gdd_total": 1850}},
        "Wheat":    {"yield_distribution": {"p10": 35.1, "p25": 50.0, "p50": 60.4, "p75": 70.0, "p90": 75.0, "std_dev": 15.1, "skewness": -0.33}, "season_stress_medians": {"heat_stress_days": 5, "critical_heat_days": 1, "precip_deficit_pct": -22.0, "consecutive_dry_max": 12, "gdd_total": 1420}},
        "Soybeans": {"yield_distribution": {"p10": 36.1, "p25": 55.0, "p50": 70.8, "p75": 82.0, "p90": 86.1, "std_dev": 19.7, "skewness": -0.69}, "season_stress_medians": {"heat_stress_days": 6, "critical_heat_days": 1, "precip_deficit_pct": -18.0, "consecutive_dry_max": 11, "gdd_total": 2100}},
    }

    market_data_mock = {
        "Rice":     {"has_futures": True, "price_usd": 16.50,  "unit": "cwt",    "vol_proxy_pct": 22.0, "typical_yield_per_acre": 78.0, "revenue_per_acre": 1287.0},
        "Wheat":    {"has_futures": True, "price_usd": 5.915,  "unit": "bushel", "vol_proxy_pct": 12.0, "typical_yield_per_acre": 49.7, "revenue_per_acre": 293.98},
        "Soybeans": {"has_futures": True, "price_usd": 11.70,  "unit": "bushel", "vol_proxy_pct": 12.0, "typical_yield_per_acre": 51.7, "revenue_per_acre": 604.89},
    }

    soil_anomaly_mock = {
        "current_anomaly": {"anomaly_signal": "dry", "precip_anomaly_pct": -54.9,
                            "temp_anomaly_c": 0.62, "max_dry_streak_30d": 11, "current_dry_streak": 9},
        "crops": {
            "Rice":     {"soil": {"soil_score": 0.6811, "drought_buffer_days": 5.1, "properties": {"clay_pct": 19.3, "soc_gkg": 25.11, "ph": 5.4, "sand_pct": 23.2}}},
            "Wheat":    {"soil": {"soil_score": 0.7819, "drought_buffer_days": 5.1, "properties": {"clay_pct": 19.3, "soc_gkg": 25.11, "ph": 5.4, "sand_pct": 23.2}}},
            "Soybeans": {"soil": {"soil_score": 0.7530, "drought_buffer_days": 5.1, "properties": {"clay_pct": 19.3, "soc_gkg": 25.11, "ph": 5.4, "sand_pct": 23.2}}},
        },
    }

    seasonal_outlook_mock = {
        "enso":       {"enso_desc": "Weak/Moderate La Niña", "oni_value": -0.55},
        "enso_phase": "weak_la_nina",
        "seasonal_outlook": {
            "drought_risk_elevated": True,
            "temp_above_normal_prob_pct": 50,
            "precip_below_normal_prob_pct": 55,
            "risk_narrative": "Weak La Niña increases drought and heat risk for the Delta region.",
        },
        "forecast_14d": {"total_precip_mm": 56.1, "heat_stress_days": 0,
                         "dry_days_forecast": 7, "extreme_heat_flag": False},
    }

    print("=" * 60)
    print("AgriShield yield_risk  |  NW Arkansas  |  Granite Call 1")
    print("=" * 60)

    result = run_risk_assessment(
        farm_input       = farm_input,
        soil_anomaly     = soil_anomaly_mock,
        monte_carlo_all  = monte_carlo_all,
        market_data      = market_data_mock,
        seasonal_outlook = seasonal_outlook_mock,
    )

    print(f"\nFallback used: {result['fallback_used']}")
    print(f"\n--- Granite Assessment Summary ---")
    g = result["granite_assessment"]
    print(g.get("assessment_summary", "N/A"))
    print(f"\nDominant risk : {g.get('farm_level',{}).get('dominant_risk','N/A')}")
    print(f"Confidence    : {g.get('farm_level',{}).get('confidence','N/A')}")

    print("\n--- Per-Crop Risk Coefficients ---")
    for name, crop in g.get("crops", {}).items():
        rc = crop.get("risk_coefficients", {})
        ye = crop.get("yield_loss_estimate", {})
        print(f"\n  {name}:")
        print(f"    Risk score  : {rc.get('overall_risk_score','N/A')}/100")
        print(f"    Drought     : {rc.get('drought_risk','N/A')}  Heat: {rc.get('heat_risk','N/A')}  Soil: {rc.get('soil_penalty','N/A')}  ENSO: {rc.get('enso_impact','N/A')}")
        print(f"    P10/P50/P90 : {ye.get('p10_yield_pct','N/A')}% / {ye.get('p50_yield_pct','N/A')}% / {ye.get('p90_yield_pct','N/A')}%")
        print(f"    Driver      : {crop.get('primary_risk_driver','N/A')}")
        print(f"    Insight     : {crop.get('key_insight','N/A')}")

    print("\n--- Dollar Loss Scenarios ---")
    for name, loss in result["loss_scenarios"].items():
        if not isinstance(loss, dict) or "error" in loss:
            continue
        s = loss["scenarios"]
        print(f"\n  {name} ({loss['acres']} acres)  |  Full revenue: ${loss['full_revenue']:,.0f}")
        print(f"    P90: ${s['good_year_p90']['revenue']:,.0f}  |  P50: ${s['median_p50']['revenue']:,.0f} (loss ${s['median_p50']['loss']:,.0f})  |  P10: ${s['bad_year_p10']['revenue']:,.0f} (loss ${s['bad_year_p10']['loss']:,.0f})")
        print(f"    Severity: {loss['severity']}")

    s = result["farm_risk_summary"]
    print(f"\n--- Farm Summary ---")
    print(f"  Total revenue  : ${s['total_full_revenue']:,.0f}")
    print(f"  P50 total loss : ${s['total_p50_loss']:,.0f}  ({s['total_p50_loss_pct']:.1f}%)")
    print(f"  P10 total loss : ${s['total_p10_loss']:,.0f}  ({s['total_p10_loss_pct']:.1f}%)")
    print(f"  Severity       : {s['overall_severity']}")

    print("\n[Done]")
