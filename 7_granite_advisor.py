"""
granite_advisor.py
==================
AgriShield – Granite Call 2: Plain-Language Farmer Narrative
"""

import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

import requests

log = logging.getLogger("agrishield.granite_advisor")

# ---------------------------------------------------------------------------
# Watsonx config
# ---------------------------------------------------------------------------

_WATSONX_URL        = "https://us-south.ml.cloud.ibm.com"
_WATSONX_API_KEY    = "mSOreiGvCoHMN1IPW4opoGMGRNc_NFLn1dInxQ4t38YE"
_WATSONX_PROJECT_ID = "e4164bef-4b12-4681-933c-d3ad03941cb5"
_GRANITE_MODEL      = "ibm/granite-3-8b-instruct"
_IAM_URL            = "https://iam.cloud.ibm.com/identity/token"
_INFER_URL          = f"{_WATSONX_URL}/ml/v1/text/chat?version=2023-05-29"
_TIMEOUT            = 60


# ---------------------------------------------------------------------------
# Watsonx helpers
# ---------------------------------------------------------------------------

def _get_iam_token() -> Optional[str]:
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
        log.warning("IAM token failed: %s", exc)
        return None


def _call_granite(prompt: str, max_tokens: int = 800) -> Optional[str]:
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
                "messages": [{"role": "user", "content": prompt}],
                "parameters": {
                    "decoding_method": "greedy",
                    "max_new_tokens":  max_tokens,
                    "min_new_tokens":  50,
                    "temperature":     0.3,
                    "top_p":           0.9,
                },
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.HTTPError as exc:
        log.warning("Granite Call 2 HTTP %s: %s", exc.response.status_code, exc.response.text[:200])
        return None
    except Exception as exc:
        log.warning("Granite Call 2 failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_advisory_prompt(
    farm_input: dict,
    risk_result: dict,
    hedge_result: dict,
    seasonal_outlook: dict,
) -> str:
    crops  = farm_input.get("crops", [])
    frs    = risk_result.get("farm_risk_summary", {})
    fhs    = hedge_result.get("farm_hedge_summary", {})
    recs   = hedge_result.get("recommendations", [])
    enso   = seasonal_outlook.get("enso", {}).get("enso_desc", "neutral")
    outlook_text = seasonal_outlook.get("seasonal_outlook", {}).get("risk_narrative", "")

    crop_lines = []
    for c in crops:
        name = c["name"]
        ls   = risk_result.get("loss_scenarios", {}).get(name, {})
        sc   = ls.get("scenarios", {})
        p50  = sc.get("median_p50", {}).get("loss", 0)
        p10  = sc.get("bad_year_p10", {}).get("loss", 0)
        fr   = ls.get("full_revenue", 0)
        sev  = ls.get("severity", "unknown")
        rec  = next((r for r in recs if r["crop"] == name), {})
        hedge_line = rec.get("recommendation", "No hedge available.")
        crop_lines.append(
            f"  {name}: full revenue=${fr:,.0f}  median loss=${p50:,.0f}  "
            f"worst-case loss=${p10:,.0f}  severity={sev}\n"
            f"  Hedge action: {hedge_line}"
        )

    crops_block = "\n".join(crop_lines)

    return f"""You are AgriShield, a plain-language agricultural risk advisor.
A farmer has the following climate risk profile for the upcoming season.
Write in plain, direct language — no financial jargon, no filler phrases.

SEASONAL CONTEXT:
  ENSO phase: {enso}
  Outlook: {outlook_text}

FARM RISK:
  Total expected revenue: ${frs.get('total_full_revenue', 0):,.0f}
  Median season loss: ${frs.get('total_p50_loss', 0):,.0f} ({frs.get('total_p50_loss_pct', 0):.1f}%)
  Worst-case loss (1-in-10 year): ${frs.get('total_p10_loss', 0):,.0f} ({frs.get('total_p10_loss_pct', 0):.1f}%)
  Dominant risk: {frs.get('dominant_risk', 'drought')}

PER-CROP BREAKDOWN:
{crops_block}

HEDGE BUDGET:
  Total hedge premium recommended: ${fhs.get('total_hedge_premium', 0):,.0f}
  Budget available: ${fhs.get('total_budget', 0):,.0f}

Write two sections:

SECTION 1 — HEDGING SUMMARY (3–5 sentences, plain English):
Tell the farmer exactly what they should do this season and why, in the same
tone you'd use talking to a neighbor. Lead with the most important action.
No bullet points. No headers. Just clear prose.

SECTION 2 — DETAILED ADVISORY (for a printed report):
Write 4–6 paragraphs covering:
  a) The seasonal climate setup and what it means for this farm specifically
  b) The risk for each crop and which one worries you most
  c) The specific hedge actions to take, in plain terms
  d) What to watch for as the season progresses (triggers to act or adjust)

Keep both sections factual and grounded. Avoid generic boilerplate."""


# ---------------------------------------------------------------------------
# Template fallback (no Granite)
# ---------------------------------------------------------------------------

def _template_narrative(
    farm_input: dict,
    risk_result: dict,
    hedge_result: dict,
    seasonal_outlook: dict,
) -> tuple[str, str]:
    frs    = risk_result.get("farm_risk_summary", {})
    fhs    = hedge_result.get("farm_hedge_summary", {})
    recs   = hedge_result.get("recommendations", [])
    enso   = seasonal_outlook.get("enso", {}).get("enso_desc", "neutral conditions")
    crops  = farm_input.get("crops", [])
    ls_all = risk_result.get("loss_scenarios", {})

    p50_loss   = frs.get("total_p50_loss", 0)
    p10_loss   = frs.get("total_p10_loss", 0)
    p50_pct    = frs.get("total_p50_loss_pct", 0)
    dominant   = frs.get("dominant_risk", "drought").replace("_", " ")
    severity   = frs.get("overall_severity", "moderate")
    total_prem = fhs.get("total_hedge_premium", 0)
    hedged_crops = fhs.get("crops_hedged_with_puts", [])

    hedge_action = (
        f"The most important action right now is to buy protective put contracts on "
        f"{' and '.join(hedged_crops)}, costing approximately ${total_prem:,.0f} in total premium. "
        if hedged_crops and total_prem > 0
        else "Your crops lack direct futures coverage — focus on crop insurance and locking in input costs before prices rise. "
    )

    short_summary = (
        f"This season looks {severity} risk driven by {dominant}, amplified by {enso}. "
        f"Your farm stands to lose a median of ${p50_loss:,.0f} ({p50_pct:.0f}% of expected revenue) "
        f"in a typical bad year, and up to ${p10_loss:,.0f} in a worst-case scenario. "
        f"{hedge_action}"
        f"Act before planting — options premiums rise as the season progresses and uncertainty resolves."
    )

    seasonal_para = (
        f"The current climate setup features {enso}. "
        f"{seasonal_outlook.get('seasonal_outlook', {}).get('risk_narrative', '')} "
        f"Recent weather data shows a precipitation deficit of "
        f"{abs(seasonal_outlook.get('current_anomaly', {}).get('precip_anomaly_pct', 0)):.0f}% "
        f"below normal for the past 30 days, which compounds the seasonal dryness signal."
    )

    crop_paras = []
    for c in crops:
        name = c["name"]
        ls   = ls_all.get(name, {})
        sc   = ls.get("scenarios", {})
        fr   = ls.get("full_revenue", 0)
        p50  = sc.get("median_p50", {}).get("loss", 0)
        p10  = sc.get("bad_year_p10", {}).get("loss", 0)
        sev  = ls.get("severity", "unknown")
        rec  = next((r for r in recs if r["crop"] == name), {})
        crop_paras.append(
            f"{name} ({c['acres']} acres): Expected revenue ${fr:,.0f}. "
            f"Median loss ${p50:,.0f}, worst-case ${p10:,.0f}. Risk level: {sev}. "
            f"Recommendation: {rec.get('recommendation', 'No hedge available.')}"
        )

    detailed = "\n\n".join([
        "SEASONAL CONTEXT\n" + seasonal_para,
        "PER-CROP RISK\n" + "\n\n".join(crop_paras),
        "WHAT TO WATCH\nAs the season progresses, monitor soil moisture weekly. "
        "If a second consecutive dry month develops or a heat event exceeds 7 consecutive days "
        "during the critical window, consider rolling your put options to a higher strike or adding contracts. "
        "Conversely, if significant rain arrives in the first 45 days of the growing season, "
        "you may be able to sell your puts back at a profit.",
    ])

    return short_summary, detailed


def _parse_granite_response(text: str) -> tuple[str, str]:
    s1_match = re.search(r"SECTION\s*1[^:]*:(.*?)(?:SECTION\s*2|$)", text, re.DOTALL | re.IGNORECASE)
    s2_match = re.search(r"SECTION\s*2[^:]*:(.*?)$", text, re.DOTALL | re.IGNORECASE)

    short    = s1_match.group(1).strip() if s1_match else ""
    detailed = s2_match.group(1).strip() if s2_match else ""

    if not short and text:
        paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
        if paragraphs:
            short    = paragraphs[0]
            detailed = "\n\n".join(paragraphs[1:]) if len(paragraphs) > 1 else paragraphs[0]

    return short, detailed


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_granite_advisor(
    farm_input: dict,
    risk_result: dict,
    hedge_result: dict,
    seasonal_outlook: dict,
) -> dict:
    prompt       = _build_advisory_prompt(farm_input, risk_result, hedge_result, seasonal_outlook)
    raw_text     = _call_granite(prompt, max_tokens=900)
    granite_used = bool(raw_text)

    if raw_text:
        short, detailed = _parse_granite_response(raw_text)
        if not short or not detailed:
            short, detailed = _template_narrative(farm_input, risk_result, hedge_result, seasonal_outlook)
            granite_used = False
        log.info("Granite advisor: parsed  granite_used=%s", granite_used)
    else:
        short, detailed = _template_narrative(farm_input, risk_result, hedge_result, seasonal_outlook)
        log.info("Granite advisor: using template fallback")

    return {
        "short_summary":      short,
        "detailed_narrative": detailed,
        "granite_used":       granite_used,
        "generated_utc":      datetime.now(timezone.utc).isoformat(),
    }