"""
granite_advisor.py
==================
AgriShield – Granite Call 2: Plain-Language Farmer Narrative
-------------------------------------------------------------
Takes the full quantitative output (risk assessment + hedge recommendations)
and asks Granite to produce:

  1. A SHORT hedging summary (3-5 sentences, farmer-readable, no jargon)
     — what the farmer should actually DO this season.

  2. A DETAILED narrative breakdown (suitable for PDF report):
     — seasonal context (ENSO, current anomaly)
     — per-crop risk summary with reasoning
     — specific hedge actions recommended
     — what to watch for as the season progresses

If Granite is unavailable, falls back to a structured template-based
narrative using the quantitative data directly. Output is always complete.

Dependencies: requests + standard library only.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

import requests

log = logging.getLogger("agrishield.granite_advisor")

# ---------------------------------------------------------------------------
# Watsonx config — reads from env first, falls back to hardcoded values
# ---------------------------------------------------------------------------

_WATSONX_API_KEY    = os.getenv("WATSONX_API_KEY",    "b3d5f298-5dd1-4f8d-a868-b3c4b223a517")
_WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "e4164bef-4b12-4681-933c-d3ad03941cb5")
_GRANITE_MODEL      = "ibm/granite-4-0-tiny-preview"
_WATSONX_URL        = "https://us-south.ml.cloud.ibm.com"
_IAM_URL            = "https://iam.cloud.ibm.com/identity/token"
_INFER_URL          = f"{_WATSONX_URL}/ml/v1/text/chat?version=2023-05-29"
_TIMEOUT            = 60


# ---------------------------------------------------------------------------
# Watsonx helpers (same pattern as yield_risk.py)
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
    prefs  = farm_input.get("preferences", {})
    crops  = farm_input.get("crops", [])
    farm   = farm_input.get("farm", {})
    frs    = risk_result.get("farm_risk_summary", {})
    fhs    = hedge_result.get("farm_hedge_summary", {})
    recs   = hedge_result.get("recommendations", [])
    enso   = seasonal_outlook.get("enso_desc", "neutral")
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

    prompt = f"""You are AgriShield, a plain-language agricultural risk advisor.
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

    return prompt


# ---------------------------------------------------------------------------
# Template fallback (no Granite)
# ---------------------------------------------------------------------------

def _template_narrative(
    farm_input: dict,
    risk_result: dict,
    hedge_result: dict,
    seasonal_outlook: dict,
) -> tuple[str, str]:
    """Returns (short_summary, detailed_narrative) as strings."""
    frs  = risk_result.get("farm_risk_summary", {})
    fhs  = hedge_result.get("farm_hedge_summary", {})
    recs = hedge_result.get("recommendations", [])
    enso = seasonal_outlook.get("enso_desc", "neutral conditions")
    crops = farm_input.get("crops", [])
    ls_all = risk_result.get("loss_scenarios", {})

    total_rev  = frs.get("total_full_revenue", 0)
    p50_loss   = frs.get("total_p50_loss", 0)
    p10_loss   = frs.get("total_p10_loss", 0)
    p50_pct    = frs.get("total_p50_loss_pct", 0)
    dominant   = frs.get("dominant_risk", "drought").replace("_", " ")
    severity   = frs.get("overall_severity", "moderate")
    total_prem = fhs.get("total_hedge_premium", 0)

    # Short summary
    hedged_crops = fhs.get("crops_hedged_with_puts", [])
    hedge_action = ""
    if hedged_crops and total_prem > 0:
        hedge_action = (
            f"The most important action right now is to buy protective put contracts on "
            f"{' and '.join(hedged_crops)}, costing approximately ${total_prem:,.0f} in total premium. "
        )
    else:
        hedge_action = (
            "Your crops lack direct futures coverage — focus on crop insurance and "
            "locking in input costs before prices rise. "
        )

    short_summary = (
        f"This season looks {severity} risk driven by {dominant}, amplified by {enso}. "
        f"Your farm stands to lose a median of ${p50_loss:,.0f} ({p50_pct:.0f}% of expected revenue) "
        f"in a typical bad year, and up to ${p10_loss:,.0f} in a worst-case scenario. "
        f"{hedge_action}"
        f"Act before planting — options premiums rise as the season progresses and uncertainty resolves."
    )

    # Detailed narrative
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
        hedge_text = rec.get("recommendation", "No hedge available.")
        crop_paras.append(
            f"{name} ({c['acres']} acres): Expected revenue ${fr:,.0f}. "
            f"Median loss ${p50:,.0f}, worst-case ${p10:,.0f}. Risk level: {sev}. "
            f"Recommendation: {hedge_text}"
        )

    watch_para = (
        "As the season progresses, monitor soil moisture weekly. "
        "If a second consecutive dry month develops or a heat event exceeds 7 consecutive days "
        "during the critical window, consider rolling your put options to a higher strike "
        "or adding contracts. Conversely, if significant rain arrives in the first 45 days "
        "of the growing season, you may be able to sell your puts back at a profit."
    )

    detailed = "\n\n".join([
        "SEASONAL CONTEXT\n" + seasonal_para,
        "PER-CROP RISK\n" + "\n\n".join(crop_paras),
        "WHAT TO WATCH\n" + watch_para,
    ])

    return short_summary, detailed


def _parse_granite_response(text: str) -> tuple[str, str]:
    """Extract SECTION 1 and SECTION 2 from Granite response."""
    short = ""
    detailed = ""

    # Try to split on section markers
    s1_match = re.search(r"SECTION\s*1[^:]*:(.*?)(?:SECTION\s*2|$)", text, re.DOTALL | re.IGNORECASE)
    s2_match = re.search(r"SECTION\s*2[^:]*:(.*?)$", text, re.DOTALL | re.IGNORECASE)

    if s1_match:
        short = s1_match.group(1).strip()
    if s2_match:
        detailed = s2_match.group(1).strip()

    # Fallback: use first paragraph as short, rest as detailed
    if not short and text:
        paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
        if paragraphs:
            short = paragraphs[0]
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
    """
    Granite Call 2: produce plain-language farmer advisory.

    Returns
    -------
    dict with:
      short_summary  : 3-5 sentence plain English summary
      detailed_narrative : full report text (for PDF)
      granite_used   : bool
    """
    prompt    = _build_advisory_prompt(farm_input, risk_result, hedge_result, seasonal_outlook)
    raw_text  = _call_granite(prompt, max_tokens=900)
    granite_used = bool(raw_text)

    if raw_text:
        short, detailed = _parse_granite_response(raw_text)
        if not short or not detailed:
            # Granite replied but unparseable — use template
            short, detailed = _template_narrative(farm_input, risk_result, hedge_result, seasonal_outlook)
            granite_used = False
        log.info("Granite advisor: response parsed  granite_used=%s", granite_used)
    else:
        short, detailed = _template_narrative(farm_input, risk_result, hedge_result, seasonal_outlook)
        log.info("Granite advisor: fallback to template narrative")

    return {
        "short_summary":       short,
        "detailed_narrative":  detailed,
        "granite_used":        granite_used,
        "generated_utc":       datetime.now(timezone.utc).isoformat(),
    }
