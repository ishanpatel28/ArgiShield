"""
hedge_optimizer.py
==================
AgriShield – Commodity Hedge Optimizer
---------------------------------------
Takes risk assessment output (loss scenarios + market data) and computes
practical protective put strategies for each crop with a futures contract.

For crops without futures, recommends crop insurance / input cost hedging.

Math:
  - Black-Scholes put pricing (European approximation; close enough for
    short-dated agricultural options on annual contracts)
  - Strike = current price × (1 - downside_buffer)
    where downside_buffer is calibrated to the P50 loss percentage
  - Number of contracts = ceil(revenue_at_risk / (contract_size × price))
  - Total premium = contracts × BSM_put_price × contract_size
  - Hedge ratio = premium / total_expected_revenue

Output feeds into:
  - granite_advisor.py  (for plain-language narrative)
  - pipeline.py         (assembled into final result)

Dependencies: math (stdlib only — no scipy needed; BSM in pure Python).
"""

import logging
import math
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("agrishield.hedge_optimizer")

# ---------------------------------------------------------------------------
# Contract sizes (CBOT/CME standard)
# ---------------------------------------------------------------------------

_CONTRACT_SIZES = {
    "corn":     5000,   # bushels
    "soybeans": 5000,   # bushels
    "wheat":    5000,   # bushels
    "rice":     2000,   # cwt  (CBOT Rough Rice is 2,000 cwt; ZR contract)
    "cotton":   50000,  # pounds
    "oats":     5000,   # bushels
    "sorghum":  5000,   # bushels (use corn contract as proxy)
    "canola":   20000,  # kg  (ICE proxy)
    "barley":   5000,   # bushels (proxy)
}

_DEFAULT_CONTRACT_SIZE = 5000

# Minimum hedge threshold — don't recommend options if premium > this
# fraction of expected revenue (too expensive to justify)
_MAX_HEDGE_COST_PCT = 0.15  # 15%


# ---------------------------------------------------------------------------
# Black-Scholes put (European)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc (no scipy needed)."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def bsm_put_price(
    S: float,   # current futures price (USD)
    K: float,   # strike price (USD)
    T: float,   # time to expiry in years
    sigma: float,  # annualized volatility (decimal, e.g. 0.20)
    r: float = 0.05,  # risk-free rate (5% default)
) -> float:
    """
    Black-Scholes-Merton put price.
    For futures options: replace S with F·e^{-rT} (F = futures price).
    Using Black's model approximation: treat futures as the underlying
    with r=0 in the dividend term (standard for commodity futures).
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    put = (K * math.exp(-r * T) * _norm_cdf(-d2)) - (S * _norm_cdf(-d1))
    return max(put, 0.0)


# ---------------------------------------------------------------------------
# Per-crop hedge recommendation
# ---------------------------------------------------------------------------

def _recommend_crop(
    crop_name: str,
    market: dict,
    loss_scenario: dict,
    risk_score: int,
    budget_pct: float,
    risk_aversion: float,
) -> dict:
    """
    Compute a protective put recommendation for one crop.

    Parameters
    ----------
    crop_name       : e.g. "Rice"
    market          : dict from market_data.get_futures_price()
    loss_scenario   : dict from yield_risk loss_scenarios[crop_name]
    risk_score      : 0-100 from Granite or fallback
    budget_pct      : hedge budget as % of total revenue (e.g. 3.0)
    risk_aversion   : 0-100 slider value

    Returns
    -------
    dict with recommendation, pricing, and plain-language summary.
    """
    key = crop_name.lower()
    has_futures = market.get("has_futures", False)

    # Crops without futures → alternatives
    if not has_futures:
        alternatives = market.get("alternatives", {})
        return {
            "crop": crop_name,
            "has_futures": False,
            "strategy": "no_futures_contract",
            "recommendation": _no_futures_narrative(crop_name, alternatives, loss_scenario),
            "alternatives": alternatives,
        }

    price       = market.get("price_usd", 0.0)
    vol         = (market.get("vol_proxy_pct", 20.0) or 20.0) / 100.0
    acres       = loss_scenario.get("acres", 0.0)
    full_rev    = loss_scenario.get("full_revenue", 0.0)
    scenarios   = loss_scenario.get("scenarios", {})
    p50_loss    = scenarios.get("median_p50", {}).get("loss", 0.0)
    p10_loss    = scenarios.get("bad_year_p10", {}).get("loss", 0.0)
    p50_yld_pct = scenarios.get("median_p50", {}).get("yield_pct", 90.0)

    if price <= 0 or acres <= 0:
        return {"crop": crop_name, "has_futures": True, "strategy": "insufficient_data",
                "recommendation": "Insufficient market data to size a hedge."}

    contract_size = _CONTRACT_SIZES.get(key, _DEFAULT_CONTRACT_SIZE)
    typical_yield = market.get("typical_yield_per_acre", 0.0)

    # Bushels/units produced by this farm at full yield
    farm_units = typical_yield * acres

    # Strike: target protecting revenue at the P50 scenario
    # (don't go too far OTM, stay meaningful)
    downside = max(0.05, (100.0 - p50_yld_pct) / 100.0)
    strike = round(price * (1.0 - downside * 0.5), 4)  # 50% of expected downside
    strike = max(strike, price * 0.70)  # floor at 30% OTM

    # Time to expiry: assume ~8-month growing season contract
    T = 8 / 12  # years

    # BSM put premium per unit
    premium_per_unit = bsm_put_price(S=price, K=strike, T=T, sigma=vol)

    # Contracts needed to cover expected P50 loss
    units_at_risk = farm_units * downside
    n_contracts_raw = units_at_risk / contract_size if contract_size > 0 else 0
    # Risk aversion scales from minimum (0.5x) to maximum (1.5x) coverage
    ra_scale = 0.5 + (risk_aversion / 100.0)
    n_contracts = max(1, math.ceil(n_contracts_raw * ra_scale))

    # Total premium cost
    total_premium = n_contracts * premium_per_unit * contract_size
    hedge_cost_pct = (total_premium / full_rev * 100) if full_rev > 0 else 0.0

    # Revenue protected at strike
    protected_units = n_contracts * contract_size
    protection_value = protected_units * max(strike - price * p50_yld_pct / 100, 0) + \
                       protected_units * strike  # max payout if price falls to zero

    # Budget check
    budget_dollars = full_rev * budget_pct / 100.0
    affordable = total_premium <= budget_dollars

    # Scale down contracts to budget if needed
    if not affordable:
        n_contracts_budget = max(1, int(budget_dollars / (premium_per_unit * contract_size + 1e-9)))
        n_contracts_used = n_contracts_budget
        total_premium_used = n_contracts_budget * premium_per_unit * contract_size
        hedge_cost_pct_used = (total_premium_used / full_rev * 100) if full_rev > 0 else 0.0
        budget_constrained = True
    else:
        n_contracts_used = n_contracts
        total_premium_used = total_premium
        hedge_cost_pct_used = hedge_cost_pct
        budget_constrained = False

    # Viability check — don't recommend if cost exceeds max threshold
    if hedge_cost_pct_used > _MAX_HEDGE_COST_PCT * 100:
        viable = False
    else:
        viable = True

    severity = loss_scenario.get("severity", "low")

    result = {
        "crop": crop_name,
        "has_futures": True,
        "strategy": "protective_put",
        "viable": viable,
        "budget_constrained": budget_constrained,
        "inputs": {
            "current_price_usd": round(price, 4),
            "strike_usd": round(strike, 4),
            "volatility_pct": round(vol * 100, 1),
            "time_to_expiry_months": round(T * 12, 0),
            "contract_size": contract_size,
            "unit": market.get("unit", "bushel"),
            "farm_units_total": round(farm_units, 1),
        },
        "sizing": {
            "contracts_recommended": n_contracts_used,
            "premium_per_unit_usd": round(premium_per_unit, 4),
            "total_premium_usd": round(total_premium_used, 2),
            "hedge_cost_pct_revenue": round(hedge_cost_pct_used, 2),
            "budget_dollars": round(budget_dollars, 2),
        },
        "risk_profile": {
            "full_revenue": round(full_rev, 2),
            "p50_loss": round(p50_loss, 2),
            "p10_loss": round(p10_loss, 2),
            "severity": severity,
            "risk_score": risk_score,
        },
        "exchange": market.get("exchange", "CBOT"),
    }

    result["recommendation"] = _put_narrative(
        crop_name, result, market, p50_loss, p10_loss, affordable
    )

    log.info(
        "Hedge %s: %d put contracts @ $%.2f strike  premium=$%.0f  cost=%.1f%%  viable=%s",
        crop_name, n_contracts_used, strike, total_premium_used,
        hedge_cost_pct_used, viable
    )

    return result


def _put_narrative(
    crop: str, rec: dict, market: dict, p50_loss: float, p10_loss: float, affordable: bool
) -> str:
    inp = rec["inputs"]
    siz = rec["sizing"]
    unit = inp["unit"]
    exchange = rec.get("exchange", "CBOT")

    lines = [
        f"Buy {siz['contracts_recommended']} {crop} put contract{'s' if siz['contracts_recommended'] > 1 else ''} "
        f"on {exchange} at a ${inp['strike_usd']:.2f}/{unit} strike.",
        f"Estimated premium: ${siz['total_premium_usd']:,.0f} total "
        f"({siz['hedge_cost_pct_revenue']:.1f}% of expected revenue).",
        f"This hedge covers your downside if prices drop below ${inp['strike_usd']:.2f}/{unit}, "
        f"partially offsetting a median-scenario loss of ${p50_loss:,.0f}.",
    ]
    if rec.get("budget_constrained"):
        lines.append(
            "Note: full coverage would exceed your hedge budget — position has been scaled to fit."
        )
    if not rec.get("viable"):
        lines.append(
            "Warning: options premium is high relative to revenue. Consider crop insurance instead."
        )
    return " ".join(lines)


def _no_futures_narrative(crop: str, alternatives: dict, loss_scenario: dict) -> str:
    p50_loss = loss_scenario.get("scenarios", {}).get("median_p50", {}).get("loss", 0.0)
    parts = [
        f"{crop} has no active exchange-traded futures contract.",
        f"With a median-scenario revenue loss of ${p50_loss:,.0f}, consider:",
    ]
    for alt_key, alt in alternatives.items():
        parts.append(f"  • {alt.get('description', alt_key)}")
    if not alternatives:
        parts.append("  • USDA crop insurance (MPCI or revenue protection policy)")
        parts.append("  • Input cost forward contracts to cap seed/fertilizer expenses")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Farm-level summary
# ---------------------------------------------------------------------------

def _farm_summary(recommendations: list, farm_risk_summary: dict, budget_pct: float) -> dict:
    total_premium = sum(
        r.get("sizing", {}).get("total_premium_usd", 0.0)
        for r in recommendations
        if r.get("has_futures") and r.get("viable", True)
    )
    total_rev = farm_risk_summary.get("total_full_revenue", 0.0)
    total_p50_loss = farm_risk_summary.get("total_p50_loss", 0.0)
    total_p10_loss = farm_risk_summary.get("total_p10_loss", 0.0)
    budget_dollars = total_rev * budget_pct / 100.0

    crops_with_puts = [r["crop"] for r in recommendations
                       if r.get("strategy") == "protective_put" and r.get("viable", True)]
    crops_no_futures = [r["crop"] for r in recommendations if not r.get("has_futures")]

    return {
        "total_hedge_premium": round(total_premium, 2),
        "total_budget": round(budget_dollars, 2),
        "hedge_cost_pct_revenue": round(total_premium / total_rev * 100, 2) if total_rev > 0 else 0,
        "crops_hedged_with_puts": crops_with_puts,
        "crops_needing_alternatives": crops_no_futures,
        "total_p50_loss_protected_against": round(total_p50_loss, 2),
        "total_p10_loss_unhedged": round(max(total_p10_loss - total_premium, 0), 2),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_hedge_optimizer(
    farm_input: dict,
    market_data: dict,
    risk_result: dict,
) -> dict:
    """
    Compute hedge recommendations for all crops.

    Parameters
    ----------
    farm_input   : original farm payload (crops, preferences)
    market_data  : output of market_data.get_market_data_all_crops()
    risk_result  : output of yield_risk.run_risk_assessment()

    Returns
    -------
    dict with per-crop recommendations and farm summary.
    """
    prefs = farm_input.get("preferences", {})
    budget_pct    = prefs.get("hedge_budget_pct_revenue", 3.0)
    risk_aversion = prefs.get("risk_aversion_0_100", 60.0)

    farm_risk_summary = risk_result.get("farm_risk_summary", {})
    loss_scenarios    = risk_result.get("loss_scenarios", {})
    granite_crops     = risk_result.get("granite_assessment", {}).get("crops", {})

    recommendations = []
    for crop_entry in farm_input.get("crops", []):
        name = crop_entry.get("name", "")
        market       = market_data.get(name, {})
        loss_scenario = loss_scenarios.get(name, {})
        risk_score    = granite_crops.get(name, {}).get(
            "risk_coefficients", {}
        ).get("overall_risk_score", 50)

        rec = _recommend_crop(
            crop_name     = name,
            market        = market,
            loss_scenario = loss_scenario,
            risk_score    = int(risk_score),
            budget_pct    = budget_pct,
            risk_aversion = risk_aversion,
        )
        recommendations.append(rec)

    summary = _farm_summary(recommendations, farm_risk_summary, budget_pct)

    log.info(
        "Hedge optimizer complete: %d crops  total_premium=$%.0f  budget=$%.0f",
        len(recommendations), summary["total_hedge_premium"], summary["total_budget"]
    )

    return {
        "recommendations": recommendations,
        "farm_hedge_summary": summary,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
