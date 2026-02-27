"""
market_data.py
==============
AgriShield – Commodity Futures Market Data
-------------------------------------------
Fetches current futures prices for known agricultural commodities
from Yahoo Finance (free, no API key required).

Key notes on the data:
  - Corn, soybeans, wheat return prices in USX (US cents per bushel).
    We convert to USD automatically: USX / 100 = USD.
  - Rice (ZR=F) returns unreliable data from Yahoo — we use a USDA
    benchmark fallback instead.
  - Cotton (CT=F) returns USX (cents per pound).
  - No implied volatility is available from free sources. We compute
    a historical volatility proxy from the 52-week range as a substitute.
    This is less precise than true IV but directionally correct and
    sufficient for strike price estimation in the hedge optimizer.

For crops with no futures market (specialty crops), returns a structured
alternatives dict that Granite uses to recommend crop insurance or
input cost hedging instead.

Output feeds into:
  - yield_risk.py   (price × yield loss = dollar exposure)
  - pipeline.py     (assembled into Granite prompt)

Dependencies: requests + standard library only.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Optional

import requests

log = logging.getLogger("agrishield.market_data")

_TIMEOUT = 15
_YF_URL  = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

# ---------------------------------------------------------------------------
# Futures contract map
# ---------------------------------------------------------------------------
# Maps crop name → Yahoo Finance ticker and unit metadata.
#
# price_unit     : what Yahoo returns ("USX" = US cents, "USD" = dollars)
# usd_divisor    : divide Yahoo price by this to get USD per unit
# unit           : the physical unit (bushels, pounds, cwt)
# bushels_per_acre_typical: USDA average yield — used when farmer doesn't
#                  provide their own yield figure
# usda_benchmark : fallback price (USD) when Yahoo data is unreliable
#                  Updated from USDA WAOB monthly supply/demand estimates

_FUTURES_MAP = {
    "corn": {
        "ticker":                 "ZC=F",
        "exchange":               "CME/CBOT",
        "price_unit":             "USX",
        "usd_divisor":            100.0,
        "unit":                   "bushel",
        "bushels_per_acre_typical": 181.0,   # USDA 2024 US average
        "usda_benchmark":         4.40,
    },
    "soybeans": {
        "ticker":                 "ZS=F",
        "exchange":               "CME/CBOT",
        "price_unit":             "USX",
        "usd_divisor":            100.0,
        "unit":                   "bushel",
        "bushels_per_acre_typical": 51.7,    # USDA 2024 US average
        "usda_benchmark":         10.80,
    },
    "wheat": {
        "ticker":                 "ZW=F",
        "exchange":               "CME/CBOT",
        "price_unit":             "USX",
        "usd_divisor":            100.0,
        "unit":                   "bushel",
        "bushels_per_acre_typical": 49.7,    # USDA 2024 US average
        "usda_benchmark":         5.70,
    },
    "cotton": {
        "ticker":                 "CT=F",
        "exchange":               "ICE",
        "price_unit":             "USX",
        "usd_divisor":            100.0,
        "unit":                   "pound",
        "bushels_per_acre_typical": 900.0,   # lbs per acre USDA average
        "usda_benchmark":         0.72,
    },
    "oats": {
        "ticker":                 "ZO=F",
        "exchange":               "CME/CBOT",
        "price_unit":             "USX",
        "usd_divisor":            100.0,
        "unit":                   "bushel",
        "bushels_per_acre_typical": 66.0,
        "usda_benchmark":         3.80,
    },
    "rice": {
        # ZR=F returns unreliable currency data from Yahoo Finance
        # Using USDA benchmark directly — rough milled rice, $/cwt
        "ticker":                 "ZR=F",
        "exchange":               "CME/CBOT",
        "price_unit":             "USD",
        "usd_divisor":            1.0,
        "unit":                   "cwt",     # hundredweight (100 lbs)
        "bushels_per_acre_typical": 78.0,    # cwt per acre USDA average
        "usda_benchmark":         16.50,     # $/cwt USDA 2024 estimate
        "use_benchmark":          True,      # skip Yahoo, unreliable for rice
    },
    "canola": {
        "ticker":                 "RS=F",
        "exchange":               "ICE Canada",
        "price_unit":             "CAD",
        "usd_divisor":            1.36,      # approximate CAD/USD — not perfect
        "unit":                   "tonne",
        "bushels_per_acre_typical": 40.0,    # bushels per acre
        "usda_benchmark":         18.50,     # $/bushel USD approx
    },
}

# ---------------------------------------------------------------------------
# Historical volatility proxy
# ---------------------------------------------------------------------------
# True implied volatility requires options chain data (not free).
# We estimate annualised volatility from the 52-week high/low range.
#
# Method: Parkinson's range-based volatility estimator (simplified)
#   vol_proxy = (high - low) / (2 × price × sqrt(252)) × 100
#
# This systematically underestimates true IV (range doesn't capture
# intraday movement) but gives a directionally correct relative measure.
# Corn's typical IV is 20-30%, this method typically returns 15-25%.
# Good enough for strike price guidance in the hedge recommendation.

def _compute_vol_proxy(price: float, high_52w: float, low_52w: float) -> float:
    """
    Estimate annualised price volatility from 52-week range.
    Returns percentage (e.g. 22.5 means 22.5% annualised vol).
    """
    if price <= 0 or high_52w <= low_52w:
        return 25.0   # default fallback for typical ag commodity vol
    try:
        range_pct = (high_52w - low_52w) / price
        # Parkinson range estimator: for a 52-week range, divide by 2.0
        # Agricultural commodities typically show 20-35% annualised vol
        vol = (range_pct / 2.0) * 100.0
        # Clamp to realistic range
        return round(max(12.0, min(60.0, vol)), 1)
    except Exception:
        return 25.0


# ---------------------------------------------------------------------------
# Fetch one futures contract
# ---------------------------------------------------------------------------

def _fetch_yahoo(ticker: str) -> Optional[dict]:
    """
    Fetch a single futures contract from Yahoo Finance.
    Returns the raw meta dict or None on any failure.
    """
    try:
        url  = _YF_URL.format(symbol=ticker)
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data["chart"]["result"][0]["meta"]
    except Exception as exc:
        log.warning("Yahoo Finance fetch failed for %s: %s", ticker, exc)
        return None


def get_futures_price(crop_name: str) -> dict:
    """
    Fetch current futures price and market context for a known crop.

    Returns a dict with:
      price_usd          : current price in USD per unit
      unit               : "bushel", "pound", "cwt", "tonne"
      contract           : contract description e.g. "Corn Futures, May-2026"
      vol_proxy_pct      : estimated annualised volatility (%)
      high_52w_usd       : 52-week high in USD
      low_52w_usd        : 52-week low in USD
      typical_yield      : USDA average yield per acre for this crop
      revenue_per_acre   : price × typical_yield (expected gross revenue)
      data_source        : "yahoo_finance" or "usda_benchmark"
      fetched_utc        : timestamp
    """
    crop_key = crop_name.lower().strip()
    contract = _FUTURES_MAP.get(crop_key)

    if contract is None:
        log.info("Crop '%s' has no futures contract — returning alternatives", crop_name)
        return _no_futures_result(crop_name)

    # Skip Yahoo for unreliable tickers
    use_benchmark = contract.get("use_benchmark", False)
    meta          = None

    if not use_benchmark:
        meta = _fetch_yahoo(contract["ticker"])

    # Build result from Yahoo data or fall back to benchmark
    if meta and not use_benchmark:
        raw_price   = meta.get("regularMarketPrice", 0)
        raw_high    = meta.get("fiftyTwoWeekHigh",   raw_price * 1.1)
        raw_low     = meta.get("fiftyTwoWeekLow",    raw_price * 0.9)
        divisor     = contract["usd_divisor"]
        short_name  = meta.get("shortName", contract["ticker"])
        volume      = meta.get("regularMarketVolume", 0)

        price_usd   = round(raw_price / divisor, 4)
        high_usd    = round(raw_high  / divisor, 4)
        low_usd     = round(raw_low   / divisor, 4)
        vol_proxy   = _compute_vol_proxy(price_usd, high_usd, low_usd)
        data_source = "yahoo_finance"

        log.info("Futures %s (%s): $%.4f/%s  vol_proxy=%.1f%%  contract=%s",
                 crop_name, contract["ticker"], price_usd,
                 contract["unit"], vol_proxy, short_name)
    else:
        # Benchmark fallback
        price_usd   = contract["usda_benchmark"]
        high_usd    = round(price_usd * 1.12, 4)
        low_usd     = round(price_usd * 0.88, 4)
        vol_proxy   = 22.0   # typical ag commodity vol as default
        short_name  = f"{crop_name.title()} Futures (benchmark)"
        volume      = 0
        data_source = "usda_benchmark"

        log.info("Futures %s: using USDA benchmark $%.4f/%s",
                 crop_name, price_usd, contract["unit"])

    typical_yield      = contract["bushels_per_acre_typical"]
    revenue_per_acre   = round(price_usd * typical_yield, 2)

    return {
        "has_futures":         True,
        "crop":                crop_name,
        "ticker":              contract["ticker"],
        "exchange":            contract["exchange"],
        "contract":            short_name,
        "price_usd":           price_usd,
        "unit":                contract["unit"],
        "vol_proxy_pct":       vol_proxy,
        "high_52w_usd":        high_usd,
        "low_52w_usd":         low_usd,
        "typical_yield_per_acre": typical_yield,
        "revenue_per_acre":    revenue_per_acre,
        "volume":              volume,
        "data_source":         data_source,
        "fetched_utc":         datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# No-futures fallback for specialty crops
# ---------------------------------------------------------------------------

# Proxy commodity suggestions for specialty crops
# Based on demand-side correlations — what competing product does a
# buyer substitute when this crop is unavailable or expensive?
_PROXY_SUGGESTIONS = {
    "hemp":        {"proxy": "soybeans",  "reason": "competing oilseed/protein crop"},
    "hemp seed":   {"proxy": "soybeans",  "reason": "competing oilseed"},
    "sunflower":   {"proxy": "soybeans",  "reason": "competing oilseed"},
    "sorghum":     {"proxy": "corn",      "reason": "competing feed grain"},
    "rye":         {"proxy": "wheat",     "reason": "competing small grain"},
    "barley":      {"proxy": "wheat",     "reason": "competing small grain"},
    "millet":      {"proxy": "corn",      "reason": "competing grain"},
    "peanuts":     {"proxy": "soybeans",  "reason": "competing protein/oil crop"},
    "flaxseed":    {"proxy": "soybeans",  "reason": "competing oilseed"},
    "safflower":   {"proxy": "soybeans",  "reason": "competing oilseed"},
    "dry beans":   {"proxy": "soybeans",  "reason": "competing legume"},
    "chickpeas":   {"proxy": "soybeans",  "reason": "competing legume protein"},
}


def _no_futures_result(crop_name: str) -> dict:
    """
    Return a structured alternatives dict for crops with no futures market.
    Granite uses this to recommend crop insurance or proxy hedging.
    """
    crop_lower = crop_name.lower().strip()
    proxy_info = _PROXY_SUGGESTIONS.get(crop_lower)

    alternatives = [
        {
            "type":        "usda_crop_insurance",
            "description": "USDA RMA Actual Production History (APH) policy — "
                           "covers revenue shortfalls for many specialty crops. "
                           "Check eligibility at rma.usda.gov for your county.",
            "recommended": True,
        },
        {
            "type":        "input_cost_hedging",
            "description": "Hedge input costs (diesel, fertilizer) via energy futures "
                           "to protect the cost side of the farm P&L even when "
                           "crop revenue cannot be directly hedged.",
            "recommended": True,
        },
    ]

    if proxy_info:
        alternatives.append({
            "type":        "proxy_futures",
            "description": f"Partial proxy hedge via {proxy_info['proxy'].title()} futures "
                           f"({proxy_info['reason']}). Correlation is real but imperfect — "
                           f"treat as partial protection only, not a full hedge.",
            "proxy_crop":  proxy_info["proxy"],
            "recommended": False,   # directionally useful but unreliable
        })

    return {
        "has_futures":   False,
        "crop":          crop_name,
        "ticker":        None,
        "price_usd":     None,
        "unit":          None,
        "vol_proxy_pct": None,
        "alternatives":  alternatives,
        "note":          f"{crop_name} has no direct exchange-traded futures contract. "
                         f"Granite will recommend alternative risk management strategies.",
        "fetched_utc":   datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Multi-crop entry point
# ---------------------------------------------------------------------------

def get_market_data_all_crops(crops: list) -> dict:
    """
    Fetch market data for all crops in the farm input.

    Parameters
    ----------
    crops : list of crop dicts from frontend JSON
            e.g. [{"name": "Rice", "acres": 50}, ...]

    Returns
    -------
    dict keyed by crop name, each value is the get_futures_price() result.
    Also includes a farm_summary with total expected revenue.
    """
    results      = {}
    total_revenue = 0.0

    for crop_entry in crops:
        crop_name = crop_entry.get("name", "unknown")
        acres     = crop_entry.get("acres", 0)

        market = get_futures_price(crop_name)
        market["acres"] = acres

        # Compute expected gross revenue for this crop at this farm
        if market.get("has_futures") and market.get("revenue_per_acre"):
            crop_revenue = round(market["revenue_per_acre"] * acres, 2)
            market["expected_gross_revenue"] = crop_revenue
            total_revenue += crop_revenue
        else:
            market["expected_gross_revenue"] = None

        results[crop_name] = market

    results["_farm_summary"] = {
        "total_expected_revenue": round(total_revenue, 2),
        "crops_with_futures":     sum(1 for k, v in results.items()
                                      if k != "_farm_summary" and v.get("has_futures")),
        "crops_without_futures":  sum(1 for k, v in results.items()
                                      if k != "_farm_summary" and not v.get("has_futures")),
    }

    return results


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    crops = [
        {"name": "Rice",     "acres": 50.0},
        {"name": "Wheat",    "acres": 20.0},
        {"name": "Soybeans", "acres": 30.0},
    ]

    print("=" * 60)
    print("AgriShield market_data  |  NW Arkansas farm")
    print("=" * 60)

    result = get_market_data_all_crops(crops)

    for crop_name, data in result.items():
        if crop_name == "_farm_summary":
            continue
        print(f"\n--- {crop_name} ({data['acres']} acres) ---")
        if data["has_futures"]:
            print(f"  Price          : ${data['price_usd']:.4f}/{data['unit']}")
            print(f"  Contract       : {data['contract']}")
            print(f"  Vol proxy      : {data['vol_proxy_pct']}%")
            print(f"  52wk range     : ${data['low_52w_usd']:.2f} – ${data['high_52w_usd']:.2f}")
            print(f"  Typical yield  : {data['typical_yield_per_acre']} {data['unit']}s/acre")
            print(f"  Revenue/acre   : ${data['revenue_per_acre']}")
            print(f"  Expected rev   : ${data['expected_gross_revenue']:,.2f}")
            print(f"  Data source    : {data['data_source']}")
        else:
            print(f"  No futures market")
            print(f"  Alternatives   : {len(data['alternatives'])} options")

    summary = result["_farm_summary"]
    print(f"\n--- Farm Summary ---")
    print(f"  Total expected revenue : ${summary['total_expected_revenue']:,.2f}")
    print(f"  Crops with futures     : {summary['crops_with_futures']}")
    print(f"  Crops without futures  : {summary['crops_without_futures']}")
    print("\n[Done]")