"""
crop_intelligence.py  —  AgriShield
-------------------------------------
Category-based crop parameter lookup. No AI, no hallucinations, instant.

When a farmer enters a crop not in the known profiles (corn/wheat/rice/etc),
this module maps it to a category and returns sensible stress params + revenue.
"""

import logging
log = logging.getLogger("agrishield.crop_intelligence")

# ---------------------------------------------------------------------------
# Category stress profiles — same field format as CROP_STRESS_PARAMS
# ---------------------------------------------------------------------------

_CATEGORY_STRESS = {
    "fruit_tree": {
        "heat_threshold_c": 35.0, "extreme_heat_c": 40.0,
        "dry_day_mm": 3.0, "critical_window_start": 30, "critical_window_end": 90,
        "gdd_base_c": 10.0, "optimal_season_rain_mm": 500.0,
        "planting_doy_offset": 0, "futures_symbol": None,
    },
    "berry": {
        "heat_threshold_c": 30.0, "extreme_heat_c": 35.0,
        "dry_day_mm": 2.0, "critical_window_start": 20, "critical_window_end": 60,
        "gdd_base_c": 7.0, "optimal_season_rain_mm": 450.0,
        "planting_doy_offset": -10, "futures_symbol": None,
    },
    "vegetable": {
        "heat_threshold_c": 32.0, "extreme_heat_c": 36.0,
        "dry_day_mm": 2.0, "critical_window_start": 20, "critical_window_end": 50,
        "gdd_base_c": 10.0, "optimal_season_rain_mm": 400.0,
        "planting_doy_offset": 0, "futures_symbol": None,
    },
    "root_crop": {
        "heat_threshold_c": 30.0, "extreme_heat_c": 35.0,
        "dry_day_mm": 3.0, "critical_window_start": 40, "critical_window_end": 80,
        "gdd_base_c": 7.0, "optimal_season_rain_mm": 500.0,
        "planting_doy_offset": -10, "futures_symbol": None,
    },
    "legume": {
        "heat_threshold_c": 34.0, "extreme_heat_c": 38.0,
        "dry_day_mm": 3.0, "critical_window_start": 45, "critical_window_end": 70,
        "gdd_base_c": 10.0, "optimal_season_rain_mm": 420.0,
        "planting_doy_offset": 0, "futures_symbol": None,
    },
    "oilseed": {
        "heat_threshold_c": 32.0, "extreme_heat_c": 36.0,
        "dry_day_mm": 2.5, "critical_window_start": 40, "critical_window_end": 65,
        "gdd_base_c": 8.0, "optimal_season_rain_mm": 380.0,
        "planting_doy_offset": 0, "futures_symbol": None,
    },
    "fiber": {
        "heat_threshold_c": 33.0, "extreme_heat_c": 37.0,
        "dry_day_mm": 3.0, "critical_window_start": 50, "critical_window_end": 80,
        "gdd_base_c": 13.0, "optimal_season_rain_mm": 550.0,
        "planting_doy_offset": 10, "futures_symbol": None,
    },
    "grain_proxy": {
        "heat_threshold_c": 32.0, "extreme_heat_c": 36.0,
        "dry_day_mm": 3.0, "critical_window_start": 50, "critical_window_end": 70,
        "gdd_base_c": 10.0, "optimal_season_rain_mm": 450.0,
        "planting_doy_offset": 0, "futures_symbol": None,
    },
    "default": {
        "heat_threshold_c": 32.0, "extreme_heat_c": 36.0,
        "dry_day_mm": 3.0, "critical_window_start": 50, "critical_window_end": 75,
        "gdd_base_c": 10.0, "optimal_season_rain_mm": 480.0,
        "planting_doy_offset": 0, "futures_symbol": None,
    },
}

# ---------------------------------------------------------------------------
# Crop → category
# ---------------------------------------------------------------------------

_CROP_CATEGORY = {
    # Fruit trees
    "avocado": "fruit_tree", "avocados": "fruit_tree",
    "apple": "fruit_tree", "apples": "fruit_tree",
    "peach": "fruit_tree", "peaches": "fruit_tree",
    "cherry": "fruit_tree", "cherries": "fruit_tree",
    "mango": "fruit_tree", "mangos": "fruit_tree", "mangoes": "fruit_tree",
    "orange": "fruit_tree", "oranges": "fruit_tree",
    "lemon": "fruit_tree", "lemons": "fruit_tree",
    "lime": "fruit_tree", "limes": "fruit_tree",
    "grapefruit": "fruit_tree", "grape": "fruit_tree", "grapes": "fruit_tree",
    "citrus": "fruit_tree", "olive": "fruit_tree", "olives": "fruit_tree",
    "walnut": "fruit_tree", "walnuts": "fruit_tree",
    "almond": "fruit_tree", "almonds": "fruit_tree",
    "pistachio": "fruit_tree", "pistachios": "fruit_tree",
    "pecan": "fruit_tree", "pecans": "fruit_tree",
    "hops": "fruit_tree",
    # Berries
    "strawberry": "berry", "strawberries": "berry",
    "blueberry": "berry", "blueberries": "berry",
    "raspberry": "berry", "raspberries": "berry",
    "blackberry": "berry", "blackberries": "berry",
    "cranberry": "berry", "cranberries": "berry",
    # Vegetables
    "tomato": "vegetable", "tomatoes": "vegetable",
    "pepper": "vegetable", "peppers": "vegetable",
    "lettuce": "vegetable", "spinach": "vegetable",
    "kale": "vegetable", "broccoli": "vegetable",
    "cauliflower": "vegetable", "cabbage": "vegetable",
    "onion": "vegetable", "onions": "vegetable",
    "garlic": "vegetable", "carrot": "vegetable", "carrots": "vegetable",
    "celery": "vegetable", "cucumber": "vegetable", "cucumbers": "vegetable",
    "squash": "vegetable", "zucchini": "vegetable",
    "pumpkin": "vegetable", "eggplant": "vegetable",
    # Root crops
    "potato": "root_crop", "potatoes": "root_crop",
    "sweet potato": "root_crop", "sweet potatoes": "root_crop",
    "yam": "root_crop", "yams": "root_crop",
    "beet": "root_crop", "beets": "root_crop",
    "turnip": "root_crop", "turnips": "root_crop",
    # Legumes
    "peanut": "legume", "peanuts": "legume",
    "chickpea": "legume", "chickpeas": "legume",
    "lentil": "legume", "lentils": "legume",
    "dry bean": "legume", "dry beans": "legume",
    # Oilseeds
    "sunflower": "oilseed", "sunflowers": "oilseed",
    "flaxseed": "oilseed", "safflower": "oilseed", "hemp seed": "oilseed",
    # Fiber / other
    "tobacco": "fiber", "hemp": "fiber",
    "sugarcane": "default", "lavender": "default", "mint": "default",
    # Grain proxies
    "sorghum": "grain_proxy", "rye": "grain_proxy",
    "barley": "grain_proxy", "millet": "grain_proxy", "triticale": "grain_proxy",
}

# ---------------------------------------------------------------------------
# Revenue per acre (USD, typical US commercial)
# ---------------------------------------------------------------------------

_CROP_REVENUE = {
    "avocado": 5000, "avocados": 5000,
    "apple": 4000, "apples": 4000,
    "peach": 3200, "peaches": 3200,
    "cherry": 6500, "cherries": 6500,
    "mango": 4500, "mangos": 4500, "mangoes": 4500,
    "orange": 3500, "oranges": 3500,
    "lemon": 4000, "lemons": 4000,
    "lime": 3800, "limes": 3800,
    "grapefruit": 3000,
    "grape": 4500, "grapes": 4500,
    "citrus": 3500, "olive": 3000, "olives": 3000,
    "walnut": 5500, "walnuts": 5500,
    "almond": 6000, "almonds": 6000,
    "pistachio": 7000, "pistachios": 7000,
    "pecan": 4000, "pecans": 4000,
    "hops": 7000,
    "strawberry": 5500, "strawberries": 5500,
    "blueberry": 8000, "blueberries": 8000,
    "raspberry": 7000, "raspberries": 7000,
    "blackberry": 5000, "blackberries": 5000,
    "cranberry": 4500, "cranberries": 4500,
    "tomato": 2800, "tomatoes": 2800,
    "pepper": 3500, "peppers": 3500,
    "lettuce": 3000, "spinach": 2500, "kale": 2200,
    "broccoli": 2800, "cauliflower": 2600, "cabbage": 1800,
    "onion": 2000, "onions": 2000,
    "garlic": 5000, "carrot": 2200, "carrots": 2200,
    "celery": 3500, "cucumber": 2500, "cucumbers": 2500,
    "squash": 1800, "zucchini": 2000, "pumpkin": 1500, "eggplant": 2500,
    "potato": 1800, "potatoes": 1800,
    "sweet potato": 2200, "sweet potatoes": 2200,
    "yam": 1600, "yams": 1600,
    "beet": 1500, "beets": 1500, "turnip": 1200, "turnips": 1200,
    "peanut": 800, "peanuts": 800,
    "chickpea": 700, "chickpeas": 700,
    "lentil": 600, "lentils": 600,
    "dry bean": 700, "dry beans": 700,
    "sunflower": 500, "sunflowers": 500,
    "flaxseed": 450, "safflower": 480, "hemp seed": 900,
    "tobacco": 4000, "hemp": 600,
    "sugarcane": 1200, "lavender": 5000, "mint": 4500,
    "sorghum": 550, "rye": 400, "barley": 500, "millet": 380,
}

# ---------------------------------------------------------------------------
# Cache + main entry point
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def get_crop_params(crop_name: str) -> dict:
    """
    Return Monte Carlo stress params for any unknown crop.
    Instant lookup — no API calls, no hallucinations.
    """
    crop_key = crop_name.lower().strip()

    if crop_key in _CACHE:
        return _CACHE[crop_key]

    category = _CROP_CATEGORY.get(crop_key, "default")
    params   = dict(_CATEGORY_STRESS[category])
    revenue  = _CROP_REVENUE.get(crop_key, 800)

    params["revenue_per_acre"]       = float(revenue)
    params["typical_yield_per_acre"] = 80.0
    params["_source"]                = "category_lookup"
    params["_category"]              = category

    log.info(
        "Crop intelligence: %s → category=%s  rev=$%d/acre  [lookup]",
        crop_name, category, revenue,
    )

    _CACHE[crop_key] = params
    return params

