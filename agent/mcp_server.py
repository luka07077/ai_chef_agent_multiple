import sys
import os
import json
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from fridge_manager import fridge_db
from fridge_manager.warning_system import check_expiring_items, check_allergen_conflict
from utils.logger_handler import log_tool_call
from conf import get_agent_config

"""
Custom MCP server.
Handles core business logic: fridge management, alerts, grocery ordering, nutrition lookup, weather.
All config values come from agent_config.yaml — no hardcoded values.
"""

config = get_agent_config()
mcp_config = config["mcp"]

mcp = FastMCP("AIChef_Core_Service")


@mcp.tool()
@log_tool_call
def get_fridge_inventory(user_id: str) -> str:
    """
    Call this when you need to know what ingredients are in the user's fridge,
    how much of each there is, and whether anything is about to expire.

    Args:
        user_id: the user's unique identifier, e.g. "user_001"
    """
    inventory = fridge_db.get_active_inventory(user_id)
    if not inventory:
        return "The fridge is empty — no ingredients available."
    return json.dumps(inventory, ensure_ascii=False, indent=2)


@mcp.tool()
@log_tool_call
def add_food_to_fridge(user_id: str, item_name: str, quantity: float, unit: str, days_to_expire: int = 7) -> str:
    """
    Call this when you need to add a new ingredient to the user's virtual fridge.
    Supports deduplication: same-name ingredients will have their quantity combined instead of duplicated.

    Args:
        user_id: the user's unique identifier
        item_name: ingredient name, e.g. "tomato"
        quantity: amount
        unit: unit of measurement, e.g. "pcs", "kg"
        days_to_expire: estimated shelf life in days, default 7
    """
    fridge_db.add_food_item(user_id, item_name, quantity, unit, days_to_expire)
    return f"Added {item_name} {quantity}{unit} to fridge. Expires in {days_to_expire} days."


@mcp.tool()
@log_tool_call
def check_fridge_warnings(user_id: str) -> str:
    """
    Scan the fridge for items that are about to expire or already expired, and generate a warning report.
    Call this when the user asks "what's about to expire?", "is my fridge safe?",
    or proactively before suggesting recipes.

    Args:
        user_id: the user's unique identifier
    """
    warnings = check_expiring_items(user_id)

    parts = []
    if warnings["expired"]:
        expired_names = [f"{i['item_name']}({i['quantity']}{i['unit']})" for i in warnings["expired"]]
        parts.append(f"[EXPIRED] {', '.join(expired_names)} — please throw these away immediately!")

    if warnings["expiring_soon"]:
        soon_names = [f"{i['item_name']}({i['days_left']} days left)" for i in warnings["expiring_soon"]]
        parts.append(f"[EXPIRING SOON] {', '.join(soon_names)} — use these first!")

    if not parts:
        return "All fridge items are in good shape, no warnings."

    return "\n".join(parts)


@mcp.tool()
@log_tool_call
def check_allergen_safety(user_id: str, ingredients: str) -> str:
    """
    Allergen safety check: see if any of the given ingredients match the user's known allergens.
    Must be called when recommending recipes or when the user mentions specific ingredients.

    Args:
        user_id: the user's unique identifier
        ingredients: comma-separated ingredient list, e.g. "peanut oil,garlic,shrimp,cabbage"
    """
    ingredient_list = [i.strip() for i in ingredients.split(",") if i.strip()]
    result = check_allergen_conflict(user_id, ingredient_list)

    if not result["is_safe"]:
        conflicts = ", ".join(result["conflicting_items"])
        return f"[SAFETY ALERT] Allergen detected: {conflicts}! Please replace these ingredients immediately — do not use them!"

    return "Safety check passed — none of the ingredients match the user's known allergens."


@mcp.tool()
@log_tool_call
def remove_food_from_fridge(item_name: str, quantity: float = None) -> str:
    """
    Remove a single type of ingredient from the fridge (e.g. when it's used up or expired).
    Note: this tool only handles one ingredient at a time. item_name must be a specific ingredient
    (e.g. "tomato"), not a bulk description like "all ingredients".
    To clear the whole fridge, use the clear_fridge_inventory tool instead.

    Args:
        item_name: specific ingredient name, e.g. "tomato", "beef"
        quantity: how much to remove; if not specified, removes all of that ingredient
    """
    user_id = config["user"]["default_user_id"]
    success = fridge_db.consume_food_item(user_id, item_name, quantity)
    if success:
        qty_str = f"{quantity}" if quantity is not None else "all"
        return f"Removed {item_name} ({qty_str}) from the fridge."
    return f"{item_name} not found in the fridge, nothing to do."


@mcp.tool()
@log_tool_call
def clear_fridge_inventory() -> str:
    """
    Clear the fridge: remove all ingredients at once.
    Call this when the user says "clear the fridge", "throw everything away", etc.
    Don't use remove_food_from_fridge one by one for this.
    """
    user_id = config["user"]["default_user_id"]
    count = fridge_db.clear_all_items(user_id)
    if count == 0:
        return "The fridge was already empty, nothing to do."
    return f"Fridge cleared — removed {count} items."


@mcp.tool()
@log_tool_call
def order_fresh_groceries(item_name: str, quantity: float, unit: str) -> str:
    """
    Call this when an ingredient is missing and needs to be ordered from an external supplier.

    Args:
        item_name: the ingredient to order
        quantity: how much to order
        unit: unit of measurement
    """
    test_user = config["user"]["default_user_id"]
    api_url = mcp_config["order_api_url"]
    timeout = mcp_config["request_timeout"]

    payload = {
        "action": "place_order",
        "user_id": test_user,
        "items": [{"name": item_name, "qty": quantity, "unit": unit}]
    }

    # Always add to the local fridge first so the demo works even if the external API is down
    default_days = config["fridge"]["default_expire_days"]
    fridge_db.add_food_item(test_user, item_name, quantity, unit, days_to_expire=default_days)

    # Try to notify the external supplier (demo only — failure doesn't affect local inventory)
    try:
        response = requests.post(api_url, json=payload, timeout=timeout)
        api_note = "and notified the supplier system." if response.status_code == 200 else "(supplier API unreachable, saved locally)"
    except requests.exceptions.RequestException:
        api_note = "(supplier API unreachable, saved locally)"

    return f"Order placed: {item_name} {quantity}{unit} added to fridge {api_note}"


@mcp.tool()
@log_tool_call
def get_nutrition_info(food_name: str) -> str:
    """
    Look up nutritional info for an ingredient (calories, protein, fat, carbs).

    Args:
        food_name: ingredient name, supports Chinese/English (e.g. "tomato", "chicken")
    """
    # api_key = os.environ.get("SPOONACULAR_API_KEY")
    api_key = mcp_config["nutrition_api_key"]
    api_base = mcp_config["nutrition_api_base"]
    timeout = mcp_config["request_timeout"]

    fallback_db = {
        "tomato": "18 kcal/100g | protein 0.9g, fat 0.2g, carbs 3.9g",
        "egg": "143 kcal/100g | protein 12.6g, fat 9.9g, carbs 1.1g",
        "beef": "250 kcal/100g | protein 26g, fat 17g, carbs 0g",
        "chicken breast": "165 kcal/100g | protein 31g, fat 3.6g, carbs 0g",
        "rice": "130 kcal/100g | protein 2.7g, fat 0.3g, carbs 28g",
        "milk": "42 kcal/100ml | protein 3.4g, fat 1.5g, carbs 5g",
        "potato": "77 kcal/100g | protein 2g, fat 0.1g, carbs 17g",
    }

    def _fallback_lookup(name):
        for k, v in fallback_db.items():
            if k in name.lower() or name.lower() in k:
                return f"[Local data] {k} nutrition: {v}"
        return None

    if not api_key:
        result = _fallback_lookup(food_name)
        return result or f"SPOONACULAR_API_KEY not set and no local cache — can't look up {food_name}."

    headers = {"User-Agent": "AIChef_Agent/1.0"}

    try:
        # Step 1: search endpoint to get the ingredient's unique ID in Spoonacular
        search_resp = requests.get(
            f"{api_base}/food/ingredients/search",
            params={"apiKey": api_key, "query": food_name, "number": 1},
            headers=headers,
            timeout=timeout
        )
        if search_resp.status_code != 200:
            raise Exception(f"Search endpoint error, HTTP {search_resp.status_code}")

        results = search_resp.json().get("results", [])
        if not results:
            return _fallback_lookup(food_name) or f"No nutrition data found for '{food_name}', try an English name."

        ingredient_id   = results[0]["id"]
        ingredient_name = results[0]["name"]

        # Step 2: detail endpoint — get nutrition data per 100g using the ID
        # Must pass amount + unit + nutrition=true, otherwise nutrition field is missing
        detail_resp = requests.get(
            f"{api_base}/food/ingredients/{ingredient_id}/information",
            params={"apiKey": api_key, "amount": 100, "unit": "g", "nutrition": "true"},
            headers=headers,
            timeout=timeout
        )
        if detail_resp.status_code != 200:
            raise Exception(f"Detail endpoint error, HTTP {detail_resp.status_code}")

        # The detail response has lots of fields — we only grab the 4 core ones
        nutrients = detail_resp.json().get("nutrition", {}).get("nutrients", [])
        nd = {n["name"].lower(): round(n["amount"], 1) for n in nutrients}

        calories = nd.get("calories", "unknown")
        protein  = nd.get("protein", "unknown")
        fat      = nd.get("fat", "unknown")
        carbs    = nd.get("carbohydrates", nd.get("net carbohydrates", "unknown"))

        return (f"【Spoonacular】{food_name}（{ingredient_name}）per 100g: "
                f"calories {calories} kcal | protein {protein}g | fat {fat}g | carbs {carbs}g")

    except Exception as e:
        return _fallback_lookup(food_name) or f"Nutrition API error: {str(e)}, and no local cache for {food_name}."


@mcp.tool()
@log_tool_call
def get_local_weather(city: str) -> str:
    """
    Call this when you need real-time weather data to recommend food, recipes, or soups.

    Args:
        city: city name, e.g. "Beijing", "Shanghai"
    """
    api_base = mcp_config["weather_api_base"]
    timeout = mcp_config["request_timeout"]

    try:
        response = requests.get(f"{api_base}/{city}?format=j1", timeout=timeout)

        if response.status_code == 200:
            data = response.json()
            cc = data['current_condition'][0]
            temp_c = cc['temp_C']
            desc = cc['lang_zh'][0]['value'] if 'lang_zh' in cc else cc['weatherDesc'][0]['value']
            feels_like = cc['FeelsLikeC']
            return f"【Real-time weather】{city}: {desc}, temp: {temp_c}°C (feels like {feels_like}°C)."
        else:
            return f"Weather service rate-limited. Assume {city} is currently rainy and cool (~15°C) and recommend accordingly."

    except requests.exceptions.Timeout:
        return f"Weather request timed out. Assume {city} is currently rainy and cool, and recommend accordingly."
    except Exception as e:
        return f"Weather lookup error: {str(e)}"


if __name__ == "__main__":
    print("AIChef_Core_Service MCP server starting (stdio transport)...", file=sys.stderr)
    mcp.run(transport='stdio')
