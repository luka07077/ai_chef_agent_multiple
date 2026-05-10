import datetime

from fridge_manager import fridge_db
from conf import get_agent_config
from utils.logger_handler import get_logger

"""
Warning system module.
Handles the "alerting" part of the business loop: scanning for soon-to-expire items
and checking for allergen conflicts.
"""

logger = get_logger("ai_chef.warning")


def check_expiring_items(user_id: str, warning_days: int = None) -> dict:
    """
    Check the user's fridge for items that are expired or about to expire.

    Returns:
        dict with 'expired' and 'expiring_soon' lists
    """
    if warning_days is None:
        config = get_agent_config()
        warning_days = config["fridge"]["warning_days"]

    inventory = fridge_db.get_active_inventory(user_id)

    result = {"expired": [], "expiring_soon": []}
    today = datetime.datetime.now()

    for item in inventory:
        exp_date = datetime.datetime.strptime(item['expiration_date'], "%Y-%m-%d")
        delta_days = (exp_date - today).days

        if delta_days < 0:
            result["expired"].append(item)
        elif 0 <= delta_days <= warning_days:
            item_copy = dict(item)
            item_copy['days_left'] = delta_days
            result["expiring_soon"].append(item_copy)

    return result


def check_allergen_conflict(user_id: str, proposed_ingredients: list) -> dict:
    """
    Check if any of the proposed ingredients match the user's known allergens.

    Args:
        user_id: user identifier
        proposed_ingredients: list of ingredient names to check

    Returns:
        dict with 'is_safe' and 'conflicting_items'
    """
    prefs = fridge_db.get_user_preferences(user_id)
    allergies_str = prefs.get("allergies", "")

    if not allergies_str or allergies_str == "none":
        return {"is_safe": True, "conflicting_items": []}

    user_allergies = [a.strip().lower() for a in allergies_str.split(",")]
    conflicts = []

    for ingredient in proposed_ingredients:
        ing_words = ingredient.lower().split()   # e.g. "peanut oil" → ["peanut", "oil"]
        matched = False
        for allergy in user_allergies:
            allergy_words = allergy.split()      # e.g. "peanuts" → ["peanuts"]
            for ing_word in ing_words:
                for allergy_word in allergy_words:
                    # Word-level bidirectional substring match (case-insensitive):
                    # "peanut" in "peanuts" ✓  "peanuts" in "peanut oil" ✗ but "peanut" in "peanuts" ✓
                    if allergy_word in ing_word or ing_word in allergy_word:
                        matched = True
                        break
                if matched:
                    break
            if matched:
                conflicts.append(ingredient)
                break

    if conflicts:
        logger.warning(f"Allergen conflict: user {user_id}, conflicting items: {conflicts}")

    return {
        "is_safe": len(conflicts) == 0,
        "conflicting_items": conflicts
    }


if __name__ == "__main__":
    test_user = "user_001"

    print("--- 1. Expiry warning check ---")
    warnings = check_expiring_items(test_user)
    print(f"  Expiring soon: {[i['item_name'] for i in warnings['expiring_soon']]}")
    print(f"  Expired: {[i['item_name'] for i in warnings['expired']]}")

    print("\n--- 2. Allergen conflict check ---")
    check = check_allergen_conflict(test_user, ["peanut oil", "pork"])
    if not check["is_safe"]:
        print(f"  Blocked! Allergen items: {check['conflicting_items']}")
    else:
        print("  Safe — no allergens found.")
