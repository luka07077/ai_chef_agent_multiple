import sqlite3
import os
from datetime import datetime, timedelta

from conf import get_project_root, get_agent_config
from utils.logger_handler import get_logger, log_tool_call

"""
Virtual fridge database module.
Manages ingredient lifecycle and user preferences. Data is stored in data/fridge.db.
"""

logger = get_logger("ai_chef.fridge")

# DB path: stored under data/ in the project root
DB_PATH = os.path.join(get_project_root(), "data", "fridge.db")


def _get_conn():
    """Get a database connection (creates the data/ directory if needed)."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database tables if they don't exist yet."""
    conn = _get_conn()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            item_name TEXT NOT NULL,
            quantity REAL NOT NULL,
            unit TEXT NOT NULL,
            add_date TEXT NOT NULL,
            expiration_date TEXT NOT NULL,
            status INTEGER DEFAULT 0
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS preferences (
            user_id TEXT PRIMARY KEY,
            allergies TEXT,
            dietary_goals TEXT
        )
    ''')

    conn.commit()
    conn.close()
    logger.info(f"Fridge database initialized, path: {DB_PATH}")


@log_tool_call
def add_food_item(user_id: str, item_name: str, quantity: float, unit: str, days_to_expire: int):
    """
    Add an ingredient to the virtual fridge.
    UPSERT logic: if the same item (same name + unit) already exists, quantities are combined
    and the expiry date is extended to the later one.
    """
    conn = _get_conn()
    cursor = conn.cursor()

    add_date = datetime.now().strftime("%Y-%m-%d")
    expiration_date = (datetime.now() + timedelta(days=days_to_expire)).strftime("%Y-%m-%d")

    # Check if there's already an active (not used up) item with the same name and unit
    cursor.execute('''
        SELECT id, quantity FROM inventory
        WHERE user_id = ? AND item_name = ? AND unit = ? AND status = 0
    ''', (user_id, item_name, unit))
    existing = cursor.fetchone()

    if existing:
        # UPSERT: add quantities together, keep the later expiry date
        new_qty = existing["quantity"] + quantity
        cursor.execute('''
            UPDATE inventory
            SET quantity = ?, expiration_date = MAX(expiration_date, ?), add_date = ?
            WHERE id = ?
        ''', (new_qty, expiration_date, add_date, existing["id"]))
        logger.info(f"Item merged: {item_name} -> new quantity {new_qty}{unit}")
    else:
        cursor.execute('''
            INSERT INTO inventory (user_id, item_name, quantity, unit, add_date, expiration_date, status)
            VALUES (?, ?, ?, ?, ?, ?, 0)
        ''', (user_id, item_name, quantity, unit, add_date, expiration_date))
        logger.info(f"New item added: {item_name} {quantity}{unit}")

    conn.commit()
    conn.close()


@log_tool_call
def consume_food_item(user_id: str, item_name: str, quantity: float = None):
    """
    Use up an ingredient: reduce its quantity or mark it as fully consumed.
    If quantity is not specified, the item is marked as fully used up.
    """
    conn = _get_conn()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, quantity FROM inventory
        WHERE user_id = ? AND item_name = ? AND status = 0
        ORDER BY expiration_date ASC LIMIT 1
    ''', (user_id, item_name))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return False

    if quantity is None or quantity >= row["quantity"]:
        cursor.execute('UPDATE inventory SET status = 1 WHERE id = ?', (row["id"],))
    else:
        new_qty = row["quantity"] - quantity
        cursor.execute('UPDATE inventory SET quantity = ? WHERE id = ?', (new_qty, row["id"]))

    conn.commit()
    conn.close()
    return True


def clear_all_items(user_id: str) -> int:
    """Mark all active (not used up) items in the user's fridge as removed. Returns the count."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE inventory SET status = 1 WHERE user_id = ? AND status = 0',
        (user_id,)
    )
    count = cursor.rowcount
    conn.commit()
    conn.close()
    logger.info(f"Fridge cleared: user {user_id}, removed {count} items")
    return count


def get_active_inventory(user_id: str) -> list:
    """Get all active (not used up) items in the user's fridge, including expired ones, for UI and warning system."""
    conn = _get_conn()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT item_name, quantity, unit, expiration_date
        FROM inventory
        WHERE user_id = ? AND status = 0
        ORDER BY expiration_date ASC
    ''', (user_id,))

    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def update_user_preferences(user_id: str, allergies: str, dietary_goals: str):
    """Update a user's preferences (allergies and dietary goals)."""
    conn = _get_conn()
    cursor = conn.cursor()

    cursor.execute('''
        REPLACE INTO preferences (user_id, allergies, dietary_goals)
        VALUES (?, ?, ?)
    ''', (user_id, allergies, dietary_goals))

    conn.commit()
    conn.close()


def get_user_preferences(user_id: str) -> dict:
    """Get a user's stored preferences."""
    conn = _get_conn()
    cursor = conn.cursor()

    cursor.execute('SELECT allergies, dietary_goals FROM preferences WHERE user_id = ?', (user_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return {"allergies": "none", "dietary_goals": "none"}


# Init: make sure tables exist when this module is imported
init_db()


if __name__ == "__main__":
    test_user = "user_001"

    # Test UPSERT: add the same item twice, should combine instead of duplicate
    add_food_item(test_user, "egg", 10, "pcs", 14)
    add_food_item(test_user, "egg", 5, "pcs", 14)  # should become 15
    add_food_item(test_user, "beef", 0.5, "kg", 3)
    add_food_item(test_user, "tomato", 3, "pcs", 5)

    update_user_preferences(test_user, allergies="peanut,seafood", dietary_goals="high-protein,low-fat")

    print(f"\n--- {test_user}'s fridge inventory ---")
    for item in get_active_inventory(test_user):
        print(f"  {item['item_name']}: {item['quantity']}{item['unit']}, expires: {item['expiration_date']}")

    print(f"\n--- {test_user}'s preferences ---")
    print(get_user_preferences(test_user))
