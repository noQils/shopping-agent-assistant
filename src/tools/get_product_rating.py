import sqlite3

from src.db import DB_PATH

def get_product_rating(product_id: int) -> dict:
    """Return average rating and review count for a single product."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT AVG(rating), COUNT(*) FROM reviews WHERE product_id = ?",
        (product_id,),
    )
    row = cursor.fetchone()
    conn.close()

    avg = round(row[0], 2) if row and row[0] is not None else 0.0
    count = row[1] if row else 0
    return {"product_id": product_id, "average_rating": avg, "review_count": count}
