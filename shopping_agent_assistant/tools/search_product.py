import json
import sqlite3
from typing import Optional

from langchain.tools import tool

from shopping_agent_assistant.db import DB_PATH

@tool
def search_product(
    query: str,
    max_price: Optional[float] = None,
    is_organic: Optional[bool] = None,
    min_rating: Optional[float] = None,
) -> str:
    """
    Search the product database by keyword (matched against name, description, and category).
    Optionally filter by maximum price, organic status, and minimum average rating.
    Returns a JSON array of matching products, each with: id, name, category, price,
    description, is_organic, average_rating, review_count.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    sql = """
        SELECT
            p.id,
            p.name,
            p.category,
            p.price,
            p.description,
            p.is_organic,
            COALESCE(r.average_rating, 0) AS average_rating,
            COALESCE(r.review_count, 0) AS review_count
        FROM products p
        LEFT JOIN (
            SELECT
                product_id,
                ROUND(AVG(rating), 2) AS average_rating,
                COUNT(*) AS review_count
            FROM reviews
            GROUP BY product_id
        ) r ON r.product_id = p.id
        WHERE 1=1
    """
    params: list = []

    if query:
        sql += " AND (p.name LIKE ? OR p.description LIKE ? OR p.category LIKE ?)"
        like = f"%{query}%"
        params.extend([like, like, like])

    if max_price is not None:
        sql += " AND p.price <= ?"
        params.append(max_price)

    if is_organic is not None:
        sql += " AND p.is_organic = ?"
        params.append(1 if is_organic else 0)

    if min_rating is not None:
        sql += " AND COALESCE(r.average_rating, 0) >= ?"
        params.append(min_rating)

    sql += " ORDER BY average_rating DESC, review_count DESC, p.price ASC"

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    products = [
        {
            "id":          row[0],
            "name":        row[1],
            "category":    row[2],
            "price":       row[3],
            "description": row[4],
            "is_organic":  bool(row[5]),
            "average_rating": row[6],
            "review_count": row[7],
        }
        for row in rows
    ]
    return json.dumps(products)
