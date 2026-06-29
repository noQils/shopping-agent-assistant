import sqlite3
import json

from langchain.tools import tool

from shopping_agent_assistant.db import DB_PATH
from shopping_agent_assistant.runtime import debug_log

@tool
def get_product_rating(product_id: int) -> str:
    """
    Get the average customer rating and total review count for a product by its ID.
    Returns a JSON object with: product_id, average_rating, review_count.
    """
    debug_log(tool_name="get_product_rating", product_id=product_id)

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
    result = {"product_id": product_id, "average_rating": avg, "review_count": count}
    debug_log(result=json.dumps(result))

    return json.dumps(result)
