import sqlite3
from collections import Counter

from langchain.tools import tool

from shopping_agent_assistant.db import DB_PATH
from shopping_agent_assistant.runtime import debug_log

@tool
def checkout(product_ids: list[int]) -> str:
    """
    Place an order for the given product IDs. Saves one order with multiple order
    items and returns a confirmation message with the order ID and total price.
    """
    if isinstance(product_ids, int):
        product_ids = [product_ids]

    if not product_ids:
        return "Error: checkout requires at least one product ID."

    product_counts = Counter(product_ids)
    debug_log(tool_name="checkout", product_ids=product_ids)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    placeholders = ", ".join("?" for _ in product_counts)
    cursor.execute(
        f"SELECT id, name, price FROM products WHERE id IN ({placeholders})",
        tuple(product_counts),
    )
    product_rows = cursor.fetchall()
    products_by_id = {product_id: (name, price) for product_id, name, price in product_rows}

    missing_product_ids = [
        product_id for product_id in product_counts if product_id not in products_by_id
    ]
    if missing_product_ids:
        conn.close()
        missing = ", ".join(str(product_id) for product_id in missing_product_ids)
        return f"Error: product ID(s) not found: {missing}."

    cursor.execute("INSERT INTO orders DEFAULT VALUES")
    order_id = cursor.lastrowid

    order_items = []
    total_price = 0.0
    for product_id, quantity in product_counts.items():
        name, price = products_by_id[product_id]
        order_items.append((order_id, product_id, name, price, quantity))
        total_price += price * quantity

    cursor.executemany(
        """
        INSERT INTO order_items (order_id, product_id, product_name, price, quantity)
        VALUES (?, ?, ?, ?, ?)
        """,
        order_items,
    )

    conn.commit()
    conn.close()

    item_summary = ", ".join(
        f"{quantity}x {products_by_id[product_id][0]}"
        for product_id, quantity in product_counts.items()
    )
    debug_log(order_id=order_id, items=item_summary, total_price=f"${total_price:.2f}")

    return (
        f"Order #{order_id} confirmed! Items: {item_summary}. Total: ${total_price:.2f}. "
        f"Your order will arrive in 1-3 business days. Thank you for shopping with us!"
    )
