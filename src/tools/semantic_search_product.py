import json
import sqlite3
from functools import lru_cache
from collections import defaultdict
from typing import Optional

import numpy as np
from langchain.tools import tool
from sentence_transformers import SentenceTransformer

from db import DB_PATH

@lru_cache(maxsize=1)
def get_embed_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def to_float32_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def rrf_fuse(result_sets, k: int = 60):
    """
    Reciprocal Rank Fusion.
    Higher score = better rank across multiple retrieval methods.
    """
    fused = defaultdict(lambda: {"score": 0.0, "product": None})

    for results in result_sets:
        for rank, item in enumerate(results, start=1):
            pid = item["id"]
            fused[pid]["product"] = item
            fused[pid]["score"] += 1.0 / (k + rank)

    return sorted(fused.values(), key=lambda x: x["score"], reverse=True)


@tool
def semantic_search_product(
    query: str,
    max_price: Optional[float] = None,
    is_organic: Optional[bool] = None,
    limit: int = 5,
) -> str:
    """
    Hybrid semantic search over the catalog.
    Uses FTS5 for keyword recall and Python cosine similarity over stored embeddings.
    Returns top matching products as JSON.
    """
    try:
        conn = get_conn()
        cursor = conn.cursor()
        embed_model = get_embed_model()

        # Apply hard filters in SQL
        base_where = "WHERE 1=1"
        params = []

        if max_price is not None:
            base_where += " AND p.price <= ?"
            params.append(max_price)

        if is_organic is not None:
            base_where += " AND p.is_organic = ?"
            params.append(1 if is_organic else 0)

        # 1) FTS search
        fts_results = []
        if query.strip():
            fts_sql = f"""
                SELECT
                    p.id,
                    p.name,
                    p.category,
                    p.price,
                    p.description,
                    p.is_organic,
                    bm25(products_fts) AS fts_score
                FROM products_fts
                JOIN products p ON p.id = products_fts.rowid
                {base_where}
                AND products_fts MATCH ?
                ORDER BY fts_score
                LIMIT ?
            """
            cursor.execute(fts_sql, params + [query, max(limit * 3, 10)])
            fts_results = [dict(row) for row in cursor.fetchall()]

        # 2) Vector search in Python to avoid sqlite-vec binding issues
        query_vec = embed_model.encode([query], normalize_embeddings=True)[0].astype(np.float32)

        vec_sql = f"""
            SELECT
                p.id,
                p.name,
                p.category,
                p.price,
                p.description,
                p.is_organic,
                pe.embedding
            FROM product_embeddings pe
            JOIN products p ON p.id = pe.product_id
            {base_where}
        """
        cursor.execute(vec_sql, params)
        vec_rows = cursor.fetchall()

        vec_results = []
        for row in vec_rows:
            item_vec = to_float32_vector(row["embedding"])
            vec_results.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "category": row["category"],
                    "price": row["price"],
                    "description": row["description"],
                    "is_organic": row["is_organic"],
                    "vec_score": cosine_similarity(query_vec, item_vec),
                }
            )

        vec_results.sort(key=lambda item: item["vec_score"], reverse=True)
        vec_results = vec_results[: max(limit * 3, 10)]

        conn.close()

        # Normalize for fusion
        for row in fts_results:
            row["source"] = "fts"
        for row in vec_results:
            row["source"] = "vector"

        fused = rrf_fuse([fts_results, vec_results])

        final = []
        for item in fused[:limit]:
            product = item["product"]
            final.append(
                {
                    "id": product["id"],
                    "name": product["name"],
                    "category": product["category"],
                    "price": product["price"],
                    "description": product["description"],
                    "is_organic": bool(product["is_organic"]),
                    "score": round(item["score"], 6),
                }
            )

        return json.dumps(final)
    except Exception as exc:
        return json.dumps(
            {
                "error": "semantic_search_product_failed",
                "message": str(exc),
            }
        )
