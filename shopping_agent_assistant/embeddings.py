from functools import lru_cache

from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embed_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)
