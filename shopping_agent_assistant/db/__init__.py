from pathlib import Path

from .setup_db import create_database, DB_PATH

def ensure_database():
    if not DB_PATH.exists():
        create_database()
