import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional
from src.config.settings import settings


def get_db_path() -> Path:
    settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return settings.DATABASE_PATH


@contextmanager
def get_db_connection(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for SQLite connection with WAL mode and row factory."""
    target_path = db_path or get_db_path()
    conn = sqlite3.connect(str(target_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
