"""
db_connection.py
================
Connection factory for the hub's SQLite database.

Provides a single context manager — get_connection() — that returns a
properly-configured sqlite3.Connection with:
  • PRAGMA foreign_keys = ON       (FK constraints enforced)
  • PRAGMA journal_mode = WAL      (concurrent reads alongside a writer)
  • PRAGMA synchronous = NORMAL    (good safety/speed balance under WAL)
  • row_factory = sqlite3.Row      (dict-like row access in repositories)

Transaction Semantics
---------------------
The context manager commits on successful exit and rolls back on any
exception. Repository code should therefore raise on failure rather than
silently catching errors, to ensure rollback occurs.

Threading Model
---------------
SQLite connections are NOT shared across threads. Each `with get_connection()`
block creates and tears down a fresh connection, which is the safest pattern
for the hub's multi-threaded runtime (reminder scheduler + Flask REST server
+ SMS handler all running concurrently — Section 3.5 of the project report).

Configuration
-------------
The database path is resolved in this order:
  1. Explicit `db_path` argument to get_connection() (used for tests).
  2. HUB_DB_PATH environment variable (set in .env or the shell).
  3. Default: "data/geriatric_hub.db"

If the optional python-dotenv package is installed, a .env file at the
project root is loaded automatically.

Author: Wise (Asumang Pobi Godwin) — KNUST COE 497
"""

import os
import sqlite3
import logging
from contextlib import contextmanager
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional .env loading
# ---------------------------------------------------------------------------
# python-dotenv is optional; if it's installed we use it, otherwise we rely
# on the regular shell environment. Add `python-dotenv` to requirements.txt
# to enable .env file loading.
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.debug("Loaded .env via python-dotenv.")
except ImportError:
    logger.debug("python-dotenv not installed — using os.environ directly.")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = "data/geriatric_hub.db"


def resolve_db_path() -> str:
    """
    Determine which SQLite file the hub should connect to.
    Exposed so tests and admin scripts can introspect the active path.
    """
    return os.environ.get("HUB_DB_PATH", DEFAULT_DB_PATH)


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory of `path` if it doesn't already exist."""
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
        logger.info("Created database parent directory: %s", parent)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@contextmanager
def get_connection(
    db_path: Optional[str] = None,
    timeout_seconds: float = 5.0,
) -> Iterator[sqlite3.Connection]:
    """
    Yield a configured sqlite3.Connection. Auto-commits on success,
    rolls back on exception, always closes.

    Parameters
    ----------
    db_path : str, optional
        Override the resolved DB path — primarily for tests pointing at
        temporary files.
    timeout_seconds : float
        How long to wait for a database lock before raising
        sqlite3.OperationalError. Important under WAL with multiple writers.

    Usage
    -----
        from src.data_management.db_connection import get_connection

        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM ElderProfile")
            rows = cur.fetchall()
    """
    path = db_path or resolve_db_path()
    _ensure_parent_dir(path)

    conn = sqlite3.connect(path, timeout=timeout_seconds)
    try:
        # Dict-like row access — repositories can do row["drug_name"]
        # rather than row[2], which is far less error-prone.
        conn.row_factory = sqlite3.Row

        # Apply pragmas on every connection. PRAGMA foreign_keys is a
        # per-connection setting; PRAGMA journal_mode is persistent on
        # the database file but cheap to re-apply.
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")

        yield conn
        conn.commit()

    except Exception:
        # Any exception inside the `with` block triggers a rollback,
        # ensuring the database is left in a consistent state.
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Health check helper — useful for the systemd watchdog
# ---------------------------------------------------------------------------

def healthcheck(db_path: Optional[str] = None) -> bool:
    """
    Verify that the database is reachable and responsive.
    Returns True on success, False on any failure.
    """
    try:
        with get_connection(db_path) as conn:
            conn.execute("SELECT 1;").fetchone()
        return True
    except Exception as exc:
        logger.error("Database healthcheck failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Resolved DB path: {resolve_db_path()}")
    if healthcheck():
        print("✓ Database is reachable and responding.")
    else:
        print("✗ Database healthcheck failed — has db_init.py been run?")
        