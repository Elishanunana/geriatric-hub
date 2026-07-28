"""
db_init.py
==========
Database initialization script for the Resilient, Offline-First Assistive
Ecosystem for Geriatric Care.

Implements the SQLite schema defined in Table 3.3 of the project report:
    1. ElderProfile         - Elder identity and emergency contacts
    2. MedicationSchedule   - Active and historical medication schedules
    3. EventLog             - Append-only audit log of all system events
    4. SyncQueue            - Pending bidirectional sync changes (hub <-> app)
    5. SystemConfig         - Configurable runtime parameters

Design Notes
------------
- The EventLog table is enforced as append-only at the database level using
  SQLite triggers that prevent UPDATE and DELETE operations on its rows.
  This guarantees that the system's audit trail cannot be tampered with by
  application logic, even in the event of a programming error elsewhere in
  the stack (Ref: Section 3.5.3 of the project report).

- Foreign keys are enabled per-connection (SQLite default is OFF) to enforce
  referential integrity between MedicationSchedule and ElderProfile.

- WAL (Write-Ahead Logging) mode is enabled to improve concurrency between
  the reminder scheduler thread, the SMS handler thread, and the Flask REST
  server thread, all of which may read/write the database simultaneously.

- The script is idempotent: running it repeatedly on an existing database
  will not destroy existing data, thanks to the use of CREATE ... IF NOT
  EXISTS clauses.

Author: Wise (Asumang Pobi Godwin) - KNUST COE 497
Supervisor: Dr. Theresa S. A. Adjaidoo
"""

import sqlite3
import os
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Database file location. On the Raspberry Pi deployment target, this will
# resolve to a path under the hub service's working directory. For local
# development, it sits alongside this script.
DB_PATH = os.environ.get("HUB_DB_PATH", "data/geriatric_hub.db")

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

SCHEMA_ELDER_PROFILE = """
CREATE TABLE IF NOT EXISTS ElderProfile (
    elder_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name              TEXT    NOT NULL,
    language          TEXT    NOT NULL DEFAULT 'twi',
    -- Comma-separated list of registered caregiver phone numbers.
    -- Used both for outbound SOS dispatch and for inbound SMS origin
    -- verification in the SMS payload handler (Section 3.5.2).
    caregiver_phones  TEXT    NOT NULL,
    created_at        TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    last_modified     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
"""

SCHEMA_MEDICATION_SCHEDULE = """
CREATE TABLE IF NOT EXISTS MedicationSchedule (
    schedule_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    elder_id         INTEGER NOT NULL,
    drug_name        TEXT    NOT NULL,
    dosage           TEXT    NOT NULL,
    -- ISO-8601 time-of-day for the dose (e.g. '08:00').
    time_due         TEXT    NOT NULL,
    -- Comma-separated day codes: 'MON,TUE,WED,...' or 'DAILY'.
    days_of_week     TEXT    NOT NULL DEFAULT 'DAILY',
    active           INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0,1)),
    -- Originator of the entry: 'caregiver', 'pharmacist', or 'hub_local'.
    -- Supports clinical traceability (Section 3.5.3).
    prescribed_by    TEXT    NOT NULL DEFAULT 'caregiver',
    -- Pathway through which the record was delivered to the hub:
    -- 'app_wifi', 'app_sms', or 'hub_local'. Used for audit/diagnostic.
    sync_method      TEXT    NOT NULL DEFAULT 'hub_local',
    last_modified    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    FOREIGN KEY (elder_id) REFERENCES ElderProfile(elder_id) ON DELETE CASCADE
);
"""

# EventLog is the system's append-only audit trail. Records are inserted
# but NEVER updated or deleted. This invariant is enforced both at the
# application layer and via SQLite triggers (defined below).
SCHEMA_EVENT_LOG = """
CREATE TABLE IF NOT EXISTS EventLog (
    event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Event categories include: 'reminder_issued', 'dose_confirmed',
    -- 'dose_missed', 'sos_triggered', 'appliance_on', 'appliance_off',
    -- 'power_on_battery', 'power_on_mains', 'sms_payload_accepted',
    -- 'sms_payload_rejected', 'system_boot', 'system_fault'.
    event_type   TEXT    NOT NULL,
    -- ISO-8601 UTC timestamp with millisecond precision.
    timestamp    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    -- Free-form JSON or text payload describing the event details.
    details      TEXT,
    -- 0 = pending sync to caregiver app; 1 = synced.
    synced_flag  INTEGER NOT NULL DEFAULT 0 CHECK (synced_flag IN (0,1))
);
"""

# Append-only enforcement triggers.
# Note: synced_flag updates are necessary for the sync subsystem, so we
# permit a narrow exception: only the synced_flag column may be updated,
# and only from 0 -> 1. All other UPDATEs and all DELETEs are blocked.
TRIGGER_EVENT_LOG_NO_DELETE = """
CREATE TRIGGER IF NOT EXISTS trg_eventlog_no_delete
BEFORE DELETE ON EventLog
BEGIN
    SELECT RAISE(ABORT, 'EventLog is append-only: DELETE is not permitted');
END;
"""

TRIGGER_EVENT_LOG_NO_UPDATE = """
CREATE TRIGGER IF NOT EXISTS trg_eventlog_no_update
BEFORE UPDATE ON EventLog
BEGIN
    SELECT CASE
        -- Permit only the synced_flag transition 0 -> 1; block everything else.
        WHEN OLD.event_id    IS NOT NEW.event_id
          OR OLD.event_type  IS NOT NEW.event_type
          OR OLD.timestamp   IS NOT NEW.timestamp
          OR OLD.details     IS NOT NEW.details
          OR (OLD.synced_flag = 1 AND NEW.synced_flag = 0)
        THEN RAISE(ABORT, 'EventLog is append-only: only synced_flag may be advanced from 0 to 1')
    END;
END;
"""

SCHEMA_SYNC_QUEUE = """
CREATE TABLE IF NOT EXISTS SyncQueue (
    -- change_id is a UUID4 string generated by the originator (hub or app).
    -- Used for idempotent rejection of duplicate SMS payloads (Section 3.5.2).
    change_id     TEXT    PRIMARY KEY,
    -- Target table: 'MedicationSchedule', 'EventLog', 'ElderProfile', etc.
    entity_type   TEXT    NOT NULL,
    entity_id     INTEGER NOT NULL,
    -- 'INSERT' | 'UPDATE' | 'DELETE'
    change_type   TEXT    NOT NULL CHECK (change_type IN ('INSERT','UPDATE','DELETE')),
    timestamp     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    -- 'pending' | 'in_flight' | 'synced' | 'failed'
    sync_state    TEXT    NOT NULL DEFAULT 'pending'
                          CHECK (sync_state IN ('pending','in_flight','synced','failed')),
    -- 'Hub->App' or 'App->Hub'
    direction     TEXT    NOT NULL CHECK (direction IN ('Hub->App','App->Hub')),
    -- 'wifi_rest' or 'sms'
    transport     TEXT    NOT NULL CHECK (transport IN ('wifi_rest','sms')),
    -- Serialized JSON payload of the change for transmission.
    payload       TEXT    NOT NULL,
    attempts      INTEGER NOT NULL DEFAULT 0
);
"""

SCHEMA_SYSTEM_CONFIG = """
CREATE TABLE IF NOT EXISTS SystemConfig (
    config_key     TEXT    PRIMARY KEY,
    config_value   TEXT    NOT NULL,
    last_modified  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
"""

# DevCommandQueue is a debug-only IPC channel between the dev_console process
# and the running hub process. Console writes commands; the hub's
# DevCommandPoller drains and applies them against the live in-memory mocks.
# This table is NOT used by any production code path.
SCHEMA_DEV_COMMAND_QUEUE = """
CREATE TABLE IF NOT EXISTS DevCommandQueue (
    cmd_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    command       TEXT    NOT NULL,
    args_json     TEXT,
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    applied_at    TEXT,
    result        TEXT
);
"""

# ---------------------------------------------------------------------------
# Indexes for performance
# ---------------------------------------------------------------------------
# These indexes accelerate the hot query paths:
#   - The reminder scheduler scans MedicationSchedule by (active, time_due).
#   - The sync engine scans SyncQueue by (sync_state, direction).
#   - The caregiver-facing event view filters EventLog by (timestamp, event_type).
INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_medsched_active_time ON MedicationSchedule(active, time_due);",
    "CREATE INDEX IF NOT EXISTS idx_eventlog_timestamp   ON EventLog(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_eventlog_synced      ON EventLog(synced_flag);",
    "CREATE INDEX IF NOT EXISTS idx_syncqueue_state      ON SyncQueue(sync_state, direction);",
]

# ---------------------------------------------------------------------------
# Default SystemConfig seed values
# ---------------------------------------------------------------------------
# These mirror the parameters described in Section 3.5.3 of the report.
# Seeded only on first initialization; existing values are preserved.
DEFAULT_CONFIG = [
    ("reminder_timeout_seconds",   "120"),
    ("reminder_retry_interval_min","15"),
    ("reminder_max_retries",       "3"),
    ("voice_confidence_threshold", "0.75"),
    ("speaker_volume_percent",     "80"),
    ("ap_ssid",                    "GeriatricHub"),
    ("ap_psk",                     "CHANGE_ME_ON_PAIRING"),
    ("hmac_key",                   "CHANGE_ME_ON_PAIRING"),
    ("pairing_token",              "CHANGE_ME_ON_PAIRING"),
    ("sms_poll_interval_seconds",  "30"),
]


# ---------------------------------------------------------------------------
# Initialization routine
# ---------------------------------------------------------------------------

def initialize_database(db_path: str = DB_PATH, verbose: bool = True) -> None:
    """
    Create all schema objects in the target SQLite database.

    Idempotent: safe to run repeatedly. Existing data is preserved.

    Parameters
    ----------
    db_path : str
        Filesystem path to the SQLite database file. Created if absent.
    verbose : bool
        If True, prints progress messages to stdout.
    """
    if verbose:
        print(f"[db_init] Opening database at: {db_path}")

    # `isolation_level=None` would put us in autocommit; we instead use an
    # explicit transaction to keep schema creation atomic.
    conn = sqlite3.connect(db_path)
    try:
        # Enable foreign key enforcement (off by default in SQLite).
        conn.execute("PRAGMA foreign_keys = ON;")
        # WAL mode improves concurrent read/write performance, important for
        # the multi-threaded hub runtime (scheduler + REST + SMS handler).
        conn.execute("PRAGMA journal_mode = WAL;")
        # Synchronous=NORMAL is a good balance of safety and write speed
        # under WAL on the SD card storage of the Raspberry Pi 4.
        conn.execute("PRAGMA synchronous = NORMAL;")

        cur = conn.cursor()

        # --- Tables ---
        if verbose: print("[db_init] Creating ElderProfile ...")
        cur.execute(SCHEMA_ELDER_PROFILE)

        if verbose: print("[db_init] Creating MedicationSchedule ...")
        cur.execute(SCHEMA_MEDICATION_SCHEDULE)

        if verbose: print("[db_init] Creating EventLog (append-only) ...")
        cur.execute(SCHEMA_EVENT_LOG)

        if verbose: print("[db_init] Installing append-only triggers on EventLog ...")
        cur.execute(TRIGGER_EVENT_LOG_NO_DELETE)
        cur.execute(TRIGGER_EVENT_LOG_NO_UPDATE)

        if verbose: print("[db_init] Creating SyncQueue ...")
        cur.execute(SCHEMA_SYNC_QUEUE)

        if verbose: print("[db_init] Creating SystemConfig ...")
        cur.execute(SCHEMA_SYSTEM_CONFIG)

        if verbose: print("[db_init] Creating DevCommandQueue (debug IPC) ...")
        cur.execute(SCHEMA_DEV_COMMAND_QUEUE)

        # --- Indexes ---
        if verbose: print("[db_init] Creating indexes ...")
        for stmt in INDEXES:
            cur.execute(stmt)

        # --- Seed default configuration (only if absent) ---
        if verbose: print("[db_init] Seeding default SystemConfig values (if absent) ...")
        cur.executemany(
            "INSERT OR IGNORE INTO SystemConfig (config_key, config_value) VALUES (?, ?);",
            DEFAULT_CONFIG
        )

        conn.commit()
        if verbose:
            print("[db_init] Schema initialization complete.")

    except sqlite3.Error as exc:
        conn.rollback()
        print(f"[db_init] ERROR: {exc}")
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Self-test: verify append-only enforcement on EventLog
# ---------------------------------------------------------------------------

def _self_test(db_path: str = DB_PATH) -> None:
    """
    Smoke test confirming that:
      1. An INSERT into EventLog succeeds.
      2. An UPDATE of an immutable column on EventLog raises an error.
      3. A DELETE on EventLog raises an error.
    """
    print("\n[db_init] Running self-test on EventLog append-only invariants ...")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    cur = conn.cursor()

    # 1. Insert should succeed.
    cur.execute(
        "INSERT INTO EventLog (event_type, details) VALUES (?, ?);",
        ("system_boot", "self-test boot event")
    )
    inserted_id = cur.lastrowid
    print(f"  [PASS] INSERT succeeded (event_id={inserted_id})")

    # 2. Disallowed UPDATE (modifying event_type) should raise.
    try:
        cur.execute(
            "UPDATE EventLog SET event_type = 'tampered' WHERE event_id = ?;",
            (inserted_id,)
        )
        print("  [FAIL] UPDATE of event_type was NOT blocked!")
    except sqlite3.IntegrityError as e:
        print(f"  [PASS] UPDATE of event_type correctly blocked: {e}")

    # 3. Permitted UPDATE (advancing synced_flag from 0 to 1) should succeed.
    try:
        cur.execute(
            "UPDATE EventLog SET synced_flag = 1 WHERE event_id = ?;",
            (inserted_id,)
        )
        print("  [PASS] UPDATE of synced_flag (0->1) correctly permitted")
    except sqlite3.IntegrityError as e:
        print(f"  [FAIL] UPDATE of synced_flag was unexpectedly blocked: {e}")

    # 4. DELETE should raise.
    try:
        cur.execute("DELETE FROM EventLog WHERE event_id = ?;", (inserted_id,))
        print("  [FAIL] DELETE was NOT blocked!")
    except sqlite3.IntegrityError as e:
        print(f"  [PASS] DELETE correctly blocked: {e}")

    # Roll back the self-test row so we don't pollute the live DB.
    conn.rollback()
    conn.close()
    print("[db_init] Self-test complete.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    initialize_database(DB_PATH, verbose=True)
    _self_test(DB_PATH)
