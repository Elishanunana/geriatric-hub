"""
repositories.py
===============
Data Access Layer for the hub. One repository per table from Table 3.3 of
the project report, plus dataclasses representing each domain entity.

Design Principles
-----------------
1. Each repository operates against the schema defined in db_init.py.
2. All SQL uses parameterized queries — no string interpolation of values.
3. EventLogRepo deliberately exposes NO update or delete method, mirroring
   the append-only triggers installed at the database layer (Section 3.5.3).
4. Each method opens its own connection via the get_connection() context
   manager, so each operation is its own atomic transaction.
5. Dataclasses serve as transport objects between repos and the rest of
   the codebase, keeping SQL column names contained inside this module.

Author: Wise (Asumang Pobi Godwin) — KNUST COE 497
"""

import json
import uuid
import logging
import sqlite3
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, List, Optional

from src.data_management.db_connection import get_connection

logger = logging.getLogger(__name__)


# ===========================================================================
# Domain entities
# ===========================================================================
# These dataclasses mirror the columns of each table 1:1. Optional fields
# (e.g. auto-generated IDs and timestamps) carry None defaults so callers
# can construct partial instances for INSERT operations.

@dataclass
class ElderProfile:
    elder_id: Optional[int] = None
    name: str = ""
    language: str = "twi"
    caregiver_phones: str = ""        # Comma-separated list
    created_at: Optional[str] = None
    last_modified: Optional[str] = None


@dataclass
class MedicationSchedule:
    schedule_id: Optional[int] = None
    elder_id: int = 0
    drug_name: str = ""
    dosage: str = ""
    time_due: str = ""                # 'HH:MM' (24-hour)
    days_of_week: str = "DAILY"       # 'DAILY' or 'MON,TUE,WED'
    active: int = 1
    prescribed_by: str = "caregiver"  # 'caregiver' | 'pharmacist' | 'hub_local'
    sync_method: str = "hub_local"    # 'app_wifi' | 'app_sms' | 'hub_local'
    last_modified: Optional[str] = None


@dataclass
class EventLogEntry:
    event_id: Optional[int] = None
    event_type: str = ""
    timestamp: Optional[str] = None
    details: Optional[str] = None     # Free-form text or JSON string
    synced_flag: int = 0


@dataclass
class SyncQueueEntry:
    change_id: str = ""               # UUID4 string (PK)
    entity_type: str = ""
    entity_id: int = 0
    change_type: str = ""             # 'INSERT' | 'UPDATE' | 'DELETE'
    timestamp: Optional[str] = None
    sync_state: str = "pending"       # 'pending' | 'in_flight' | 'synced' | 'failed'
    direction: str = ""               # 'Hub->App' | 'App->Hub'
    transport: str = ""               # 'wifi_rest' | 'sms'
    payload: str = ""                 # Serialized JSON
    attempts: int = 0


# ===========================================================================
# Helpers
# ===========================================================================

def _utcnow_iso() -> str:
    """Return the current UTC time as ISO-8601 with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _serialize(value: Any) -> Optional[str]:
    """Serialize dicts/lists to JSON; pass through strings and None unchanged."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


# ===========================================================================
# Base repository
# ===========================================================================

class BaseRepository:
    """
    Shared infrastructure for all repositories. Holds a connection factory
    so tests can inject a custom one (e.g. pointing at a temp DB).
    """

    def __init__(self, connection_factory: Callable = get_connection):
        self._cf = connection_factory


# ===========================================================================
# ElderProfileRepo
# ===========================================================================

class ElderProfileRepo(BaseRepository):
    """CRUD operations on the ElderProfile table."""

    def insert(self, profile: ElderProfile) -> int:
        """Create a new elder profile. Returns the new elder_id."""
        with self._cf() as conn:
            cur = conn.execute(
                """
                INSERT INTO ElderProfile (name, language, caregiver_phones)
                VALUES (?, ?, ?);
                """,
                (profile.name, profile.language, profile.caregiver_phones),
            )
            return cur.lastrowid

    def fetch_by_id(self, elder_id: int) -> Optional[ElderProfile]:
        with self._cf() as conn:
            row = conn.execute(
                "SELECT * FROM ElderProfile WHERE elder_id = ?;", (elder_id,)
            ).fetchone()
            return ElderProfile(**dict(row)) if row else None

    def fetch_first(self) -> Optional[ElderProfile]:
        """
        Most deployments will have exactly one elder per hub. This helper
        simplifies the common case for the reminder scheduler and the
        SOS handler.
        """
        with self._cf() as conn:
            row = conn.execute(
                "SELECT * FROM ElderProfile ORDER BY elder_id LIMIT 1;"
            ).fetchone()
            return ElderProfile(**dict(row)) if row else None

    def update_caregiver_phones(self, elder_id: int, phones_csv: str) -> bool:
        """Replace the caregiver_phones field. Returns True if a row was updated."""
        with self._cf() as conn:
            cur = conn.execute(
                """
                UPDATE ElderProfile
                   SET caregiver_phones = ?,
                       last_modified = ?
                 WHERE elder_id = ?;
                """,
                (phones_csv, _utcnow_iso(), elder_id),
            )
            return cur.rowcount > 0

    def caregiver_phones(self, elder_id: int) -> List[str]:
        """Return the caregiver phone numbers as a clean list."""
        prof = self.fetch_by_id(elder_id)
        if prof is None or not prof.caregiver_phones:
            return []
        return [p.strip() for p in prof.caregiver_phones.split(",") if p.strip()]

    def is_caregiver_number(self, phone: str, elder_id: Optional[int] = None) -> bool:
        """
        Check whether `phone` is a registered caregiver number. Used by the
        SMS payload handler for origin verification (Section 3.5.2: "Messages
        from unrecognised numbers are immediately discarded").

        If elder_id is None, all elders' caregiver lists are searched —
        useful in single-elder deployments.
        """
        phone_norm = phone.strip()
        with self._cf() as conn:
            if elder_id is None:
                rows = conn.execute(
                    "SELECT caregiver_phones FROM ElderProfile;"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT caregiver_phones FROM ElderProfile WHERE elder_id = ?;",
                    (elder_id,),
                ).fetchall()

        for row in rows:
            registered = [p.strip() for p in (row["caregiver_phones"] or "").split(",")]
            if phone_norm in registered:
                return True
        return False


# ===========================================================================
# MedicationScheduleRepo
# ===========================================================================

class MedicationScheduleRepo(BaseRepository):
    """CRUD plus the reminder-scheduler's `due-within` query."""

    def insert(self, sched: MedicationSchedule) -> int:
        """Create a new medication schedule. Returns the new schedule_id."""
        with self._cf() as conn:
            cur = conn.execute(
                """
                INSERT INTO MedicationSchedule (
                    elder_id, drug_name, dosage, time_due, days_of_week,
                    active, prescribed_by, sync_method, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    sched.elder_id, sched.drug_name, sched.dosage,
                    sched.time_due, sched.days_of_week, sched.active,
                    sched.prescribed_by, sched.sync_method, _utcnow_iso(),
                ),
            )
            return cur.lastrowid

    def fetch_by_id(self, schedule_id: int) -> Optional[MedicationSchedule]:
        with self._cf() as conn:
            row = conn.execute(
                "SELECT * FROM MedicationSchedule WHERE schedule_id = ?;",
                (schedule_id,),
            ).fetchone()
            return MedicationSchedule(**dict(row)) if row else None

    def fetch_all_active(self, elder_id: Optional[int] = None) -> List[MedicationSchedule]:
        """
        Return every active schedule, optionally scoped to a single elder.
        Backs the READ_SCHEDULE voice command (Table 3.2).
        """
        with self._cf() as conn:
            if elder_id is None:
                rows = conn.execute(
                    "SELECT * FROM MedicationSchedule WHERE active = 1 ORDER BY time_due;"
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM MedicationSchedule
                     WHERE active = 1 AND elder_id = ?
                     ORDER BY time_due;
                    """,
                    (elder_id,),
                ).fetchall()
            return [MedicationSchedule(**dict(r)) for r in rows]

    def fetch_due_within(
        self,
        lookahead_seconds: int = 60,
        current_time: Optional[datetime] = None,
    ) -> List[MedicationSchedule]:
        """
        Return active schedules whose time_due falls within the window
        [current_time, current_time + lookahead_seconds] AND whose
        days_of_week includes today's day code.

        This is the central query of the reminder scheduler (Section 3.5.2:
        "queries the SQLite database at regular intervals for reminders due
        within the next 60 seconds").

        Parameters
        ----------
        lookahead_seconds : int
            Window size. Default of 60 matches the report.
        current_time : datetime, optional
            Override "now" — primarily for deterministic unit testing.
        """
        now = current_time or datetime.now()
        end = now + timedelta(seconds=lookahead_seconds)

        # Day codes match the strftime('%a') three-letter abbreviation
        # uppercased: MON, TUE, WED, THU, FRI, SAT, SUN.
        today_code = now.strftime("%a").upper()

        now_str = now.strftime("%H:%M")
        end_str = end.strftime("%H:%M")

        # Edge case: if the lookahead window crosses midnight (now > end_str
        # in lex order, e.g., now=23:59 → end=00:00), we'd miss schedules.
        # For the 60-second default this self-heals on the next scheduler
        # tick after midnight, so we accept this and document it.
        if now_str > end_str:
            logger.debug(
                "Due-within window crosses midnight — single tick may miss "
                "early-morning schedules; next tick will catch them."
            )
            time_clause = "(time_due >= ? OR time_due <= ?)"
        else:
            time_clause = "(time_due >= ? AND time_due <= ?)"

        # Day filter: 'DAILY' is a wildcard, otherwise check that the
        # comma-separated day list contains today's code (with comma
        # boundaries to prevent substring false-matches).
        day_pattern = f"%,{today_code},%"

        sql = f"""
            SELECT * FROM MedicationSchedule
             WHERE active = 1
               AND {time_clause}
               AND (
                       days_of_week = 'DAILY'
                    OR (',' || days_of_week || ',') LIKE ?
                   )
             ORDER BY time_due;
        """

        with self._cf() as conn:
            rows = conn.execute(sql, (now_str, end_str, day_pattern)).fetchall()
            return [MedicationSchedule(**dict(r)) for r in rows]

    def update_active(self, schedule_id: int, active: bool) -> bool:
        """Activate or deactivate a schedule (soft delete)."""
        with self._cf() as conn:
            cur = conn.execute(
                """
                UPDATE MedicationSchedule
                   SET active = ?, last_modified = ?
                 WHERE schedule_id = ?;
                """,
                (1 if active else 0, _utcnow_iso(), schedule_id),
            )
            return cur.rowcount > 0

    def deactivate(self, schedule_id: int) -> bool:
        """Convenience: soft-delete a schedule by setting active = 0."""
        return self.update_active(schedule_id, False)

    def upsert_from_payload(
        self,
        sched: MedicationSchedule,
        sync_method: str,
        prescribed_by: str,
    ) -> int:
        """
        Apply a schedule received via either sync pathway. If a schedule with
        the same (elder_id, drug_name, time_due, days_of_week) already exists,
        it is updated; otherwise it is inserted. Sets sync_method and
        prescribed_by per the originating channel (Section 3.5.2).
        """
        with self._cf() as conn:
            existing = conn.execute(
                """
                SELECT schedule_id FROM MedicationSchedule
                 WHERE elder_id = ? AND drug_name = ?
                   AND time_due = ? AND days_of_week = ?;
                """,
                (sched.elder_id, sched.drug_name, sched.time_due, sched.days_of_week),
            ).fetchone()

            ts = _utcnow_iso()
            if existing:
                conn.execute(
                    """
                    UPDATE MedicationSchedule
                       SET dosage = ?, active = ?, prescribed_by = ?,
                           sync_method = ?, last_modified = ?
                     WHERE schedule_id = ?;
                    """,
                    (sched.dosage, sched.active, prescribed_by,
                     sync_method, ts, existing["schedule_id"]),
                )
                return existing["schedule_id"]
            else:
                cur = conn.execute(
                    """
                    INSERT INTO MedicationSchedule (
                        elder_id, drug_name, dosage, time_due, days_of_week,
                        active, prescribed_by, sync_method, last_modified
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (sched.elder_id, sched.drug_name, sched.dosage, sched.time_due,
                     sched.days_of_week, sched.active, prescribed_by,
                     sync_method, ts),
                )
                return cur.lastrowid


# ===========================================================================
# EventLogRepo  —  APPEND-ONLY
# ===========================================================================

class EventLogRepo(BaseRepository):
    """
    Append-only repository for the system audit log.

    Notable absences (deliberate, not oversight):
      • No update() method — events are immutable once written.
      • No delete() method — the log is permanent.
    The only mutation permitted is mark_synced(), which advances
    synced_flag from 0 to 1; this is whitelisted by the trigger
    `trg_eventlog_no_update` installed in db_init.py.
    """

    # Canonical event_type constants — used by callers throughout the
    # codebase to avoid magic strings.
    REMINDER_ISSUED       = "reminder_issued"
    DOSE_CONFIRMED        = "dose_confirmed"
    DOSE_MISSED           = "dose_missed"
    SOS_TRIGGERED         = "sos_triggered"
    APPLIANCE_ON          = "appliance_on"
    APPLIANCE_OFF         = "appliance_off"
    POWER_ON_BATTERY      = "power_on_battery"
    POWER_ON_MAINS        = "power_on_mains"
    SMS_PAYLOAD_ACCEPTED  = "sms_payload_accepted"
    SMS_PAYLOAD_REJECTED  = "sms_payload_rejected"
    SYSTEM_BOOT           = "system_boot"
    SYSTEM_FAULT          = "system_fault"

    def insert(self, event_type: str, details: Optional[Any] = None) -> int:
        """
        Append a new event. Returns the new event_id.

        `details` may be a string OR a dict/list — dicts and lists are
        JSON-serialized automatically for ergonomic caller code.
        """
        details_str = _serialize(details)
        with self._cf() as conn:
            cur = conn.execute(
                """
                INSERT INTO EventLog (event_type, details)
                VALUES (?, ?);
                """,
                (event_type, details_str),
            )
            return cur.lastrowid

    def fetch_recent(
        self,
        limit: int = 100,
        since: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> List[EventLogEntry]:
        """
        Return the most recent log entries, newest first. Optional filters:
          • since: ISO-8601 timestamp lower bound (exclusive)
          • event_type: restrict to a single event category
        """
        clauses = []
        params: list = []
        if since:
            clauses.append("timestamp > ?")
            params.append(since)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM EventLog{where} ORDER BY event_id DESC LIMIT ?;"
        params.append(limit)

        with self._cf() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [EventLogEntry(**dict(r)) for r in rows]

    def fetch_unsynced(self, limit: int = 100) -> List[EventLogEntry]:
        """Return events not yet sent to the caregiver app."""
        with self._cf() as conn:
            rows = conn.execute(
                """
                SELECT * FROM EventLog
                 WHERE synced_flag = 0
                 ORDER BY event_id ASC
                 LIMIT ?;
                """,
                (limit,),
            ).fetchall()
            return [EventLogEntry(**dict(r)) for r in rows]

    def mark_synced(self, event_id: int) -> bool:
        """
        Advance synced_flag from 0 → 1 after successful sync to the app.
        This is the *only* permitted mutation on EventLog rows; the
        append-only triggers in db_init.py block all others.
        """
        with self._cf() as conn:
            cur = conn.execute(
                "UPDATE EventLog SET synced_flag = 1 WHERE event_id = ? AND synced_flag = 0;",
                (event_id,),
            )
            return cur.rowcount > 0


# ===========================================================================
# SyncQueueRepo
# ===========================================================================

class SyncQueueRepo(BaseRepository):
    """
    Tracks pending bidirectional sync changes between the hub and the
    caregiver mobile app, across both transports (wifi_rest and sms).
    See Section 3.5.4 of the project report.
    """

    def enqueue(self, entry: SyncQueueEntry) -> str:
        """
        Insert a new pending sync record. If entry.change_id is empty,
        a UUID4 is generated automatically.
        Returns the change_id of the inserted row.
        """
        change_id = entry.change_id or str(uuid.uuid4())
        ts = entry.timestamp or _utcnow_iso()

        with self._cf() as conn:
            conn.execute(
                """
                INSERT INTO SyncQueue (
                    change_id, entity_type, entity_id, change_type,
                    timestamp, sync_state, direction, transport,
                    payload, attempts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    change_id, entry.entity_type, entry.entity_id,
                    entry.change_type, ts, entry.sync_state,
                    entry.direction, entry.transport,
                    entry.payload, entry.attempts,
                ),
            )
        return change_id

    def enqueue_change(
        self,
        entity_type: str,
        entity_id: int,
        change_type: str,
        direction: str,
        transport: str,
        payload: Any,
    ) -> str:
        """
        High-level convenience wrapper: builds the SyncQueueEntry, serializes
        the payload to JSON if necessary, and enqueues it.
        """
        entry = SyncQueueEntry(
            change_id=str(uuid.uuid4()),
            entity_type=entity_type,
            entity_id=entity_id,
            change_type=change_type,
            direction=direction,
            transport=transport,
            payload=_serialize(payload) or "",
        )
        return self.enqueue(entry)

    def fetch_pending(
        self,
        direction: str,
        transport: Optional[str] = None,
        limit: int = 100,
    ) -> List[SyncQueueEntry]:
        """
        Fetch records awaiting transmission. Filter by direction (required)
        and optionally by transport. Used by the sync engine to assemble a
        batch for the next available channel.
        """
        if transport:
            sql = """
                SELECT * FROM SyncQueue
                 WHERE sync_state = 'pending'
                   AND direction = ?
                   AND transport = ?
                 ORDER BY timestamp ASC
                 LIMIT ?;
            """
            params = (direction, transport, limit)
        else:
            sql = """
                SELECT * FROM SyncQueue
                 WHERE sync_state = 'pending'
                   AND direction = ?
                 ORDER BY timestamp ASC
                 LIMIT ?;
            """
            params = (direction, limit)

        with self._cf() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [SyncQueueEntry(**dict(r)) for r in rows]

    def fetch_by_id(self, change_id: str) -> Optional[SyncQueueEntry]:
        with self._cf() as conn:
            row = conn.execute(
                "SELECT * FROM SyncQueue WHERE change_id = ?;",
                (change_id,),
            ).fetchone()
            return SyncQueueEntry(**dict(row)) if row else None

    def is_duplicate(self, change_id: str) -> bool:
        """
        Idempotency check used by the SMS payload handler (Section 3.5.2:
        "receipt of a duplicate payload bearing the same change_id is
        detected and ignored").
        """
        with self._cf() as conn:
            row = conn.execute(
                "SELECT 1 FROM SyncQueue WHERE change_id = ?;",
                (change_id,),
            ).fetchone()
            return row is not None
    
    def update_transport(self, change_id: str, new_transport: str) -> bool:
        """
        Switch the transport pathway of a pending record. Used by the
        ConnectivityArbiter to promote items from wifi_rest to sms when
        Wi-Fi is unavailable and the entry has exceeded its urgency
        budget. Only mutates records currently in 'pending' state — once
        an entry is in_flight or synced its transport is historical and
        must not change.
        """
        if new_transport not in ("wifi_rest", "sms"):
            raise ValueError(f"invalid transport: {new_transport}")
        with self._cf() as conn:
            cur = conn.execute(
                """
                UPDATE SyncQueue
                   SET transport = ?
                 WHERE change_id = ? AND sync_state = 'pending';
                """,
                (new_transport, change_id),
            )
            return cur.rowcount > 0

    def mark_in_flight(self, change_id: str) -> bool:
        """Transition pending → in_flight when the sync engine begins transmission."""
        return self._transition(change_id, "pending", "in_flight")

    def mark_synced(self, change_id: str) -> bool:
        """Transition in_flight → synced upon recipient acknowledgement."""
        return self._transition(change_id, "in_flight", "synced")

    def mark_failed(self, change_id: str) -> bool:
        """
        Transition in_flight → failed and increment the attempt counter.
        The sync engine's retry policy may later reset failed → pending.
        """
        with self._cf() as conn:
            cur = conn.execute(
                """
                UPDATE SyncQueue
                   SET sync_state = 'failed',
                       attempts   = attempts + 1
                 WHERE change_id = ?
                   AND sync_state = 'in_flight';
                """,
                (change_id,),
            )
            return cur.rowcount > 0

    def _transition(self, change_id: str, from_state: str, to_state: str) -> bool:
        """Generic state-machine helper — only transitions if currently in `from_state`."""
        with self._cf() as conn:
            cur = conn.execute(
                """
                UPDATE SyncQueue
                   SET sync_state = ?
                 WHERE change_id = ? AND sync_state = ?;
                """,
                (to_state, change_id, from_state),
            )
            return cur.rowcount > 0


# ===========================================================================
# Standalone smoke test
# ===========================================================================

if __name__ == "__main__":
    """
    Tiny end-to-end smoke test. Run this directly:
        python -m src.data_management.repositories
    Assumes db_init.py has been executed.
    """
    elder_repo  = ElderProfileRepo()
    med_repo    = MedicationScheduleRepo()
    log_repo    = EventLogRepo()
    sync_repo   = SyncQueueRepo()

    print("--- ElderProfileRepo ---")
    elder_id = elder_repo.insert(ElderProfile(
        name="Test Elder",
        language="twi",
        caregiver_phones="+233244123456,+233244999999",
    ))
    print(f"Inserted elder_id = {elder_id}")
    print(f"Fetch first      : {elder_repo.fetch_first()}")
    print(f"Is +233244123456 a caregiver? {elder_repo.is_caregiver_number('+233244123456')}")
    print(f"Is +233200000000 a caregiver? {elder_repo.is_caregiver_number('+233200000000')}")

    print("\n--- MedicationScheduleRepo ---")
    sched_id = med_repo.insert(MedicationSchedule(
        elder_id=elder_id,
        drug_name="Paracetamol",
        dosage="500mg",
        time_due="08:00",
        days_of_week="DAILY",
        prescribed_by="pharmacist",
        sync_method="hub_local",
    ))
    print(f"Inserted schedule_id = {sched_id}")

    # Use a fixed datetime that falls inside the medication's window.
    test_now = datetime.now().replace(hour=7, minute=59, second=30, microsecond=0)
    due = med_repo.fetch_due_within(lookahead_seconds=60, current_time=test_now)
    print(f"Due within 60s of {test_now.strftime('%H:%M:%S')}: {len(due)} schedule(s)")
    for s in due:
        print(f"  • {s.drug_name} {s.dosage} @ {s.time_due}")

    print("\n--- EventLogRepo ---")
    eid = log_repo.insert(EventLogRepo.SYSTEM_BOOT, details={"version": "0.1.0-smoke"})
    print(f"Inserted event_id = {eid}")
    print(f"Recent events    : {len(log_repo.fetch_recent(limit=5))}")
    print(f"Unsynced events  : {len(log_repo.fetch_unsynced())}")
    print(f"Mark synced      : {log_repo.mark_synced(eid)}")

    print("\n--- SyncQueueRepo ---")
    cid = sync_repo.enqueue_change(
        entity_type="MedicationSchedule",
        entity_id=sched_id,
        change_type="INSERT",
        direction="Hub->App",
        transport="wifi_rest",
        payload={"drug_name": "Paracetamol", "dosage": "500mg", "time_due": "08:00"},
    )
    print(f"Enqueued change_id = {cid}")
    print(f"Is duplicate?       {sync_repo.is_duplicate(cid)}")
    print(f"Pending Hub->App   : {len(sync_repo.fetch_pending('Hub->App'))}")
    print(f"Mark in_flight     : {sync_repo.mark_in_flight(cid)}")
    print(f"Mark synced        : {sync_repo.mark_synced(cid)}")
    print(f"After sync, pending Hub->App: {len(sync_repo.fetch_pending('Hub->App'))}")
    