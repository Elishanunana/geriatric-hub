"""
sync_engine.py
==============
Outbound sync orchestrator. Polls the SyncQueue for pending Hub→App
entries and routes each through the appropriate transport based on the
ConnectivityArbiter's decision.

Why this asymmetric design?
---------------------------
The two transport pathways have fundamentally different shapes:

    Wi-Fi REST  →  PULL.  The caregiver app initiates by polling the hub's
                          REST endpoints; the hub merely answers when asked.
                          So Wi-Fi entries can wait passively in the queue.

    SMS         →  PUSH.  The hub initiates by sending a payload via the
                          SIM800L; nothing happens unless we actively dispatch.

The SyncEngine therefore primarily drives the PUSH side. Items destined
for wifi_rest are left for the REST API to serve when the app polls; only
items destined for (or promoted to) sms are actively dispatched here.

Author: Wise (Asumang Pobi Godwin) — KNUST COE 497
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.data_management.repositories import (
    EventLogRepo,
    SyncQueueRepo,
    SyncQueueEntry,
)
from src.communication.connectivity_arbiter import (
    ConnectivityArbiter,
    TRANSPORT_SMS,
    TRANSPORT_WIFI_REST,
    TRANSPORT_WAIT,
)
from src.communication.sms_transport import SMSTransport, DispatchResult

logger = logging.getLogger(__name__)


# ===========================================================================
# Configuration
# ===========================================================================

DEFAULT_POLL_INTERVAL_SECONDS = 30
DEFAULT_BATCH_LIMIT           = 20


# ===========================================================================
# SyncEngine
# ===========================================================================

class SyncEngine:
    """
    Polls SyncQueue for pending Hub→App entries; routes each through
    SMSTransport when SMS is the chosen pathway, leaves wifi_rest items
    for the REST API. Threading-safe and dependency-injected.
    """

    def __init__(
        self,
        sms_transport: SMSTransport,
        arbiter:       ConnectivityArbiter,
        sync_repo:     Optional[SyncQueueRepo] = None,
        event_repo:    Optional[EventLogRepo]  = None,
        *,
        poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS,
        batch_limit:           int = DEFAULT_BATCH_LIMIT,
    ):
        self._sms        = sms_transport
        self._arbiter    = arbiter
        self._sync_repo  = sync_repo  or SyncQueueRepo()
        self._event_repo = event_repo or EventLogRepo()

        self._poll_interval = poll_interval_seconds
        self._batch_limit   = batch_limit

        # Lifecycle.
        self._stop_event = threading.Event()
        self._thread:    Optional[threading.Thread] = None
        self._lock       = threading.Lock()
        # Serialises process_once() so the polling thread cannot race
        # a manual call from a test or admin tool.
        self._pipeline_lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                logger.warning("SyncEngine already running.")
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop, name="SyncEngine", daemon=True,
            )
            self._thread.start()
        logger.info(
            "SyncEngine started (poll interval = %ds, batch limit = %d).",
            self._poll_interval, self._batch_limit,
        )

    def stop(self, join_timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=join_timeout)
        logger.info("SyncEngine stopped.")

    # -----------------------------------------------------------------------
    # Polling loop
    # -----------------------------------------------------------------------

    def _run_loop(self) -> None:
        # Brief startup delay so we don't hammer the queue at boot.
        self._stop_event.wait(timeout=2.0)

        while not self._stop_event.is_set():
            try:
                self.process_once()
            except Exception:
                logger.exception("SyncEngine tick raised; continuing.")
                self._event_repo.insert(
                    EventLogRepo.SYSTEM_FAULT,
                    details={"subsystem": "SyncEngine", "stage": "tick"},
                )
            self._stop_event.wait(timeout=self._poll_interval)

    # -----------------------------------------------------------------------
    # Public single-pass entry point
    # -----------------------------------------------------------------------

    def process_once(self) -> Dict[str, int]:
        """
        Drain one batch of pending Hub→App entries through the dispatch
        pipeline. Returns a count breakdown for callers (mainly tests).

        Counts:
          'dispatched_sms'  — successfully sent via SMS, marked synced.
          'left_for_rest'   — kept on wifi_rest, the REST API will serve.
          'promoted_then_dispatched' — was wifi_rest, promoted to SMS, sent.
          'failed'          — dispatch attempted and failed.
          'waiting'         — both pathways unavailable; left pending.
        """
        counts = {
            "dispatched_sms":           0,
            "left_for_rest":            0,
            "promoted_then_dispatched": 0,
            "failed":                   0,
            "waiting":                  0,
        }

        with self._pipeline_lock:
            entries = self._sync_repo.fetch_pending(
                direction="Hub->App", limit=self._batch_limit,
            )
            for entry in entries:
                if self._stop_event.is_set():
                    break
                outcome = self._process_entry(entry)
                counts[outcome] = counts.get(outcome, 0) + 1
        return counts

    # -----------------------------------------------------------------------
    # Per-entry routing
    # -----------------------------------------------------------------------

    def _process_entry(self, entry: SyncQueueEntry) -> str:
        """
        Apply the arbiter's decision to a single entry. Returns one of
        the count keys defined in process_once().
        """
        was_originally_wifi = (entry.transport == TRANSPORT_WIFI_REST)
        decision = self._arbiter.select_transport(entry)

        if decision == TRANSPORT_WIFI_REST:
            # Passive — REST API will serve when the app polls.
            return "left_for_rest"

        if decision == TRANSPORT_WAIT:
            return "waiting"

        # decision == TRANSPORT_SMS — possibly a promotion.
        promoted = False
        if was_originally_wifi:
            promoted = self._arbiter.promote_to_sms(entry)
            if not promoted:
                # Promotion failed (e.g., row already in_flight from a race).
                # Treat as wait — a future tick will retry.
                logger.warning(
                    "promote_to_sms returned False for change_id=%s — skipping.",
                    entry.change_id,
                )
                return "waiting"

        ok = self._dispatch_via_sms(entry)
        if ok:
            return "promoted_then_dispatched" if promoted else "dispatched_sms"
        return "failed"

    # -----------------------------------------------------------------------
    # SMS dispatch — full state-machine transition
    # -----------------------------------------------------------------------

    def _dispatch_via_sms(self, entry: SyncQueueEntry) -> bool:
        """
        Drive the SyncQueue state machine for one SMS dispatch:
            pending → in_flight → (synced | failed)

        Returns True iff the entry was successfully dispatched and
        transitioned to 'synced'.
        """
        # 1. Atomic claim.
        if not self._sync_repo.mark_in_flight(entry.change_id):
            # Someone else already claimed it (e.g. a parallel instance,
            # though we don't expect that in this design).
            logger.debug(
                "mark_in_flight returned False for change_id=%s — already claimed.",
                entry.change_id,
            )
            return False

        # 2. Dispatch.
        result: DispatchResult = self._sms.dispatch(entry)

        # 3. Resolve state.
        if result.overall_ok:
            self._sync_repo.mark_synced(entry.change_id)
            return True

        self._sync_repo.mark_failed(entry.change_id)
        return False


# ===========================================================================
# Standalone smoke test
# ===========================================================================

if __name__ == "__main__":
    """
    Run from the project root:

        python -m src.communication.sync_engine

    Demonstrates outbound routing across four scenarios:
      1. Routine entry on wifi_rest  → left for REST (Wi-Fi assumed up later).
      2. Urgent SOS entry on wifi_rest, OLD timestamp, Wi-Fi DOWN
                                       → PROMOTED to sms → DISPATCHED.
      3. Entry already on sms        → DISPATCHED directly.
      4. Urgent entry, Wi-Fi UP      → left for REST (no promotion needed).
    """
    import json
    import logging
    import uuid
    from datetime import timedelta
    from src.hardware_mocks.mock_sim800l  import MockGSMModule
    from src.data_management.repositories import (
        ElderProfileRepo, ElderProfile,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(name)s — %(message)s",
    )

    # --- Seed elder + caregiver phones ---------------------------------
    elder_repo = ElderProfileRepo()
    elder = elder_repo.fetch_first()
    if elder is None:
        elder_id = elder_repo.insert(ElderProfile(
            name="Demo Elder",
            language="twi",
            caregiver_phones="+233244111111,+233244222222",
        ))
    else:
        elder_id = elder.elder_id
        elder_repo.update_caregiver_phones(
            elder_id, "+233244111111,+233244222222"
        )

    # --- Wire up the components ----------------------------------------
    gsm        = MockGSMModule()
    sync_repo  = SyncQueueRepo()
    event_repo = EventLogRepo()

    # --- Clean slate: mark every pending Hub→App SyncQueue entry as
    # synced so the smoke test runs against a deterministic queue
    # regardless of how much state has accumulated from earlier task
    # runs (Task 7's SMS handler acks, Task 8's REST schedule syncs,
    # prior runs of this same test, etc.). In production the SyncEngine
    # naturally drains the queue; in a one-shot test we need to
    # fast-forward past the backlog before enqueuing our test fixtures.
    backlog_pending = sync_repo.fetch_pending(direction="Hub->App", limit=10000)
    backlog_count = 0
    for entry in backlog_pending:
        # Walk through the proper state machine: pending → in_flight → synced.
        if sync_repo.mark_in_flight(entry.change_id):
            sync_repo.mark_synced(entry.change_id)
            backlog_count += 1
    print(f"  [setup] Drained {backlog_count} pre-existing pending Hub→App "
          f"entries to give the test a clean slate.")

    sms_transport = SMSTransport(
        gsm=gsm,
        hmac_key="",
        dev_mock_hmac=True,    # safe: no real key needed for the demo
    )
    arbiter = ConnectivityArbiter(
        sync_repo=sync_repo,
        gsm=gsm,
        default_wifi_caregiver_connected=False,   # start with Wi-Fi DOWN
    )
    engine = SyncEngine(
        sms_transport=sms_transport,
        arbiter=arbiter,
        sync_repo=sync_repo,
        event_repo=event_repo,
    )

    # --- Helper: make a Hub→App entry directly in the queue ------------
    def enqueue_hub_to_app(*, event_type, transport, age_seconds=0):
        cid = str(uuid.uuid4())
        ts  = (datetime.now(timezone.utc) - timedelta(seconds=age_seconds)) \
                .isoformat(timespec="milliseconds").replace("+00:00", "Z")
        sync_repo.enqueue(SyncQueueEntry(
            change_id=cid,
            entity_type="EventLog",
            entity_id=999,
            change_type="INSERT",
            timestamp=ts,
            sync_state="pending",
            direction="Hub->App",
            transport=transport,
            payload=json.dumps({
                "event_type":    event_type,
                "demo_marker":   cid[:8],
                "originated_at": ts,
            }),
        ))
        return cid

    # --- Build the four test entries -----------------------------------
    cid_1 = enqueue_hub_to_app(           # routine, current
        event_type="dose_confirmed", transport="wifi_rest", age_seconds=0,
    )
    cid_2 = enqueue_hub_to_app(           # urgent, OLD timestamp (2 mins ago)
        event_type="sos_triggered",  transport="wifi_rest", age_seconds=120,
    )
    cid_3 = enqueue_hub_to_app(           # already on sms
        event_type="dose_missed",    transport="sms",       age_seconds=600,
    )
    cid_4 = enqueue_hub_to_app(           # urgent + old, but Wi-Fi will be UP
        event_type="sos_triggered",  transport="wifi_rest", age_seconds=120,
    )

    # ===================================================================
    # FIRST TICK — Wi-Fi DOWN
    # ===================================================================
    print("\n" + "=" * 70)
    print("  FIRST TICK — Wi-Fi DOWN")
    print("  Expectations:")
    print("    • cid_1 (routine, wifi_rest)        → left_for_rest")
    print("    • cid_2 (urgent + old, wifi_rest)   → promoted_then_dispatched")
    print("    • cid_3 (already on sms)            → dispatched_sms")
    print("    • cid_4 (urgent + old, wifi_rest)   → promoted_then_dispatched")
    print("=" * 70)

    counts = engine.process_once()
    print(f"\n  Counts: {counts}")

    # --- Verify ---
    e1 = sync_repo.fetch_by_id(cid_1)
    e2 = sync_repo.fetch_by_id(cid_2)
    e3 = sync_repo.fetch_by_id(cid_3)
    e4 = sync_repo.fetch_by_id(cid_4)

    print("\n  Per-entry state after first tick:")
    print(f"    cid_1: state={e1.sync_state:<10} transport={e1.transport}")
    print(f"    cid_2: state={e2.sync_state:<10} transport={e2.transport}")
    print(f"    cid_3: state={e3.sync_state:<10} transport={e3.transport}")
    print(f"    cid_4: state={e4.sync_state:<10} transport={e4.transport}")

    assert e1.sync_state == "pending"   and e1.transport == "wifi_rest"
    assert e2.sync_state == "synced"    and e2.transport == "sms"
    assert e3.sync_state == "synced"    and e3.transport == "sms"
    assert e4.sync_state == "synced"    and e4.transport == "sms"
    assert counts["left_for_rest"]            == 1
    assert counts["promoted_then_dispatched"] == 2
    assert counts["dispatched_sms"]           == 1

    # ===================================================================
    # SECOND TICK — Wi-Fi UP
    # ===================================================================
    print("\n" + "=" * 70)
    print("  SECOND TICK — Wi-Fi UP")
    print("  Expectations:")
    print("    • cid_1 still pending → left_for_rest (REST will serve)")
    print("=" * 70)

    arbiter.set_wifi_caregiver_connected(True)
    counts2 = engine.process_once()
    print(f"\n  Counts: {counts2}")

    e1_after = sync_repo.fetch_by_id(cid_1)
    print(f"  cid_1 after second tick: state={e1_after.sync_state} "
          f"transport={e1_after.transport}")

    assert e1_after.sync_state == "pending"
    assert counts2["left_for_rest"] == 1

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("\n" + "=" * 70)
    print("  GSM outbound log (SMS body sample, first dispatch):")
    print("=" * 70)
    sent = gsm.sent_log
    print(f"  Total SMS dispatched: {len(sent)}")
    if sent:
        first = sent[0]
        body  = first.body
        preview = body if len(body) <= 120 else body[:120] + "..."
        print(f"  → {first.recipient}")
        print(f"     {preview}")

    print("\n  All assertions passed. Outbound sync routing is wired correctly.")
    print("=" * 70)
