"""
sms_payload_handler.py
======================
Inbound SMS payload processor — the Hub-side terminus of the fallback
synchronisation pathway described in Sections 3.5.2 and 3.5.4 of the
project report.

The handler runs a background polling loop that periodically asks the
SIM800L for unread messages and processes each through a strict pipeline:

    1. Origin verification — sender must be a registered caregiver number;
       otherwise the message is discarded and logged as a security event.
    2. Schema parsing — the payload must conform to the delimited text
       format defined below.
    3. HMAC verification — the appended authentication tag is recomputed
       locally and compared against the carried tag.
    4. Idempotency — duplicate change_ids are rejected without reprocessing.
    5. Atomic application — the validated change is applied to the local
       SQLite database within a single transaction (achieved via the
       repository methods, which each open and commit their own connection).
    6. Acknowledgement — a confirmation is appended to the EventLog and a
       Hub→App ack record is queued in the SyncQueue.
    7. Cleanup — the processed SMS is deleted from the SIM800L's storage.

Payload Format (Mock Phase)
---------------------------
A pipe-delimited single-line text format optimised for the 160-character
single-segment SMS constraint:

    MED|<change_type>|<change_id>|<elder_id>|<drug_name>|<dosage>|<time_due>|<days_of_week>|<active>|<timestamp>|HMAC=<hex>

Example:
    MED|INSERT|7c3fa8b1-0a5e-4f5c-9e7d-b2a8c3f1d2e0|1|Paracetamol|500mg|08:00|DAILY|1|2026-04-28T09:30:00.000Z|HMAC=ab12cd34...

Fields:
    • Sentinel        — must be 'MED' (entity type marker; future expansion
                        may add 'CFG', 'PROF', etc.).
    • change_type     — INSERT | UPDATE | DELETE.
    • change_id       — UUID4 string used for idempotent rejection.
    • elder_id        — Foreign key into ElderProfile.
    • drug_name       — Free-form, no pipes.
    • dosage          — Free-form, no pipes.
    • time_due        — 'HH:MM' (24-hour).
    • days_of_week    — 'DAILY' or 'MON,TUE,...' (commas, not pipes).
    • active          — 0 or 1.
    • timestamp       — ISO-8601 UTC originating timestamp.
    • HMAC=<hex>      — Last field. Hex-encoded HMAC-SHA256 over the
                        canonical pipe-joined payload (all fields up to
                        but not including the HMAC marker), keyed with the
                        shared pairing-derived key from SystemConfig.

Mock-phase note: For terminal demos and unit tests, the HMAC check accepts
the literal string 'HMAC=MOCK' as a "blessed" tag bypassing crypto. This
behaviour is gated by the `dev_mock_hmac` constructor flag and MUST be
False in production. The standalone smoke test enables it for clarity.

Author: Wise (Asumang Pobi Godwin) — KNUST COE 497
"""

from __future__ import annotations

import hmac
import hashlib
import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

from src.data_management.repositories import (
    ElderProfileRepo,
    MedicationScheduleRepo,
    EventLogRepo,
    SyncQueueRepo,
    MedicationSchedule,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Configuration constants
# ===========================================================================

DEFAULT_POLL_INTERVAL_SECONDS = 30   # Per Section 3.5.2 of the report.
PAYLOAD_SENTINEL              = "MED"
HMAC_PREFIX                   = "HMAC="
DEV_MOCK_HMAC_TAG             = "HMAC=MOCK"   # Mock-phase bypass marker.


# ===========================================================================
# Pipeline rejection reasons
# ===========================================================================

@dataclass(frozen=True)
class RejectionReason:
    UNKNOWN_SENDER       = "unknown_sender"
    BAD_SCHEMA           = "bad_schema"
    INVALID_HMAC         = "invalid_hmac"
    DUPLICATE_CHANGE_ID  = "duplicate_change_id"
    APPLY_FAILED         = "apply_failed"
    UNSUPPORTED_ENTITY   = "unsupported_entity"
    UNSUPPORTED_CHANGE   = "unsupported_change_type"


# ===========================================================================
# Parsed-payload data class
# ===========================================================================

@dataclass
class ParsedPayload:
    sentinel:     str
    change_type:  str           # 'INSERT' | 'UPDATE' | 'DELETE'
    change_id:    str           # UUID4 from originator
    elder_id:     int
    drug_name:    str
    dosage:       str
    time_due:     str
    days_of_week: str
    active:       int
    timestamp:    str
    hmac_tag:     str           # The hex digest carried in the message
    canonical:    str           # The exact bytes that were HMAC'd


# ===========================================================================
# SMSPayloadHandler
# ===========================================================================

class SMSPayloadHandler:
    """
    Polls the GSM module on a background thread and runs each inbound
    message through the validation pipeline. Threading-safe and
    dependency-injected throughout.

    Lifecycle:
        handler = SMSPayloadHandler(gsm, hmac_key=...)
        handler.start()
        ...
        handler.stop()
    """

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(
        self,
        gsm: Any,                                      # MockGSMModule / GSMModule
        elder_repo: Optional[ElderProfileRepo]         = None,
        med_repo:   Optional[MedicationScheduleRepo]   = None,
        event_repo: Optional[EventLogRepo]             = None,
        sync_repo:  Optional[SyncQueueRepo]            = None,
        *,
        hmac_key: str = "",
        poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS,
        dev_mock_hmac: bool = False,
    ):
        """
        Parameters
        ----------
        gsm : injected GSM adapter exposing .read_unread_messages() and
              .delete_message(index).
        *_repo : repository instances; default-constructed if omitted.
        hmac_key : the shared key derived from the pairing token during
                   initial setup (Section 3.5.5). For real deployments this
                   is loaded from SystemConfig at boot.
        poll_interval_seconds : how often the polling thread wakes.
        dev_mock_hmac : if True, the literal tag 'HMAC=MOCK' bypasses the
                        cryptographic check. MUST be False in production.
                        Used by the smoke test and unit tests.
        """
        self._gsm         = gsm
        self._elder_repo  = elder_repo  or ElderProfileRepo()
        self._med_repo    = med_repo    or MedicationScheduleRepo()
        self._event_repo  = event_repo  or EventLogRepo()
        self._sync_repo   = sync_repo   or SyncQueueRepo()

        self._hmac_key            = hmac_key.encode("utf-8") if hmac_key else b""
        self._poll_interval       = poll_interval_seconds
        self._dev_mock_hmac       = dev_mock_hmac

        # Threading primitives.
        self._stop_event = threading.Event()
        self._thread:    Optional[threading.Thread] = None
        # Serializes the entire pipeline so a manual .process_once() call
        # cannot race the polling thread on the same message.
        self._pipeline_lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Begin polling on a daemon thread. Idempotent."""
        if self._thread and self._thread.is_alive():
            logger.warning("SMSPayloadHandler already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="SMSPayloadHandler",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "SMSPayloadHandler started (poll interval = %ds, dev_mock_hmac=%s).",
            self._poll_interval, self._dev_mock_hmac,
        )

    def stop(self, join_timeout: float = 5.0) -> None:
        """Signal the polling thread to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=join_timeout)
        logger.info("SMSPayloadHandler stopped.")

    # -----------------------------------------------------------------------
    # Polling loop
    # -----------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Wake every poll_interval, drain the inbox, sleep again."""
        # Small initial delay so we don't hammer the SIM module immediately
        # on boot while other subsystems are still initialising.
        self._stop_event.wait(timeout=2.0)

        while not self._stop_event.is_set():
            try:
                self.process_once()
            except Exception:
                logger.exception("SMS poll cycle raised; continuing.")
                self._event_repo.insert(
                    EventLogRepo.SYSTEM_FAULT,
                    details={
                        "subsystem": "SMSPayloadHandler",
                        "stage":     "poll_cycle",
                    },
                )

            self._stop_event.wait(timeout=self._poll_interval)

    # -----------------------------------------------------------------------
    # Public single-pass entry point — drains every unread message
    # -----------------------------------------------------------------------

    def process_once(self) -> int:
        """
        Read all unread messages from the GSM module and run each through
        the pipeline. Returns the number of messages processed (accepted
        OR rejected — 'processed' here means 'reached a terminal state').

        Exposed publicly so tests and dev tooling can drive the pipeline
        synchronously without spinning up the background thread.
        """
        with self._pipeline_lock:
            try:
                messages = self._gsm.read_unread_messages()
            except Exception:
                logger.exception("Failed to read unread messages from GSM module.")
                return 0

            count = 0
            for msg in messages:
                if self._stop_event.is_set():
                    break
                self._process_message(msg)
                count += 1
            return count

    # -----------------------------------------------------------------------
    # Per-message pipeline
    # -----------------------------------------------------------------------

    def _process_message(self, msg: Any) -> None:
        """
        Run one inbound message through all pipeline stages. Each terminal
        outcome — accepted, rejected, applied — is logged exactly once,
        and the message is deleted from SIM storage in every case to
        prevent reprocessing.

        Per the report (Section 3.5.2): "Messages that fail authentication
        or schema validation are silently discarded, logged as security
        events in the EventLog, and removed from the module's storage to
        prevent reprocessing." We follow that exactly.
        """
        sender = getattr(msg, "sender", None)
        body   = getattr(msg, "body",   None)
        index  = getattr(msg, "index",  None)

        # ----- Stage 1: Origin verification ---------------------------------
        if not self._is_known_caregiver(sender):
            self._reject(
                index, sender, body,
                reason=RejectionReason.UNKNOWN_SENDER,
                detail={"sender": sender},
            )
            return

        # ----- Stage 2: Schema parsing --------------------------------------
        parsed, parse_error = self._parse_payload(body)
        if parsed is None:
            self._reject(
                index, sender, body,
                reason=RejectionReason.BAD_SCHEMA,
                detail={"sender": sender, "parse_error": parse_error},
            )
            return

        # ----- Stage 3: HMAC verification -----------------------------------
        if not self._verify_hmac(parsed):
            self._reject(
                index, sender, body,
                reason=RejectionReason.INVALID_HMAC,
                detail={
                    "sender":    sender,
                    "change_id": parsed.change_id,
                },
            )
            return

        # ----- Stage 4: Idempotency check -----------------------------------
        if self._sync_repo.is_duplicate(parsed.change_id):
            # Per Section 3.5.2: "receipt of a duplicate payload bearing
            # the same change_id is detected and ignored". This is a
            # benign condition (likely retransmission of a delayed SMS),
            # so we log it at INFO rather than as a security event.
            self._event_repo.insert(
                EventLogRepo.SMS_PAYLOAD_REJECTED,
                details={
                    "sender":    sender,
                    "change_id": parsed.change_id,
                    "reason":    RejectionReason.DUPLICATE_CHANGE_ID,
                },
            )
            self._safe_delete(index)
            return

        # ----- Stage 5: Validate change_type / sentinel ---------------------
        if parsed.sentinel != PAYLOAD_SENTINEL:
            self._reject(
                index, sender, body,
                reason=RejectionReason.UNSUPPORTED_ENTITY,
                detail={"sentinel": parsed.sentinel},
            )
            return

        if parsed.change_type not in ("INSERT", "UPDATE", "DELETE"):
            self._reject(
                index, sender, body,
                reason=RejectionReason.UNSUPPORTED_CHANGE,
                detail={"change_type": parsed.change_type},
            )
            return

        # ----- Stage 6: Atomic application ----------------------------------
        try:
            schedule_id = self._apply_change(parsed)
        except Exception:
            logger.exception(
                "Apply-change raised for change_id=%s", parsed.change_id
            )
            self._reject(
                index, sender, body,
                reason=RejectionReason.APPLY_FAILED,
                detail={"change_id": parsed.change_id},
            )
            return

        # ----- Stage 7: Acknowledge + cleanup -------------------------------
        # Record the inbound change_id in the SyncQueue with sync_state =
        # 'synced' so future duplicates of the same change_id are detected
        # by is_duplicate(). Also enqueues a Hub→App ack so the caregiver
        # app learns its update was applied.
        self._record_acknowledgement(parsed, schedule_id, sender)

        self._event_repo.insert(
            EventLogRepo.SMS_PAYLOAD_ACCEPTED,
            details={
                "sender":       sender,
                "change_id":    parsed.change_id,
                "change_type":  parsed.change_type,
                "schedule_id":  schedule_id,
                "drug_name":    parsed.drug_name,
                "time_due":     parsed.time_due,
            },
        )
        logger.info(
            "SMS payload accepted: change_id=%s schedule_id=%s",
            parsed.change_id, schedule_id,
        )

        self._safe_delete(index)

    # -----------------------------------------------------------------------
    # Stage helpers
    # -----------------------------------------------------------------------

    def _is_known_caregiver(self, sender: Optional[str]) -> bool:
        if not sender:
            return False
        try:
            return self._elder_repo.is_caregiver_number(sender)
        except Exception:
            logger.exception("Caregiver-number lookup failed for %r", sender)
            return False

    def _parse_payload(self, body: Optional[str]) -> Tuple[Optional[ParsedPayload], Optional[str]]:
        """
        Parse the pipe-delimited payload. Returns (parsed, None) on success,
        (None, error_string) on failure.
        """
        if not body or not isinstance(body, str):
            return None, "empty_or_non_string_body"

        body_stripped = body.strip()
        parts = body_stripped.split("|")

        # We expect 11 fields: sentinel + 9 content fields + HMAC=...
        if len(parts) != 11:
            return None, f"expected 11 pipe-separated fields, got {len(parts)}"

        sentinel = parts[0]

        hmac_field = parts[-1]
        if not hmac_field.startswith(HMAC_PREFIX):
            return None, "missing HMAC= prefix in last field"
        hmac_tag = hmac_field[len(HMAC_PREFIX):]
        if not hmac_tag:
            return None, "empty HMAC tag"

        # Reconstruct the canonical bytes that the originator signed:
        # everything up to (but not including) the HMAC field, joined with '|'.
        canonical = "|".join(parts[:-1])

        try:
            # Field layout (parts indices):
            # 0: sentinel  | 1: change_type | 2: change_id | 3: elder_id
            # 4: drug_name | 5: dosage      | 6: time_due  | 7: days_of_week
            # 8: active    | 9: timestamp   |10: HMAC=...
            elder_id = int(parts[3])
            active   = int(parts[8])
        except ValueError as e:
            return None, f"non-integer field: {e}"

        # Light validation — pathological values are caught here, not in
        # apply_change, so we can reject them before mutating state.
        if active not in (0, 1):
            return None, f"active must be 0 or 1, got {active}"

        if not parts[2]:  # change_id
            return None, "empty change_id"

        # Validate change_id is a plausible UUID; we don't enforce v4
        # specifically because the report doesn't require it.
        try:
            uuid.UUID(parts[2])
        except ValueError:
            return None, "change_id is not a valid UUID"

        return ParsedPayload(
            sentinel=sentinel,
            change_type=parts[1],
            change_id=parts[2],
            elder_id=elder_id,
            drug_name=parts[4],
            dosage=parts[5],
            time_due=parts[6],
            days_of_week=parts[7],
            active=active,
            timestamp=parts[9],
            hmac_tag=hmac_tag,
            canonical=canonical,
        ), None

    def _verify_hmac(self, parsed: ParsedPayload) -> bool:
        """
        Compare the recomputed HMAC-SHA256 against the carried tag using
        constant-time comparison (hmac.compare_digest). If dev_mock_hmac
        is True AND the carried tag is exactly 'MOCK', we accept the
        message — strictly for terminal demos and unit tests.
        """
        if self._dev_mock_hmac and parsed.hmac_tag == "MOCK":
            logger.debug("HMAC bypass: dev_mock_hmac accepted MOCK tag.")
            return True

        if not self._hmac_key:
            logger.error(
                "HMAC verification requested but no key configured — rejecting."
            )
            return False

        try:
            expected = hmac.new(
                self._hmac_key,
                parsed.canonical.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
        except Exception:
            logger.exception("HMAC computation failed.")
            return False

        return hmac.compare_digest(expected, parsed.hmac_tag)

    def _apply_change(self, parsed: ParsedPayload) -> int:
        """
        Apply the parsed change to MedicationSchedule. Returns the
        affected schedule_id.

        Atomicity: each repository method opens its own connection and
        commits a single transaction, so the apply step is atomic at the
        SQLite level. If any step fails, the connection rolls back and
        no partial state is persisted.
        """
        sched = MedicationSchedule(
            elder_id=parsed.elder_id,
            drug_name=parsed.drug_name,
            dosage=parsed.dosage,
            time_due=parsed.time_due,
            days_of_week=parsed.days_of_week,
            active=parsed.active,
        )

        if parsed.change_type == "INSERT":
            return self._med_repo.upsert_from_payload(
                sched,
                sync_method="app_sms",
                prescribed_by="pharmacist",  # SMS pathway is the pharmacist channel
            )

        if parsed.change_type == "UPDATE":
            # Same as INSERT semantically — upsert_from_payload matches on
            # the natural key (elder_id, drug_name, time_due, days_of_week)
            # and updates if found, inserts otherwise.
            return self._med_repo.upsert_from_payload(
                sched,
                sync_method="app_sms",
                prescribed_by="pharmacist",
            )

        if parsed.change_type == "DELETE":
            # Find the existing schedule by natural key, then deactivate
            # (soft delete). If nothing matches, this is a no-op but still
            # treated as successful — duplicate DELETE is idempotent by design.
            existing = self._find_existing_schedule(sched)
            if existing is None:
                logger.info(
                    "DELETE payload had no matching schedule — treating as no-op."
                )
                return -1  # Sentinel: "no matching record"
            self._med_repo.deactivate(existing.schedule_id)
            return existing.schedule_id

        # Should be unreachable — gated upstream.
        raise ValueError(f"Unsupported change_type: {parsed.change_type}")

    def _find_existing_schedule(
        self, sched: MedicationSchedule
    ) -> Optional[MedicationSchedule]:
        """
        Resolve a schedule by natural key for DELETE handling. Uses the
        all-active list filtered in Python — fine for the small per-elder
        schedule counts (typically <20) we're dealing with.
        """
        for s in self._med_repo.fetch_all_active(elder_id=sched.elder_id):
            if (
                s.drug_name == sched.drug_name
                and s.time_due == sched.time_due
                and s.days_of_week == sched.days_of_week
            ):
                return s
        return None

    def _record_acknowledgement(
        self,
        parsed: ParsedPayload,
        schedule_id: int,
        sender: str,
    ) -> None:
        """
        Two SyncQueue records are written here, doing two distinct jobs:

        (a) An App→Hub record using the originator's change_id, marked
            as already 'synced'. This is what makes future duplicates of
            the same change_id detectable via is_duplicate().

        (b) A Hub→App record carrying the acknowledgement back to the
            caregiver app on the next sync window — Section 3.5.2:
            "a corresponding Hub→App acknowledgement record is queued
             in the SyncQueue for transmission during the next Wi-Fi window."
        """
        # (a) Mark the originator's change as synced. We use the raw enqueue
        # API because we want to preserve the originator's change_id verbatim.
        from src.data_management.repositories import SyncQueueEntry
        try:
            self._sync_repo.enqueue(SyncQueueEntry(
                change_id=parsed.change_id,
                entity_type="MedicationSchedule",
                entity_id=schedule_id,
                change_type=parsed.change_type,
                timestamp=parsed.timestamp,
                sync_state="synced",         # Already applied — don't re-send.
                direction="App->Hub",
                transport="sms",
                payload=parsed.canonical,
            ))
        except Exception:
            # Most common cause: race with is_duplicate() if two threads
            # process the same message. Benign — the duplicate-detection
            # invariant is preserved by the PK constraint on change_id.
            logger.exception(
                "Failed to record App->Hub change_id=%s (likely benign duplicate).",
                parsed.change_id,
            )

        # (b) Hub→App acknowledgement — fresh change_id of our own.
        # We use change_type='UPDATE' because the SyncQueue CHECK constraint
        # only permits INSERT/UPDATE/DELETE. The fact that this record is
        # specifically an acknowledgement is encoded in the payload's
        # 'ack_for_change_id' field, which the caregiver app uses to match
        # the ack back to the originating App→Hub change.
        try:
            self._sync_repo.enqueue_change(
                entity_type="MedicationSchedule",
                entity_id=schedule_id,
                change_type="UPDATE",
                direction="Hub->App",
                transport="wifi_rest",       # Acks prefer the higher-bandwidth pathway.
                payload={
                    "ack_for_change_id": parsed.change_id,
                    "schedule_id":       schedule_id,
                    "applied_at":        _utcnow_iso(),
                    "ack":               True,   # Explicit marker for the app's parser.
                },
            )
        except Exception:
            logger.exception("Failed to enqueue Hub->App ack.")

    # -----------------------------------------------------------------------
    # Rejection + cleanup helpers
    # -----------------------------------------------------------------------

    def _reject(
        self,
        index: Optional[int],
        sender: Optional[str],
        body: Optional[str],
        *,
        reason: str,
        detail: dict,
    ) -> None:
        """
        Log a rejection event and delete the offending message from SIM
        storage. The full body is NOT logged for messages rejected on
        origin/HMAC grounds — those are presumed potentially adversarial,
        and we don't want to immortalise their contents in the audit log.
        """
        log_details = {"reason": reason, **detail}
        if reason == RejectionReason.BAD_SCHEMA:
            # Bad schema is most often a bug, not an attack — include a
            # bounded preview of the body for debugging.
            preview = (body or "")[:80]
            log_details["body_preview"] = preview

        self._event_repo.insert(
            EventLogRepo.SMS_PAYLOAD_REJECTED,
            details=log_details,
        )
        logger.warning(
            "SMS payload rejected: reason=%s sender=%s", reason, sender
        )
        self._safe_delete(index)

    def _safe_delete(self, index: Optional[int]) -> None:
        """Delete a message from SIM storage; failures are logged, not raised."""
        if index is None:
            return
        try:
            self._gsm.delete_message(index)
        except Exception:
            logger.exception("Failed to delete SMS at index %s", index)


# ===========================================================================
# Module-level helpers
# ===========================================================================

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def build_signed_payload(
    *,
    change_type: str,
    change_id: Optional[str],
    elder_id: int,
    drug_name: str,
    dosage: str,
    time_due: str,
    days_of_week: str = "DAILY",
    active: int = 1,
    hmac_key: str = "",
    use_mock_hmac: bool = False,
) -> str:
    """
    Helper for building a correctly-formatted SMS payload from the
    caregiver-app side. Used by the smoke test below and reusable by
    later integration tests for the sync engine.

    If use_mock_hmac is True, the literal tag 'MOCK' is appended instead
    of a real HMAC. The handler must be configured with dev_mock_hmac=True
    to accept such payloads.
    """
    cid = change_id or str(uuid.uuid4())
    ts = _utcnow_iso()
    canonical = "|".join([
        PAYLOAD_SENTINEL, change_type, cid, str(elder_id),
        drug_name, dosage, time_due, days_of_week,
        str(active), ts,
    ])

    if use_mock_hmac:
        tag = "MOCK"
    else:
        if not hmac_key:
            raise ValueError("hmac_key is required when use_mock_hmac=False")
        tag = hmac.new(
            hmac_key.encode("utf-8"),
            canonical.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    return canonical + "|" + HMAC_PREFIX + tag


# ===========================================================================
# Standalone smoke test
# ===========================================================================

if __name__ == "__main__":
    """
    Run from the project root:

        python -m src.control_logic.sms_payload_handler

    Demonstrates the full pipeline across five scenarios:
      1. Valid payload from a registered caregiver  → ACCEPTED.
      2. Same payload again (duplicate change_id)   → REJECTED (idempotent).
      3. Payload from an unknown number             → REJECTED (origin).
      4. Malformed payload                          → REJECTED (schema).
      5. Real-HMAC mode payload (correctly signed)  → ACCEPTED.
    """
    import logging
    from src.hardware_mocks.mock_sim800l  import MockGSMModule
    from src.data_management.repositories import (
        ElderProfileRepo, ElderProfile,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(name)s — %(message)s",
    )

    # --- Seed an elder with two caregiver phones ------------------------
    elder_repo = ElderProfileRepo()
    med_repo   = MedicationScheduleRepo()
    event_repo = EventLogRepo()

    CAREGIVER_PHONE = "+233244111111"
    UNKNOWN_PHONE   = "+233200000000"

    elder = elder_repo.fetch_first()
    if elder is None:
        elder_id = elder_repo.insert(ElderProfile(
            name="Demo Elder",
            language="twi",
            caregiver_phones=f"{CAREGIVER_PHONE},+233244222222",
        ))
    else:
        elder_id = elder.elder_id
        elder_repo.update_caregiver_phones(
            elder_id, f"{CAREGIVER_PHONE},+233244222222"
        )

    gsm = MockGSMModule()

    # --- Scenario 1: Valid payload (mock-HMAC mode) --------------------
    handler_mock_mode = SMSPayloadHandler(
        gsm=gsm,
        hmac_key="",                 # no real key needed in mock mode
        dev_mock_hmac=True,
    )

    payload_1 = build_signed_payload(
        change_type="INSERT",
        change_id=str(uuid.uuid4()),
        elder_id=elder_id,
        drug_name="Paracetamol",
        dosage="500mg",
        time_due="08:00",
        days_of_week="DAILY",
        active=1,
        use_mock_hmac=True,
    )
    print("\n" + "=" * 70)
    print("  SCENARIO 1 — Valid payload from registered caregiver")
    print("=" * 70)
    gsm.inject_inbound_sms(CAREGIVER_PHONE, payload_1)
    handler_mock_mode.process_once()

    # --- Scenario 2: Duplicate change_id ------------------------------
    print("\n" + "=" * 70)
    print("  SCENARIO 2 — Replay of the same payload (duplicate change_id)")
    print("=" * 70)
    gsm.inject_inbound_sms(CAREGIVER_PHONE, payload_1)
    handler_mock_mode.process_once()

    # --- Scenario 3: Unknown sender -----------------------------------
    payload_3 = build_signed_payload(
        change_type="INSERT",
        change_id=str(uuid.uuid4()),
        elder_id=elder_id,
        drug_name="Aspirin",
        dosage="100mg",
        time_due="09:00",
        days_of_week="DAILY",
        active=1,
        use_mock_hmac=True,
    )
    print("\n" + "=" * 70)
    print("  SCENARIO 3 — Same payload sent from an UNKNOWN number")
    print("=" * 70)
    gsm.inject_inbound_sms(UNKNOWN_PHONE, payload_3)
    handler_mock_mode.process_once()

    # --- Scenario 4: Malformed payload --------------------------------
    print("\n" + "=" * 70)
    print("  SCENARIO 4 — Malformed payload (missing fields)")
    print("=" * 70)
    gsm.inject_inbound_sms(CAREGIVER_PHONE, "MED|INSERT|broken")
    handler_mock_mode.process_once()

    # --- Scenario 5: Real HMAC end-to-end -----------------------------
    REAL_KEY = "shared-secret-derived-from-pairing-token"
    handler_real_mode = SMSPayloadHandler(
        gsm=gsm,
        hmac_key=REAL_KEY,
        dev_mock_hmac=False,
    )
    payload_5 = build_signed_payload(
        change_type="INSERT",
        change_id=str(uuid.uuid4()),
        elder_id=elder_id,
        drug_name="Lisinopril",
        dosage="10mg",
        time_due="20:00",
        days_of_week="DAILY",
        active=1,
        hmac_key=REAL_KEY,
        use_mock_hmac=False,
    )
    print("\n" + "=" * 70)
    print("  SCENARIO 5 — Correctly-signed real-HMAC payload")
    print("=" * 70)
    gsm.inject_inbound_sms(CAREGIVER_PHONE, payload_5)
    handler_real_mode.process_once()

    # --- Final inspection ---------------------------------------------
    print("\n" + "=" * 70)
    print("  FINAL STATE")
    print("=" * 70)
    actives = med_repo.fetch_all_active(elder_id=elder_id)
    print(f"  Active medications for elder_id={elder_id}: {len(actives)}")
    for s in actives:
        print(f"    • [{s.schedule_id}] {s.drug_name} {s.dosage} @ {s.time_due} "
              f"(prescribed_by={s.prescribed_by}, sync_method={s.sync_method})")

    print("\n  Recent EventLog (last 10):")
    for ev in event_repo.fetch_recent(limit=10):
        print(f"    • [{ev.event_id}] {ev.event_type}  ({ev.timestamp})")
        if ev.details:
            preview = ev.details if len(ev.details) <= 100 else ev.details[:100] + "..."
            print(f"        {preview}")

    print("\n  Unread inbox: %d (should be 0 — all messages were deleted after processing)"
          % gsm.storage_used())
    print("=" * 70)
