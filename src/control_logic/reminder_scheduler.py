"""
reminder_scheduler.py
=====================
The medication-reminder orchestrator. Runs a cron-style scheduling loop on
a background thread that drives the Voice Engine and the Data Management
Layer to deliver, confirm, and log medication doses for the elder.

Implements Section 3.5.2 of the project report — Control Logic Layer:
    "The reminder scheduler operates as a cron-style scheduling loop that
     queries the SQLite database at regular intervals for reminders due
     within the next 60 seconds, generates the appropriate Twi spoken prompt
     via the TTS engine, waits for a confirmation response from the Voice
     Engine or a configurable timeout of 120 seconds, and logs the outcome
     as confirmed, missed, or timed out. If a reminder is missed, the
     scheduler re-prompts after a 15-minute interval for up to three
     additional attempts, after which the event is logged as a definitive
     missed dose and a notification is queued for the caregiver application."

Design Notes
------------
1. Dependency injection: every external dependency — the speaker, the
   microphone, the repositories — is passed in via the constructor.
   This is what enables the integration tests to substitute mocks today
   and the real Vosk/Piper drivers in Kumasi tomorrow without modifying
   the scheduler.

2. Threading model: ONE main scheduler thread polls for due meds at a
   fixed cadence (default 30s — half the lookahead window, so we can never
   miss a reminder by skipping over it). For each reminder fired, a
   dedicated worker thread runs the prompt-confirm-retry state machine
   so that a long-running confirmation wait never blocks the next tick.

3. State tracking: a thread-safe set of in-flight (schedule_id, dose_date)
   pairs prevents the same dose from being prompted twice if the polling
   interval and the lookahead window overlap a single dose's time_due.

4. Voice engine coupling: the scheduler subscribes to the microphone's
   command callback. When a worker is awaiting confirmation, it sets a
   per-worker threading.Event that the callback wakes on receipt of a
   DOSE_CONFIRMED or DOSE_MISSED action.

5. Sync queue integration: confirmed and missed doses are also written to
   the SyncQueue with direction='Hub->App' so the caregiver app receives
   them on the next sync window (Section 3.5.4).

Author: Wise (Asumang Pobi Godwin) — KNUST COE 497
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Set, Tuple, Any

from src.data_management.repositories import (
    MedicationScheduleRepo,
    EventLogRepo,
    SyncQueueRepo,
    MedicationSchedule,
)
# CommandEvent + action constants live with the microphone mock today and
# with the real keyword spotter tomorrow. The import path is symmetrical
# in both worlds, so the swap is a single line in main.py.
from src.hardware_mocks.mock_microphone import (
    CommandEvent,
    ACTION_DOSE_CONFIRMED,
    ACTION_DOSE_MISSED,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Configuration constants — defaults from Section 3.5.2 of the report
# ===========================================================================

DEFAULT_POLL_INTERVAL_SECONDS  = 30      # How often we scan for due meds.
DEFAULT_LOOKAHEAD_SECONDS      = 60      # Per Section 3.5.2.
DEFAULT_CONFIRMATION_TIMEOUT   = 120     # Per Section 3.5.2.
DEFAULT_RETRY_INTERVAL_SECONDS = 15 * 60 # 15 minutes per Section 3.5.2.
DEFAULT_MAX_ADDITIONAL_RETRIES = 3       # Per Section 3.5.2.


# ===========================================================================
# Outcome enum — what happened to a single reminder cycle
# ===========================================================================

@dataclass(frozen=True)
class ReminderOutcome:
    CONFIRMED         = "confirmed"          # Elder said "Yε, mafa m'aduru"
    EXPLICITLY_MISSED = "explicitly_missed"  # Elder said "Mfaa m'aduru nkaa"
    TIMEOUT           = "timeout"            # No response within 120s
    DEFINITIVE_MISSED = "definitive_missed"  # All retries exhausted


# ===========================================================================
# Helper: per-dose key used for in-flight tracking
# ===========================================================================

def _dose_key(schedule_id: int, when: datetime) -> Tuple[int, str]:
    """
    Compose a unique key for one (schedule, calendar-day) pair.
    Prevents duplicate prompts for the same dose on the same day.
    """
    return (schedule_id, when.strftime("%Y-%m-%d"))


# ===========================================================================
# Twi prompt builders — single source of truth for prompt phrasing
# ===========================================================================
# NOTE FOR THE PANEL: these prompts are placeholders. The final phrasing
# will be validated and refined during the User Needs Assessment phase
# (Section 3.2 of the report). Centralizing them here means the validated
# wording can be dropped in without touching scheduling logic.

def _build_initial_prompt(sched: MedicationSchedule) -> str:
    return (
        f"Akwaaba. Ɛyɛ bere a wo bɛnom wo aduru. "
        f"Fa {sched.drug_name}, {sched.dosage}. "
        f"Sɛ wama wonom a, ka sɛ: Yɛ, mafa m'aduru."
    )

def _build_retry_prompt(sched: MedicationSchedule, attempt: int) -> str:
    return (
        f"Mekae bio. Wonom wo {sched.drug_name} {sched.dosage} a? "
        f"Sɛ woanom a, mesrɛ wo nom no seesei."
    )

def _build_final_missed_prompt(sched: MedicationSchedule) -> str:
    return (
        f"Ɔhwɛfoɔ no bɛhwɛ wo. Yɛakyerɛw sɛ wonnom {sched.drug_name} ɛnnɛ."
    )


# ===========================================================================
# ReminderScheduler
# ===========================================================================

class ReminderScheduler:
    """
    Cron-style reminder driver. Threading-safe and dependency-injected.

    Lifecycle:
        scheduler = ReminderScheduler(speaker, microphone, ...)
        microphone.on_command_callback = scheduler.on_voice_command  # OR pass via mic ctor
        scheduler.start()
        ...
        scheduler.stop()
    """

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(
        self,
        speaker: Any,                                 # MockSpeaker / TTSEngine
        microphone: Any,                              # MockMicrophone / KeywordSpotter
        med_repo: Optional[MedicationScheduleRepo] = None,
        event_repo: Optional[EventLogRepo] = None,
        sync_repo: Optional[SyncQueueRepo] = None,
        *,
        poll_interval_seconds:    int = DEFAULT_POLL_INTERVAL_SECONDS,
        lookahead_seconds:        int = DEFAULT_LOOKAHEAD_SECONDS,
        confirmation_timeout_sec: int = DEFAULT_CONFIRMATION_TIMEOUT,
        retry_interval_seconds:   int = DEFAULT_RETRY_INTERVAL_SECONDS,
        max_additional_retries:   int = DEFAULT_MAX_ADDITIONAL_RETRIES,
        elder_id: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        speaker, microphone : injected hardware adapters.
            Must expose .speak(text) and a callback registration mechanism
            respectively. Using duck-typing rather than ABCs keeps the
            mock/real boundary maximally flexible.
        *_repo : repository instances. Default-constructed if omitted.
        poll_interval_seconds : how often to scan for due meds.
        lookahead_seconds : window passed to fetch_due_within().
        confirmation_timeout_sec : per-prompt confirmation wait.
        retry_interval_seconds : delay between missed-dose retries.
        max_additional_retries : retries AFTER the initial prompt.
        elder_id : restrict scheduling to a single elder. None = all elders.
        """
        self._speaker     = speaker
        self._microphone  = microphone
        self._med_repo    = med_repo   or MedicationScheduleRepo()
        self._event_repo  = event_repo or EventLogRepo()
        self._sync_repo   = sync_repo  or SyncQueueRepo()

        self._poll_interval        = poll_interval_seconds
        self._lookahead            = lookahead_seconds
        self._confirmation_timeout = confirmation_timeout_sec
        self._retry_interval       = retry_interval_seconds
        self._max_retries          = max_additional_retries
        self._elder_id             = elder_id

        # Threading primitives.
        self._stop_event       = threading.Event()
        self._main_thread:    Optional[threading.Thread] = None
        self._workers:        Set[threading.Thread] = set()
        self._workers_lock    = threading.Lock()

        # In-flight dose tracking — prevents duplicate prompts for the
        # same (schedule, day) pair. The key is (schedule_id, 'YYYY-MM-DD').
        self._in_flight:      Set[Tuple[int, str]] = set()
        self._in_flight_lock  = threading.Lock()

        # Confirmation routing: maps schedule_id → (Event, last_action).
        # The voice callback writes here; the worker reads from here.
        self._pending_confirmations: dict[int, _PendingConfirmation] = {}
        self._pending_lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Begin polling on a daemon thread. Idempotent."""
        if self._main_thread and self._main_thread.is_alive():
            logger.warning("ReminderScheduler already running.")
            return

        self._stop_event.clear()
        self._main_thread = threading.Thread(
            target=self._run_loop,
            name="ReminderScheduler-Main",
            daemon=True,
        )
        self._main_thread.start()
        self._event_repo.insert(
            EventLogRepo.SYSTEM_BOOT,
            details={
                "subsystem": "ReminderScheduler",
                "poll_interval_s":  self._poll_interval,
                "lookahead_s":      self._lookahead,
                "confirm_timeout_s":self._confirmation_timeout,
                "retry_interval_s": self._retry_interval,
                "max_retries":      self._max_retries,
            },
        )
        logger.info("ReminderScheduler started.")

    def stop(self, join_timeout: float = 5.0) -> None:
        """
        Signal the scheduler to stop. Workers in their retry sleep are
        woken via the same stop_event so shutdown is prompt.
        """
        self._stop_event.set()

        # Unblock any worker waiting on a confirmation Event.
        with self._pending_lock:
            for pending in self._pending_confirmations.values():
                pending.event.set()

        if self._main_thread:
            self._main_thread.join(timeout=join_timeout)

        # Best-effort wait for in-flight workers; they're daemons, so the
        # process can still exit even if a few are mid-retry-sleep.
        with self._workers_lock:
            workers_snapshot = list(self._workers)
        for w in workers_snapshot:
            w.join(timeout=join_timeout)

        logger.info("ReminderScheduler stopped.")

    # -----------------------------------------------------------------------
    # Voice-engine callback — wired in from main.py
    # -----------------------------------------------------------------------

    def on_voice_command(self, event: CommandEvent) -> None:
        """
        Receive a CommandEvent from the keyword spotter. If any worker is
        currently awaiting confirmation AND the action is one of the
        dose-related commands, route the action to the oldest pending
        worker and wake it.

        Routing policy: oldest-first. If multiple reminders are
        simultaneously awaiting confirmation (e.g. two meds at the same
        time_due), the elder's response is applied to the one prompted
        first. This is a deliberate simplification — true multi-prompt
        disambiguation would require slot-fill dialogue which is out of
        scope for the constrained-keyword design (Section 2.3).
        """
        if event.action not in (ACTION_DOSE_CONFIRMED, ACTION_DOSE_MISSED):
            # Not a dose-confirmation command — ignored here. SOS, appliance
            # commands, etc. are handled by other modules subscribed to the
            # same callback chain in main.py.
            return

        with self._pending_lock:
            if not self._pending_confirmations:
                logger.debug(
                    "Voice command %s received but no reminder is awaiting "
                    "confirmation — ignored.", event.action
                )
                return

            # Oldest-first by issue_time.
            oldest_sid = min(
                self._pending_confirmations,
                key=lambda sid: self._pending_confirmations[sid].issued_at,
            )
            pending = self._pending_confirmations[oldest_sid]
            pending.last_action = event.action
            pending.event.set()
            logger.info(
                "Voice command %s routed to schedule_id=%d", event.action, oldest_sid
            )

    # -----------------------------------------------------------------------
    # Main polling loop
    # -----------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Poll for due meds; spawn a worker for each one not already in flight."""
        # Initial small jitter so we don't spam the DB the instant we boot.
        self._stop_event.wait(timeout=1.0)

        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                # A repo failure or a systemic error must NEVER kill the
                # scheduler. Log and continue.
                logger.exception("ReminderScheduler tick raised; continuing.")
                self._event_repo.insert(
                    EventLogRepo.SYSTEM_FAULT,
                    details={"subsystem": "ReminderScheduler", "stage": "tick"},
                )

            # Wait either for the next tick or for the stop signal.
            self._stop_event.wait(timeout=self._poll_interval)

    def _tick(self) -> None:
        """One scan of the due-window."""
        now = datetime.now()

        due = self._med_repo.fetch_due_within(
            lookahead_seconds=self._lookahead,
            current_time=now,
        )

        if self._elder_id is not None:
            due = [s for s in due if s.elder_id == self._elder_id]

        if not due:
            return

        for sched in due:
            key = _dose_key(sched.schedule_id, now)

            with self._in_flight_lock:
                if key in self._in_flight:
                    # Already being handled by a worker — skip.
                    continue
                self._in_flight.add(key)

            self._spawn_worker(sched, key, now)

    def _spawn_worker(
        self,
        sched: MedicationSchedule,
        key: Tuple[int, str],
        issued_at: datetime,
    ) -> None:
        """Launch a daemon worker thread to handle one dose end-to-end."""
        thread = threading.Thread(
            target=self._dose_worker,
            args=(sched, key, issued_at),
            name=f"Reminder-Worker-sched{sched.schedule_id}",
            daemon=True,
        )
        with self._workers_lock:
            self._workers.add(thread)
        thread.start()

    # -----------------------------------------------------------------------
    # Per-dose worker — runs the prompt + confirmation + retry state machine
    # -----------------------------------------------------------------------

    def _dose_worker(
        self,
        sched: MedicationSchedule,
        key: Tuple[int, str],
        issued_at: datetime,
    ) -> None:
        """
        Drive a single dose through up to (1 + max_retries) prompt cycles.
        Final outcome is one of:
            CONFIRMED, EXPLICITLY_MISSED, DEFINITIVE_MISSED
        """
        try:
            outcome = ReminderOutcome.DEFINITIVE_MISSED  # Pessimistic default

            # Total attempts = 1 initial + N retries.
            total_attempts = 1 + self._max_retries

            for attempt_index in range(total_attempts):
                if self._stop_event.is_set():
                    logger.info(
                        "Worker for schedule_id=%d exiting due to shutdown",
                        sched.schedule_id,
                    )
                    return

                # 1. Build & speak the prompt.
                if attempt_index == 0:
                    prompt = _build_initial_prompt(sched)
                else:
                    prompt = _build_retry_prompt(sched, attempt_index)

                self._log_event(
                    EventLogRepo.REMINDER_ISSUED,
                    {
                        "schedule_id":  sched.schedule_id,
                        "drug_name":    sched.drug_name,
                        "dosage":       sched.dosage,
                        "attempt":      attempt_index + 1,
                        "total_attempts": total_attempts,
                    },
                )
                self._speaker.speak(prompt)

                # 2. Await confirmation.
                action = self._await_confirmation(sched.schedule_id)

                # 3. Handle outcome.
                if action == ACTION_DOSE_CONFIRMED:
                    outcome = ReminderOutcome.CONFIRMED
                    break

                if action == ACTION_DOSE_MISSED:
                    # Elder explicitly declined this prompt. Per Section
                    # 3.5.2 the system "re-prompts after a 15-minute
                    # interval" — i.e. an explicit miss still consumes
                    # one of the attempts and we proceed into the retry
                    # cycle (unless this was the last attempt).
                    self._log_event(
                        EventLogRepo.DOSE_MISSED,
                        {
                            "schedule_id": sched.schedule_id,
                            "drug_name":   sched.drug_name,
                            "attempt":     attempt_index + 1,
                            "kind":        "explicit_decline",
                        },
                    )
                    if attempt_index == total_attempts - 1:
                        outcome = ReminderOutcome.EXPLICITLY_MISSED
                        break
                else:
                    # Timeout — no recognized command within window.
                    self._log_event(
                        EventLogRepo.DOSE_MISSED,
                        {
                            "schedule_id": sched.schedule_id,
                            "drug_name":   sched.drug_name,
                            "attempt":     attempt_index + 1,
                            "kind":        "timeout",
                        },
                    )
                    if attempt_index == total_attempts - 1:
                        outcome = ReminderOutcome.DEFINITIVE_MISSED
                        break

                # 4. Wait the retry interval — but wake immediately on
                # shutdown so we don't sit on a 15-minute sleep.
                if self._stop_event.wait(timeout=self._retry_interval):
                    return

            # 5. Definitive outcome — log & queue for caregiver.
            self._record_final_outcome(sched, outcome, issued_at)

        except Exception:
            logger.exception(
                "Dose worker for schedule_id=%d failed unexpectedly",
                sched.schedule_id,
            )
            self._log_event(
                EventLogRepo.SYSTEM_FAULT,
                {
                    "subsystem": "ReminderScheduler",
                    "stage":     "dose_worker",
                    "schedule_id": sched.schedule_id,
                },
            )
        finally:
            # Release the in-flight slot so the next scheduled occurrence
            # of this dose (e.g. tomorrow) can fire.
            with self._in_flight_lock:
                self._in_flight.discard(key)
            with self._workers_lock:
                self._workers.discard(threading.current_thread())

    # -----------------------------------------------------------------------
    # Confirmation routing — bridges the voice callback and the worker
    # -----------------------------------------------------------------------

    def _await_confirmation(self, schedule_id: int) -> Optional[str]:
        """
        Block until either:
          - the voice callback delivers an action for this schedule_id, OR
          - the confirmation timeout expires, OR
          - the scheduler is shutting down.
        Returns the action string, or None on timeout/shutdown.
        """
        pending = _PendingConfirmation(
            event=threading.Event(),
            issued_at=datetime.now(timezone.utc),
        )
        with self._pending_lock:
            self._pending_confirmations[schedule_id] = pending

        try:
            got = pending.event.wait(timeout=self._confirmation_timeout)
            if not got:
                return None  # Timeout
            if self._stop_event.is_set():
                return None  # Shutdown
            return pending.last_action
        finally:
            with self._pending_lock:
                self._pending_confirmations.pop(schedule_id, None)

    # -----------------------------------------------------------------------
    # Final-outcome bookkeeping
    # -----------------------------------------------------------------------

    def _record_final_outcome(
        self,
        sched: MedicationSchedule,
        outcome: str,
        issued_at: datetime,
    ) -> None:
        """
        Write the definitive outcome to the EventLog AND queue a Hub→App
        notification (Section 3.5.2: "a notification is queued for the
        caregiver application"). Section 3.5.4: confirmed and missed doses
        flow through the SyncQueue with direction='Hub->App'.
        """
        if outcome == ReminderOutcome.CONFIRMED:
            event_type = EventLogRepo.DOSE_CONFIRMED
        else:
            event_type = EventLogRepo.DOSE_MISSED

        details = {
            "schedule_id":  sched.schedule_id,
            "drug_name":    sched.drug_name,
            "dosage":       sched.dosage,
            "outcome":      outcome,
            "issued_at":    issued_at.isoformat(timespec="seconds"),
        }
        event_id = self._log_event(event_type, details)

        # Speak a closing prompt for definitively missed doses so the elder
        # knows their caregiver will be notified.
        if outcome == ReminderOutcome.DEFINITIVE_MISSED:
            self._speaker.speak(_build_final_missed_prompt(sched))

        # Queue for sync to the caregiver app via either available pathway.
        # Transport defaults to wifi_rest; the connectivity_arbiter may
        # later promote urgent items (e.g., DEFINITIVE_MISSED) to sms.
        try:
            self._sync_repo.enqueue_change(
                entity_type="EventLog",
                entity_id=event_id,
                change_type="INSERT",
                direction="Hub->App",
                transport="wifi_rest",
                payload={
                    "event_type": event_type,
                    "details":    details,
                },
            )
        except Exception:
            # SyncQueue failure is not fatal — the EventLog row itself is
            # the source of truth, and the sync engine can later sweep
            # unsynced events on its own.
            logger.exception(
                "Failed to enqueue Hub->App sync for event_id=%d", event_id,
            )

    def _log_event(self, event_type: str, details: dict) -> int:
        """Thin wrapper that swallows nothing — caller wants the event_id."""
        return self._event_repo.insert(event_type, details=details)


# ===========================================================================
# Internal types
# ===========================================================================

@dataclass
class _PendingConfirmation:
    """Tracks one worker's wait for a voice confirmation."""
    event: threading.Event
    issued_at: datetime
    last_action: Optional[str] = None


# ===========================================================================
# Standalone smoke test
# ===========================================================================

if __name__ == "__main__":
    """
    Quick standalone exercise — wire up mocks, seed a med due in 10 seconds,
    boot the scheduler, simulate a confirmation, and watch the logs.

    Run from the project root:
        python -m src.control_logic.reminder_scheduler
    """
    import time
    from src.hardware_mocks.mock_speaker    import MockSpeaker
    from src.hardware_mocks.mock_microphone import MockMicrophone
    from src.data_management.repositories   import (
        ElderProfileRepo, ElderProfile, MedicationSchedule,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(name)s — %(message)s",
    )

    # --- Seed test elder + medication due 10 seconds from now ---
    elder_repo = ElderProfileRepo()
    med_repo   = MedicationScheduleRepo()

    elder = elder_repo.fetch_first()
    if elder is None:
        elder_id = elder_repo.insert(ElderProfile(
            name="Demo Elder",
            language="twi",
            caregiver_phones="+233244000000",
        ))
    else:
        elder_id = elder.elder_id

    due_at = datetime.now().replace(second=0, microsecond=0)
    # Push it ~30s ahead so it lands inside the next polling window.
    from datetime import timedelta
    due_at = (datetime.now() + timedelta(seconds=20)).replace(second=0, microsecond=0)

    med_repo.insert(MedicationSchedule(
        elder_id=elder_id,
        drug_name="DemoDrug",
        dosage="100mg",
        time_due=due_at.strftime("%H:%M"),
        days_of_week="DAILY",
        active=1,
        prescribed_by="hub_local",
        sync_method="hub_local",
    ))
    print(f"[TEST] Seeded DemoDrug due at {due_at.strftime('%H:%M')}")

    # --- Build the scheduler with shortened timings for the demo ---
    speaker = MockSpeaker(simulate_latency=False)
    mic     = MockMicrophone()  # callback wired below

    scheduler = ReminderScheduler(
        speaker=speaker,
        microphone=mic,
        confirmation_timeout_sec=15,    # fast demo
        retry_interval_seconds=5,        # fast demo
        max_additional_retries=2,        # fast demo
        poll_interval_seconds=5,         # fast demo
    )

    # Wire the mic's command callback to the scheduler.
    mic._on_command = scheduler.on_voice_command  # type: ignore[attr-defined]

    scheduler.start()
    print("[TEST] Scheduler running. Sleeping 30s, then injecting confirmation...")

    try:
        # Wait long enough for the prompt to fire.
        time.sleep(30)

        # Simulate the elder saying "Yε, mafa m'aduru".
        print("[TEST] Injecting DOSE_CONFIRMED")
        mic.inject_command(ACTION_DOSE_CONFIRMED)

        # Let the worker finish bookkeeping.
        time.sleep(3)
    finally:
        scheduler.stop()
        print("[TEST] Done. Inspect EventLog and SyncQueue to verify.")
        