"""
connectivity_arbiter.py
=======================
Pathway selector for the hybrid bidirectional sync protocol described in
Section 3.5.4 of the project report:

    "The choice of pathway at any given moment is governed by physical
     proximity and connectivity availability rather than by user action,
     allowing the caregiver to interact with the application in a uniform
     manner irrespective of their location."

The arbiter answers two questions for the SyncEngine on every tick:

    (a) For a pending Hub→App entry currently labelled `wifi_rest`, should
        we keep waiting for the REST API to serve it, OR has it waited
        long enough that we should promote it to SMS?

    (b) For an entry already labelled `sms`, is the GSM pathway currently
        viable?

Urgency budgets — how long an entry may wait for its primary pathway
before being promoted — are event-type specific. SOS triggers must reach
the caregiver within seconds; routine missed-dose summaries can wait
minutes. Non-urgent entries (appliance state, dose confirmations, etc.)
are NEVER promoted — they wait indefinitely for the next Wi-Fi window.

Mock-Phase Wi-Fi State
----------------------
The real wlan0 AP interface tracks associated stations via hostapd. For
the mock phase we expose .set_wifi_caregiver_connected(bool) so tests can
toggle the perceived state deterministically. When integrating with the
real Pi, replace .is_wifi_caregiver_connected() with a hostapd_cli call
or an inotify watcher on the lease file.

Author: Wise (Asumang Pobi Godwin) — KNUST COE 497
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.data_management.repositories import (
    SyncQueueRepo,
    SyncQueueEntry,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Decision constants — return values from .select_transport()
# ===========================================================================

TRANSPORT_WIFI_REST = "wifi_rest"   # Wait for the REST API to serve it.
TRANSPORT_SMS       = "sms"         # Dispatch via SMSTransport now.
TRANSPORT_WAIT      = "wait"        # Neither pathway viable — try again later.


# ===========================================================================
# Default urgency budgets (seconds)
# ===========================================================================
# Keys are event_types (or entity_types) found inside the SyncQueueEntry's
# payload. Entries whose event_type is NOT in this dict are treated as
# non-urgent — they wait indefinitely for Wi-Fi.
DEFAULT_URGENCY_BUDGETS: Dict[str, int] = {
    # Life-safety: caregiver MUST learn about this within a minute.
    "sos_triggered":          60,
    # Dispatch outcome of an SOS — useful but slightly less urgent.
    "sos_dispatch_complete":  120,
    # Definitive missed dose (all retries exhausted) — clinical concern.
    "dose_missed":            300,    # 5 minutes
}


# ===========================================================================
# ConnectivityArbiter
# ===========================================================================

class ConnectivityArbiter:
    """
    Stateful policy object. Internally tracks the current Wi-Fi caregiver
    connection state via a thread-safe flag. All decision methods are
    pure functions of (entry, internal state) and have no side effects
    EXCEPT promote_to_sms(), which mutates the SyncQueue.
    """

    def __init__(
        self,
        sync_repo: Optional[SyncQueueRepo] = None,
        gsm: Optional[Any] = None,
        *,
        urgency_budgets: Optional[Dict[str, int]] = None,
        default_wifi_caregiver_connected: bool = False,
        min_signal_strength_csq: int = 8,    # CSQ scale — ~weak but workable
    ):
        """
        Parameters
        ----------
        sync_repo : repository for promote_to_sms() mutations.
        gsm : optional GSM adapter; if provided, .signal_strength() is
              consulted in is_sms_available(). If None, SMS is assumed
              available (appropriate for the mock phase).
        urgency_budgets : per-event-type seconds-to-promote. Defaults
                          provided above.
        default_wifi_caregiver_connected : initial state of the toggle.
        min_signal_strength_csq : CSQ threshold below which SMS is
                                  considered unavailable.
        """
        self._sync_repo = sync_repo or SyncQueueRepo()
        self._gsm       = gsm
        self._urgency_budgets = dict(urgency_budgets or DEFAULT_URGENCY_BUDGETS)
        self._min_csq         = min_signal_strength_csq

        # Mock-phase toggle for Wi-Fi caregiver presence.
        self._wifi_lock = threading.Lock()
        self._wifi_caregiver_connected = default_wifi_caregiver_connected

    # -----------------------------------------------------------------------
    # Wi-Fi state — mock toggle, real implementation hooks here
    # -----------------------------------------------------------------------

    def is_wifi_caregiver_connected(self) -> bool:
        """
        Whether a caregiver app is currently associated with the hub's
        AP. In production this would query hostapd; in the mock phase it
        returns the value set by .set_wifi_caregiver_connected().
        """
        with self._wifi_lock:
            return self._wifi_caregiver_connected

    def set_wifi_caregiver_connected(self, connected: bool) -> None:
        """Mock-phase toggle. No effect when the real implementation lands."""
        with self._wifi_lock:
            previous = self._wifi_caregiver_connected
            self._wifi_caregiver_connected = bool(connected)
        if previous != bool(connected):
            logger.info(
                "Wi-Fi caregiver state: %s → %s",
                "connected" if previous else "disconnected",
                "connected" if connected else "disconnected",
            )

    # -----------------------------------------------------------------------
    # SMS availability
    # -----------------------------------------------------------------------

    def is_sms_available(self) -> bool:
        """
        Whether the GSM module is currently capable of dispatching. If
        no GSM adapter was injected, we assume yes (mock-phase default).
        With a real SIM800L wired in, this checks signal strength via CSQ.
        """
        if self._gsm is None:
            return True
        try:
            csq = int(self._gsm.signal_strength())
            return csq >= self._min_csq
        except Exception:
            logger.exception("signal_strength() raised — treating SMS as unavailable.")
            return False

    # -----------------------------------------------------------------------
    # Classification helpers
    # -----------------------------------------------------------------------

    def get_event_type(self, entry: SyncQueueEntry) -> Optional[str]:
        """
        Extract the event_type field from the entry's payload (if any).
        Returns None when the payload doesn't carry one — appliance
        commands, schedule updates, etc.
        """
        if not entry.payload:
            return None
        try:
            obj = json.loads(entry.payload)
        except (json.JSONDecodeError, TypeError):
            return None
        if isinstance(obj, dict):
            return obj.get("event_type")
        return None

    def is_urgent(self, entry: SyncQueueEntry) -> bool:
        """An entry is urgent iff its event_type has a defined budget."""
        et = self.get_event_type(entry)
        return et is not None and et in self._urgency_budgets

    def has_exceeded_urgency_budget(self, entry: SyncQueueEntry) -> bool:
        """
        Compare the entry's age (from its timestamp) against the budget
        for its event_type. Returns True if the budget has been exceeded.
        Returns False for entries with no defined budget.
        """
        event_type = self.get_event_type(entry)
        if event_type is None or event_type not in self._urgency_budgets:
            return False

        budget_seconds = self._urgency_budgets[event_type]
        age_seconds = self._age_seconds(entry)
        return age_seconds is not None and age_seconds >= budget_seconds

    @staticmethod
    def _age_seconds(entry: SyncQueueEntry) -> Optional[float]:
        """How long since the entry was created (per its timestamp)."""
        if not entry.timestamp:
            return None
        try:
            # Accept both Z and +00:00 forms.
            ts = entry.timestamp.replace("Z", "+00:00")
            t = datetime.fromisoformat(ts)
        except ValueError:
            return None
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - t).total_seconds()

    # -----------------------------------------------------------------------
    # Decision
    # -----------------------------------------------------------------------

    def select_transport(self, entry: SyncQueueEntry) -> str:
        """
        Decide which transport to use for this entry RIGHT NOW. Returns
        one of: TRANSPORT_WIFI_REST, TRANSPORT_SMS, TRANSPORT_WAIT.

        Decision matrix (assuming entry.transport is the originator's
        intended pathway):

           originator | wifi_present | sms_avail | urgent_overdue | result
           -----------+--------------+-----------+----------------+----------
           sms        |   any        |   yes     |   any          | SMS
           sms        |   any        |   no      |   any          | WAIT
           wifi_rest  |   yes        |   any     |   any          | WIFI
           wifi_rest  |   no         |   yes     |   yes          | SMS  (promote)
           wifi_rest  |   no         |   yes     |   no           | WIFI (still wait)
           wifi_rest  |   no         |   no      |   any          | WIFI (still wait — best we have)
        """
        sms_avail   = self.is_sms_available()
        wifi_avail  = self.is_wifi_caregiver_connected()

        if entry.transport == TRANSPORT_SMS:
            return TRANSPORT_SMS if sms_avail else TRANSPORT_WAIT

        # Entry is on wifi_rest.
        if wifi_avail:
            return TRANSPORT_WIFI_REST

        # No Wi-Fi. Promote only if urgent AND past budget AND SMS is up.
        if (
            self.is_urgent(entry)
            and self.has_exceeded_urgency_budget(entry)
            and sms_avail
        ):
            return TRANSPORT_SMS

        # Default: leave on wifi_rest, wait for the app to appear.
        return TRANSPORT_WIFI_REST

    # -----------------------------------------------------------------------
    # Mutation — promote a pending entry from wifi_rest to sms
    # -----------------------------------------------------------------------

    def promote_to_sms(self, entry: SyncQueueEntry) -> bool:
        """
        Persist the transport change in the SyncQueue so the next engine
        tick picks it up via fetch_pending(transport='sms'). Returns
        True if the row was actually mutated.
        """
        if entry.transport == TRANSPORT_SMS:
            return False  # Nothing to do.

        ok = self._sync_repo.update_transport(entry.change_id, TRANSPORT_SMS)
        if ok:
            entry.transport = TRANSPORT_SMS  # Keep caller's view consistent.
            logger.info(
                "Promoted change_id=%s from wifi_rest to sms (urgent + Wi-Fi unavailable).",
                entry.change_id,
            )
        return ok
    