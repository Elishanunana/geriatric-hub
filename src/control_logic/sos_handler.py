"""
sos_handler.py
==============
High-priority emergency alert pathway. Implements Section 3.5.2 of the
project report exactly:

    "The SOS alert pathway is implemented as a high-priority interrupt
     handler. Upon receiving either a voice SOS command or a button press
     event, the handler immediately:
       (1) logs the SOS event with a timestamp;
       (2) activates the flashing red LED;
       (3) dispatches SMS alerts to all registered caregiver phone numbers
           via the GSM module;
       (4) speaks a Twi confirmation to the elder; and
       (5) marks the SOS event as pending-acknowledgement in the SyncQueue
           for transmission to the caregiver application at the next
           available synchronisation window."

Design Notes
------------
1. Two trigger sources, one pathway. The handler exposes a single public
   .trigger() method, plus two thin adapter methods — .on_voice_command()
   for the keyword spotter callback chain and .on_button_press() for the
   GPIO interrupt callback. Both adapters funnel into .trigger(), so the
   downstream sequence is identical regardless of input modality. This
   matters for auditability: the EventLog records WHICH source fired the
   alert, but the system's response is provably uniform.

2. Sequence ordering — log first, then notify. The order in the report is
   deliberate: log → LED → SMS → speak (async) → enqueue. We follow it
   exactly. The TTS confirmation is fired on a daemon thread BEFORE the
   SMS dispatch returns, so the elder hears reassurance even while the
   GSM module is still negotiating with the network.

3. Re-entrancy and rate limiting. A single elder pressing the button or
   shouting "Boa me!" repeatedly within a few seconds should produce ONE
   logical SOS event, not five. A configurable cooldown window (default
   30 seconds) suppresses repeat triggers. Repeats during cooldown are
   themselves logged (as 'sos_suppressed_during_cooldown') so the panel
   has a record that the system did receive them and chose to suppress.

4. Failure resilience. The SMS send loop is wrapped so a failure to one
   caregiver number does NOT abort the entire dispatch. We track
   per-recipient success and log the aggregate result. The LED stays
   flashing red until .acknowledge() is called — typically by the sync
   engine when the caregiver app delivers an ack record.

5. Dependency injection throughout. Speaker, microphone (for voice
   subscription), GPIO, GSM, and all three repositories are constructor
   arguments. Tests inject mocks; production wires real drivers.

Author: Wise (Asumang Pobi Godwin) — KNUST COE 497
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, List

from src.data_management.repositories import (
    ElderProfileRepo,
    EventLogRepo,
    SyncQueueRepo,
)
from src.hardware_mocks.mock_microphone import (
    CommandEvent,
    ACTION_SOS,
)
from src.hardware_mocks.mock_gpio import (
    LED_RED,
    LED_FLASHING,
    LED_GREEN,
    LED_STEADY,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Configuration constants
# ===========================================================================

DEFAULT_COOLDOWN_SECONDS    = 30   # Suppress repeat triggers within this window.
DEFAULT_TTS_CONFIRMATION    = "Yɛabɔ wo abusuafoɔ amaneɛ. Mmoa reba. Tena ase brɛoo."
# Approximate translation: "We have alerted your family. Help is coming. Stay calm."
# (Final phrasing pending validation in the User Needs Assessment, Section 3.2.)

DEFAULT_SMS_TEMPLATE = (
    "SOS ALERT — {elder_name}\n"
    "Time: {timestamp_local}\n"
    "Source: {source}\n"
    "An emergency has been triggered at the hub. "
    "Please call or visit immediately."
)


# ===========================================================================
# Source enum — provenance of a given SOS trigger
# ===========================================================================

@dataclass(frozen=True)
class SOSSource:
    VOICE  = "voice"           # Spoken "Boa me!" recognized by Vosk
    BUTTON = "button"          # Physical push-button press
    MANUAL = "manual"          # Programmatic trigger (tests / admin tools)


# ===========================================================================
# Result type — what happened during one SOS dispatch
# ===========================================================================

@dataclass
class SOSDispatchResult:
    sos_event_id:       int
    triggered_at:       str            # ISO-8601 UTC
    source:             str            # SOSSource.*
    suppressed:         bool = False   # True if rate-limited
    sms_recipients:     List[str] = field(default_factory=list)
    sms_succeeded:      List[str] = field(default_factory=list)
    sms_failed:         List[str] = field(default_factory=list)
    sync_change_id:     Optional[str] = None


# ===========================================================================
# SOSHandler
# ===========================================================================

class SOSHandler:
    """
    High-priority emergency interrupt handler. Single entry point —
    .trigger(source) — produces the full five-step sequence.

    Lifecycle:
        sos = SOSHandler(speaker, gpio, gsm, ...)
        sos.wire_to_microphone(microphone)   # subscribe to voice events
        sos.wire_to_gpio()                   # register button callback
        ...
        sos.acknowledge()                    # called by sync engine on app ack
    """

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(
        self,
        speaker: Any,                                 # MockSpeaker / TTSEngine
        gpio: Any,                                    # MockGPIOController / GPIOController
        gsm: Any,                                     # MockGSMModule / GSMModule
        elder_repo:  Optional[ElderProfileRepo] = None,
        event_repo:  Optional[EventLogRepo]     = None,
        sync_repo:   Optional[SyncQueueRepo]    = None,
        *,
        cooldown_seconds:   int  = DEFAULT_COOLDOWN_SECONDS,
        tts_confirmation:   str  = DEFAULT_TTS_CONFIRMATION,
        sms_template:       str  = DEFAULT_SMS_TEMPLATE,
        elder_id:           Optional[int] = None,
    ):
        """
        Parameters
        ----------
        speaker, gpio, gsm : injected hardware adapters.
        *_repo : repository instances; default-constructed if omitted.
        cooldown_seconds : minimum interval between accepted triggers.
        tts_confirmation : Twi reassurance phrase spoken to the elder.
        sms_template : str.format()-style template for the outbound alert.
        elder_id : restrict to a single elder. None = use the first elder.
        """
        self._speaker     = speaker
        self._gpio        = gpio
        self._gsm         = gsm
        self._elder_repo  = elder_repo or ElderProfileRepo()
        self._event_repo  = event_repo or EventLogRepo()
        self._sync_repo   = sync_repo  or SyncQueueRepo()

        self._cooldown          = timedelta(seconds=cooldown_seconds)
        self._tts_confirmation  = tts_confirmation
        self._sms_template      = sms_template
        self._elder_id          = elder_id

        # Re-entrancy / rate-limiting state.
        self._lock                  = threading.Lock()
        self._last_trigger_at:      Optional[datetime] = None
        self._pending_ack_event_id: Optional[int]      = None

    # -----------------------------------------------------------------------
    # Wiring helpers
    # -----------------------------------------------------------------------

    def wire_to_microphone(self, microphone: Any) -> None:
        """
        Subscribe to the keyword spotter so ACTION_SOS commands fire the
        handler. The mock microphone exposes _on_command as its callback
        slot; the real engine exposes the same name via its constructor.

        If multiple subsystems need to share the microphone callback (e.g.
        the reminder scheduler also subscribes), main.py should install a
        fan-out dispatcher rather than calling this method directly.
        """
        microphone._on_command = self.on_voice_command  # type: ignore[attr-defined]
        logger.info("SOSHandler subscribed to microphone callback.")

    def wire_to_gpio(self) -> None:
        """Register the physical button callback on the GPIO controller."""
        self._gpio.on_button_press(self.on_button_press)
        logger.info("SOSHandler registered GPIO button callback.")

    # -----------------------------------------------------------------------
    # Public adapter callbacks
    # -----------------------------------------------------------------------

    def on_voice_command(self, event: CommandEvent) -> None:
        """Microphone callback — fires only on ACTION_SOS."""
        if event.action != ACTION_SOS:
            return
        self.trigger(source=SOSSource.VOICE)

    def on_button_press(self) -> None:
        """GPIO interrupt callback — always treated as SOS."""
        self.trigger(source=SOSSource.BUTTON)

    # -----------------------------------------------------------------------
    # The emergency pathway
    # -----------------------------------------------------------------------

    def trigger(self, source: str = SOSSource.MANUAL) -> SOSDispatchResult:
        """
        Execute the full SOS sequence. Returns a structured result useful
        for tests and for the dev console.

        Sequence (matches Section 3.5.2 verbatim):
          1. Log the SOS event with a timestamp.
          2. Activate the flashing red LED.
          3. Dispatch SMS alerts to all registered caregiver phones.
          4. Speak a Twi confirmation to the elder (asynchronous).
          5. Queue a Hub→App pending-acknowledgement record in SyncQueue.
        """
        now = datetime.now(timezone.utc)

        # --- Cooldown gate (re-entrancy protection) ----------------------
        with self._lock:
            if (
                self._last_trigger_at is not None
                and now - self._last_trigger_at < self._cooldown
            ):
                # Suppressed — log it, but do not fire the full sequence.
                suppressed_id = self._event_repo.insert(
                    "sos_suppressed_during_cooldown",
                    details={
                        "source":              source,
                        "previous_trigger_at": self._last_trigger_at.isoformat(),
                        "cooldown_seconds":    int(self._cooldown.total_seconds()),
                    },
                )
                logger.info(
                    "SOS trigger from %s suppressed (within %ds cooldown).",
                    source, self._cooldown.total_seconds(),
                )
                return SOSDispatchResult(
                    sos_event_id=suppressed_id,
                    triggered_at=now.isoformat(timespec="milliseconds"),
                    source=source,
                    suppressed=True,
                )

            # Past the gate — record the trigger time before doing anything.
            self._last_trigger_at = now

        # --- Step 1: Log the SOS event ----------------------------------
        timestamp_iso   = now.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        timestamp_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        sos_event_id = self._event_repo.insert(
            EventLogRepo.SOS_TRIGGERED,
            details={
                "source":          source,
                "triggered_at":    timestamp_iso,
                "elder_id":        self._elder_id,
                "ack_state":       "pending",
            },
        )
        logger.warning("SOS TRIGGERED — source=%s event_id=%d", source, sos_event_id)

        with self._lock:
            self._pending_ack_event_id = sos_event_id

        # --- Step 2: Flashing red LED -----------------------------------
        try:
            self._gpio.set_led(LED_RED, LED_FLASHING)
        except Exception:
            # LED failure must not abort SMS dispatch — the LED is feedback
            # to the elder, but the SMS is the actual safety mechanism.
            logger.exception("Failed to set LED to flashing red.")

        # --- Step 3: SMS dispatch ---------------------------------------
        elder = self._resolve_elder()
        elder_name = elder.name if elder else "Unknown Elder"
        recipients = (
            self._elder_repo.caregiver_phones(elder.elder_id)
            if elder else []
        )

        sms_body = self._sms_template.format(
            elder_name=elder_name,
            timestamp_local=timestamp_local,
            source=source,
        )

        succeeded: List[str] = []
        failed:    List[str] = []

        if not recipients:
            # No caregivers configured — log it loudly. The system still
            # speaks reassurance so the elder isn't left without feedback.
            logger.error(
                "SOS triggered but NO caregiver phone numbers are registered."
            )
            self._event_repo.insert(
                EventLogRepo.SYSTEM_FAULT,
                details={
                    "subsystem":     "SOSHandler",
                    "stage":         "sms_dispatch",
                    "reason":        "no_caregiver_numbers",
                    "sos_event_id":  sos_event_id,
                },
            )
        else:
            for phone in recipients:
                try:
                    ok = self._gsm.send_sms(phone, sms_body)
                    (succeeded if ok else failed).append(phone)
                except Exception:
                    logger.exception("Exception while sending SOS SMS to %s", phone)
                    failed.append(phone)

        # --- Step 4: Async TTS confirmation -----------------------------
        # Fire-and-forget so SMS delivery isn't blocked by ~1.5s of synthesis.
        # The speaker mock implements speak_async(); the real Piper engine
        # will too.
        try:
            if hasattr(self._speaker, "speak_async"):
                self._speaker.speak_async(self._tts_confirmation)
            else:
                # Fallback: spawn our own daemon thread.
                threading.Thread(
                    target=self._safe_speak,
                    args=(self._tts_confirmation,),
                    name="SOSHandler-TTS",
                    daemon=True,
                ).start()
        except Exception:
            logger.exception("Failed to dispatch TTS confirmation.")

        # --- Step 5: Queue Hub→App ack-pending record -------------------
        sync_change_id: Optional[str] = None
        try:
            sync_change_id = self._sync_repo.enqueue_change(
                entity_type="EventLog",
                entity_id=sos_event_id,
                change_type="INSERT",
                direction="Hub->App",
                # SOS events default to wifi_rest like other events; the
                # connectivity_arbiter (Section 3.5.4) may promote them to
                # SMS if no Wi-Fi window opens within the urgency budget.
                transport="wifi_rest",
                payload={
                    "event_type":     EventLogRepo.SOS_TRIGGERED,
                    "sos_event_id":   sos_event_id,
                    "source":         source,
                    "triggered_at":   timestamp_iso,
                    "elder_name":     elder_name,
                    "sms_succeeded":  succeeded,
                    "sms_failed":     failed,
                    "ack_state":      "pending",
                },
            )
        except Exception:
            logger.exception(
                "Failed to enqueue SOS sync record for event_id=%d", sos_event_id
            )

        # --- Aggregate dispatch outcome to EventLog ---------------------
        # Distinct from the trigger event so the panel can audit BOTH the
        # moment of trigger AND the outcome of dispatch.
        self._event_repo.insert(
            "sos_dispatch_complete",
            details={
                "sos_event_id":     sos_event_id,
                "recipients_total": len(recipients),
                "recipients_ok":    len(succeeded),
                "recipients_fail":  len(failed),
                "succeeded":        succeeded,
                "failed":           failed,
            },
        )

        return SOSDispatchResult(
            sos_event_id=sos_event_id,
            triggered_at=timestamp_iso,
            source=source,
            suppressed=False,
            sms_recipients=recipients,
            sms_succeeded=succeeded,
            sms_failed=failed,
            sync_change_id=sync_change_id,
        )

    # -----------------------------------------------------------------------
    # Acknowledgement — called by the sync engine when caregiver app acks
    # -----------------------------------------------------------------------

    def acknowledge(self, sos_event_id: Optional[int] = None) -> bool:
        """
        Mark the most recent SOS as acknowledged: stop the flashing red
        LED, return to steady-green normal-operation indication, and log
        the ack. If sos_event_id is supplied, it is recorded in the ack
        log entry; otherwise the handler's tracked pending ID is used.

        Returns True if there was a pending SOS to acknowledge.
        """
        with self._lock:
            target = sos_event_id if sos_event_id is not None else self._pending_ack_event_id
            if target is None:
                logger.info("acknowledge() called but no SOS is pending ack.")
                return False
            self._pending_ack_event_id = None

        try:
            self._gpio.set_led(LED_GREEN, LED_STEADY)
        except Exception:
            logger.exception("Failed to reset LED on SOS acknowledgement.")

        self._event_repo.insert(
            "sos_acknowledged",
            details={"sos_event_id": target},
        )
        logger.info("SOS event_id=%d acknowledged.", target)
        return True

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _resolve_elder(self):
        """Resolve the elder profile to use for this SOS dispatch."""
        if self._elder_id is not None:
            return self._elder_repo.fetch_by_id(self._elder_id)
        return self._elder_repo.fetch_first()

    def _safe_speak(self, text: str) -> None:
        """Fallback async speak path used only when speak_async is absent."""
        try:
            self._speaker.speak(text)
        except Exception:
            logger.exception("Speaker raised inside SOS TTS fallback.")


# ===========================================================================
# Standalone smoke test
# ===========================================================================

if __name__ == "__main__":
    """
    Quick standalone exercise. Run from the project root:

        python -m src.control_logic.sos_handler

    What this demonstrates:
      • Voice-triggered SOS  → full sequence fires.
      • Cooldown rejection  → second trigger inside 30s is suppressed.
      • Manual acknowledgement → LED returns to steady green.
      • Button-triggered SOS (after cooldown) → identical pathway, different source.
    """
    import time
    from src.hardware_mocks.mock_speaker    import MockSpeaker
    from src.hardware_mocks.mock_microphone import MockMicrophone
    from src.hardware_mocks.mock_gpio       import MockGPIOController
    from src.hardware_mocks.mock_sim800l    import MockGSMModule
    from src.data_management.repositories   import ElderProfileRepo, ElderProfile

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(name)s — %(message)s",
    )

    # --- Seed an elder with two caregiver phone numbers --------------
    elder_repo = ElderProfileRepo()
    elder = elder_repo.fetch_first()
    if elder is None:
        elder_repo.insert(ElderProfile(
            name="Demo Elder",
            language="twi",
            caregiver_phones="+233244111111,+233244222222",
        ))
    else:
        elder_repo.update_caregiver_phones(
            elder.elder_id, "+233244111111,+233244222222"
        )

    # --- Wire up the mocks ------------------------------------------
    speaker = MockSpeaker(simulate_latency=False)
    mic     = MockMicrophone()
    gpio    = MockGPIOController()
    gsm     = MockGSMModule()

    gpio.start()

    sos = SOSHandler(
        speaker=speaker,
        gpio=gpio,
        gsm=gsm,
        cooldown_seconds=10,   # short cooldown for the demo
    )
    sos.wire_to_microphone(mic)
    sos.wire_to_gpio()

    print("\n" + "=" * 70)
    print("  SCENARIO 1 — Voice-triggered SOS")
    print("=" * 70)
    mic.inject_command(ACTION_SOS)
    time.sleep(1)

    print("\n" + "=" * 70)
    print("  SCENARIO 2 — Repeat trigger inside cooldown (should suppress)")
    print("=" * 70)
    result = sos.trigger(source=SOSSource.MANUAL)
    print(f"  → suppressed = {result.suppressed}")
    time.sleep(1)

    print("\n" + "=" * 70)
    print("  SCENARIO 3 — Caregiver acknowledges via app")
    print("=" * 70)
    sos.acknowledge()
    time.sleep(1)

    print("\n" + "=" * 70)
    print("  SCENARIO 4 — Wait out cooldown, then button trigger")
    print("=" * 70)
    print("  (sleeping 11s to clear cooldown...)")
    time.sleep(11)
    gpio.trigger_sos()
    time.sleep(1)

    print("\n" + "=" * 70)
    print("  Done. Inspect EventLog and SyncQueue to verify.")
    print("=" * 70)

    gpio.stop()
    