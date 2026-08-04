"""
command_dispatcher.py
=====================
Voice-command router for the non-emergency, non-medication-confirmation
subset of Table 3.2 of the project report. Subscribes to the keyword
spotter's command callback and dispatches to the appropriate handler for:

    • ACTION_APPLIANCE_ON   — close the relay (turn on light/fan)
    • ACTION_APPLIANCE_OFF  — open the relay
    • ACTION_READ_SCHEDULE  — speak today's medication summary
    • ACTION_REPEAT_LAST    — replay the most recent TTS utterance

Explicitly NOT handled here (by design):
    • ACTION_SOS              — owned by SOSHandler (Section 3.5.2 SOS pathway)
    • ACTION_DOSE_CONFIRMED   — owned by ReminderScheduler
    • ACTION_DOSE_MISSED      — owned by ReminderScheduler

These three are silently ignored. The dispatcher is one of several
subscribers to the same callback chain that main.py installs around the
microphone — each subscriber filters for the actions it owns and ignores
the rest.

Design Notes
------------
1. Pure routing layer. No state machine, no retries, no scheduling. Each
   command is a one-shot stimulus → response with a logged outcome. This
   keeps the dispatcher simple and individually testable.

2. Today's schedule for READ_SCHEDULE. The report's Table 3.2 says the
   action should "Read today's medication schedule aloud." We honor that
   by filtering MedicationScheduleRepo's active list to schedules whose
   days_of_week includes today's day code (or 'DAILY'). The same matching
   logic the reminder scheduler uses is replicated here.

3. Twi prompt phrasing is centralized at the top of the file. Per the
   report's User-Centred Design methodology (Section 3.1) and the panel-
   credibility principle (every defended claim must trace to documented
   design), final wording will be validated in the User Needs Assessment
   phase. The placeholders here are clearly marked as such.

4. Dependency injection throughout. Speaker, GPIO, and the two
   repositories (event log + medication schedule) are constructor
   arguments. Tests inject mocks; production wires real drivers.

5. Threading safety. The dispatcher is stateless apart from the relay
   command, which delegates state to the GPIO controller itself. No
   internal locks are required at this layer.

Author: Wise (Asumang Pobi Godwin) — KNUST COE 497
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, List, Optional

from src.data_management.repositories import (
    MedicationScheduleRepo,
    EventLogRepo,
    MedicationSchedule,
)
from src.hardware_mocks.mock_microphone import (
    CommandEvent,
    ACTION_APPLIANCE_ON,
    ACTION_APPLIANCE_OFF,
    ACTION_READ_SCHEDULE,
    ACTION_REPEAT_LAST,
    ACTION_SOS,
    ACTION_DOSE_CONFIRMED,
    ACTION_DOSE_MISSED,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Twi prompt phrasing — single source of truth
# ===========================================================================
# NOTE FOR THE PANEL: these prompts are placeholders, to be validated in
# the User Needs Assessment phase per Section 3.2 of the project report.

PROMPT_APPLIANCE_ON          = "Masɔ kanea no."
# Approximate translation: "I have switched on the light."

PROMPT_APPLIANCE_OFF         = "Madum kanea no."
# Approximate translation: "I have switched off the light."

PROMPT_APPLIANCE_ALREADY_ON  = "Kanea no asɔ dedaw."
# "The light is already on."

PROMPT_APPLIANCE_ALREADY_OFF = "Kanea no adum dedaw."
# "The light is already off."

PROMPT_NO_SCHEDULE_TODAY     = "Wonni ɛduro biaa wobɛfa ɛnɛ"
# "You have no medication scheduled for today."

PROMPT_SCHEDULE_HEADER       = "Ɛnnɛ, wo aduru a wobɛnom nie:"
# "Here is your medication for today:"

PROMPT_RELAY_FAULT           = "Mentumi nyɛ no seesei. Mesrɛ wo, sɔ hwɛ bio."
# "I cannot do that right now. Please try again."


# Action sets — declared explicitly so the routing logic is self-documenting.
HANDLED_ACTIONS = frozenset({
    ACTION_APPLIANCE_ON,
    ACTION_APPLIANCE_OFF,
    ACTION_READ_SCHEDULE,
    ACTION_REPEAT_LAST,
})

IGNORED_ACTIONS = frozenset({
    ACTION_SOS,             # → SOSHandler
    ACTION_DOSE_CONFIRMED,  # → ReminderScheduler
    ACTION_DOSE_MISSED,     # → ReminderScheduler
})


# ===========================================================================
# CommandDispatcher
# ===========================================================================

class CommandDispatcher:
    """
    Routes non-emergency, non-confirmation voice commands to the
    appropriate side-effect handler. Exposes a single subscriber method —
    .on_voice_command(event) — that main.py wires into the keyword
    spotter's callback chain.

    Lifecycle:
        dispatcher = CommandDispatcher(speaker, gpio, ...)
        dispatcher.wire_to_microphone(microphone)   # subscribe
        ...
        # No explicit shutdown required — dispatcher is stateless.
    """

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(
        self,
        speaker: Any,                                     # MockSpeaker / TTSEngine
        gpio: Any,                                        # MockGPIOController / GPIOController
        med_repo:   Optional[MedicationScheduleRepo] = None,
        event_repo: Optional[EventLogRepo]           = None,
        *,
        elder_id: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        speaker, gpio : injected hardware adapters.
        med_repo, event_repo : repository instances; default-constructed if omitted.
        elder_id : restrict the schedule readback to a single elder.
                   None = all elders' active schedules (typically the same in
                   a single-elder deployment).
        """
        self._speaker     = speaker
        self._gpio        = gpio
        self._med_repo    = med_repo   or MedicationScheduleRepo()
        self._event_repo  = event_repo or EventLogRepo()
        self._elder_id    = elder_id

    # -----------------------------------------------------------------------
    # Wiring helper
    # -----------------------------------------------------------------------

    def wire_to_microphone(self, microphone: Any) -> None:
        """
        Subscribe to the keyword spotter's command callback. Note: if
        multiple subsystems need to share the microphone callback (and they
        do — the SOSHandler and ReminderScheduler also subscribe), main.py
        should install a fan-out dispatcher rather than calling this method
        directly. Provided here for unit-test convenience.
        """
        microphone._on_command = self.on_voice_command  # type: ignore[attr-defined]
        logger.info("CommandDispatcher subscribed to microphone callback.")

    # -----------------------------------------------------------------------
    # Public callback
    # -----------------------------------------------------------------------

    def on_voice_command(self, event: CommandEvent) -> None:
        """
        Microphone callback. Routes the action to its handler, or silently
        ignores actions owned by other subsystems.

        Per the report's design, this method must be cheap and non-blocking
        for the actions it does NOT handle, since it sits in the same
        callback chain as the SOS pathway (which is latency-sensitive).
        """
        action = event.action

        if action in IGNORED_ACTIONS:
            # Silently ignored — owned by another subsystem.
            return

        if action not in HANDLED_ACTIONS:
            # Unknown action — should not happen given the constrained
            # vocabulary, but log it as a system anomaly.
            logger.warning(
                "CommandDispatcher received unknown action %r — ignored.",
                action,
            )
            return

        try:
            if action == ACTION_APPLIANCE_ON:
                self._handle_appliance(turn_on=True)
            elif action == ACTION_APPLIANCE_OFF:
                self._handle_appliance(turn_on=False)
            elif action == ACTION_READ_SCHEDULE:
                self._handle_read_schedule()
            elif action == ACTION_REPEAT_LAST:
                self._handle_repeat_last()
        except Exception:
            # Any handler exception is logged but does not propagate —
            # we don't want a faulty handler to crash the callback chain
            # and starve the SOS pathway of voice events.
            logger.exception(
                "CommandDispatcher handler for action=%s raised.", action
            )
            self._event_repo.insert(
                EventLogRepo.SYSTEM_FAULT,
                details={
                    "subsystem": "CommandDispatcher",
                    "action":    action,
                    "stage":     "handler",
                },
            )

    # -----------------------------------------------------------------------
    # Handler — appliance on/off
    # -----------------------------------------------------------------------

    def _handle_appliance(self, turn_on: bool) -> None:
        """
        Drive the relay and confirm. Detects no-op transitions (already in
        the requested state) and gives the elder a distinct confirmation
        so they aren't left wondering whether the command registered.
        """
        currently_on = bool(getattr(self._gpio, "relay_state", False))

        if currently_on == turn_on:
            # No-op — speak a distinct confirmation so the elder has
            # closure on their utterance.
            self._speaker.speak(
                PROMPT_APPLIANCE_ALREADY_ON if turn_on else PROMPT_APPLIANCE_ALREADY_OFF
            )
            self._event_repo.insert(
                EventLogRepo.APPLIANCE_ON if turn_on else EventLogRepo.APPLIANCE_OFF,
                details={
                    "requested_state": "on" if turn_on else "off",
                    "previous_state":  "on" if currently_on else "off",
                    "transition":      "noop",
                },
            )
            return

        # Real transition.
        try:
            self._gpio.set_relay(turn_on)
        except Exception:
            logger.exception("Relay control failed for turn_on=%s", turn_on)
            self._speaker.speak(PROMPT_RELAY_FAULT)
            self._event_repo.insert(
                EventLogRepo.SYSTEM_FAULT,
                details={
                    "subsystem":       "CommandDispatcher",
                    "stage":           "relay_control",
                    "requested_state": "on" if turn_on else "off",
                },
            )
            return

        self._speaker.speak(PROMPT_APPLIANCE_ON if turn_on else PROMPT_APPLIANCE_OFF)
        self._event_repo.insert(
            EventLogRepo.APPLIANCE_ON if turn_on else EventLogRepo.APPLIANCE_OFF,
            details={
                "requested_state": "on" if turn_on else "off",
                "previous_state":  "on" if currently_on else "off",
                "transition":      "applied",
            },
        )

    # -----------------------------------------------------------------------
    # Handler — read today's schedule
    # -----------------------------------------------------------------------

    def _handle_read_schedule(self) -> None:
        """
        Speak a Twi summary of today's medication schedule. Filters the
        active schedules to entries whose days_of_week include today's
        day code (or 'DAILY'), sorted by time_due. If nothing is due
        today, speaks a clear "no medications today" prompt instead.
        """
        active = self._med_repo.fetch_all_active(elder_id=self._elder_id)
        today_meds = self._filter_for_today(active)

        if not today_meds:
            self._speaker.speak(PROMPT_NO_SCHEDULE_TODAY)
            self._event_repo.insert(
                "schedule_read_aloud",
                details={
                    "elder_id":       self._elder_id,
                    "schedules_read": 0,
                },
            )
            return

        summary = self._build_schedule_summary(today_meds)
        self._speaker.speak(summary)

        self._event_repo.insert(
            "schedule_read_aloud",
            details={
                "elder_id":        self._elder_id,
                "schedules_read":  len(today_meds),
                "schedule_ids":    [s.schedule_id for s in today_meds],
            },
        )

    @staticmethod
    def _filter_for_today(schedules: List[MedicationSchedule]) -> List[MedicationSchedule]:
        """
        Return schedules whose days_of_week applies to today, sorted by
        time_due. Mirrors the day-matching logic in
        MedicationScheduleRepo.fetch_due_within() so the readback and
        the reminder scheduler agree on what 'today' means.
        """
        today_code = datetime.now().strftime("%a").upper()  # 'MON', 'TUE', ...

        applicable = []
        for s in schedules:
            dow = (s.days_of_week or "").upper().strip()
            if dow == "DAILY" or today_code in [d.strip() for d in dow.split(",")]:
                applicable.append(s)

        applicable.sort(key=lambda s: s.time_due)
        return applicable

    @staticmethod
    def _build_schedule_summary(schedules: List[MedicationSchedule]) -> str:
        """
        Compose a natural Twi summary. Uses the header prompt, then one
        clause per medication of the form:
            "<drug>, <dosage>, bere a ɛyɛ <HH:MM>."
        i.e., "<drug>, <dosage>, at <time>." in Twi structure.
        """
        clauses = []
        for s in schedules:
            clauses.append(
                f"{s.drug_name}, {s.dosage} sɛ ɛbɔ {s.time_due}."
            )
        return PROMPT_SCHEDULE_HEADER + " " + " ".join(clauses)

    # -----------------------------------------------------------------------
    # Handler — repeat last utterance
    # -----------------------------------------------------------------------

    def _handle_repeat_last(self) -> None:
        """
        Replay the most recent TTS utterance. Delegates to the speaker's
        own .repeat_last() implementation, which the mock provides and
        the real Piper engine will too.
        """
        if hasattr(self._speaker, "repeat_last"):
            self._speaker.repeat_last()
        elif hasattr(self._speaker, "last_spoken") and self._speaker.last_spoken:
            self._speaker.speak(self._speaker.last_spoken)
        else:
            # Speaker doesn't expose repeat — nothing to do, but log it.
            logger.warning(
                "Speaker has no repeat_last() or last_spoken — REPEAT_LAST is a no-op."
            )

        self._event_repo.insert(
            "tts_repeated",
            details={"requested_via": "voice_command"},
        )


# ===========================================================================
# Standalone smoke test
# ===========================================================================

if __name__ == "__main__":
    """
    Run from the project root:

        python -m src.control_logic.command_dispatcher

    Demonstrates each handled command in order, plus verifies that
    ignored actions (SOS / DOSE_CONFIRMED / DOSE_MISSED) silently bypass
    the dispatcher.
    """
    import logging
    from src.hardware_mocks.mock_speaker    import MockSpeaker
    from src.hardware_mocks.mock_microphone import MockMicrophone
    from src.hardware_mocks.mock_gpio       import MockGPIOController
    from src.data_management.repositories   import (
        ElderProfileRepo, ElderProfile, MedicationSchedule,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(name)s — %(message)s",
    )

    # --- Seed an elder + a few medications ------------------------------
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

    # Seed two daily meds at distinct times so the schedule readback is
    # non-trivial. Idempotency: upsert_from_payload prevents duplicates
    # if you re-run the smoke test.
    med_repo.upsert_from_payload(
        MedicationSchedule(
            elder_id=elder_id, drug_name="Paracetamol", dosage="500mg",
            time_due="08:00", days_of_week="DAILY", active=1,
        ),
        sync_method="hub_local", prescribed_by="hub_local",
    )
    med_repo.upsert_from_payload(
        MedicationSchedule(
            elder_id=elder_id, drug_name="Lisinopril", dosage="10mg",
            time_due="20:00", days_of_week="DAILY", active=1,
        ),
        sync_method="hub_local", prescribed_by="hub_local",
    )

    # --- Wire up the mocks ---------------------------------------------
    speaker = MockSpeaker(simulate_latency=False)
    mic     = MockMicrophone()
    gpio    = MockGPIOController()
    gpio.start()

    dispatcher = CommandDispatcher(
        speaker=speaker,
        gpio=gpio,
        elder_id=elder_id,
    )
    dispatcher.wire_to_microphone(mic)

    # --- Scenario 1: appliance on (real transition) --------------------
    print("\n" + "=" * 70)
    print("  SCENARIO 1 — APPLIANCE_ON  ('Sua fitaa no')")
    print("=" * 70)
    mic.inject_command(ACTION_APPLIANCE_ON)

    # --- Scenario 2: appliance on again (no-op confirmation) -----------
    print("\n" + "=" * 70)
    print("  SCENARIO 2 — APPLIANCE_ON again  (should report 'already on')")
    print("=" * 70)
    mic.inject_command(ACTION_APPLIANCE_ON)

    # --- Scenario 3: appliance off (real transition) -------------------
    print("\n" + "=" * 70)
    print("  SCENARIO 3 — APPLIANCE_OFF  ('Sua fitaa no na')")
    print("=" * 70)
    mic.inject_command(ACTION_APPLIANCE_OFF)

    # --- Scenario 4: read today's schedule -----------------------------
    print("\n" + "=" * 70)
    print("  SCENARIO 4 — READ_SCHEDULE  ('Aduru bɛn na mefa?')")
    print("=" * 70)
    mic.inject_command(ACTION_READ_SCHEDULE)

    # --- Scenario 5: repeat last ---------------------------------------
    print("\n" + "=" * 70)
    print("  SCENARIO 5 — REPEAT_LAST  ('Mesrɛ wo, yɛ san bio')")
    print("=" * 70)
    mic.inject_command(ACTION_REPEAT_LAST)

    # --- Scenario 6: ignored actions (must be silent) ------------------
    print("\n" + "=" * 70)
    print("  SCENARIO 6 — Ignored actions (SOS / DOSE_*) should bypass dispatcher")
    print("=" * 70)
    mic.inject_command(ACTION_SOS)
    mic.inject_command(ACTION_DOSE_CONFIRMED)
    mic.inject_command(ACTION_DOSE_MISSED)
    print("  → If you see no [SPK] / [GPIO] output above this line for these")
    print("    three commands, the ignore policy is working correctly.")

    print("\n" + "=" * 70)
    print("  Done. Inspect EventLog rows for appliance_on/off, schedule_read_aloud,")
    print("  and tts_repeated — but NO entries should exist for the ignored actions.")
    print("=" * 70)

    gpio.stop()
    