"""
mock_microphone.py
==================
Terminal-based simulator for the USB cardioid microphone + Vosk Twi
keyword spotter (Sections 3.4.2 and 3.5.1 of the project report).

Behavior Contract
-----------------
The real KeywordSpotter (voice_engine/keyword_spotter.py) will:
  - Run in a dedicated background thread.
  - Continuously process 16 kHz / 16-bit mono audio frames from the USB mic.
  - On grammar-match above the configured confidence threshold (default 0.75),
    emit a CommandEvent to the Control Logic Layer via a callback.

This mock preserves that contract exactly:
  - Same public API: .listen(), .start(), .stop(), .inject_command()
  - Same event payload: CommandEvent(twi_phrase, action, confidence, timestamp)
  - Same callback registration via constructor.

Replacement Plan
----------------
When physical hardware arrives, replace the import in main.py:
    from hardware_mocks.mock_microphone import MockMicrophone as KeywordSpotter
with:
    from voice_engine.keyword_spotter import KeywordSpotter
No changes required in the Control Logic Layer.

Author: Wise (Asumang Pobi Godwin) - KNUST COE 497
"""

import threading
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Command vocabulary (mirrors Table 3.2 — preliminary, subject to validation
# in the User Needs Assessment phase, Section 3.2 of the report).
# ---------------------------------------------------------------------------
# Each command maps a canonical Twi phrase to a symbolic action name that
# the Control Logic Layer's command_dispatcher.py can switch on.

@dataclass(frozen=True)
class TwiCommand:
    twi_phrase: str       # Canonical Twi phrase (Table 3.2)
    english_gloss: str    # Approximate English meaning
    action: str           # Symbolic action name for the dispatcher


# Action name constants — also imported by command_dispatcher.py to avoid
# magic strings on either side of the boundary.
ACTION_DOSE_CONFIRMED   = "DOSE_CONFIRMED"
ACTION_DOSE_MISSED      = "DOSE_MISSED"
ACTION_SOS              = "SOS"
ACTION_APPLIANCE_ON     = "APPLIANCE_ON"
ACTION_APPLIANCE_OFF    = "APPLIANCE_OFF"
ACTION_READ_SCHEDULE    = "READ_SCHEDULE"
ACTION_REPEAT_LAST      = "REPEAT_LAST"

# Vocabulary ordered to match Table 3.2.
TWI_VOCABULARY: List[TwiCommand] = [
    TwiCommand("Aane, mafa m'aduru",     "Yes, I have taken my medicine",   ACTION_DOSE_CONFIRMED),
    TwiCommand("Menfaa m'aduru no y3",    "I have not yet taken it",         ACTION_DOSE_MISSED),
    TwiCommand("Boa me!",              "Help me!",                        ACTION_SOS),
    TwiCommand("Sɔ kanea no",         "Switch on the light",             ACTION_APPLIANCE_ON),
    TwiCommand("Dum Kanea no",      "Switch off the light",            ACTION_APPLIANCE_OFF),
    TwiCommand("Aduru bεn na mefa?",   "What medicine do I take?",        ACTION_READ_SCHEDULE),
    TwiCommand("Mesrε wo, san kano bio", "Please repeat that",              ACTION_REPEAT_LAST),
]


# ---------------------------------------------------------------------------
# CommandEvent — the payload emitted to the Control Logic Layer
# ---------------------------------------------------------------------------

@dataclass
class CommandEvent:
    twi_phrase: str
    action: str
    confidence: float
    timestamp: str   # ISO-8601 UTC


# ---------------------------------------------------------------------------
# MockMicrophone
# ---------------------------------------------------------------------------

class MockMicrophone:
    """
    Simulates the Vosk-driven keyword spotter using terminal input.
    """

    def __init__(
        self,
        on_command_callback: Optional[Callable[[CommandEvent], None]] = None,
        confidence_threshold: float = 0.75,
        simulated_confidence: float = 0.95,
    ):
        """
        Parameters
        ----------
        on_command_callback : callable
            Invoked with a CommandEvent whenever a recognized command is
            detected. Mirrors the queue/callback pattern of the real engine.
        confidence_threshold : float
            Below this threshold, the (mock) engine drops the command silently.
            Real engine: see Section 3.5.1, default 0.75.
        simulated_confidence : float
            The confidence value attached to recognized mock commands.
        """
        self._on_command = on_command_callback
        self._confidence_threshold = confidence_threshold
        self._simulated_confidence = simulated_confidence
        self._listener_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Build a lookup map: accept either the full Twi phrase OR the index (1-7).
        self._lookup: Dict[str, TwiCommand] = {}
        for idx, cmd in enumerate(TWI_VOCABULARY, start=1):
            self._lookup[str(idx)] = cmd
            self._lookup[cmd.twi_phrase.lower()] = cmd

    # -- Public API ---------------------------------------------------------

    @property
    def vocabulary(self) -> List[TwiCommand]:
        """The recognized Twi command set (Table 3.2)."""
        return list(TWI_VOCABULARY)

    def listen(self, prompt: str = "[MIC] > ") -> Optional[CommandEvent]:
        """
        Blocking single-shot listen. Reads one line from stdin, attempts to
        match it against the Twi vocabulary, and returns a CommandEvent
        (or None if unrecognized).

        This is the simplest interaction pattern — useful for unit tests
        and quick scripted demos. For continuous operation matching the
        real engine, use .start() instead.
        """
        try:
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return None

        return self._dispatch(user_input)

    def start(self) -> None:
        """
        Begin continuous listening on a background thread. The thread reads
        stdin line-by-line and dispatches recognized commands through the
        registered callback. Mirrors the real engine's continuous operation.
        """
        if self._listener_thread and self._listener_thread.is_alive():
            logger.warning("MockMicrophone already started.")
            return

        self._stop_event.clear()
        self._listener_thread = threading.Thread(
            target=self._listen_loop, name="MockMic-Listener", daemon=True
        )
        self._listener_thread.start()
        self._print_menu()

    def stop(self) -> None:
        """Signal the listener thread to exit on its next input cycle."""
        self._stop_event.set()
        print("[MIC] Listener stopping (press Enter to release stdin).")

    def inject_command(
        self,
        twi_phrase_or_action: str,
        confidence: Optional[float] = None,
    ) -> Optional[CommandEvent]:
        """
        Programmatic command injection — bypasses stdin entirely.
        Used by unit tests and by integration tests that need to script
        a deterministic command sequence.

        Accepts either a Twi phrase from the vocabulary OR an action constant
        (e.g. ACTION_SOS).
        """
        # Try as Twi phrase first.
        cmd = self._lookup.get(twi_phrase_or_action.lower())
        if cmd is None:
            # Try as action constant.
            for c in TWI_VOCABULARY:
                if c.action == twi_phrase_or_action:
                    cmd = c
                    break

        if cmd is None:
            logger.warning("inject_command: unknown phrase/action %r", twi_phrase_or_action)
            return None

        return self._emit(cmd, confidence if confidence is not None else self._simulated_confidence)

    # -- Internal -----------------------------------------------------------

    def _print_menu(self) -> None:
        print("\n" + "=" * 70)
        print("  MOCK MICROPHONE — Twi Keyword Spotter Simulator")
        print("  (Vocabulary from Table 3.2 of the project report)")
        print("=" * 70)
        for idx, cmd in enumerate(TWI_VOCABULARY, start=1):
            print(f"  [{idx}]  {cmd.twi_phrase:<25}  →  {cmd.action}")
            print(f"        ({cmd.english_gloss})")
        print("\n  Type a number (1-7), the full Twi phrase, or 'quit' to stop.")
        print("=" * 70 + "\n")

    def _listen_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                user_input = input("[MIC] > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if user_input.lower() in ("quit", "exit", "q"):
                self._stop_event.set()
                break

            if not user_input:
                continue

            self._dispatch(user_input)

    def _dispatch(self, user_input: str) -> Optional[CommandEvent]:
        cmd = self._lookup.get(user_input.lower())
        if cmd is None:
            print(f"[MIC]  ✗  No grammar match for '{user_input}' (treated as silence)")
            return None
        return self._emit(cmd, self._simulated_confidence)

    def _emit(self, cmd: TwiCommand, confidence: float) -> Optional[CommandEvent]:
        # Simulate the confidence-threshold gate from the real engine.
        if confidence < self._confidence_threshold:
            print(f"[MIC]  ✗  Below threshold (conf={confidence:.2f} < {self._confidence_threshold}) — dropped")
            return None

        event = CommandEvent(
            twi_phrase=cmd.twi_phrase,
            action=cmd.action,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds") + "Z",
        )
        print(f"[MIC]  ✓  Recognized: '{cmd.twi_phrase}'  →  {cmd.action}  (conf={confidence:.2f})")

        if self._on_command:
            try:
                self._on_command(event)
            except Exception:
                logger.exception("on_command_callback raised")

        return event


# ---------------------------------------------------------------------------
# Standalone test harness — run directly to test the mock in isolation.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def _print_event(ev: CommandEvent):
        print(f"   ↳ DISPATCHED to control logic: action={ev.action}")

    mic = MockMicrophone(on_command_callback=_print_event)
    mic.start()
    try:
        mic._listener_thread.join()
    except KeyboardInterrupt:
        mic.stop()
        