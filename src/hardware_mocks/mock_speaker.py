"""
mock_speaker.py
===============
Terminal-based simulator for the 5W speaker + Piper Twi TTS engine
(Sections 3.4.2 and 3.5.1 of the project report).

Behavior Contract
-----------------
The real TTSEngine (voice_engine/tts_engine.py) will:
  - Load a GhanaNLP-sourced Twi voice model into Piper.
  - Synthesize a WAV buffer for any input text (typical latency 1.2–1.8 s
    for a 10–15 word reminder sentence on the Pi 4).
  - Play the buffer through ALSA on the 3.5mm audio jack.

This mock preserves the public API so the swap is transparent:
  - .speak(text)               — synthesize + play (blocking)
  - .speak_async(text)         — fire-and-forget
  - .last_spoken               — for the REPEAT_LAST command path

Replacement Plan
----------------
Swap the import in main.py:
    from hardware_mocks.mock_speaker import MockSpeaker as TTSEngine
to:
    from voice_engine.tts_engine import TTSEngine

Author: Wise (Asumang Pobi Godwin) - KNUST COE 497
"""

import threading
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MockSpeaker:
    """
    Prints simulated TTS output to the terminal in a visually distinct format.
    Maintains a record of the last utterance for the REPEAT_LAST command.
    """

    def __init__(self, simulate_latency: bool = True, latency_per_word: float = 0.10):
        """
        Parameters
        ----------
        simulate_latency : bool
            If True, .speak() sleeps for an interval proportional to the text
            length, mimicking the real Piper synthesis + playback time.
            Set to False in unit tests to keep them fast.
        latency_per_word : float
            Approximate seconds per word. Tuned to ~1.5s for a 15-word
            sentence — consistent with the real Pi 4 measurement in
            Section 3.5.1 of the report.
        """
        self._simulate_latency = simulate_latency
        self._latency_per_word = latency_per_word
        self._last_spoken: Optional[str] = None
        self._lock = threading.Lock()

    # -- Public API ---------------------------------------------------------

    @property
    def last_spoken(self) -> Optional[str]:
        """The most recent utterance — used by the REPEAT_LAST command."""
        return self._last_spoken

    def speak(self, text: str) -> None:
        """
        Synthesize and play a Twi (or English) utterance. Blocks until
        playback completes — matches the real engine's synchronous behavior
        used by the reminder scheduler (Section 3.5.2).
        """
        if not text:
            logger.warning("speak() called with empty text — ignored.")
            return

        with self._lock:
            self._last_spoken = text

        word_count = len(text.split())
        latency = word_count * self._latency_per_word if self._simulate_latency else 0.0

        # Visually distinct two-line block so TTS output is unmissable in a
        # busy terminal log alongside scheduler ticks and SMS handler output.
        border = "─" * 70
        print(f"\n[SPK] {border}")
        print(f"[SPK]  🔊  TTS OUTPUT  ({word_count} words, ~{latency:.1f}s synthesis)")
        print(f"[SPK]      \"{text}\"")
        print(f"[SPK] {border}\n")

        if self._simulate_latency:
            time.sleep(latency)

    def speak_async(self, text: str) -> threading.Thread:
        """
        Fire-and-forget speak. Returns the worker thread for callers that
        need to join later. Used by the SOS handler so the alert SMS dispatch
        is not blocked by TTS playback.
        """
        t = threading.Thread(
            target=self.speak,
            args=(text,),
            name=f"MockSpeaker-Async",
            daemon=True,
        )
        t.start()
        return t

    def repeat_last(self) -> None:
        """Re-play the most recent utterance. Backs the REPEAT_LAST command."""
        if self._last_spoken is None:
            self.speak("Mensesee biribiara.")  # "I haven't said anything yet."
            return
        print("[SPK]  ↻  Repeating last utterance...")
        self.speak(self._last_spoken)

    def stop(self) -> None:
        """No-op for the mock; real engine would halt ALSA playback."""
        logger.debug("MockSpeaker.stop() — no-op")


# ---------------------------------------------------------------------------
# Standalone test harness
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    spk = MockSpeaker(simulate_latency=True)
    spk.speak("Mema wo akye. Bere a wo bεnom wo aduru no aduru.")
    spk.speak("Yε wo medaase.")
    spk.repeat_last()
    