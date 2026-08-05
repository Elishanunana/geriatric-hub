"""
wav_speaker.py
==============
Pre-recorded-audio speaker: plays fixed Twi WAV clips through the Pi's 3.5mm
jack, as a drop-in replacement for hardware_mocks.MockSpeaker. It preserves
the mock's public contract exactly, so the swap in main.py is a single
import-line alias change and nothing downstream (SOSHandler, CommandDispatcher,
ReminderScheduler) needs editing:

    # from src.hardware_mocks.mock_speaker import MockSpeaker
    from src.voice_engine.wav_speaker import WavSpeaker as MockSpeaker

Why pre-recorded (and not live TTS)
-----------------------------------
The neural VAHA TTS model is still training, so for the demo the hub speaks a
fixed set of pre-recorded prompts (the report's original TTS-output approach).
When a trained VAHA model is ready, this is a one-line swap for a VahaSpeaker;
the .speak(text) contract is identical, so nothing else changes.

How a prompt is matched to a WAV
--------------------------------
speak(text) receives the exact Twi sentence the control logic built. This class
maps that text to a WAV file two ways:

  1. FIXED prompts (SOS confirmation, appliance on/off, etc.) — matched by
     importing the real prompt constants from the control-logic modules, so the
     mapping can never drift from the code. Whitespace-normalised, so minor
     spacing differences still match.

  2. MEDICATION prompts (reminder / retry / missed / schedule read) — these are
     built dynamically with the drug name embedded, so they're matched by
     pattern + drug name (e.g. any "Mmere aso …" sentence containing
     "Amlodipine" -> reminder_amlodipine.wav). Adding a new drug later only
     needs a new recording named after it; nothing here changes.

If no WAV matches (e.g. a drug with no recording yet), the prompt is printed to
the console and the action still completes — the hub never crashes for a missing
clip.

Audio playback uses `aplay` (ALSA). The default output follows the system ALSA
default (set to the headphone jack via ~/.asoundrc); pass device=... to force a
specific one, e.g. device="plughw:2,0".

WAV files live in <project_root>/voice_prompts/ by default.

Author: Wise (Akabua Elisha Nunana) - KNUST COE 497
"""

from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Drugs we have (or may have) recordings for. Extend as recordings are added.
KNOWN_DRUGS: List[str] = ["Amlodipine", "Donepezil"]


def _project_root() -> Path:
    # this file: <root>/src/voice_engine/wav_speaker.py -> parents[2] == <root>
    return Path(__file__).resolve().parents[2]


class WavSpeaker:
    """
    Plays pre-recorded Twi WAV clips. Drop-in for MockSpeaker. See module
    docstring for the prompt->WAV matching rules and graceful fallback.
    """

    def __init__(
        self,
        prompts_dir: Optional[str] = None,
        device: Optional[str] = "plughw:2,0",
        *,
        simulate_latency: bool = False,   # accepted for MockSpeaker parity (unused)
        latency_per_word: float = 0.0,    # accepted for MockSpeaker parity (unused)
    ):
        """
        Parameters
        ----------
        prompts_dir : str, optional
            Folder containing the .wav clips. Defaults to
            <project_root>/voice_prompts/.
        device : str, optional
            ALSA device for aplay (e.g. "plughw:2,0" to force the headphone
            jack). None = system ALSA default (recommended; set via ~/.asoundrc).
        simulate_latency, latency_per_word :
            Accepted only so this constructs identically to MockSpeaker; unused
            (real playback duration is the clip length).
        """
        self._dir = Path(prompts_dir) if prompts_dir else (_project_root() / "voice_prompts")
        self._device = device
        self._last_spoken: Optional[str] = None
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None

        self._fixed: Dict[str, str] = {}
        self._schedule_header: Optional[str] = None
        self._build_fixed_map()

        n = sum(1 for f in self._fixed.values() if (self._dir / f).exists())
        logger.info("WavSpeaker: prompts_dir=%s (%d/%d fixed clips present)",
                    self._dir, n, len(self._fixed))
        if not self._dir.exists():
            print(f"[SPK]  !  voice prompts folder not found: {self._dir} "
                  f"(hub will still run; prompts print to console)")

    # -- Build the fixed-prompt -> filename map from the real constants ------

    def _build_fixed_map(self) -> None:
        """Import the actual prompt constants so the mapping matches the code
        exactly. Degrades to an empty fixed map if imports fail (dynamic +
        fallback still work)."""
        try:
            from src.control_logic.command_dispatcher import (
                PROMPT_APPLIANCE_ON, PROMPT_APPLIANCE_OFF,
                PROMPT_APPLIANCE_ALREADY_ON, PROMPT_APPLIANCE_ALREADY_OFF,
                PROMPT_NO_SCHEDULE_TODAY, PROMPT_RELAY_FAULT,
                PROMPT_SCHEDULE_HEADER,
            )
            from src.control_logic.sos_handler import DEFAULT_TTS_CONFIRMATION

            self._fixed = {
                DEFAULT_TTS_CONFIRMATION:      "sos_confirm.wav",
                PROMPT_APPLIANCE_ON:           "appliance_on.wav",
                PROMPT_APPLIANCE_OFF:          "appliance_off.wav",
                PROMPT_APPLIANCE_ALREADY_ON:   "appliance_already_on.wav",
                PROMPT_APPLIANCE_ALREADY_OFF:  "appliance_already_off.wav",
                PROMPT_NO_SCHEDULE_TODAY:      "no_schedule_today.wav",
                PROMPT_RELAY_FAULT:            "relay_fault.wav",
            }
            self._schedule_header = PROMPT_SCHEDULE_HEADER
        except Exception:  # noqa: BLE001 - never let a mapping import break boot
            logger.exception("WavSpeaker: could not import prompt constants; "
                             "fixed-prompt audio disabled (dynamic still works).")
            self._fixed = {}
            self._schedule_header = None

    # -- Prompt -> WAV resolution -------------------------------------------

    @staticmethod
    def _norm(s: str) -> str:
        return " ".join(s.split())

    def _detect_drug(self, text: str) -> Optional[str]:
        low = text.lower()
        for drug in KNOWN_DRUGS:
            if drug.lower() in low:
                return drug.lower()
        return None

    def _resolve(self, text: str) -> Optional[Path]:
        """Return the WAV path for this prompt, or None if we have no clip."""
        norm = self._norm(text)

        # 1. Fixed prompts (whitespace-normalised exact match).
        for phrase, fname in self._fixed.items():
            if self._norm(phrase) == norm:
                return self._dir / fname

        # 2. Schedule read: header + per-drug clauses.
        if self._schedule_header and norm.startswith(self._norm(self._schedule_header)):
            return self._dir / "schedule_read.wav"

        # 3. Medication prompts, keyed by pattern + drug name.
        drug = self._detect_drug(norm)
        if drug:
            if norm.startswith("Mmere aso"):
                return self._dir / f"reminder_{drug}.wav"
            if norm.startswith("Merekae wo bio"):
                return self._dir / f"reminder_retry_{drug}.wav"
            if "Yɛakyerɛw sɛ wonnom" in norm:
                return self._dir / f"missed_{drug}.wav"

        return None

    # -- Public API (matches MockSpeaker) -----------------------------------

    @property
    def last_spoken(self) -> Optional[str]:
        return self._last_spoken

    def speak(self, text: str) -> None:
        """Play the matching WAV (blocking). Prints the prompt to the console
        for demo visibility, then plays audio if a clip exists."""
        if not text:
            logger.warning("speak() called with empty text — ignored.")
            return
        with self._lock:
            self._last_spoken = text

        wav = self._resolve(text)
        self._print_block(text, wav)

        if wav is None:
            logger.info("No WAV for prompt (console-only): %r", text)
            return
        if not wav.exists():
            logger.warning("Prompt matched %s but file is missing.", wav.name)
            print(f"[SPK]  !  missing clip: {wav.name} (not recorded yet)")
            return
        self._play(wav)

    def _play(self, wav: Path) -> None:
        cmd = ["aplay", "-q"]
        if self._device:
            cmd += ["-D", self._device]
        cmd.append(str(wav))
        try:
            self._proc = subprocess.Popen(cmd)
            self._proc.wait()   # blocking, matches MockSpeaker.speak
        except FileNotFoundError:
            print("[SPK]  X  'aplay' not found — install alsa-utils "
                  "(sudo apt install alsa-utils).")
        except Exception as exc:  # noqa: BLE001
            logger.warning("aplay failed for %s: %s", wav.name, exc)
        finally:
            self._proc = None

    def speak_async(self, text: str) -> threading.Thread:
        """Fire-and-forget speak (used by the SOS path so SMS isn't blocked)."""
        t = threading.Thread(target=self.speak, args=(text,),
                             name="WavSpeaker-Async", daemon=True)
        t.start()
        return t

    def repeat_last(self) -> None:
        """Replay the last utterance (backs the REPEAT_LAST command)."""
        if self._last_spoken is None:
            print("[SPK]  ↻  nothing to repeat yet.")
            return
        print("[SPK]  ↻  Repeating last utterance...")
        self.speak(self._last_spoken)

    def stop(self) -> None:
        """Halt any current playback."""
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                p.terminate()
            except Exception:  # noqa: BLE001
                pass

    # -- Console output (keeps demo terminal readable) ----------------------

    def _print_block(self, text: str, wav: Optional[Path]) -> None:
        border = "─" * 70
        tag = f"▶ {wav.name}" if wav is not None else "(no recording — console only)"
        print(f"\n[SPK] {border}")
        print(f"[SPK]  🔊  TTS OUTPUT   {tag}")
        print(f"[SPK]      \"{text}\"")
        print(f"[SPK] {border}\n")


# ---------------------------------------------------------------------------
# Standalone test harness — plays a couple of clips if the folder exists.
#   python -m src.voice_engine.wav_speaker
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    spk = WavSpeaker(device=sys.argv[1] if len(sys.argv) > 1 else None)
    print("prompts dir:", spk._dir)
    # Try to speak the real SOS confirmation + an appliance prompt.
    from src.control_logic.sos_handler import DEFAULT_TTS_CONFIRMATION
    from src.control_logic.command_dispatcher import PROMPT_APPLIANCE_ON
    spk.speak(DEFAULT_TTS_CONFIRMATION)
    spk.speak(PROMPT_APPLIANCE_ON)
    spk.repeat_last()
