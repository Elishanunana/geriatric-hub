"""
whisper_spotter.py
==================
Real voice-engine adapter: a fine-tuned Whisper (whisper.cpp / ggml) model
wrapped behind the EXACT public contract of hardware_mocks.MockMicrophone,
so it drops into main.py's Step-3 hardware wiring with a single import-line
change and no downstream edits.

Relationship to the report
---------------------------
The report (Section 2.3, citation [19]) specifies a *constrained keyword
spotting* paradigm for a small, fixed command vocabulary. The original
design named a grammar-constrained Vosk recogniser; no usable Vosk Twi
model exists, so this implementation substitutes a fine-tuned Whisper model
that transcribes to free text, then projects that text onto the closed
7-command vocabulary via string similarity. The transcription engine
changed; the constrained-vocabulary principle did not.

Recognition pipeline (mirrors scripts/whisper_test.py, which was validated
empirically at 7/8 correct action mapping, ~3 s inference on the Pi 4):

    mic audio (3 s)  ->  16 kHz float32  ->  Whisper transcription (free text)
                     ->  diacritic-normalised string-similarity match
                     ->  best command + ratio  ->  gate at threshold
                     ->  CommandEvent  ->  self._on_command (fan-out)

Design decisions baked in (from empirical test results)
-------------------------------------------------------
* Confidence == the difflib.SequenceMatcher ratio (0.0-1.0) between the
  normalised transcription and the matched command phrase. Same 0-1 scale
  as the existing confidence_threshold, and fully traceable to whisper_test.
* Default confidence_threshold = 0.65. This rejects the 0.58 misrecognition
  observed during testing while accepting the 0.66-1.00 correct band.
* Press-to-record / prompt-driven, NOT continuous listening. Inference is
  ~3 s, so a continuous stream would lag; the user presses Enter, then speaks.
  (On the Pi this Enter can later be replaced by a GPIO push-to-talk button
  without changing this class's contract.)
* SOS via voice is a *secondary* trigger only. The physical SOS button wired
  in main.py Step 6c remains the primary, safety-critical path; this class
  simply emits SOS like any other command when the voice path recognises it.

Drop-in contract preserved from MockMicrophone
----------------------------------------------
* Constructible with zero arguments: WhisperSpotter().
* self._on_command is a public attribute slot, assigned by main.py AFTER
  construction and read at emit time (never captured at construction).
* Public API: .listen(), .start(), .stop(), .inject_command(), .vocabulary.
* Emits the identical CommandEvent dataclass (imported, not redefined).
* .inject_command() needs NO model and NO audio stack — the dev-console
  injection fallback and the 30/30 integration tests keep working unchanged.

Heavy dependencies (pywhispercpp, numpy, scipy, sounddevice) and the model
file are imported/loaded LAZILY, inside the methods that actually record or
transcribe. Importing this module on a laptop or in CI with none of that
installed is safe.

    python -m pip install pywhispercpp numpy scipy sounddevice

Standalone smoke test (from the project root, venv active):

    python -m src.voice_engine.whisper_spotter            # live press-to-record
    python -m src.voice_engine.whisper_spotter --file clip.wav

Author: Wise (Akabua Elisha Nunana) - KNUST COE 497
"""

from __future__ import annotations

import logging
import os
import re
import threading
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from math import gcd
from typing import Any, Callable, List, Optional, Tuple

# The CommandEvent, TwiCommand, vocabulary and action constants live with the
# microphone mock today. Importing them here (rather than redefining) means
# WhisperSpotter emits the SAME dataclass type the whole system already
# consumes -- the contract is identical by construction.
from src.hardware_mocks.mock_microphone import (
    CommandEvent,
    TwiCommand,
    TWI_VOCABULARY,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (match scripts/whisper_test.py)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
MODEL_FILENAME = "ggml-model-q4_0.bin"


# ===========================================================================
# Module-level helpers (audio + text). Heavy libs imported lazily inside.
# ===========================================================================

def find_model(explicit: Optional[str]) -> str:
    """
    Locate the ggml model. Searches: explicit path, ./models/<name>, ./<name>.
    Raises FileNotFoundError with a helpful message if none is found.
    """
    candidates: List[str] = []
    if explicit:
        candidates.append(explicit)
    candidates.append(os.path.join("models", MODEL_FILENAME))
    candidates.append(MODEL_FILENAME)
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    searched = "\n      ".join(os.path.abspath(c) for c in candidates)
    raise FileNotFoundError(
        f"Whisper model not found. Looked in:\n      {searched}\n"
        f"Put {MODEL_FILENAME} in a 'models/' folder at the project root, "
        f"or pass model_path=<path>."
    )


def _to_16k_f32(sig: "Any", sr: int) -> "Any":
    """Resample a float signal to 16 kHz float32 (what Whisper expects)."""
    import numpy as np
    from scipy.signal import resample_poly
    if sr != SAMPLE_RATE:
        g = gcd(int(sr), SAMPLE_RATE)
        sig = resample_poly(sig, SAMPLE_RATE // g, int(sr) // g)
    return sig.astype(np.float32)


def load_wav_16k(path: str) -> "Any":
    """Load a WAV file as 16 kHz float32 mono."""
    import numpy as np
    from scipy.io import wavfile
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if data.dtype == np.int16:
        x = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        x = (data.astype(np.float32) - 128.0) / 128.0
    else:
        x = data.astype(np.float32)
    return _to_16k_f32(x, sr)


def record_from_mic(seconds: float, in_idx: Any) -> "Any":
    """Record from the mic; return 16 kHz float32 mono. Imports sd lazily."""
    import numpy as np
    import sounddevice as sd
    dev = sd.query_devices(in_idx) if in_idx is not None else sd.query_devices(kind="input")
    sr = int(round(dev["default_samplerate"]))
    frames = int(round(seconds * sr))
    rec = sd.rec(frames, samplerate=sr, channels=1, dtype="int16", device=in_idx)
    sd.wait()
    return _to_16k_f32(rec.reshape(-1).astype(np.float32) / 32768.0, sr)


def list_devices() -> None:
    import sounddevice as sd
    print(sd.query_devices())


def _norm(s: str) -> str:
    """Lower-case, strip diacritics and punctuation for robust matching."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


# ===========================================================================
# WhisperSpotter
# ===========================================================================

class WhisperSpotter:
    """
    Fine-tuned Whisper keyword spotter, drop-in for MockMicrophone.

    See the module docstring for the full contract. In brief: construct it,
    let main.py assign ._on_command, then .start() the press-to-record loop.
    Every recognised utterance becomes a CommandEvent delivered to
    ._on_command, exactly as the mock did.
    """

    def __init__(
        self,
        on_command_callback: Optional[Callable[[CommandEvent], None]] = None,
        confidence_threshold: float = 0.65,
        *,
        model_path: Optional[str] = None,
        language: str = "auto",
        record_seconds: float = 3.0,
        input_device: Any = None,
        n_threads: int = 4,
        simulated_confidence: float = 1.0,   # accepted for MockMicrophone parity
        energy_threshold: float = 0.02,      # RMS gate for speech onset (autonomous)
        frame_seconds: float = 0.4,          # length of each onset-detection frame
        post_command_cooldown: float = 1.0,  # quiet period after each recognition
        calibrate_seconds: float = 1.0,      # ambient noise sampling at startup
    ):
        """
        Parameters
        ----------
        on_command_callback : callable, optional
            Invoked with a CommandEvent on each recognised command. main.py
            assigns the fan-out to ._on_command AFTER construction, so this
            may be left None here.
        confidence_threshold : float
            Similarity ratio below which a recognition is rejected as silence
            / out-of-vocabulary. Default 0.65 (see module docstring).
        model_path : str, optional
            Path to ggml-model-q4_0.bin. If None, searched at load time.
        language : str
            Whisper language code, or "auto". Default "auto".
        record_seconds : float
            Seconds of audio captured per press-to-record cycle. Default 3.0.
        input_device : int | str | None
            sounddevice input index/name, or None for the system default.
        n_threads : int
            CPU threads for whisper.cpp inference. Default 4 (Pi 4 has 4 cores).
        simulated_confidence : float
            Confidence assigned to programmatically injected commands
            (.inject_command). Named to mirror MockMicrophone's constructor;
            for real voice, confidence comes from the model, not this value.
        """
        self._on_command = on_command_callback
        self._confidence_threshold = confidence_threshold
        self._model_path = model_path
        self._language = language
        self._record_seconds = record_seconds
        self._input_device = input_device
        self._n_threads = n_threads
        self._injected_confidence = simulated_confidence
        self._energy_threshold = energy_threshold
        self._frame_seconds = frame_seconds
        self._post_command_cooldown = post_command_cooldown
        self._calibrate_seconds = calibrate_seconds
        self._effective_threshold = energy_threshold

        # Lazily-loaded whisper.cpp model handle.
        self._model: Any = None
        self._model_load_failed = False

        # Background press-to-record thread + stop flag.
        self._listener_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # (action, phrase) pairs used for the similarity projection, plus a
        # lookup for inject_command (phrase / index / action -> TwiCommand).
        self._vocab: List[Tuple[str, str]] = [
            (c.action, c.twi_phrase) for c in TWI_VOCABULARY
        ]
        self._lookup = {}
        for idx, cmd in enumerate(TWI_VOCABULARY, start=1):
            self._lookup[str(idx)] = cmd
            self._lookup[cmd.twi_phrase.lower()] = cmd

    # -- Public API ---------------------------------------------------------

    @property
    def vocabulary(self) -> List[TwiCommand]:
        """The recognised Twi command set (Table 3.2)."""
        return list(TWI_VOCABULARY)

    def listen(self, prompt: str = "[VOICE] Press Enter, then speak > ") -> Optional[CommandEvent]:
        """
        Blocking single-shot capture: wait for Enter, record one utterance,
        transcribe, project onto the vocabulary, and return a CommandEvent
        (or None if nothing cleared the threshold). Mirrors MockMicrophone.listen.
        """
        try:
            input(prompt)
        except (EOFError, KeyboardInterrupt):
            return None
        return self._capture_and_recognise()

    def start(self) -> None:
        """
        Begin the press-to-record loop on a background daemon thread. The
        model is loaded once at the top of the loop (off the main thread, so
        boot is not blocked). Mirrors MockMicrophone.start's threading model.
        """
        if self._listener_thread and self._listener_thread.is_alive():
            logger.warning("WhisperSpotter already started.")
            return
        self._stop_event.clear()
        self._listener_thread = threading.Thread(
            target=self._listen_loop, name="WhisperSpotter-Listener", daemon=True
        )
        self._listener_thread.start()

    def stop(self) -> None:
        """Signal the listener thread to exit on its next frame."""
        self._stop_event.set()
        print("[VOICE] Listener stopping.")

    def inject_command(
        self,
        twi_phrase_or_action: str,
        confidence: Optional[float] = None,
    ) -> Optional[CommandEvent]:
        """
        Programmatic command injection -- bypasses audio and the model
        entirely. Accepts a Twi phrase, a 1-7 index, or an action constant
        (e.g. "SOS"). Used by the dev console and integration tests, and
        requires no whisper/audio dependency. Fires ._on_command, exactly
        like the mock, so the fan-out runs.
        """
        cmd = self._lookup.get(twi_phrase_or_action.lower())
        if cmd is None:
            for c in TWI_VOCABULARY:
                if c.action == twi_phrase_or_action:
                    cmd = c
                    break
        if cmd is None:
            logger.warning("inject_command: unknown phrase/action %r", twi_phrase_or_action)
            return None
        return self._emit(
            cmd,
            confidence if confidence is not None else self._injected_confidence,
            source="inject",
        )

    # -- Internal: recognition ---------------------------------------------

    def _ensure_model(self) -> bool:
        """Load the whisper.cpp model once. Returns False (and logs) on failure."""
        if self._model is not None:
            return True
        if self._model_load_failed:
            return False
        try:
            from pywhispercpp.model import Model
        except ImportError:
            self._model_load_failed = True
            print("[VOICE]  X pywhispercpp not installed. Run: "
                  "python -m pip install pywhispercpp numpy scipy sounddevice")
            return False
        try:
            path = find_model(self._model_path)
        except FileNotFoundError as exc:
            self._model_load_failed = True
            print(f"[VOICE]  X {exc}")
            return False
        print(f"[VOICE] Loading model: {path}")
        try:
            self._model = Model(
                path,
                redirect_whispercpp_logs_to=open(os.devnull, "w"),
                n_threads=self._n_threads,
                print_realtime=False,
                print_progress=False,
                print_timestamps=False,
            )
        except Exception as exc:  # noqa: BLE001 - want to keep the hub alive
            self._model_load_failed = True
            print(f"[VOICE]  X Failed to load model: {exc}")
            return False
        print("[VOICE] Model loaded.")
        return True

    def _transcribe(self, audio: "Any") -> Tuple[str, float]:
        """Transcribe 16 kHz float32 audio. Returns (text, seconds_elapsed)."""
        import time
        kwargs = {}
        if self._language and self._language.lower() not in ("", "auto"):
            kwargs["language"] = self._language
        t0 = time.perf_counter()
        segments = self._model.transcribe(audio, **kwargs)
        elapsed = time.perf_counter() - t0
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, elapsed

    def _best_match(self, text: str) -> Tuple[float, Optional[TwiCommand]]:
        """Project free text onto the vocabulary; return (best_ratio, command)."""
        nt = _norm(text)
        if not nt:
            return 0.0, None
        best_ratio = 0.0
        best_cmd: Optional[TwiCommand] = None
        for action, phrase in self._vocab:
            ratio = SequenceMatcher(None, nt, _norm(phrase)).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                # Resolve the TwiCommand for this action.
                for c in TWI_VOCABULARY:
                    if c.action == action:
                        best_cmd = c
                        break
        return best_ratio, best_cmd

    def _capture_and_recognise(self, audio: Optional["Any"] = None) -> Optional[CommandEvent]:
        """One full cycle: (record if needed) -> transcribe -> match -> emit.

        If `audio` is supplied (autonomous listen path), it is used as-is;
        otherwise a fresh recording is captured (interactive listen path).
        """
        if not self._ensure_model():
            return None
        if audio is None:
            print(f"[VOICE] Recording {self._record_seconds:g}s -- speak now...", flush=True)
            try:
                audio = record_from_mic(self._record_seconds, self._input_device)
            except ImportError:
                print("[VOICE]  X sounddevice not installed. Run: "
                      "python -m pip install sounddevice")
                return None
            except Exception as exc:  # noqa: BLE001
                print(f"[VOICE]  X Recording failed: {exc}")
                return None

        print("[VOICE] Transcribing...", flush=True)
        text, elapsed = self._transcribe(audio)
        ratio, cmd = self._best_match(text)

        display = text if text else "(empty)"
        print(f'[VOICE] Heard: "{display}"  ({elapsed:.2f}s)')
        if cmd is None:
            print("[VOICE]  X No vocabulary match -- treated as silence")
            return None
        print(f"[VOICE] Closest: {cmd.action}  (ratio={ratio:.2f})")
        return self._emit(cmd, ratio, source="voice")

    # -- Internal: emit -----------------------------------------------------

    def _emit(self, cmd: TwiCommand, confidence: float, source: str) -> Optional[CommandEvent]:
        """Gate on the threshold, build a CommandEvent, deliver to ._on_command."""
        if confidence < self._confidence_threshold:
            print(f"[VOICE]  X Below threshold "
                  f"(conf={confidence:.2f} < {self._confidence_threshold}) -- dropped")
            return None

        event = CommandEvent(
            twi_phrase=cmd.twi_phrase,
            action=cmd.action,
            confidence=confidence,
            # Timestamp format mirrors MockMicrophone exactly so the emitted
            # event is byte-for-byte contract-identical downstream.
            timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds") + "Z",
        )
        tag = "inject" if source == "inject" else "voice"
        print(f"[VOICE]  OK  Recognized ({tag}): '{cmd.twi_phrase}'  ->  "
              f"{cmd.action}  (conf={confidence:.2f})")

        if self._on_command:
            try:
                self._on_command(event)
            except Exception:  # noqa: BLE001 - isolate downstream faults
                logger.exception("on_command_callback raised")
        return event

    # -- Internal: listen loop ---------------------------------------------

    def _listen_loop(self) -> None:
        # Preload the model up front (on this worker thread) so the first
        # recognition is not delayed by the load. A failure here is non-fatal:
        # the dev-console injection pathway still works without the model.
        self._ensure_model()
        self._effective_threshold = self._energy_threshold
        self._print_ready_banner()

        # Sample the ambient noise floor once so a loud room doesn't trigger
        # constant false captures. In a quiet room the static default wins.
        try:
            self._calibrate_noise_floor()
        except Exception as exc:  # noqa: BLE001
            logger.debug("noise calibration skipped: %s", exc)

        # Autonomous energy-gated loop: cheap and silent until speech is heard,
        # then capture one utterance and recognise it. No stdin — runs headless
        # under systemd at boot.
        while not self._stop_event.is_set():
            trigger = self._wait_for_speech()
            if trigger is None:
                break  # stop requested (or audio stack unavailable)
            audio = self._capture_utterance(prefix=trigger)
            if audio is not None:
                self._capture_and_recognise(audio=audio)
            # Quiet period so the utterance tail — and any spoken response the
            # hub plays back — doesn't immediately re-trigger the mic.
            if self._stop_event.wait(self._post_command_cooldown):
                break

    def _read_frame(self, seconds: float) -> Tuple[Optional["Any"], float]:
        """Record one short frame; return (audio_16k_f32, rms_energy).
        Returns (None, 0.0) if the audio stack is unavailable or errors."""
        try:
            import numpy as np
            audio = record_from_mic(seconds, self._input_device)
        except ImportError:
            return None, 0.0
        except Exception as exc:  # noqa: BLE001
            logger.debug("frame capture failed: %s", exc)
            return None, 0.0
        if audio is None or len(audio) == 0:
            return None, 0.0
        rms = float(np.sqrt(np.mean(np.square(audio))))
        return audio, rms

    def _calibrate_noise_floor(self) -> None:
        """Measure ambient RMS briefly and raise the speech threshold above the
        noise floor if the room is loud; never drops below the static default.
        Logs the chosen threshold for tuning."""
        _, floor = self._read_frame(self._calibrate_seconds)
        adaptive = floor * 2.5 + 0.008
        self._effective_threshold = max(self._energy_threshold, adaptive)
        print(f"[VOICE] Ambient noise floor RMS={floor:.4f}  ->  "
              f"speech threshold={self._effective_threshold:.4f}")

    def _wait_for_speech(self) -> Optional["Any"]:
        """Block in short frames until RMS crosses the speech threshold or a
        stop is signalled. Returns the triggering audio frame (so the onset
        isn't clipped from the capture) or None when stopping."""
        while not self._stop_event.is_set():
            audio, rms = self._read_frame(self._frame_seconds)
            if audio is None:
                # Audio stack unavailable — back off instead of spinning hot.
                if self._stop_event.wait(1.0):
                    return None
                continue
            if rms >= self._effective_threshold:
                logger.debug("speech onset detected (rms=%.4f)", rms)
                return audio
        return None

    def _capture_utterance(self, prefix: Optional["Any"] = None) -> Optional["Any"]:
        """Record the remainder of an utterance and prepend the trigger frame
        so the first syllable isn't lost. Returns 16 kHz float32 audio."""
        import numpy as np
        remainder = max(0.0, self._record_seconds - self._frame_seconds)
        tail = None
        if remainder > 0:
            try:
                tail = record_from_mic(remainder, self._input_device)
            except Exception as exc:  # noqa: BLE001
                logger.debug("utterance capture failed: %s", exc)
                tail = None
        parts = [p for p in (prefix, tail) if p is not None and len(p) > 0]
        if not parts:
            return None
        return np.concatenate(parts)

    def _print_ready_banner(self) -> None:
        print("\n" + "=" * 70)
        print("  WHISPER SPOTTER -- Twi voice command engine (autonomous)")
        print("  Listening continuously; acts only on in-vocabulary commands.")
        print("  (Vocabulary from Table 3.2 of the project report)")
        print("=" * 70)
        for idx, cmd in enumerate(TWI_VOCABULARY, start=1):
            print(f"  [{idx}]  {cmd.twi_phrase:<25}  ->  {cmd.action}")
        print("=" * 70 + "\n")


# ===========================================================================
# Standalone harness -- run directly to exercise the spotter in isolation.
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test WhisperSpotter in isolation.")
    parser.add_argument("--model", default=None, help=f"Path to {MODEL_FILENAME}.")
    parser.add_argument("--file", default=None, help="Transcribe one WAV and exit.")
    parser.add_argument("--language", default="auto", help="Language code or 'auto'.")
    parser.add_argument("--input", default=None, help="Input device index/name.")
    parser.add_argument("--duration", type=float, default=3.0, help="Seconds per utterance.")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads.")
    args = parser.parse_args()

    def _print_event(ev: CommandEvent) -> None:
        print(f"   -> DISPATCHED to control logic: action={ev.action} "
              f"conf={ev.confidence:.2f}")

    in_idx: Any = args.input
    if isinstance(args.input, str) and args.input.isdigit():
        in_idx = int(args.input)

    spotter = WhisperSpotter(
        on_command_callback=_print_event,
        model_path=args.model,
        language=args.language,
        record_seconds=args.duration,
        input_device=in_idx,
        n_threads=args.threads,
    )

    if args.file:
        if not spotter._ensure_model():
            raise SystemExit(1)
        _audio = load_wav_16k(args.file)
        _text, _elapsed = spotter._transcribe(_audio)
        _ratio, _cmd = spotter._best_match(_text)
        print(f'File: {args.file}')
        print(f'Transcription: "{_text}"  ({_elapsed:.2f}s)')
        if _cmd:
            spotter._emit(_cmd, _ratio, source="voice")
        else:
            print("No vocabulary match.")
        raise SystemExit(0)

    spotter.start()
    try:
        if spotter._listener_thread:
            spotter._listener_thread.join()
    except KeyboardInterrupt:
        spotter.stop()
