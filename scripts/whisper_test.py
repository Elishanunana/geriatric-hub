"""
whisper_test.py
===============
Evaluation harness for the fine-tuned Whisper model (the ggml /
whisper.cpp file a colleague trained on the seven Twi command recordings).

This is a STANDALONE EXPERIMENT, separate from the hub and from the
MFCC+DTW recogniser. Its only job is to answer one question honestly:
how well does this model transcribe the seven commands on FRESH speech —
audio it did not train on?

Why "fresh speech" matters
--------------------------
The model was trained on the recorded command audio. If you test it on
those same recordings (e.g. the voice_templates files), it will look
near-perfect and tell you nothing — that is the training set. The only
meaningful test is new audio: speak live, or record new clips. This tool
defaults to LIVE microphone capture for exactly that reason.

What it reports
---------------
For each utterance: the raw transcription text, how long inference took
(latency matters on the Raspberry Pi), and — since the model outputs free
text, not a command label — which of the seven commands the text most
closely matches. That mapping is what a real integration would need on
top of the model.

The model file
--------------
Put `ggml-model-q4_0.bin` in a `models/` folder at the project root:

    geriatric-hub1/models/ggml-model-q4_0.bin

…or pass its path with --model. The tool searches ./models/ and the
project root automatically.

Usage (from the project root, venv active; -m form keeps the command map
in sync with the project vocabulary)
------------------------------------------------------------------------
    python -m scripts.whisper_test                 # live mic, auto language
    python -m scripts.whisper_test --file clip.wav # transcribe one WAV
    python -m scripts.whisper_test --language en   # force a language code
    python -m scripts.whisper_test --list          # list audio devices

Dependencies
------------
    python -m pip install pywhispercpp numpy scipy sounddevice

Author: Wise (Asumang Pobi Godwin) — KNUST COE 497
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
import time
import unicodedata
from math import gcd

try:
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from pywhispercpp.model import Model
except ImportError:
    print(
        "\n  X Missing dependency. Install the Whisper test libraries first:\n\n"
        "      python -m pip install pywhispercpp numpy scipy sounddevice\n",
        file=sys.stderr,
    )
    sys.exit(1)

# Command vocabulary (optional) — enables the text->command mapping.
try:
    from src.hardware_mocks.mock_microphone import TWI_VOCABULARY
    VOCAB = [(c.action, c.twi_phrase) for c in TWI_VOCABULARY]
except Exception:
    VOCAB = []


SAMPLE_RATE   = 16000
MODEL_FILENAME = "ggml-model-q4_0.bin"


# ===========================================================================
# Model location
# ===========================================================================

def find_model(explicit: str | None) -> str:
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.append(os.path.join("models", MODEL_FILENAME))
    candidates.append(MODEL_FILENAME)
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    searched = "\n      ".join(os.path.abspath(c) for c in candidates)
    raise SystemExit(
        f"  X Model file not found. Looked in:\n      {searched}\n\n"
        f"    Put {MODEL_FILENAME} in a 'models' folder at the project root, "
        f"or pass --model <path>."
    )


# ===========================================================================
# Audio helpers
# ===========================================================================

def to_16k_f32(sig: "np.ndarray", sr: int) -> "np.ndarray":
    """Resample a float signal to 16 kHz float32 (what Whisper expects)."""
    if sr != SAMPLE_RATE:
        g = gcd(int(sr), SAMPLE_RATE)
        sig = resample_poly(sig, SAMPLE_RATE // g, int(sr) // g)
    return sig.astype(np.float32)


def load_wav_16k(path: str) -> "np.ndarray":
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
    return to_16k_f32(x, sr)


def record_from_mic(seconds: float, in_idx) -> "np.ndarray":
    """Record from the mic; return 16 kHz float32 mono. Imports sd lazily."""
    import sounddevice as sd
    dev = sd.query_devices(in_idx) if in_idx is not None else sd.query_devices(kind="input")
    sr = int(round(dev["default_samplerate"]))
    frames = int(round(seconds * sr))
    rec = sd.rec(frames, samplerate=sr, channels=1, dtype="int16", device=in_idx)
    sd.wait()
    return to_16k_f32(rec.reshape(-1).astype(np.float32) / 32768.0, sr)


def list_devices() -> None:
    import sounddevice as sd
    print(sd.query_devices())


# ===========================================================================
# Text normalisation + command mapping
# ===========================================================================

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))   # drop diacritics
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def map_to_commands(text: str, top: int = 3):
    """Return [(ratio, action, phrase), ...] best-first by string similarity."""
    if not VOCAB:
        return []
    nt = _norm(text)
    scored = [
        (difflib.SequenceMatcher(None, nt, _norm(phrase)).ratio(), action, phrase)
        for action, phrase in VOCAB
    ]
    scored.sort(reverse=True)
    return scored[:top]


# ===========================================================================
# Transcription
# ===========================================================================

def transcribe(model: "Model", audio: "np.ndarray", language: str | None) -> tuple[str, float]:
    """Transcribe 16 kHz float32 audio. Returns (text, seconds_elapsed)."""
    kwargs = {}
    if language and language.lower() not in ("", "auto"):
        kwargs["language"] = language
    t0 = time.perf_counter()
    segments = model.transcribe(audio, **kwargs)
    elapsed = time.perf_counter() - t0
    text = " ".join(s.text.strip() for s in segments).strip()
    return text, elapsed


def show(text: str, elapsed: float) -> None:
    print("  " + "-" * 56)
    if text:
        print(f'  TRANSCRIPTION:  "{text}"')
    else:
        print("  TRANSCRIPTION:  (empty — no speech detected)")
    print(f"  inference time: {elapsed:.2f}s")
    mapped = map_to_commands(text)
    if mapped:
        print("  closest commands:")
        for ratio, action, phrase in mapped:
            print(f"    {ratio*100:5.1f}%  {action:<16} \"{phrase}\"")
    print("  " + "-" * 56)


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test the fine-tuned Whisper command model on fresh speech."
    )
    parser.add_argument("--model", default=None, help=f"Path to {MODEL_FILENAME}.")
    parser.add_argument("--file", default=None, help="Transcribe a WAV file and exit.")
    parser.add_argument("--language", default="auto",
                        help="Language code (e.g. en). Default: auto-detect. Ask your "
                             "colleague which code the model was trained with.")
    parser.add_argument("--input", default=None,
                        help="(live) Input device index. Default: system default.")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="(live) Seconds to record per utterance.")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for inference.")
    parser.add_argument("--list", action="store_true", help="List audio devices and exit.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show whisper.cpp's internal load logs.")
    args = parser.parse_args()

    print("\n" + "=" * 64)
    print("  WHISPER MODEL TESTER  (fine-tuned Twi command model)")
    print("=" * 64)

    if args.list:
        list_devices()
        return 0

    try:
        model_path = find_model(args.model)
    except SystemExit as exc:
        print(exc)
        return 1

    print(f"  Model    : {model_path}")
    print(f"  Language : {args.language}")
    if not VOCAB:
        print("  (command vocabulary not importable — run with -m from the project "
              "root to enable text->command mapping)")

    # Load the model once. Suppress whisper.cpp's verbose C++ logs unless --verbose.
    redirect = False if args.verbose else open(os.devnull, "w")
    print("  Loading model...")
    try:
        model = Model(
            model_path,
            redirect_whispercpp_logs_to=redirect,
            n_threads=args.threads,
            print_realtime=False,
            print_progress=False,
            print_timestamps=False,
        )
    except Exception as exc:
        print(f"\n  X Failed to load model: {exc}\n")
        return 1
    print("  Model loaded.\n")

    # --- Single file --------------------------------------------------
    if args.file:
        if not os.path.isfile(args.file):
            print(f"  X File not found: {args.file}\n")
            return 1
        audio = load_wav_16k(args.file)
        text, elapsed = transcribe(model, audio, args.language)
        print(f"  File: {args.file}")
        show(text, elapsed)
        print()
        return 0

    # --- Live loop ----------------------------------------------------
    in_idx = None
    if args.input is not None:
        try:
            in_idx = int(args.input)
        except ValueError:
            in_idx = args.input

    print("  LIVE MODE — press Enter, then speak a command. (q to quit.)")
    print("  Reminder: test FRESH speech, not the files the model trained on.")
    while True:
        try:
            choice = input("\n  [Enter]=record  q=quit > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if choice == "q":
            break
        print(f"  >> recording {args.duration:g}s — say a command now...", flush=True)
        try:
            audio = record_from_mic(args.duration, in_idx)
        except ImportError:
            print("\n  X sounddevice not installed. Run: python -m pip install sounddevice\n")
            return 1
        except Exception as exc:
            print(f"\n  X Recording failed: {exc}\n")
            return 1
        print("  >> transcribing...", flush=True)
        text, elapsed = transcribe(model, audio, args.language)
        show(text, elapsed)

    print("\n  Done.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    