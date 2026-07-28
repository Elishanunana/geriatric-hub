"""
record_templates.py
====================
Records the Asante Twi command templates that the MFCC+DTW keyword spotter
will match against at runtime. This is the data-collection step that turns
the closed 7-command vocabulary (Table 3.2) into a set of reference
recordings.

For each command it captures several takes, lets you keep/redo each one
based on an immediate signal report, resamples every kept take to the
pipeline's target format (16 kHz, mono, int16), and saves it under:

    voice_templates/<ACTION>/<speaker>_<NN>.wav

e.g.  voice_templates/SOS/wise_01.wav

The vocabulary is imported from src.hardware_mocks.mock_microphone so this
tool, the mock, and the real keyword spotter all share ONE source of truth
for the command set — add or change a command in one place and everything
stays aligned.

How to run
----------
From the project root, venv activated. It MUST be run with -m (it imports
from `src`, which only resolves when the project root is the working dir):

    python -m scripts.record_templates --speaker wise

Useful flags:
    --speaker  wise        label for this voice (subdir-safe; required)
    --input    1           input device index (run sanity_check --list)
    --takes    5           takes to record per command this session
    --duration 3.0         seconds per take (raise for longer phrases)
    --only     SOS,3       restrict to some commands (action name or index)
    --output   4           device for the optional [p]layback

Per-take controls
-----------------
After each take you'll see a peak/RMS report, then:
    [Enter] keep    [p] play it back    [r] redo    [s] skip command    [q] quit

Resumable
---------
Re-running continues numbering after whatever already exists for that
speaker — it never overwrites earlier takes. So you can record in several
sittings, or top up a command that needs more samples.

Dependencies
------------
    pip install sounddevice numpy scipy

(You already have sounddevice + numpy from the sanity check; scipy is new —
it provides the high-quality resampler.)

Author: Wise (Asumang Pobi Godwin) — KNUST COE 497
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
import time
import wave

# Twi phrases contain non-ASCII characters (e.g. ε). Force UTF-8 stdout so
# printing the vocabulary can't crash on a legacy cp1252 console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    import numpy as np
    import sounddevice as sd
    from scipy.signal import resample_poly
except ImportError:
    print(
        "\n  X Missing dependency. Install the audio + DSP libraries:\n\n"
        "      pip install sounddevice numpy scipy\n",
        file=sys.stderr,
    )
    sys.exit(1)

# Single source of truth for the command set. Lightweight import — defines
# dataclasses + constants only, no DB or hardware side effects.
from src.hardware_mocks.mock_microphone import TWI_VOCABULARY


# ===========================================================================
# Configuration
# ===========================================================================

TARGET_SR          = 16000     # Pipeline target: 16 kHz mono int16.
DEFAULT_TAKES      = 5
DEFAULT_DURATION   = 3.0
DEFAULT_CHANNELS   = 1
SAMPLE_WIDTH_BYTES = 2
FULL_SCALE_INT16   = 32767
LEAD_IN_SECONDS    = 0.4       # Pause after Enter so the keypress isn't captured.

SILENCE_PEAK_FRACTION  = 0.01
CLIPPING_PEAK_FRACTION = 0.99

TEMPLATES_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "voice_templates")
)


# ===========================================================================
# Device resolution (same scheme as the sanity-check script)
# ===========================================================================

def _resolve_device(arg, want_input: bool) -> int:
    """index/name/None -> concrete device index for the given direction."""
    devices = sd.query_devices()
    kind    = "input" if want_input else "output"
    ch_key  = "max_input_channels" if want_input else "max_output_channels"

    if arg is None:
        idx = sd.default.device[0 if want_input else 1]
        if idx is None or idx < 0:
            raise SystemExit(
                f"  X No default {kind} device. Pass --{kind} <index> "
                f"(run: python -m scripts.sanity_check_audio --list)."
            )
        return int(idx)

    try:
        return int(arg)
    except ValueError:
        pass

    matches = [
        i for i, d in enumerate(devices)
        if arg.lower() in d["name"].lower() and d[ch_key] > 0
    ]
    if not matches:
        raise SystemExit(f"  X No {kind} device name contains {arg!r}.")
    if len(matches) > 1:
        names = ", ".join(f"[{i}] {devices[i]['name']}" for i in matches)
        raise SystemExit(f"  X {arg!r} is ambiguous: {names}. Pass an index.")
    return matches[0]


# ===========================================================================
# Capture, analysis, resample, save
# ===========================================================================

def record_take(in_idx: int, src_sr: int, channels: int, duration: float) -> "np.ndarray":
    """Record one fixed-length take of int16 audio from `in_idx`."""
    time.sleep(LEAD_IN_SECONDS)               # Dodge the Enter-key transient.
    print("     >> recording...", end="", flush=True)
    rec = sd.rec(
        int(round(duration * src_sr)),
        samplerate=src_sr,
        channels=channels,
        dtype="int16",
        device=in_idx,
    )
    sd.wait()
    print(" done.")
    return rec


def signal_status(rec: "np.ndarray"):
    """Return (peak_fraction, rms_fraction, status) for a take."""
    # int32 before abs() so a lone -32768 sample can't overflow int16.
    peak = int(np.max(np.abs(rec.astype(np.int32)))) if rec.size else 0
    rms  = (float(np.sqrt(np.mean(rec.astype(np.float64) ** 2)))
            if rec.size else 0.0)
    peak_frac = peak / FULL_SCALE_INT16
    rms_frac  = rms  / FULL_SCALE_INT16

    if peak_frac < SILENCE_PEAK_FRACTION:
        status = "SILENCE"
    elif peak_frac > CLIPPING_PEAK_FRACTION:
        status = "CLIP"
    else:
        status = "OK"
    return peak_frac, rms_frac, status


def print_report(peak_frac: float, rms_frac: float, status: str) -> None:
    tag = {"OK": "OK ", "SILENCE": "!SIL", "CLIP": "!CLP"}[status]
    print(f"     {tag}  peak {peak_frac * 100:5.1f}%   rms {rms_frac * 100:5.1f}%")
    if status == "SILENCE":
        print("           almost nothing captured — check the mic, then redo (r)")
    elif status == "CLIP":
        print("           clipped — lower the Windows mic level a notch, then redo (r)")


def to_16k_mono(rec: "np.ndarray", src_sr: int) -> "np.ndarray":
    """Downmix to mono and resample to 16 kHz int16."""
    x = rec[:, 0] if rec.ndim == 2 else rec
    x = x.astype(np.float64)
    if src_sr != TARGET_SR:
        g = math.gcd(TARGET_SR, src_sr)
        x = resample_poly(x, TARGET_SR // g, src_sr // g)
    x = np.clip(np.round(x), -32768, 32767).astype(np.int16)
    return x


def next_index(action: str, speaker: str) -> int:
    """Lowest unused take number for this speaker+action (1-based)."""
    folder   = os.path.join(TEMPLATES_ROOT, action)
    existing = glob.glob(os.path.join(folder, f"{speaker}_*.wav"))
    nums = []
    for p in existing:
        m = re.search(rf"{re.escape(speaker)}_(\d+)\.wav$", os.path.basename(p))
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1


def save_template(action: str, speaker: str, idx: int, samples16k: "np.ndarray") -> str:
    folder = os.path.join(TEMPLATES_ROOT, action)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{speaker}_{idx:02d}.wav")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(SAMPLE_WIDTH_BYTES)
        wf.setframerate(TARGET_SR)
        wf.writeframes(samples16k.tobytes())
    return path


def play(rec: "np.ndarray", src_sr: int, out_idx: int) -> None:
    sd.play(rec, samplerate=src_sr, device=out_idx)
    sd.wait()


def prompt_decision(rec: "np.ndarray", src_sr: int, out_idx: int) -> str:
    """Return one of: keep | redo | skip | quit. Handles [p]layback inline."""
    while True:
        choice = input(
            "     [Enter]=keep  [p]=play  [r]=redo  [s]=skip command  [q]=quit: "
        ).strip().lower()
        if choice == "":
            return "keep"
        if choice == "p":
            print("     >> playing back...")
            try:
                play(rec, src_sr, out_idx)
            except sd.PortAudioError as exc:
                print(f"     (playback failed: {exc})")
            continue
        if choice == "r":
            return "redo"
        if choice == "s":
            return "skip"
        if choice in ("q", "quit", "exit"):
            return "quit"
        print("     ? press Enter, or one of p / r / s / q.")


# ===========================================================================
# Command selection
# ===========================================================================

def select_commands(only):
    """Filter the vocabulary by --only tokens (1-based index or action substr)."""
    if not only:
        return list(TWI_VOCABULARY)
    tokens = [t.strip().lower() for t in only.split(",") if t.strip()]
    chosen = []
    for i, cmd in enumerate(TWI_VOCABULARY, start=1):
        if str(i) in tokens or any(tok in cmd.action.lower() for tok in tokens):
            chosen.append(cmd)
    return chosen


def print_vocabulary() -> None:
    print("\n  Command vocabulary (Table 3.2):")
    for i, cmd in enumerate(TWI_VOCABULARY, start=1):
        print(f"    [{i}]  {cmd.action:<16}  {cmd.twi_phrase}")
        print(f"         ({cmd.english_gloss})")


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record Twi command templates for the keyword spotter."
    )
    parser.add_argument("--speaker", default=None,
                        help="Voice label, e.g. wise. Prompted if omitted.")
    parser.add_argument("--input", default=None,
                        help="Input device index or name substring "
                             "(default: system default input).")
    parser.add_argument("--output", default=None,
                        help="Output device for [p]layback "
                             "(default: system default output).")
    parser.add_argument("--takes", type=int, default=DEFAULT_TAKES,
                        help=f"Takes per command this session (default {DEFAULT_TAKES}).")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                        help=f"Seconds per take (default {DEFAULT_DURATION}).")
    parser.add_argument("--samplerate", type=int, default=None,
                        help="Record rate in Hz (default: the mic's own rate). "
                             "Saved templates are always resampled to 16000.")
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS,
                        help=f"Channels to record (default {DEFAULT_CHANNELS} = mono).")
    parser.add_argument("--only", default=None,
                        help="Restrict to commands, e.g. 'SOS,3' (action or index).")
    args = parser.parse_args()

    print("\n" + "=" * 78)
    print("  VOICE ENGINE — TWI COMMAND TEMPLATE RECORDER")
    print("=" * 78)
    print_vocabulary()

    # Speaker label.
    speaker = args.speaker or input("\n  Speaker label (e.g. wise): ").strip()
    if not speaker:
        print("  X A speaker label is required.")
        return 1
    speaker = speaker.lower().replace(" ", "_")

    # Devices.
    try:
        in_idx  = _resolve_device(args.input,  want_input=True)
        out_idx = _resolve_device(args.output, want_input=False)
    except SystemExit as exc:
        print(exc)
        return 1

    in_dev = sd.query_devices(in_idx)
    if in_dev["max_input_channels"] < args.channels:
        print(
            f"  X Input device [{in_idx}] '{in_dev['name']}' supports only "
            f"{in_dev['max_input_channels']} channel(s); asked {args.channels}."
        )
        return 1

    src_sr   = args.samplerate or int(round(in_dev["default_samplerate"]))
    commands = select_commands(args.only)
    if not commands:
        print(f"  X --only={args.only!r} matched no commands.")
        return 1

    print(f"\n  Speaker : {speaker}")
    print(f"  Input   : [{in_idx}] {in_dev['name']}")
    print(f"  Record  : {src_sr} Hz -> resampled to {TARGET_SR} Hz mono int16")
    print(f"  Takes   : {args.takes} per command   Duration: {args.duration:g}s")
    print(f"  Saving  : {TEMPLATES_ROOT}{os.sep}<ACTION>{os.sep}{speaker}_NN.wav")
    print(f"  Commands: {len(commands)} of {len(TWI_VOCABULARY)}")

    summary = {}
    quit_all = False

    for cmd in commands:
        if quit_all:
            break

        action = cmd.action
        print("\n" + "-" * 78)
        print(f"  COMMAND: {action}    \"{cmd.twi_phrase}\"")
        print(f"           ({cmd.english_gloss})")
        print("-" * 78)

        idx          = next_index(action, speaker)
        already_have = idx - 1
        if already_have:
            print(f"     ({already_have} take(s) already saved for {speaker})")

        made = 0
        while made < args.takes:
            print(f"\n     Take {made + 1}/{args.takes}  ->  index {idx:02d}")
            ready = input("     Press Enter to record  (s=skip command, q=quit): ").strip().lower()
            if ready in ("q", "quit", "exit"):
                quit_all = True
                break
            if ready in ("s", "skip"):
                break

            try:
                rec = record_take(in_idx, src_sr, args.channels, args.duration)
            except sd.PortAudioError as exc:
                print(f"     X recording failed: {exc}")
                print("       Try a different --input device, or --samplerate 44100.")
                return 1

            peak_frac, rms_frac, status = signal_status(rec)
            print_report(peak_frac, rms_frac, status)

            decision = prompt_decision(rec, src_sr, out_idx)
            if decision == "redo":
                continue                       # same index, record again
            if decision == "skip":
                break
            if decision == "quit":
                quit_all = True
                break

            # keep
            samples16k = to_16k_mono(rec, src_sr)
            path = save_template(action, speaker, idx, samples16k)
            secs = len(samples16k) / TARGET_SR
            print(f"     saved -> {os.path.relpath(path, TEMPLATES_ROOT)}  "
                  f"({secs:.2f}s @ {TARGET_SR} Hz)")
            made += 1
            idx += 1

        summary[action] = next_index(action, speaker) - 1   # total now on disk

    # ---- Session summary --------------------------------------------------
    print("\n" + "=" * 78)
    print(f"  SESSION SUMMARY — speaker '{speaker}'")
    print("=" * 78)
    for cmd in commands:
        total = summary.get(cmd.action, next_index(cmd.action, speaker) - 1)
        print(f"    {cmd.action:<16}  {total} template(s) total")
    print("=" * 78)
    print(f"  Files under: {TEMPLATES_ROOT}")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())