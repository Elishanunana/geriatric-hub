"""
sanity_check_audio.py
=====================
Pre-flight audio check for the voice-engine integration. Before we build
the command-recording utility or the MFCC+DTW keyword spotter, this script
proves the two things everything else depends on:

    1. The USB microphone is visible to Python and actually captures sound.
    2. The output device (your laptop speaker, for now) can play it back.

It does NOT touch the hub, the database, or any mocks. It is a standalone
diagnostic you run on its own.

Workflow
--------
From the project root (venv activated), first LIST your audio devices so
you can find the USB microphone's index:

    python -m scripts.sanity_check_audio --list

Then run the full record -> playback check, pointing --input at the USB mic:

    python -m scripts.sanity_check_audio --input 2

(Replace 2 with the index shown for your USB microphone. If you omit
--input entirely, the system default input device is used — which on a
laptop is usually the BUILT-IN mic, not the USB one, so passing --input
is recommended.)

What you should see
-------------------
The device table, then a 3-2-1 countdown, then a few seconds of recording
while you speak, then a signal report (peak / RMS level), then your own
voice played back through the laptop speaker.

The signal report is the key diagnostic: a working mic shows a healthy
peak; a muted or wrong device shows near-zero ("SILENCE") and warns you
BEFORE you waste time wondering why playback is quiet.

Dependencies
------------
    pip install sounddevice numpy

On Windows the sounddevice wheel bundles PortAudio, so no system-level
audio packages are required here. (On the Raspberry Pi later you WILL need
`sudo apt install portaudio19-dev` first — but that is a Pi concern, not a
laptop one.)

Author: Wise (Asumang Pobi Godwin) — KNUST COE 497
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import wave

# sounddevice + numpy are the only third-party requirements. Fail loudly
# with the exact install command if they are missing, since this is likely
# the first time they are being installed in this venv.
try:
    import numpy as np
    import sounddevice as sd
except ImportError:
    print(
        "\n  X Missing dependency. Install the audio libraries first:\n\n"
        "      pip install sounddevice numpy\n",
        file=sys.stderr,
    )
    sys.exit(1)


# ===========================================================================
# Configuration
# ===========================================================================

DEFAULT_DURATION_SECONDS = 4.0
DEFAULT_CHANNELS         = 1          # Mono — what the pipeline wants.
SAMPLE_WIDTH_BYTES       = 2          # int16 = 2 bytes/sample.
FULL_SCALE_INT16         = 32767

RECORDINGS_DIR  = os.path.join(os.path.dirname(__file__), "..", "recordings")
OUTPUT_FILENAME = "sanity_check.wav"

SILENCE_PEAK_FRACTION  = 0.01   # Below ~1% of full scale -> probably silence.
CLIPPING_PEAK_FRACTION = 0.99   # Above ~99% -> probably clipping/too loud.


# ===========================================================================
# Device discovery
# ===========================================================================

def print_device_table() -> None:
    """Print every audio device with its index, capabilities, and host API."""
    devices  = sd.query_devices()
    hostapis = sd.query_hostapis()
    try:
        default_in, default_out = sd.default.device
    except Exception:
        default_in, default_out = (-1, -1)

    print("\n" + "=" * 78)
    print("  AUDIO DEVICES")
    print("=" * 78)
    print(f"  {'idx':>3}  {'in':>3} {'out':>3}  {'rate':>7}  {'host API':<14}  name")
    print("  " + "-" * 74)

    for idx, d in enumerate(devices):
        in_ch  = d["max_input_channels"]
        out_ch = d["max_output_channels"]
        rate   = int(round(d["default_samplerate"]))
        host   = hostapis[d["hostapi"]]["name"]

        markers = []
        if idx == default_in:
            markers.append("default-IN")
        if idx == default_out:
            markers.append("default-OUT")
        marker_str = ("  <- " + ", ".join(markers)) if markers else ""

        print(
            f"  {idx:>3}  {in_ch:>3} {out_ch:>3}  {rate:>7}  "
            f"{host:<14}  {d['name']}{marker_str}"
        )

    print("=" * 78)
    print("  'in'/'out' = number of input/output channels the device supports.")
    print("  A microphone has in > 0; a speaker has out > 0.")
    print("  On Windows the same physical device often appears several times")
    print("  under different host APIs (MME, WASAPI, DirectSound) — any will do.")
    print("=" * 78 + "\n")


def _resolve_device(arg, want_input: bool) -> int:
    """
    Turn a --input/--output argument into a concrete device index.

    `arg` may be:
      • None              -> use the system default for that direction.
      • an integer string -> used directly as the device index.
      • a name substring  -> matched (case-insensitive) against device names.
    """
    devices = sd.query_devices()
    kind    = "input" if want_input else "output"
    ch_key  = "max_input_channels" if want_input else "max_output_channels"

    if arg is None:
        idx = sd.default.device[0 if want_input else 1]
        if idx is None or idx < 0:
            raise SystemExit(
                f"  X No default {kind} device found. Run with --list and pass "
                f"--{kind} <index> explicitly."
            )
        return int(idx)

    # Numeric index?
    try:
        return int(arg)
    except ValueError:
        pass

    # Substring match among devices that support this direction.
    matches = [
        i for i, d in enumerate(devices)
        if arg.lower() in d["name"].lower() and d[ch_key] > 0
    ]
    if not matches:
        raise SystemExit(f"  X No {kind} device name contains {arg!r}. Try --list.")
    if len(matches) > 1:
        names = ", ".join(f"[{i}] {devices[i]['name']}" for i in matches)
        raise SystemExit(
            f"  X {arg!r} matches several {kind} devices: {names}. "
            f"Pass the index instead."
        )
    return matches[0]


# ===========================================================================
# Recording + analysis + playback
# ===========================================================================

def describe_signal(recording: "np.ndarray") -> None:
    """Print a peak/RMS report and warn about silence or clipping."""
    # Cast to int32 before abs() so a lone -32768 sample can't overflow int16.
    peak = int(np.max(np.abs(recording.astype(np.int32)))) if recording.size else 0
    rms  = (float(np.sqrt(np.mean(recording.astype(np.float64) ** 2)))
            if recording.size else 0.0)

    peak_frac = peak / FULL_SCALE_INT16
    rms_frac  = rms  / FULL_SCALE_INT16
    peak_dbfs = 20 * np.log10(peak_frac) if peak_frac > 0 else float("-inf")

    print("\n  Signal report")
    print("  " + "-" * 40)
    print(f"    peak level : {peak_frac * 100:6.2f}%  ({peak_dbfs:6.1f} dBFS)")
    print(f"    rms  level : {rms_frac * 100:6.2f}%")

    if peak_frac < SILENCE_PEAK_FRACTION:
        print(
            "    ! SILENCE — the mic captured almost nothing. Likely causes:\n"
            "        - wrong --input device (run --list and check the index)\n"
            "        - the mic is muted or its level is at zero in Windows Sound\n"
            "        - Windows microphone privacy setting is blocking access\n"
            "          (Settings > Privacy & security > Microphone)"
        )
    elif peak_frac > CLIPPING_PEAK_FRACTION:
        print(
            "    ! CLIPPING — the signal is maxed out. Lower the mic input\n"
            "      level in Windows Sound settings, or speak further away."
        )
    else:
        print("    OK — looks healthy, real audio was captured.")
    print()


def record(in_idx: int, samplerate: int, channels: int, duration: float) -> "np.ndarray":
    """Record `duration` seconds of int16 audio from device `in_idx`."""
    frames = int(round(duration * samplerate))

    print("\n  Get ready to speak into the microphone...")
    for n in (3, 2, 1):
        print(f"    {n}...", end=" ", flush=True)
        time.sleep(1)
    print("\n  >> RECORDING — speak now!", flush=True)

    recording = sd.rec(
        frames,
        samplerate=samplerate,
        channels=channels,
        dtype="int16",
        device=in_idx,
    )
    sd.wait()
    print("  OK Recording complete.")
    return recording


def save_wav(recording: "np.ndarray", samplerate: int, channels: int) -> str:
    """Write the recording to a WAV file and return its absolute path."""
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    path = os.path.abspath(os.path.join(RECORDINGS_DIR, OUTPUT_FILENAME))
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(SAMPLE_WIDTH_BYTES)
        wf.setframerate(samplerate)
        wf.writeframes(recording.tobytes())
    return path


def playback(recording: "np.ndarray", samplerate: int, out_idx: int) -> None:
    """Play the recording back through `out_idx` (your laptop speaker)."""
    print("  >> Playing it back through the speaker now...")
    sd.play(recording, samplerate=samplerate, device=out_idx)
    sd.wait()
    print("  OK Playback complete.")


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audio sanity check: list devices, record from the mic, "
                    "and play it back."
    )
    parser.add_argument("--list", action="store_true",
                        help="List audio devices and exit.")
    parser.add_argument("--input", default=None,
                        help="Input device: index (e.g. 2) or name substring "
                             "(e.g. USB). Default: system default input.")
    parser.add_argument("--output", default=None,
                        help="Output device: index or name substring. "
                             "Default: system default output (laptop speaker).")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SECONDS,
                        help=f"Seconds to record (default {DEFAULT_DURATION_SECONDS}).")
    parser.add_argument("--samplerate", type=int, default=None,
                        help="Sample rate in Hz. Default: the mic's own default rate.")
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS,
                        help=f"Channels to record (default {DEFAULT_CHANNELS} = mono).")
    args = parser.parse_args()

    print("\n" + "=" * 78)
    print("  VOICE ENGINE — AUDIO SANITY CHECK")
    print("=" * 78)

    # Always show the device table — it's the map for everything else.
    print_device_table()

    if args.list:
        print("  (--list given: stopping here. Re-run with --input <index> to "
              "record.)\n")
        return 0

    try:
        in_idx  = _resolve_device(args.input,  want_input=True)
        out_idx = _resolve_device(args.output, want_input=False)
    except SystemExit as exc:
        print(exc)
        return 1

    in_dev  = sd.query_devices(in_idx)
    out_dev = sd.query_devices(out_idx)

    # Validate channel count against the device's capability.
    if in_dev["max_input_channels"] < args.channels:
        print(
            f"  X Input device [{in_idx}] '{in_dev['name']}' supports only "
            f"{in_dev['max_input_channels']} input channel(s); you asked for "
            f"{args.channels}. Try --channels {in_dev['max_input_channels']}."
        )
        return 1

    # Resolve sample rate: explicit flag wins, else the mic's default rate.
    samplerate = args.samplerate or int(round(in_dev["default_samplerate"]))

    print(f"  Input  : [{in_idx}] {in_dev['name']}")
    print(f"  Output : [{out_idx}] {out_dev['name']}")
    print(f"  Format : {samplerate} Hz, {args.channels} channel(s), int16, "
          f"{args.duration:g}s")

    try:
        recording = record(in_idx, samplerate, args.channels, args.duration)
    except sd.PortAudioError as exc:
        print(f"\n  X Recording failed: {exc}")
        print("    Try a different --input device (run --list), or a standard")
        print("    --samplerate like 44100 or 48000.")
        return 1

    describe_signal(recording)

    wav_path = save_wav(recording, samplerate, args.channels)
    print(f"  Saved recording -> {wav_path}\n")

    try:
        playback(recording, samplerate, out_idx)
    except sd.PortAudioError as exc:
        print(f"\n  X Playback failed: {exc}")
        print("    Try a different --output device (run --list).")
        return 1

    print("\n" + "=" * 78)
    print("  OK SANITY CHECK PASSED — mic captures and speaker plays.")
    print("    If you heard your own voice clearly, the hardware path is ready")
    print("    and we can build the command-recording utility next.")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    