"""
dtw_matcher.py
==============
The recognition core of the Twi voice engine: closed-set keyword spotting
by MFCC feature extraction + Dynamic Time Warping (DTW) template matching.

This is the algorithm the project report commits to (Section 2.6 / 3.5.1):
rather than open-vocabulary ASR — which has no reliable Twi model — we
match a spoken utterance against a small bank of pre-recorded example
templates for each of the seven commands. Whichever command's templates
are closest (and close enough) is the recognised command.

Pipeline (identical for templates AND live audio — symmetry is the point)
------------------------------------------------------------------------
    WAV / mic  ->  16 kHz mono float
               ->  trim leading/trailing silence (energy gate)
               ->  MFCC (13 coeffs) + cepstral mean normalisation (CMN)
               ->  append delta features  =>  26-dim per frame
               ->  DTW distance to every template, length-normalised
               ->  nearest command wins; reject if best distance > threshold

Why these choices
-----------------
• CMN (subtracting each coefficient's mean per utterance) removes the
  recording-level / channel offset, so the loud and quiet takes in your
  template set are put on equal footing. Deltas add level-invariant
  dynamics. Together they make matching robust to the volume variation
  we saw during recording.
• Silence is trimmed here, in the shared feature path, so a 3-second clip
  that is mostly silence is reduced to just its speech — and templates and
  live audio are trimmed the same way.
• DTW length-normalisation (dividing the warp cost by the path length)
  makes distances comparable across utterances of different durations.

This module is import-safe: the matcher needs only numpy / scipy /
python_speech_features / fastdtw. The microphone (sounddevice) is imported
lazily, ONLY in the --listen CLI mode, so the class can be reused by the
hub without a mic dependency.

CLI
---
Run from the project root:

    python -m src.voice_engine.dtw_matcher --evaluate      # score templates (no mic)
    python -m src.voice_engine.dtw_matcher --listen        # speak and recognise live
    python -m src.voice_engine.dtw_matcher --file some.wav  # classify one file

Dependencies
------------
    pip install numpy scipy python_speech_features fastdtw

Author: Wise (Asumang Pobi Godwin) — KNUST COE 497
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from dataclasses import dataclass, field
from math import gcd
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from scipy.spatial.distance import euclidean
    from python_speech_features import mfcc, delta
    from fastdtw import fastdtw
except ImportError:
    print(
        "\n  X Missing dependency. Install the recognition libraries first:\n\n"
        "      pip install numpy scipy python_speech_features fastdtw\n",
        file=sys.stderr,
    )
    sys.exit(1)

logger = logging.getLogger(__name__)


# ===========================================================================
# Configuration — shared by templates and live audio so they match
# ===========================================================================

SAMPLE_RATE   = 16000          # Canonical pipeline rate.
NUM_CEPSTRA   = 13             # MFCC coefficients (C0..C12).
WIN_LEN       = 0.025          # 25 ms analysis window.
WIN_STEP      = 0.010          # 10 ms hop -> 100 frames/sec.
NFFT          = 512
PREEMPH       = 0.97
DELTA_SPAN    = 2              # Frames each side for the delta computation.

# Silence trimming.
TRIM_FRAME_MS   = 20
TRIM_RATIO      = 0.10         # Keep frames above 10% of the loudest frame.
TRIM_PAD_MS     = 60
TRIM_FLOOR      = 1.0e-4       # Absolute floor on the [-1,1] scale.

# Minimum amount of speech (post-trim) to attempt a match, in seconds.
MIN_SPEECH_SEC  = 0.12

# Default templates directory: <project_root>/voice_templates
DEFAULT_TEMPLATES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "voice_templates"
)

# Default acceptance threshold. inf = never reject (always return the
# nearest command). Run --evaluate to get a calibrated value to pass via
# --threshold, since the absolute distance scale depends on the features.
DEFAULT_THRESHOLD = float("inf")


# Optional: pull the canonical action -> Twi phrase map from the project so
# CLI output can show the phrase. Degrades gracefully if run standalone.
try:
    from src.hardware_mocks.mock_microphone import TWI_VOCABULARY
    ACTION_PHRASE = {c.action: c.twi_phrase for c in TWI_VOCABULARY}
except Exception:
    ACTION_PHRASE = {}


# ===========================================================================
# Audio loading / resampling / trimming
# ===========================================================================

def load_wav(path: str) -> Tuple["np.ndarray", int]:
    """Load a WAV as mono float64 in [-1, 1]. Returns (signal, samplerate)."""
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]                      # First channel if stereo.
    dt = data.dtype
    if dt == np.int16:
        x = data.astype(np.float64) / 32768.0
    elif dt == np.int32:
        x = data.astype(np.float64) / 2147483648.0
    elif dt == np.uint8:
        x = (data.astype(np.float64) - 128.0) / 128.0
    else:
        x = data.astype(np.float64)            # Already float.
    return x, sr


def to_target_rate(sig: "np.ndarray", sr: int) -> "np.ndarray":
    """Resample a float signal to SAMPLE_RATE (16 kHz)."""
    if sr == SAMPLE_RATE:
        return sig
    g    = gcd(int(sr), SAMPLE_RATE)
    up   = SAMPLE_RATE // g
    down = int(sr) // g
    return resample_poly(sig, up, down)


def trim_silence(sig: "np.ndarray", sr: int = SAMPLE_RATE) -> "np.ndarray":
    """
    Energy-gate the signal down to its spoken portion. Keeps frames whose
    RMS exceeds TRIM_RATIO of the loudest frame (with an absolute floor),
    then pads slightly. Returns the original signal if it can't find any
    voiced region (e.g. near-silence).
    """
    frame_len = int(sr * TRIM_FRAME_MS / 1000)
    if frame_len <= 0 or len(sig) < frame_len:
        return sig

    n = len(sig) // frame_len
    frames = sig[: n * frame_len].reshape(n, frame_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)

    peak = float(rms.max())
    if peak <= 0:
        return sig

    thresh = max(peak * TRIM_RATIO, TRIM_FLOOR)
    voiced = np.where(rms > thresh)[0]
    if voiced.size == 0:
        return sig

    pad = int(sr * TRIM_PAD_MS / 1000)
    start = max(0, voiced[0] * frame_len - pad)
    end   = min(len(sig), (voiced[-1] + 1) * frame_len + pad)
    return sig[start:end]


# ===========================================================================
# Feature extraction
# ===========================================================================

def extract_features(sig: "np.ndarray", sr: int = SAMPLE_RATE) -> Optional["np.ndarray"]:
    """
    Turn a raw signal into the (frames x 26) feature matrix used for DTW:
    CMN'd MFCCs concatenated with their deltas. Returns None if there isn't
    enough speech to be meaningful.
    """
    if sr != SAMPLE_RATE:
        sig = to_target_rate(sig, sr)

    sig = trim_silence(sig, SAMPLE_RATE)
    if len(sig) < int(MIN_SPEECH_SEC * SAMPLE_RATE):
        return None

    m = mfcc(
        sig,
        samplerate=SAMPLE_RATE,
        winlen=WIN_LEN,
        winstep=WIN_STEP,
        numcep=NUM_CEPSTRA,
        nfft=NFFT,
        preemph=PREEMPH,
        appendEnergy=False,
    )
    if m.shape[0] < 3:
        return None

    # Cepstral mean normalisation — removes per-utterance level/channel offset.
    m = m - np.mean(m, axis=0, keepdims=True)

    d = delta(m, DELTA_SPAN)
    feat = np.hstack([m, d]).astype(np.float64)
    return feat


def features_from_file(path: str) -> Optional["np.ndarray"]:
    sig, sr = load_wav(path)
    return extract_features(sig, sr)


# ===========================================================================
# DTW distance
# ===========================================================================

def dtw_distance(a: "np.ndarray", b: "np.ndarray") -> float:
    """
    Length-normalised DTW distance between two feature sequences. Lower is
    more similar. fastdtw is O(N) (radius-limited), keeping per-utterance
    recognition quick enough for the Raspberry Pi.
    """
    distance, path = fastdtw(a, b, dist=euclidean)
    return distance / max(1, len(path))


# ===========================================================================
# Result type
# ===========================================================================

@dataclass
class MatchResult:
    action:   Optional[str]                 # Best command, or None if rejected.
    distance: float                         # Distance to the nearest template.
    accepted: bool                          # distance <= threshold?
    ranked:   List[Tuple[str, float]] = field(default_factory=list)  # all, asc.

    @property
    def phrase(self) -> str:
        return ACTION_PHRASE.get(self.action or "", "")


# ===========================================================================
# The matcher
# ===========================================================================

class TwiKeywordMatcher:
    """
    Loads the template bank once, then classifies utterances against it.

    Usage
    -----
        matcher = TwiKeywordMatcher(threshold=...)
        result  = matcher.classify_audio(signal, samplerate)   # live audio
        result  = matcher.classify_file("clip.wav")            # a WAV
        matcher.evaluate_leave_one_out()                       # score templates
    """

    def __init__(
        self,
        templates_dir: str = DEFAULT_TEMPLATES_DIR,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.threshold = threshold
        self.templates_dir = os.path.abspath(templates_dir)
        # action -> list of (template_name, feature_matrix)
        self.templates: Dict[str, List[Tuple[str, "np.ndarray"]]] = {}
        self._load()

    # -----------------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------------

    def _load(self) -> None:
        if not os.path.isdir(self.templates_dir):
            raise FileNotFoundError(
                f"Templates directory not found: {self.templates_dir}\n"
                f"Record templates first with scripts/record_templates.py."
            )

        n_files = 0
        for action_dir in sorted(glob.glob(os.path.join(self.templates_dir, "*"))):
            if not os.path.isdir(action_dir):
                continue
            action = os.path.basename(action_dir)
            feats: List[Tuple[str, "np.ndarray"]] = []
            for wav in sorted(glob.glob(os.path.join(action_dir, "*.wav"))):
                f = features_from_file(wav)
                if f is None:
                    logger.warning("Skipping unusable template (too short/quiet): %s", wav)
                    continue
                feats.append((os.path.basename(wav), f))
                n_files += 1
            if feats:
                self.templates[action] = feats

        if not self.templates:
            raise RuntimeError(
                f"No usable templates loaded from {self.templates_dir}."
            )
        logger.info(
            "Loaded %d templates across %d commands.", n_files, len(self.templates)
        )

    @property
    def commands(self) -> List[str]:
        return sorted(self.templates.keys())

    def template_count(self) -> int:
        return sum(len(v) for v in self.templates.values())

    # -----------------------------------------------------------------------
    # Classification
    # -----------------------------------------------------------------------

    def classify_features(self, feat: "np.ndarray") -> MatchResult:
        """Classify a pre-computed feature matrix against the template bank."""
        scores: Dict[str, float] = {}
        for action, items in self.templates.items():
            # Nearest-neighbour: distance to the closest template of this command.
            scores[action] = min(dtw_distance(feat, f) for _, f in items)

        ranked = sorted(scores.items(), key=lambda kv: kv[1])
        best_action, best_dist = ranked[0]
        accepted = best_dist <= self.threshold
        return MatchResult(
            action=best_action if accepted else None,
            distance=best_dist,
            accepted=accepted,
            ranked=ranked,
        )

    def classify_audio(self, sig: "np.ndarray", sr: int) -> MatchResult:
        """Classify a raw signal (any sample rate)."""
        feat = extract_features(sig, sr)
        if feat is None:
            return MatchResult(action=None, distance=float("inf"),
                               accepted=False, ranked=[])
        return self.classify_features(feat)

    def classify_file(self, path: str) -> MatchResult:
        sig, sr = load_wav(path)
        return self.classify_audio(sig, sr)

    # -----------------------------------------------------------------------
    # Evaluation — leave-one-out over the template bank
    # -----------------------------------------------------------------------

    def evaluate_leave_one_out(self) -> Dict[str, object]:
        """
        For each template, classify it against all OTHER templates and check
        whether the predicted command matches its true command. Prints a
        per-command breakdown, overall accuracy, and a suggested acceptance
        threshold derived from the distance distributions.
        """
        actions = self.commands
        total = correct = 0
        per_true_correct: Dict[str, int] = {a: 0 for a in actions}
        per_true_total:   Dict[str, int] = {a: 0 for a in actions}
        confusions: List[Tuple[str, str, str]] = []   # (template, true, pred)
        correct_dists: List[float] = []               # nearest same-command distance
        impostor_dists: List[float] = []              # nearest other-command distance

        for true_action in actions:
            for i, (name, feat) in enumerate(self.templates[true_action]):
                scores: Dict[str, float] = {}
                for a in actions:
                    dists = [
                        dtw_distance(feat, f)
                        for j, (_, f) in enumerate(self.templates[a])
                        if not (a == true_action and j == i)   # leave self out
                    ]
                    if dists:
                        scores[a] = min(dists)

                ranked = sorted(scores.items(), key=lambda kv: kv[1])
                pred_action, _ = ranked[0]

                total += 1
                per_true_total[true_action] += 1
                if pred_action == true_action:
                    correct += 1
                    per_true_correct[true_action] += 1
                else:
                    confusions.append((name, true_action, pred_action))

                if true_action in scores:
                    correct_dists.append(scores[true_action])
                wrong = [d for a, d in scores.items() if a != true_action]
                if wrong:
                    impostor_dists.append(min(wrong))

        accuracy = correct / total if total else 0.0

        # ---- Report -------------------------------------------------------
        print("\n" + "=" * 78)
        print("  LEAVE-ONE-OUT EVALUATION  (each template classified vs all others)")
        print("=" * 78)
        print(f"  Templates: {self.template_count()}   Commands: {len(actions)}")
        print("  " + "-" * 74)
        for a in actions:
            c, t = per_true_correct[a], per_true_total[a]
            bar = "OK " if c == t else "!! "
            phrase = ACTION_PHRASE.get(a, "")
            extra = f'   "{phrase}"' if phrase else ""
            print(f"  {bar}{a:<16} {c}/{t} correct{extra}")
        print("  " + "-" * 74)
        print(f"  OVERALL ACCURACY: {correct}/{total} = {accuracy * 100:.1f}%")

        if confusions:
            print("\n  Misclassifications:")
            for name, ta, pa in confusions:
                print(f"    {ta}/{name}  ->  predicted {pa}")
        else:
            print("\n  No misclassifications — every command is separable from the rest.")

        # ---- Threshold calibration ---------------------------------------
        print("\n  Distance calibration")
        print("  " + "-" * 74)
        if correct_dists and impostor_dists:
            cd = np.array(correct_dists)
            idd = np.array(impostor_dists)
            p95_correct  = float(np.percentile(cd, 95))
            p05_impostor = float(np.percentile(idd, 5))
            # Place the threshold in the gap between the correct-match and
            # impostor distributions when they separate cleanly; if they
            # overlap, bias slightly toward accepting genuine commands.
            if p05_impostor > p95_correct:
                suggested = round((p95_correct + p05_impostor) / 2.0, 2)
                gap_note = "(midpoint of the correct/impostor gap)"
            else:
                suggested = round(p95_correct * 1.15, 2)
                gap_note = "(distributions overlap — biased toward accepting commands)"
            print(f"    correct-match distance : min {cd.min():.2f}  "
                  f"mean {cd.mean():.2f}  max {cd.max():.2f}")
            print(f"    nearest-impostor dist  : min {idd.min():.2f}  "
                  f"mean {idd.mean():.2f}  max {idd.max():.2f}")
            print(f"\n    Suggested --threshold  : {suggested}  {gap_note}")
            print( "    NOTE: this separates COMMANDS from each other. Rejecting")
            print( "    non-speech noise is primarily the job of the energy gate")
            print( "    in the live capture loop; tune the final value with --listen")
            print( "    by comparing real-command distances against silence.")
        else:
            suggested = None
            print("    Not enough data to calibrate a threshold.")
        print("=" * 78 + "\n")

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "confusions": confusions,
            "suggested_threshold": suggested,
        }


# ===========================================================================
# CLI helpers
# ===========================================================================

def _print_result(result: MatchResult, threshold: float, top: int = 3) -> None:
    print("  " + "-" * 50)
    if result.accepted and result.action is not None:
        phrase = f'  "{result.phrase}"' if result.phrase else ""
        print(f"  RECOGNISED: {result.action}{phrase}")
        print(f"  distance {result.distance:.2f}  (threshold {threshold})")
    else:
        nearest = result.ranked[0][0] if result.ranked else "n/a"
        print(f"  REJECTED (nearest was {nearest} at {result.distance:.2f}, "
              f"threshold {threshold})")
    if result.ranked:
        print("  ranked:")
        for action, dist in result.ranked[:top]:
            print(f"    {dist:8.2f}  {action}")
    print("  " + "-" * 50)


def _record_from_mic(seconds: float, in_idx) -> "np.ndarray":
    """Record from the mic and return float mono at 16 kHz. Imports sd lazily."""
    import sounddevice as sd
    dev = sd.query_devices(in_idx) if in_idx is not None else sd.query_devices(kind="input")
    sr = int(round(dev["default_samplerate"]))
    frames = int(round(seconds * sr))
    rec = sd.rec(frames, samplerate=sr, channels=1, dtype="int16", device=in_idx)
    sd.wait()
    sig = rec.reshape(-1).astype(np.float64) / 32768.0
    return to_target_rate(sig, sr)


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="MFCC+DTW Twi keyword matcher: evaluate templates or "
                    "recognise speech live."
    )
    parser.add_argument("--evaluate", action="store_true",
                        help="Score the template bank by leave-one-out (no mic).")
    parser.add_argument("--listen", action="store_true",
                        help="Record from the mic and recognise, in a loop.")
    parser.add_argument("--file", default=None,
                        help="Classify a single WAV file and exit.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Acceptance threshold; above it, an utterance is "
                             "rejected. Default: inf (never reject).")
    parser.add_argument("--templates-dir", default=DEFAULT_TEMPLATES_DIR,
                        help="Folder containing the per-command template subfolders.")
    parser.add_argument("--input", default=None,
                        help="(listen mode) Input device index. Default: system default.")
    parser.add_argument("--duration", type=float, default=2.5,
                        help="(listen mode) Seconds to record per utterance.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("\n" + "=" * 78)
    print("  TWI KEYWORD MATCHER  (MFCC + DTW)")
    print("=" * 78)

    try:
        matcher = TwiKeywordMatcher(
            templates_dir=args.templates_dir,
            threshold=args.threshold,
        )
    except Exception as exc:
        print(f"\n  X {exc}\n")
        return 1

    print(f"  Loaded {matcher.template_count()} templates: "
          f"{', '.join(matcher.commands)}")

    # --- Single file ---------------------------------------------------
    if args.file:
        if not os.path.isfile(args.file):
            print(f"\n  X File not found: {args.file}\n")
            return 1
        result = matcher.classify_file(args.file)
        print(f"\n  File: {args.file}")
        _print_result(result, args.threshold)
        print()
        return 0

    # --- Live mic loop -------------------------------------------------
    if args.listen:
        in_idx = None
        if args.input is not None:
            try:
                in_idx = int(args.input)
            except ValueError:
                in_idx = args.input
        print("\n  LISTEN MODE — press Enter to record, then say a command.")
        print("  (Type q then Enter to quit.)")
        if args.threshold == float("inf"):
            print("  Note: no threshold set, so the nearest command is always")
            print("  returned. Run --evaluate to get a --threshold to reject noise.")
        while True:
            try:
                choice = input("\n  [Enter]=record  q=quit > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if choice == "q":
                break
            print(f"  >> recording {args.duration:g}s — say it now...", flush=True)
            try:
                sig = _record_from_mic(args.duration, in_idx)
            except ImportError:
                print("\n  X sounddevice not installed. Run: pip install sounddevice\n")
                return 1
            except Exception as exc:
                print(f"\n  X Recording failed: {exc}\n")
                return 1
            result = matcher.classify_audio(sig, SAMPLE_RATE)
            _print_result(result, args.threshold)
        print("\n  Done.\n")
        return 0

    # --- Default: evaluate ---------------------------------------------
    matcher.evaluate_leave_one_out()
    return 0


if __name__ == "__main__":
    sys.exit(main())
    