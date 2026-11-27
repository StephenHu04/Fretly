"""Make a sound profile for one guitar from six calibration recordings.

Record each string on its own, open and then frets 1 to 12, one note at a
time, with the same phone and in the same place as the clips to be tabbed.
Name the files ``string_e``, ``string_B``, ``string_G``, ``string_D``,
``string_A`` and ``string_lowE`` (any supported audio format) in one folder.

It checks each string's tuning, fits the string-from-sound model (decision
023), says how well it tells strings apart on frets it didn't see, and writes
a JSON profile for ``make_tab.py --profile``.

Usage:
    python scripts/calibrate.py ../samples/calibration
    python scripts/calibrate.py ../samples/calibration --out profile.json --max-fret 19
"""

from __future__ import annotations

import argparse
import statistics
import sys
from datetime import date
from pathlib import Path

import numpy as np

from pipeline.audio_validation import SUPPORTED_EXTENSIONS
from pipeline.errors import PipelineError
from pipeline.run import preprocessed_path, transcribe_file
from tab.fretboard import DEFAULT_MAX_FRET, Fretboard
from tab.string_model import Profile, calibration_notes, cents_off, fit_profile, note_features, read_samples

STRING_FILES = ("string_e", "string_B", "string_G", "string_D", "string_A", "string_lowE")
OPEN_TOLERANCE_CENTS = 15  # an open string further off than this needs tuning
FRETTED_TOLERANCE_CENTS = 40  # fretted notes this sharp are close to being heard a semitone up


def find_recording(folder: Path, stem: str) -> Path | None:
    for ext in sorted(SUPPORTED_EXTENSIONS):
        if (folder / (stem + ext)).is_file():
            return folder / (stem + ext)
    return None


def tuning_warnings(name: str, cents: dict[int, float]) -> list[str]:
    warnings = []
    if 0 in cents and abs(cents[0]) > OPEN_TOLERANCE_CENTS:
        warnings.append(f"{name}: open string is {cents[0]:+.0f} cents out; tune it and record again")
    fretted = [c for f, c in cents.items() if f > 0]
    if fretted and abs(statistics.median(fretted)) > FRETTED_TOLERANCE_CENTS:
        warnings.append(
            f"{name}: fretted notes are {statistics.median(fretted):+.0f} cents out (median); "
            "some may be heard as the next note"
        )
    return warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="Folder with the six string_* recordings")
    parser.add_argument("--out", type=Path, help="Profile to write (default: <folder>/profile.json)")
    parser.add_argument("--frets", type=int, default=12, help="Highest fret recorded on each string")
    parser.add_argument("--max-fret", type=int, default=DEFAULT_MAX_FRET, help="Highest fret on the guitar")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/calibration"), help="Where to write audio/MIDI artifacts")
    args = parser.parse_args(argv)
    fretboard = Fretboard(max_fret=args.max_fret)
    if not 1 <= args.frets <= args.max_fret:
        parser.error("--frets must be between 1 and --max-fret")

    features, strings, frets, warnings = [], [], [], []
    print(f"{'string':<8}{'notes':>6}  {'open':>6}  {'fretted (median)':>17}")
    for string, stem in enumerate(STRING_FILES, start=1):
        audio = find_recording(args.folder, stem)
        if audio is None:
            print(f"error: no {stem} recording in {args.folder}", file=sys.stderr)
            return 1
        try:
            _, result, _ = transcribe_file(audio, args.out_dir)
        except PipelineError as exc:
            print(f"error [{exc.code}] in {audio.name}: {exc.message}", file=sys.stderr)
            return 1
        samples = read_samples(preprocessed_path(audio, args.out_dir))
        found, missing = calibration_notes(result.notes, fretboard.open_pitch(string), args.frets)
        cents = {f: c for f, n in found.items() if (c := cents_off(samples, n)) is not None}
        for fret, note in found.items():
            features.append(note_features(samples, note))
            strings.append(string)
            frets.append(fret)
        fretted = [c for f, c in cents.items() if f > 0]
        print(f"{stem[7:]:<8}{len(found):>6}  {cents.get(0, float('nan')):>+5.0f}c  "
              f"{statistics.median(fretted) if fretted else float('nan'):>+16.0f}c")
        if missing:
            warnings.append(f"{stem[7:]}: no note heard at fret {', '.join(map(str, missing))}")
        warnings += tuning_warnings(stem[7:], cents)

    try:
        model, check = fit_profile(np.array(features), strings, frets, fretboard)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    labels = dict(zip(range(1, 7), ("e", "B", "G", "D", "A", "lowE"), strict=True))
    print(f"\nTold strings apart on frets it hadn't seen: {check.accuracy:.0%} of notes "
          f"(guessing: {check.chance:.0%})")
    print("  " + ", ".join(f"{labels[s]} {a:.0%}" for s, a in check.per_string.items()))
    for w in warnings:
        print(f"warning: {w}")

    out = args.out or args.folder / "profile.json"
    Profile(model, fretboard.tuning, {
        "created": date.today().isoformat(),
        "notes": len(strings),
        "frets": args.frets,
        "held_out_accuracy": round(check.accuracy, 3),
        "per_string_accuracy": {labels[s]: round(a, 3) for s, a in check.per_string.items()},
        "warnings": warnings,
    }).save(out)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
