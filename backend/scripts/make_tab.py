"""Turn a guitar recording or a MIDI file into tablature.

Pipeline: [audio -> validate -> FFmpeg -> Basic Pitch ->] notes ->
cleanup (octave ghosts, ringing open strings, low thumps, shared onsets) -> string/fret
assignment ->
plain-text tab + metrics.

Audio files run the Phase 2 transcription first; .mid/.midi files skip it.

Usage:
    python scripts/make_tab.py ../samples/audio_clips/no_chord1.m4a
    python scripts/make_tab.py clip.mid --method both     # baseline vs optimizer
    python scripts/make_tab.py clip.m4a --json
    python scripts/make_tab.py clip.m4a --style lead      # open, balanced or lead
    python scripts/make_tab.py clip.m4a --alternatives 3  # up to 3 different tabs
    python scripts/make_tab.py clip.m4a --list-notes      # numbers for --pin
    python scripts/make_tab.py clip.m4a --start-fret 5 --pin 3:G5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import partial
from pathlib import Path
from typing import Any

from pipeline.errors import PipelineError
from pipeline.midi_parser import NoteEvent, parse_midi
from pipeline.run import transcribe_file
from pipeline.transcribe import TranscriptionSettings
from tab.alternatives import alternatives
from tab.assign import Assigner, Assignment, assign_lowest_fret
from tab.cleanup import clean_notes
from tab.fretboard import DEFAULT_MAX_FRET, Fretboard, Position
from tab.metrics import PlayabilityMetrics, measure
from tab.optimizer import STYLES, assign_optimized
from tab.render import DEFAULT_WIDTH, render_tab, string_labels

MIDI_EXTENSIONS = frozenset({".mid", ".midi"})
ASSIGNERS: dict[str, Assigner] = {"optimized": assign_optimized, "baseline": assign_lowest_fret}

METRIC_LABELS = {
    "position_shifts": "position shifts",
    "shifts_per_minute": "shifts / minute",
    "max_stretch": "max stretch (frets)",
    "fret_travel": "fret travel",
    "string_changes": "string changes",
    "unplayable": "unplayable notes",
}


def parse_pins(values: list[str], fretboard: Fretboard) -> dict[int, Position]:
    """``3:G5`` -> note 3 on the G string, fret 5. ``e`` is the high E string, ``E`` the low one."""
    strings = {label.strip(): i for i, label in enumerate(string_labels(fretboard), start=1)}
    pins: dict[int, Position] = {}
    for value in values:
        match = re.fullmatch(r"(\d+):([A-Za-z]#?)(\d+)", value)
        if not match or match.group(2) not in strings:
            raise ValueError(f"can't read pin {value!r}; use NOTE:STRINGFRET, e.g. 3:G5 (strings {' '.join(strings)})")
        index = int(match.group(1)) - 1
        if index in pins:
            raise ValueError(f"note {index + 1} is pinned twice")
        pins[index] = Position(strings[match.group(2)], int(match.group(3)))
    return pins


def _notes_to_dict(assignment: Assignment) -> list[dict[str, object]]:
    return [
        {**t.note.to_dict(), "string": t.position.string, "fret": t.position.fret}
        for t in assignment.notes
    ]


def _metrics_table(metrics: dict[str, PlayabilityMetrics]) -> str:
    names = list(metrics)
    lines = [f"{'':<22}" + "".join(f"{n:>12}" for n in names)]
    for key, label in METRIC_LABELS.items():
        values = [getattr(m, key) for m in metrics.values()]
        cells = "".join(f"{v:>12.1f}" if isinstance(v, float) else f"{v:>12}" for v in values)
        lines.append(f"{label:<22}{cells}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="Audio file (.wav .m4a ...) or MIDI file (.mid)")
    parser.add_argument(
        "--method", choices=["optimized", "baseline", "both"], default="optimized",
        help="String/fret assigner (default: optimized)",
    )
    parser.add_argument("--max-fret", type=int, default=DEFAULT_MAX_FRET, help="Highest fret on the guitar (classical: usually 19)")
    parser.add_argument("--style", choices=list(STYLES), default="balanced", help="Playing style for the optimizer")
    parser.add_argument("--start-fret", type=int, help="Fret the hand starts at (optimizer)")
    parser.add_argument("--pin", action="append", default=[], metavar="NOTE:STRINGFRET",
                        help="Fix a note's position, e.g. 3:G5; repeatable (optimizer)")
    parser.add_argument("--alternatives", type=int, default=1, metavar="N", help="Show up to N different tabs (optimizer)")
    parser.add_argument("--list-notes", action="store_true", help="Number the notes, for --pin")
    parser.add_argument("--no-ghost-filter", action="store_true", help="Keep octave-ghost notes")
    parser.add_argument("--no-ring-filter", action="store_true", help="Keep ringing open-string notes")
    parser.add_argument("--no-thump-filter", action="store_true", help="Keep quiet low notes under louder ones")
    parser.add_argument("--no-onset-filter", action="store_true", help="Audio only: keep quieter notes that start together with another")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Wrap tab lines at this width")
    parser.add_argument("--out-dir", default="outputs", help="Where to write audio/MIDI artifacts")
    parser.add_argument("--skip-preprocess", action="store_true", help="Audio only: skip FFmpeg")
    parser.add_argument("--onset-threshold", type=float, default=0.5, help="Audio only")
    parser.add_argument("--frame-threshold", type=float, default=0.3, help="Audio only")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args(argv)
    if args.alternatives < 1:
        parser.error("--alternatives must be at least 1")
    hints = args.pin or args.start_fret is not None or args.style != "balanced" or args.alternatives > 1
    if args.method == "baseline" and hints:
        parser.error("--style, --pin, --start-fret and --alternatives need the optimizer")

    if not 1 <= args.max_fret <= DEFAULT_MAX_FRET:
        parser.error(f"--max-fret must be 1-{DEFAULT_MAX_FRET}")
    fretboard = Fretboard(max_fret=args.max_fret)
    try:
        pins = parse_pins(args.pin, fretboard)
    except ValueError as exc:
        parser.error(str(exc))

    src = Path(args.path)
    source: dict[str, Any] = {"path": str(src)}
    try:
        if src.suffix.lower() in MIDI_EXTENSIONS:
            notes: list[NoteEvent] = parse_midi(src)
            source["type"] = "midi"
        else:
            settings = TranscriptionSettings(
                onset_threshold=args.onset_threshold, frame_threshold=args.frame_threshold
            )
            info, result, timings = transcribe_file(src, Path(args.out_dir), settings, args.skip_preprocess)
            notes = result.notes
            source.update(type="audio", audio=info.to_dict(), midi_path=str(result.midi_path), timings=timings)
    except PipelineError as exc:
        print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
        return 1

    cleanup = clean_notes(
        notes,
        fretboard,
        octave_ghosts=not args.no_ghost_filter,
        ringing_open_strings=not args.no_ring_filter,
        low_thumps=not args.no_thump_filter,
        # A MIDI file's simultaneous notes are written, not misheard.
        shared_onsets=source["type"] == "audio" and not args.no_onset_filter,
    )
    detected, notes = len(notes), cleanup.kept
    methods = list(ASSIGNERS) if args.method == "both" else [args.method]
    options = partial(alternatives, settings=STYLES[args.style], pins=pins, start_fret=args.start_fret)
    assignments: dict[str, Assignment] = {}
    try:
        for m in methods:
            if m == "optimized":
                for i, option in enumerate(options(notes, fretboard, args.alternatives), start=1):
                    assignments[m if i == 1 else f"option {i}"] = option
            else:
                assignments[m] = ASSIGNERS[m](notes, fretboard)
    except ValueError as exc:
        print(f"error [invalid_hint]: {exc}", file=sys.stderr)
        return 1
    metrics = {m: measure(a) for m, a in assignments.items()}
    tabs = {m: render_tab(a, fretboard, args.width) for m, a in assignments.items()}

    if args.json:
        print(json.dumps({
            "source": source,
            "removed": {reason: [n.to_dict() for n in ns] for reason, ns in cleanup.removed.items()},
            "results": {
                m: {
                    "tab": tabs[m],
                    "notes": _notes_to_dict(assignments[m]),
                    "unplayable": [n.to_dict() for n in assignments[m].unplayable],
                    "metrics": metrics[m].to_dict(),
                }
                for m in assignments
            },
        }, indent=2))
        return 0

    if source["type"] == "audio":
        audio = source["audio"]
        print(f"Input: {src.name} ({audio['duration_s']:.2f} s, {audio['format']}), "
              f"transcribed in {sum(source['timings'].values()):.1f} s")
    else:
        print(f"Input: {src.name} (MIDI)")
    removed = ", ".join(
        f"{len(ns)} {reason.replace('_', ' ')}{'s' if len(ns) > 1 else ''}"
        for reason, ns in cleanup.removed.items()
        if ns
    )
    print(f"Notes: {detected} detected" + (f", removed {removed}" if removed else ""))
    for reason, ns in cleanup.removed.items():
        for n in ns:
            print(f"  {reason.replace('_', ' ')}: {n.name} at {n.start_s:.2f} s")
    if args.list_notes:
        for i, n in enumerate(sorted(notes), start=1):
            print(f"  note {i}: {n.name} at {n.start_s:.2f} s")
    for m in assignments:
        print(f"\n{m.capitalize()}:\n{tabs[m]}")
    print("\n" + _metrics_table(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
