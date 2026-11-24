"""Turn a guitar recording or a MIDI file into tablature.

Pipeline: [audio -> validate -> FFmpeg -> Basic Pitch ->] notes ->
cleanup (octave ghosts, ringing open strings, low thumps) -> string/fret assignment ->
plain-text tab + metrics.

Audio files run the Phase 2 transcription first; .mid/.midi files skip it.

Usage:
    python scripts/make_tab.py ../samples/audio_clips/no_chord1.m4a
    python scripts/make_tab.py clip.mid --method both     # baseline vs optimizer
    python scripts/make_tab.py clip.m4a --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pipeline.errors import PipelineError
from pipeline.midi_parser import NoteEvent, parse_midi
from pipeline.run import transcribe_file
from pipeline.transcribe import TranscriptionSettings
from tab.assign import Assigner, Assignment, assign_lowest_fret
from tab.cleanup import clean_notes
from tab.fretboard import Fretboard
from tab.metrics import PlayabilityMetrics, measure
from tab.optimizer import assign_optimized
from tab.render import DEFAULT_WIDTH, render_tab

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
    parser.add_argument("--no-ghost-filter", action="store_true", help="Keep octave-ghost notes")
    parser.add_argument("--no-ring-filter", action="store_true", help="Keep ringing open-string notes")
    parser.add_argument("--no-thump-filter", action="store_true", help="Keep quiet low notes under louder ones")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Wrap tab lines at this width")
    parser.add_argument("--out-dir", default="outputs", help="Where to write audio/MIDI artifacts")
    parser.add_argument("--skip-preprocess", action="store_true", help="Audio only: skip FFmpeg")
    parser.add_argument("--onset-threshold", type=float, default=0.5, help="Audio only")
    parser.add_argument("--frame-threshold", type=float, default=0.3, help="Audio only")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args(argv)

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

    fretboard = Fretboard()
    cleanup = clean_notes(
        notes,
        fretboard,
        octave_ghosts=not args.no_ghost_filter,
        ringing_open_strings=not args.no_ring_filter,
        low_thumps=not args.no_thump_filter,
    )
    detected, notes = len(notes), cleanup.kept
    methods = list(ASSIGNERS) if args.method == "both" else [args.method]
    assignments = {m: ASSIGNERS[m](notes, fretboard) for m in methods}
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
                for m in methods
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
    for m in methods:
        print(f"\n{m.capitalize()}:\n{tabs[m]}")
    print("\n" + _metrics_table(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
