"""Score the pipeline on recordings that have a reference tab.

For every ``<name>.tab.txt`` it finds the matching audio (``<name>.m4a`` etc.),
runs transcription and cleanup, and reports:

- detection: played notes found, missed and extra (precision / recall);
- positions: with the reference notes as input, how often each assigner picks
  the same string and fret as the player.

Usage:
    python scripts/evaluate.py                       # every reference under ../samples
    python scripts/evaluate.py ../samples/audio_clips/no_chord3.tab.txt -v
    python scripts/evaluate.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pipeline.audio_validation import SUPPORTED_EXTENSIONS
from pipeline.errors import PipelineError
from pipeline.midi_parser import NoteEvent, midi_to_note_name
from pipeline.run import transcribe_file
from tab.assign import assign_lowest_fret
from tab.cleanup import clean_notes
from tab.evaluation import parse_reference, position_agreement, score_detection
from tab.fretboard import Fretboard
from tab.optimizer import assign_optimized

REFERENCE_SUFFIX = ".tab.txt"


def find_references(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.rglob(f"*{REFERENCE_SUFFIX}")))
        else:
            found.append(path)
    return found


def find_audio(reference: Path) -> Path | None:
    stem = reference.name[: -len(REFERENCE_SUFFIX)]
    for ext in sorted(SUPPORTED_EXTENSIONS):
        candidate = reference.with_name(stem + ext)
        if candidate.is_file():
            return candidate
    return None


def evaluate_clip(
    reference: Path, audio: Path, fretboard: Fretboard, out_dir: Path, ghost_filter: bool, ring_filter: bool
) -> dict[str, Any]:
    positions = parse_reference(reference.read_text(encoding="utf-8"), fretboard)
    pitches = [fretboard.pitch_at(p) for p in positions]

    _, result, _ = transcribe_file(audio, out_dir)
    cleanup = clean_notes(result.notes, fretboard, octave_ghosts=ghost_filter, ringing_open_strings=ring_filter)
    detection = score_detection(pitches, cleanup.kept)

    # Fret choices are scored on the reference notes, so transcription
    # mistakes can't affect them. Timing is synthetic (one note every 0.3 s).
    ref_notes = [NoteEvent(i * 0.3, (i + 1) * 0.3, p, 80) for i, p in enumerate(pitches)]
    agreement = {
        name: position_agreement(positions, [t.position for t in assign(ref_notes, fretboard).notes])
        for name, assign in (("optimized", assign_optimized), ("baseline", assign_lowest_fret))
    }
    return {
        "clip": audio.stem,
        "played": detection.played,
        "detected": detection.detected,
        "correct": detection.correct,
        "missed": [midi_to_note_name(p) for p in detection.missed],
        "extra": [{"name": n.name, "start_s": round(n.start_s, 3), "velocity": n.velocity} for n in detection.extra],
        "removed": {
            reason: [{"name": n.name, "start_s": round(n.start_s, 3)} for n in ns]
            for reason, ns in cleanup.removed.items()
        },
        "positions_matching": agreement,
    }


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "-"


def format_report(clips: list[dict[str, Any]], verbose: bool) -> str:
    header = f"{'clip':<14}{'played':>7}{'found':>7}{'missed':>7}{'extra':>7}   {'optimizer':>10}{'baseline':>10}"
    lines = ["Detection (notes) and string/fret agreement with the reference", "", header]
    total = dict.fromkeys(("played", "correct", "missed", "extra", "opt", "base", "detected"), 0)
    for c in clips:
        opt, base = c["positions_matching"]["optimized"], c["positions_matching"]["baseline"]
        lines.append(
            f"{c['clip']:<14}{c['played']:>7}{c['correct']:>7}{len(c['missed']):>7}{len(c['extra']):>7}"
            f"   {opt:>4}/{c['played']:<5}{base:>4}/{c['played']:<5}"
        )
        if verbose:
            if c["missed"]:
                lines.append(f"{'':<14}missed: {', '.join(c['missed'])}")
            if c["extra"]:
                extras = ", ".join(f"{e['name']}@{e['start_s']:.2f}s" for e in c["extra"])
                lines.append(f"{'':<14}extra:  {extras}")
            for reason, removed in c["removed"].items():
                if removed:
                    names = ", ".join(f"{r['name']}@{r['start_s']:.2f}s" for r in removed)
                    lines.append(f"{'':<14}removed ({reason.replace('_', ' ')}): {names}")
        for key, value in (("played", c["played"]), ("correct", c["correct"]), ("missed", len(c["missed"])),
                           ("extra", len(c["extra"])), ("opt", opt), ("base", base), ("detected", c["detected"])):
            total[key] += value
    lines.append(
        f"{'total':<14}{total['played']:>7}{total['correct']:>7}{total['missed']:>7}{total['extra']:>7}"
        f"   {total['opt']:>4}/{total['played']:<5}{total['base']:>4}/{total['played']:<5}"
    )
    lines += [
        "",
        f"Recall    {_pct(total['correct'], total['played'])}  (played notes that were found)",
        f"Precision {_pct(total['correct'], total['detected'])}  (detected notes that were really played)",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("../samples")],
                        help="Reference .tab.txt files or folders to search (default: ../samples)")
    parser.add_argument("--no-ghost-filter", action="store_true", help="Score without the octave-ghost filter")
    parser.add_argument("--no-ring-filter", action="store_true", help="Score without the ringing open-string filter")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/evaluate"))
    parser.add_argument("-v", "--verbose", action="store_true", help="List missed and extra notes")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args(argv)

    fretboard = Fretboard()
    clips = []
    for reference in find_references(args.paths):
        audio = find_audio(reference)
        if audio is None:
            print(f"skipping {reference.name}: no matching audio file", file=sys.stderr)
            continue
        try:
            clips.append(evaluate_clip(
                reference, audio, fretboard, args.out_dir, not args.no_ghost_filter, not args.no_ring_filter
            ))
        except PipelineError as exc:
            print(f"error [{exc.code}] in {audio.name}: {exc.message}", file=sys.stderr)
            return 1
        except ValueError as exc:
            print(f"error in {reference.name}: {exc}", file=sys.stderr)
            return 1

    if not clips:
        print("error: no reference tabs with matching audio found", file=sys.stderr)
        return 1
    print(json.dumps({"clips": clips}, indent=2) if args.json else format_report(clips, args.verbose))
    return 0


if __name__ == "__main__":
    sys.exit(main())
