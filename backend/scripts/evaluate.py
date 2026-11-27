"""Score the pipeline on recordings that have a reference tab.

For every ``<name>.tab.txt`` it finds the matching audio (``<name>.m4a`` etc.),
runs transcription and cleanup, and reports:

- detection: played notes found, missed and extra (precision / recall);
- frets: with the reference notes as input, how often each assigner picks the
  same string and fret as the player;
  These use no audio, so ``--profile`` doesn't change them;
- in tab: end to end, played notes that appear in the final tab exactly as
  played (right note, right string and fret).

Clips in a ``holdout`` folder are skipped when searching, so they can be kept
for a final check that nothing was tuned on. Score them with ``--holdout``.

Usage:
    python scripts/evaluate.py                       # every reference under ../samples, except holdout/
    python scripts/evaluate.py ../samples/audio_clips/no_chord3.tab.txt -v
    python scripts/evaluate.py --holdout             # only ../samples/holdout
    python scripts/evaluate.py --json
    python scripts/evaluate.py --profile ../samples/calibration/profile.json  # optimizer hears strings
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

from pipeline.audio_validation import SUPPORTED_EXTENSIONS
from pipeline.errors import PipelineError
from pipeline.midi_parser import NoteEvent, midi_to_note_name
from pipeline.run import preprocessed_path, transcribe_file
from tab.assign import assign_lowest_fret
from tab.cleanup import clean_notes
from tab.evaluation import exact_in_tab, parse_reference, position_agreement, score_detection
from tab.fretboard import DEFAULT_MAX_FRET, Fretboard
from tab.optimizer import STYLES, OptimizerSettings, assign_optimized
from tab.string_model import Profile, read_samples

REFERENCE_SUFFIX = ".tab.txt"
HOLDOUT_DIR = "holdout"


def find_references(paths: list[Path]) -> list[Path]:
    """Reference tabs in ``paths``; folder searches skip ``holdout`` folders below them.

    A holdout folder passed directly (or a file inside one) is still used.
    """
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(
                ref
                for ref in sorted(path.rglob(f"*{REFERENCE_SUFFIX}"))
                if HOLDOUT_DIR not in ref.relative_to(path).parts[:-1]
            )
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
    reference: Path,
    audio: Path,
    fretboard: Fretboard,
    out_dir: Path,
    filters: dict[str, bool],
    settings: OptimizerSettings,
    profile: Profile | None = None,
) -> dict[str, Any]:
    positions = parse_reference(reference.read_text(encoding="utf-8"), fretboard)
    pitches = [fretboard.pitch_at(p) for p in positions]

    _, result, _ = transcribe_file(audio, out_dir)
    cleanup = clean_notes(result.notes, fretboard, **filters)
    detection = score_detection(pitches, cleanup.kept)

    # Fret choices are scored on the reference notes, so transcription
    # mistakes can't affect them. Timing is synthetic (one note every 0.3 s).
    ref_notes = [NoteEvent(i * 0.3, (i + 1) * 0.3, p, 80) for i, p in enumerate(pitches)]
    assigners = (("optimized", partial(assign_optimized, settings=settings)), ("baseline", assign_lowest_fret))
    agreement = {
        name: position_agreement(positions, [t.position for t in assign(ref_notes, fretboard).notes])
        for name, assign in assigners
    }
    in_tab = {name: exact_in_tab(positions, assign(cleanup.kept, fretboard), fretboard) for name, assign in assigners}
    if profile:
        sound = profile.string_probabilities(read_samples(preprocessed_path(audio, out_dir)), cleanup.kept, fretboard)
        heard = assign_optimized(cleanup.kept, fretboard, settings, sound=sound)
        in_tab["optimized"] = exact_in_tab(positions, heard, fretboard)
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
        "exact_in_tab": in_tab,
    }


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "-"


def format_report(clips: list[dict[str, Any]], verbose: bool) -> str:
    def row(label: str, c: dict[str, int]) -> str:
        n = c["played"]
        return (
            f"{label:<12}{n:>7}{c['correct']:>7}{c['missed']:>7}{c['extra']:>7}"
            f"   {c['frets_opt']:>3}/{n:<4}{c['frets_base']:>3}/{n:<4}"
            f"   {c['tab_opt']:>3}/{n:<4}{c['tab_base']:>3}/{n:<4}"
        )

    lines = [
        f"{'':<40}   {'frets':<16}   {'in tab':<16}",
        f"{'clip':<12}{'played':>7}{'found':>7}{'missed':>7}{'extra':>7}   {'optim.':<8}{'base':<8}   {'optim.':<8}{'base':<8}",
    ]
    total = dict.fromkeys(
        ("played", "detected", "correct", "missed", "extra", "frets_opt", "frets_base", "tab_opt", "tab_base"), 0
    )
    for c in clips:
        counts = {
            "played": c["played"],
            "detected": c["detected"],
            "correct": c["correct"],
            "missed": len(c["missed"]),
            "extra": len(c["extra"]),
            "frets_opt": c["positions_matching"]["optimized"],
            "frets_base": c["positions_matching"]["baseline"],
            "tab_opt": c["exact_in_tab"]["optimized"],
            "tab_base": c["exact_in_tab"]["baseline"],
        }
        lines.append(row(c["clip"], counts))
        if verbose:
            if c["missed"]:
                lines.append(f"{'':<12}missed: {', '.join(c['missed'])}")
            if c["extra"]:
                extras = ", ".join(f"{e['name']}@{e['start_s']:.2f}s" for e in c["extra"])
                lines.append(f"{'':<12}extra:  {extras}")
            for reason, removed in c["removed"].items():
                if removed:
                    names = ", ".join(f"{r['name']}@{r['start_s']:.2f}s" for r in removed)
                    lines.append(f"{'':<12}removed ({reason.replace('_', ' ')}): {names}")
        for key in total:
            total[key] += counts[key]
    lines.append(row("total", total))
    n = total["played"]
    lines += [
        "",
        f"Recall      {_pct(total['correct'], n)}  played notes that were heard",
        f"Precision   {_pct(total['correct'], total['detected'])}  detected notes that were really played",
        f"Frets       optimizer {_pct(total['frets_opt'], n)}, baseline {_pct(total['frets_base'], n)}"
        "  player's string/fret, given the correct notes",
        f"In tab      optimizer {_pct(total['tab_opt'], n)}, baseline {_pct(total['tab_base'], n)}"
        "  played notes exactly right in the final tab",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path,
                        help="Reference .tab.txt files or folders to search (default: ../samples)")
    parser.add_argument("--holdout", action="store_true",
                        help=f"Score only the held-out clips in ../samples/{HOLDOUT_DIR}")
    parser.add_argument("--max-fret", type=int, default=DEFAULT_MAX_FRET, help="Highest fret on the guitar (classical: usually 19)")
    parser.add_argument("--style", choices=list(STYLES), default="balanced", help="Playing style for the optimizer")
    parser.add_argument("--profile", type=Path, help="Sound profile from calibrate.py; the optimizer's in-tab score uses it")
    parser.add_argument("--no-ghost-filter", action="store_true", help="Score without the octave-ghost filter")
    parser.add_argument("--no-ring-filter", action="store_true", help="Score without the ringing open-string filter")
    parser.add_argument("--no-thump-filter", action="store_true", help="Score without the low-thump filter")
    parser.add_argument("--no-onset-filter", action="store_true", help="Score without the shared-onset filter")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/evaluate"))
    parser.add_argument("-v", "--verbose", action="store_true", help="List missed and extra notes")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args(argv)

    if not 1 <= args.max_fret <= DEFAULT_MAX_FRET:
        parser.error(f"--max-fret must be 1-{DEFAULT_MAX_FRET}")
    fretboard = Fretboard(max_fret=args.max_fret)
    filters = {
        "octave_ghosts": not args.no_ghost_filter,
        "ringing_open_strings": not args.no_ring_filter,
        "low_thumps": not args.no_thump_filter,
        "shared_onsets": not args.no_onset_filter,
    }
    profile = None
    if args.profile:
        try:
            profile = Profile.load(args.profile)
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"can't read profile: {exc}")
    clips = []
    if args.holdout and args.paths:
        parser.error("--holdout can't be combined with paths")
    paths = args.paths or [Path("../samples") / HOLDOUT_DIR if args.holdout else Path("../samples")]
    for reference in find_references(paths):
        audio = find_audio(reference)
        if audio is None:
            print(f"skipping {reference.name}: no matching audio file", file=sys.stderr)
            continue
        try:
            clips.append(evaluate_clip(reference, audio, fretboard, args.out_dir, filters, STYLES[args.style], profile))
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
