"""Score fret choice on GuitarSet's single-note solo passages.

Compares each assigner's string and fret with the player's, using the dataset's
labelled notes (no audio, no transcription). Download annotation.zip from
https://zenodo.org/records/3371780 into data/guitarset/ first.

Usage:
    python scripts/score_guitarset.py
    python scripts/score_guitarset.py --holdout     # held-out players; final check only
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from tab.assign import assign_lowest_fret
from tab.evaluation import position_agreement
from tab.fretboard import Fretboard
from tab.guitarset import load_passages
from tab.optimizer import assign_optimized

ASSIGNERS = (assign_optimized, assign_lowest_fret)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotations", type=Path, default=Path("../data/guitarset/annotation.zip"))
    parser.add_argument("--holdout", action="store_true", help="Score the held-out players instead")
    args = parser.parse_args(argv)
    if not args.annotations.is_file():
        print(f"error: {args.annotations} not found; see the README for the download", file=sys.stderr)
        return 1

    fretboard = Fretboard()
    rows: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])  # passages, notes, optimizer, baseline
    for passage in load_passages(args.annotations, fretboard, args.holdout):
        notes = [p.note for p in passage.played]
        reference = [p.position for p in passage.played]
        matches = [
            position_agreement(reference, [t.position for t in assign(notes, fretboard).notes])
            for assign in ASSIGNERS
        ]
        for key in (passage.player, "total"):
            for i, value in enumerate((1, len(notes), *matches)):
                rows[key][i] += value

    print(f"{'player':<8}{'passages':>10}{'notes':>8}{'optimizer':>11}{'baseline':>10}")
    for key in sorted(rows):
        passages, total_notes, opt, base = rows[key]
        print(f"{key:<8}{passages:>10}{total_notes:>8}{opt / total_notes:>11.1%}{base / total_notes:>10.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
