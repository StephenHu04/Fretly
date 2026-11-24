"""Compare the optimizer with the lowest-fret baseline on fixed test phrases.

Prints the playability metrics for both assigners on every phrase in
tab/phrases.py. Agreement with real players is scored separately by
scripts/evaluate.py on recordings with reference tabs.

Usage:
    python scripts/compare_assigners.py
    python scripts/compare_assigners.py -v        # also print both tabs per phrase
"""

from __future__ import annotations

import argparse
import sys

from tab.assign import assign_lowest_fret
from tab.fretboard import Fretboard
from tab.metrics import measure
from tab.optimizer import assign_optimized
from tab.phrases import PHRASES
from tab.render import render_tab

METRICS = (
    ("position_shifts", "shifts"),
    ("fret_travel", "travel"),
    ("max_stretch", "stretch"),
    ("string_changes", "strings"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="store_true", help="Print both tabs for each phrase")
    args = parser.parse_args(argv)

    fretboard = Fretboard()
    header = f"{'phrase':<14}" + "".join(f"{label:>14}" for _, label in METRICS)
    lines = ["Optimizer / baseline (lower is better)", "", header]
    totals = {key: [0, 0] for key, _ in METRICS}
    for phrase in PHRASES:
        assignments = [assign(phrase.notes(), fretboard) for assign in (assign_optimized, assign_lowest_fret)]
        opt, base = (measure(a) for a in assignments)
        cells = ""
        for key, _ in METRICS:
            o, b = getattr(opt, key), getattr(base, key)
            totals[key][0] += o
            totals[key][1] += b
            cells += f"{o:>9} / {b:<2}"
        lines.append(f"{phrase.name:<14}{cells}")
        if args.verbose:
            lines.append(f"  {phrase.description}")
            for label, assignment in zip(("optimizer", "baseline"), assignments, strict=True):
                lines.append(f"  {label}:")
                lines.extend("    " + row for row in render_tab(assignment, fretboard).splitlines())
            lines.append("")
    lines.append(f"{'total':<14}" + "".join(f"{o:>9} / {b:<2}" for o, b in totals.values()))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
