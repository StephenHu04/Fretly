"""Offer several different tabs for the same notes (decision 021).

The notes alone often fit more than one hand position, so besides the
optimizer's first choice this proposes fingerings that differ from it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace

from pipeline.midi_parser import NoteEvent
from tab.assign import Assignment
from tab.fretboard import Fretboard, Position
from tab.optimizer import STYLES, OptimizerSettings, assign_optimized, rests_before, sequence_cost

HOME_FRETS = (1, 3, 5, 7, 9, 12)
MIN_DIFFERENCE = 0.3  # share of notes that must move for a tab to count as different


def _difference(a: Assignment, b: Assignment) -> float:
    if not a.notes:
        return 0.0
    return sum(x.position != y.position for x, y in zip(a.notes, b.notes, strict=True)) / len(a.notes)


def alternatives(
    notes: Iterable[NoteEvent],
    fretboard: Fretboard,
    count: int,
    settings: OptimizerSettings | None = None,
    pins: Mapping[int, Position] | None = None,
    start_fret: int | None = None,
) -> list[Assignment]:
    """Up to ``count`` clearly different assignments, the optimizer's choice first.

    The others come from the playing styles and from other home frets, cheapest
    first under ``settings``. Fewer are returned when the notes don't allow more.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    ordered = sorted(notes)
    settings = settings or OptimizerSettings()

    def solve(candidate: OptimizerSettings) -> Assignment:
        return assign_optimized(ordered, fretboard, candidate, pins=pins, start_fret=start_fret)

    def cost(assignment: Assignment) -> float:
        rests = rests_before([t.note for t in assignment.notes])
        return sequence_cost([t.position for t in assignment.notes], settings, rests, start_fret)

    chosen = [solve(settings)]
    variants = [*STYLES.values(), *(replace(STYLES["lead"], home_fret=fret) for fret in HOME_FRETS)]
    for option in sorted((solve(v) for v in variants), key=cost):
        if len(chosen) == count:
            break
        if all(_difference(option, other) >= MIN_DIFFERENCE for other in chosen):
            chosen.append(option)
    return chosen
