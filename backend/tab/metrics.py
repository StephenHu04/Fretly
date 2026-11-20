"""Playability metrics for an assignment, used to compare assigners.

Open strings don't need the fretting hand, so they're ignored for shifts,
stretch and fret travel (but still count as string changes).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise

from tab.assign import Assignment

# A hand in one position covers four frets, one per finger.
HAND_SPAN_FRETS = 4
# Consecutive notes closer than this are played without time to move the hand.
STRETCH_WINDOW_S = 0.25


@dataclass(frozen=True)
class PlayabilityMetrics:
    notes: int
    unplayable: int
    position_shifts: int
    shifts_per_minute: float
    max_stretch: int
    fret_travel: int
    string_changes: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def measure(assignment: Assignment) -> PlayabilityMetrics:
    notes = assignment.notes
    fretted = [n for n in notes if not n.position.is_open]

    # A shift happens when a fretted note can't share a 4-fret window with
    # every fretted note played since the last shift.
    shifts = 0
    low = high = None
    for n in fretted:
        fret = n.position.fret
        if low is None or high is None:
            low = high = fret
        elif max(high, fret) - min(low, fret) >= HAND_SPAN_FRETS:
            shifts += 1
            low = high = fret
        else:
            low, high = min(low, fret), max(high, fret)

    pairs = list(pairwise(fretted))
    travel = sum(abs(b.position.fret - a.position.fret) for a, b in pairs)
    stretch = max(
        (
            abs(b.position.fret - a.position.fret)
            for a, b in pairs
            if b.note.start_s - a.note.start_s < STRETCH_WINDOW_S
        ),
        default=0,
    )
    string_changes = sum(a.position.string != b.position.string for a, b in pairwise(notes))

    duration_s = notes[-1].note.end_s - notes[0].note.start_s if notes else 0.0
    per_minute = shifts / duration_s * 60 if duration_s > 0 else 0.0

    return PlayabilityMetrics(
        notes=len(notes),
        unplayable=len(assignment.unplayable),
        position_shifts=shifts,
        shifts_per_minute=per_minute,
        max_stretch=stretch,
        fret_travel=travel,
        string_changes=string_changes,
    )
