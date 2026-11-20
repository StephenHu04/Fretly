"""String/fret assignment: shared types and the naive baseline.

Every assigner has the same signature, ``(notes, fretboard) -> Assignment``, so
metrics, rendering and comparisons don't depend on which one ran.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from pipeline.midi_parser import NoteEvent
from tab.fretboard import Fretboard, Position


@dataclass(frozen=True)
class TabNote:
    note: NoteEvent
    position: Position


@dataclass(frozen=True)
class Assignment:
    notes: list[TabNote]  # playable notes in time order
    unplayable: list[NoteEvent]  # outside the fretboard's range; never dropped silently


Assigner = Callable[[Iterable[NoteEvent], Fretboard], Assignment]


def assign_lowest_fret(notes: Iterable[NoteEvent], fretboard: Fretboard) -> Assignment:
    """Give each note its lowest-fret position, ignoring its neighbours.

    This is what a naive converter does, and it's the baseline the optimizer
    has to beat. Ties (only possible with repeated open-string pitches) go to
    the higher string. Simultaneous notes may land on the same string;
    chords are handled later.
    """
    placed: list[TabNote] = []
    unplayable: list[NoteEvent] = []
    for note in sorted(notes):
        candidates = fretboard.candidates(note.pitch)
        if candidates:
            placed.append(TabNote(note, candidates[0]))
        else:
            unplayable.append(note)
    return Assignment(placed, unplayable)
