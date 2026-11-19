"""Filters that remove unreliable notes before string/fret assignment.

Basic Pitch sometimes reports a note's second harmonic as an extra note one
octave up, starting at the same time and at about half the velocity (observed
on synthetic audio in Phase 2). Left in, these become phantom notes in the tab.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from pipeline.midi_parser import NoteEvent

OCTAVE = 12


@dataclass(frozen=True)
class GhostFilterSettings:
    # Starting values from synthetic audio; retune on real recordings.
    max_onset_gap_s: float = 0.05
    max_velocity_ratio: float = 0.6

    def __post_init__(self) -> None:
        if self.max_onset_gap_s < 0:
            raise ValueError("max_onset_gap_s must be non-negative")
        if not 0 < self.max_velocity_ratio <= 1:
            raise ValueError("max_velocity_ratio must be in (0, 1]")


@dataclass(frozen=True)
class FilterResult:
    kept: list[NoteEvent]
    removed: list[NoteEvent]


def _is_ghost_of(candidate: NoteEvent, base: NoteEvent, settings: GhostFilterSettings) -> bool:
    return (
        candidate.pitch == base.pitch + OCTAVE
        and abs(candidate.start_s - base.start_s) <= settings.max_onset_gap_s
        and candidate.velocity <= settings.max_velocity_ratio * base.velocity
    )


def remove_octave_ghosts(
    notes: Iterable[NoteEvent], settings: GhostFilterSettings | None = None
) -> FilterResult:
    """Drop notes that look like the second harmonic of a louder note an octave below.

    Every note is compared against the original input, so a ghost can't
    protect another note by being removed first. Removed notes are returned
    rather than discarded so callers can show what was filtered.
    """
    settings = settings or GhostFilterSettings()
    ordered = sorted(notes)
    kept: list[NoteEvent] = []
    removed: list[NoteEvent] = []
    for note in ordered:
        # Quadratic, but clips are short (tens to hundreds of notes).
        if any(_is_ghost_of(note, base, settings) for base in ordered):
            removed.append(note)
        else:
            kept.append(note)
    return FilterResult(kept, removed)
