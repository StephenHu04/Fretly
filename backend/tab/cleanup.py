"""Filters that remove unreliable notes before string/fret assignment.

- Octave ghosts: Basic Pitch sometimes reports a note's second harmonic as an
  extra note one octave up, at the same time and about half the velocity. On
  real recordings quiet ghosts an octave *below* a loud note also appear.
- Ringing open strings: lifting a finger off a fretted note lets the open
  string ring, and the model hears that as a new note (decision 011).
- Low thumps: short, quiet low notes (mostly G2) under a louder note, from
  the guitar body or the pick attack (decision 014).

Left in, all of them become notes nobody played.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from pipeline.midi_parser import NoteEvent
from tab.fretboard import Fretboard

OCTAVE = 12


@dataclass(frozen=True)
class GhostFilterSettings:
    # Starting values from synthetic audio; retune on real recordings.
    max_onset_gap_s: float = 0.05
    max_velocity_ratio: float = 0.6
    below: bool = True  # also remove ghosts an octave below (seen on real recordings)

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
    octave_away = candidate.pitch == base.pitch + OCTAVE or (settings.below and candidate.pitch == base.pitch - OCTAVE)
    return (
        octave_away
        and abs(candidate.start_s - base.start_s) <= settings.max_onset_gap_s
        and candidate.velocity <= settings.max_velocity_ratio * base.velocity
    )


def remove_octave_ghosts(
    notes: Iterable[NoteEvent], settings: GhostFilterSettings | None = None
) -> FilterResult:
    """Drop quiet notes exactly an octave from a louder note starting at the same time.

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


@dataclass(frozen=True)
class RingingFilterSettings:
    # Fitted on the four no_chord clips (2026-11-22): catches 6 of 12 ringing
    # notes and no real ones. Retune as more reference clips are added.
    max_lead_s: float = 0.05  # open note may start this long before the fretted note ends
    max_lag_s: float = 0.07  # ...or this long after it
    max_velocity_ratio: float = 0.75

    def __post_init__(self) -> None:
        if self.max_lead_s < 0 or self.max_lag_s < 0:
            raise ValueError("max_lead_s and max_lag_s must be non-negative")
        if not 0 < self.max_velocity_ratio <= 1:
            raise ValueError("max_velocity_ratio must be in (0, 1]")


def _is_ringing_after(
    candidate: NoteEvent, fretted: NoteEvent, open_pitch: int, fretboard: Fretboard, settings: RingingFilterSettings
) -> bool:
    return (
        open_pitch < fretted.pitch <= open_pitch + fretboard.max_fret  # could be fretted on that string
        and fretted.start_s < candidate.start_s
        and -settings.max_lead_s <= candidate.start_s - fretted.end_s <= settings.max_lag_s
        and candidate.velocity <= settings.max_velocity_ratio * fretted.velocity
    )


def remove_ringing_open_strings(
    notes: Iterable[NoteEvent], fretboard: Fretboard, settings: RingingFilterSettings | None = None
) -> FilterResult:
    """Drop notes that look like an open string ringing after a finger lifted off it.

    A note is removed when its pitch is an open string's, a note that could be
    fretted on that string ends right around its onset, and it's clearly
    quieter than that note. The check uses every string a note *could* be on,
    not an assignment, because the assigner may pick a different string from
    the one the player used.
    """
    settings = settings or RingingFilterSettings()
    ordered = sorted(notes)
    kept: list[NoteEvent] = []
    removed: list[NoteEvent] = []
    for note in ordered:
        if note.pitch in fretboard.tuning and any(
            _is_ringing_after(note, other, note.pitch, fretboard, settings) for other in ordered if other is not note
        ):
            removed.append(note)
        else:
            kept.append(note)
    return FilterResult(kept, removed)


@dataclass(frozen=True)
class ThumpFilterSettings:
    # Fitted on the eight reference clips (2026-11-23); see decision 014.
    min_interval: int = OCTAVE  # semitones below the louder note
    max_lead_s: float = 0.08  # may start this long before the louder note
    max_velocity_ratio: float = 0.8
    max_duration_s: float = 0.35

    def __post_init__(self) -> None:
        if self.min_interval < 1:
            raise ValueError("min_interval must be at least 1")
        if self.max_lead_s < 0 or self.max_duration_s <= 0:
            raise ValueError("max_lead_s must be non-negative and max_duration_s positive")
        if not 0 < self.max_velocity_ratio <= 1:
            raise ValueError("max_velocity_ratio must be in (0, 1]")


def _is_thump_under(candidate: NoteEvent, louder: NoteEvent, settings: ThumpFilterSettings) -> bool:
    return (
        candidate.pitch <= louder.pitch - settings.min_interval
        and louder.start_s - settings.max_lead_s <= candidate.start_s < louder.end_s
        and candidate.end_s - candidate.start_s <= settings.max_duration_s
        and candidate.velocity <= settings.max_velocity_ratio * louder.velocity
    )


def remove_low_thumps(notes: Iterable[NoteEvent], settings: ThumpFilterSettings | None = None) -> FilterResult:
    """Drop short, quiet notes well below a louder note that's sounding at the time.

    On phone recordings the model reports the guitar body's resonance or the
    pick's attack as a low note (mostly G2) whatever note was played. In
    single-note playing nothing else sounds under a note, so a short quiet note
    an octave or more below a louder one is noise. With chords (milestone 3.9)
    a quiet bass note under a melody can be real, so this filter will need to
    be off there.
    """
    settings = settings or ThumpFilterSettings()
    ordered = sorted(notes)
    kept: list[NoteEvent] = []
    removed: list[NoteEvent] = []
    for note in ordered:
        if any(_is_thump_under(note, other, settings) for other in ordered if other is not note):
            removed.append(note)
        else:
            kept.append(note)
    return FilterResult(kept, removed)


@dataclass(frozen=True)
class CleanupResult:
    kept: list[NoteEvent]
    removed: dict[str, list[NoteEvent]]  # reason -> notes removed for it


def clean_notes(
    notes: Iterable[NoteEvent],
    fretboard: Fretboard,
    octave_ghosts: bool = True,
    ringing_open_strings: bool = True,
    low_thumps: bool = True,
) -> CleanupResult:
    """Run the enabled filters in order and report what each one removed."""
    kept = sorted(notes)
    removed: dict[str, list[NoteEvent]] = {}
    if octave_ghosts:
        result = remove_octave_ghosts(kept)
        kept, removed["octave_ghost"] = result.kept, result.removed
    if ringing_open_strings:
        result = remove_ringing_open_strings(kept, fretboard)
        kept, removed["ringing_open_string"] = result.kept, result.removed
    if low_thumps:
        result = remove_low_thumps(kept)
        kept, removed["low_thump"] = result.kept, result.removed
    return CleanupResult(kept, removed)
