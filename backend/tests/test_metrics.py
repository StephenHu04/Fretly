from __future__ import annotations

import pytest

from pipeline.midi_parser import NoteEvent
from tab.assign import Assignment, TabNote
from tab.fretboard import Position
from tab.metrics import measure


def tab(*positions: tuple[int, int], step_s: float = 0.5) -> Assignment:
    """Build an assignment from (string, fret) pairs, one note every ``step_s``."""
    notes = [
        TabNote(NoteEvent(i * step_s, (i + 1) * step_s, 60, 100), Position(s, f))
        for i, (s, f) in enumerate(positions)
    ]
    return Assignment(notes, [])


def test_staying_in_one_position_has_no_shifts() -> None:
    # Frets 5-8 fit under one hand.
    m = measure(tab((6, 5), (6, 8), (5, 5), (5, 7), (4, 6)))
    assert m.position_shifts == 0
    assert m.fret_travel == 3 + 3 + 2 + 1
    assert m.string_changes == 2


def test_leaving_the_window_counts_one_shift_each_time() -> None:
    m = measure(tab((1, 1), (1, 3), (1, 7), (1, 9), (1, 2)))
    assert m.position_shifts == 2  # 1-3 -> 7-9 -> 2
    assert m.fret_travel == 2 + 4 + 2 + 7


def test_window_is_four_frets_wide() -> None:
    assert measure(tab((1, 5), (1, 8))).position_shifts == 0
    assert measure(tab((1, 5), (1, 9))).position_shifts == 1


def test_open_strings_do_not_move_the_hand() -> None:
    m = measure(tab((1, 7), (2, 0), (1, 8)))
    assert m.position_shifts == 0
    assert m.fret_travel == 1
    assert m.string_changes == 2  # open strings still change string


def test_stretch_only_counts_quick_consecutive_notes() -> None:
    assert measure(tab((1, 2), (1, 7), step_s=0.1)).max_stretch == 5
    assert measure(tab((1, 2), (1, 7), step_s=0.5)).max_stretch == 0


def test_shifts_per_minute() -> None:
    # Two shifts over 6 notes x 0.5 s = 3 s of music -> 40 per minute.
    m = measure(tab((1, 1), (1, 9), (1, 1), (1, 1), (1, 1), (1, 1)))
    assert m.position_shifts == 2
    assert m.shifts_per_minute == pytest.approx(40.0)


def test_counts_unplayable_notes() -> None:
    lost = NoteEvent(0.0, 0.5, 30, 100)
    m = measure(Assignment(tab((1, 1)).notes, [lost]))
    assert (m.notes, m.unplayable) == (1, 1)


def test_empty_assignment() -> None:
    m = measure(Assignment([], []))
    assert m.to_dict() == {
        "notes": 0,
        "unplayable": 0,
        "position_shifts": 0,
        "shifts_per_minute": 0.0,
        "max_stretch": 0,
        "fret_travel": 0,
        "string_changes": 0,
    }
