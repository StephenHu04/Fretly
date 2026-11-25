from __future__ import annotations

import pytest

from pipeline.midi_parser import NoteEvent
from tab.alternatives import MIN_DIFFERENCE, _difference, alternatives
from tab.fretboard import Fretboard, Position
from tab.optimizer import assign_optimized

BOARD = Fretboard()
RIFF = [NoteEvent(i * 0.25, (i + 1) * 0.25, p, 90) for i, p in enumerate([59, 56, 59, 61, 59, 57, 56, 54])]


def test_first_is_the_optimizers_choice_and_the_rest_differ() -> None:
    options = alternatives(RIFF, BOARD, 3)
    assert len(options) == 3
    assert options[0] == assign_optimized(RIFF, BOARD)
    for i, a in enumerate(options):
        assert all(_difference(a, b) >= MIN_DIFFERENCE for b in options[:i])
        assert all(BOARD.pitch_at(t.position) == t.note.pitch for t in a.notes)


def test_pins_hold_in_every_option() -> None:
    pin = Position(3, 4)
    assert all(a.notes[0].position == pin for a in alternatives(RIFF, BOARD, 3, pins={0: pin}))


def test_fewer_when_the_notes_allow_only_one_fingering() -> None:
    low = [NoteEvent(i * 0.25, (i + 1) * 0.25, p, 90) for i, p in enumerate([40, 41, 42, 43])]
    assert len(alternatives(low, BOARD, 3)) == 1
    assert len(alternatives([], BOARD, 3)) == 1


def test_count_must_be_positive() -> None:
    with pytest.raises(ValueError):
        alternatives(RIFF, BOARD, 0)
