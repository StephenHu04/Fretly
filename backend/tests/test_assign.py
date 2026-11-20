from __future__ import annotations

from pipeline.midi_parser import NoteEvent
from tab.assign import assign_lowest_fret
from tab.fretboard import Fretboard, Position

BOARD = Fretboard()


def melody(pitches: list[int], step_s: float = 0.25) -> list[NoteEvent]:
    return [NoteEvent(i * step_s, (i + 1) * step_s, p, 100) for i, p in enumerate(pitches)]


def test_every_position_sounds_the_note() -> None:
    notes = melody(list(range(40, 89)))
    result = assign_lowest_fret(notes, BOARD)
    assert result.unplayable == []
    assert [t.note for t in result.notes] == notes
    assert all(BOARD.pitch_at(t.position) == t.note.pitch for t in result.notes)


def test_picks_the_lowest_fret() -> None:
    # E4 is playable on all six strings; the baseline always takes the open string.
    result = assign_lowest_fret(melody([64, 60, 67]), BOARD)
    assert [t.position for t in result.notes] == [Position(1, 0), Position(2, 1), Position(1, 3)]


def test_unplayable_notes_are_reported_not_dropped() -> None:
    notes = melody([39, 52, 89])
    result = assign_lowest_fret(notes, BOARD)
    assert [t.note.pitch for t in result.notes] == [52]
    assert result.unplayable == [notes[0], notes[2]]


def test_output_is_in_time_order() -> None:
    notes = melody([40, 45, 50])
    result = assign_lowest_fret(reversed(notes), BOARD)
    assert [t.note for t in result.notes] == notes


def test_respects_the_fretboard_configuration() -> None:
    short = Fretboard(max_fret=12)
    result = assign_lowest_fret(melody([76, 77]), short)
    assert [t.position for t in result.notes] == [Position(1, 12)]
    assert [n.pitch for n in result.unplayable] == [77]


def test_empty_input() -> None:
    result = assign_lowest_fret([], BOARD)
    assert result.notes == [] and result.unplayable == []
