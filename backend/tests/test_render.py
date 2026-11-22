from __future__ import annotations

from textwrap import dedent

from pipeline.midi_parser import NoteEvent
from tab.assign import Assignment, TabNote
from tab.fretboard import Fretboard, Position
from tab.optimizer import assign_optimized
from tab.render import render_tab, string_labels

BOARD = Fretboard()


def tab(*columns: dict[int, int], step_s: float = 0.25) -> Assignment:
    """Build an assignment from columns of {string: fret}, one column every ``step_s``."""
    notes = []
    for i, column in enumerate(columns):
        for string, fret in column.items():
            pos = Position(string, fret)
            notes.append(TabNote(NoteEvent(i * step_s, (i + 1) * step_s, BOARD.pitch_at(pos), 80), pos))
    return Assignment(sorted(notes, key=lambda t: (t.note.start_s, t.position.string)), [])


def golden(text: str) -> str:
    return dedent(text).strip("\n")


def test_no_chord1_from_the_model_notes() -> None:
    # The notes Basic Pitch detected in samples/audio_clips/no_chord1.m4a
    # (start, end, pitch, velocity), run through the optimizer.
    detected = [
        (0.975, 1.196, 64, 84), (1.265, 1.498, 63, 70), (1.614, 1.916, 59, 58),
        (1.916, 2.149, 64, 66), (2.254, 2.405, 63, 53), (2.567, 3.055, 59, 84),
        (2.846, 3.078, 64, 74), (3.159, 3.368, 63, 69), (3.484, 3.728, 59, 73),
        (3.809, 4.043, 64, 92), (4.031, 4.461, 59, 55), (4.449, 4.809, 68, 87),
        (4.461, 4.693, 59, 43),
    ]  # fmt: skip
    notes = [NoteEvent(*n) for n in detected]
    assert render_tab(assign_optimized(notes, BOARD), BOARD) == golden("""
        e|-0-----0-----0-----0---4-|
        B|---4-0---4-0---4-0---0-0-|
        G|-------------------------|
        D|-------------------------|
        A|-------------------------|
        E|-------------------------|
    """)


def test_no_chord1_reference_tab() -> None:
    phrase = [{1: 0}, {2: 4}, {2: 0}] * 3 + [{2: 5}, {1: 2}, {1: 4}]
    assert render_tab(tab(*phrase), BOARD) == golden("""
        e|-0-----0-----0-------2-4-|
        B|---4-0---4-0---4-0-5-----|
        G|-------------------------|
        D|-------------------------|
        A|-------------------------|
        E|-------------------------|
    """)


def test_two_digit_frets_stay_aligned_and_double_stops_share_a_column() -> None:
    assert render_tab(tab({1: 12}, {1: 10}, {2: 13}, {1: 8, 2: 10}), BOARD) == golden("""
        e|-12-10----8--|
        B|-------13-10-|
        G|-------------|
        D|-------------|
        A|-------------|
        E|-------------|
    """)


def test_notes_within_30_ms_share_a_column() -> None:
    a = Position(1, 3)
    b = Position(3, 2)
    notes = [
        TabNote(NoteEvent(0.0, 0.5, BOARD.pitch_at(a), 80), a),
        TabNote(NoteEvent(0.02, 0.5, BOARD.pitch_at(b), 80), b),
    ]
    assert render_tab(Assignment(notes, []), BOARD).splitlines()[:3] == ["e|-3-|", "B|---|", "G|-2-|"]
    notes[1] = TabNote(NoteEvent(0.05, 0.5, BOARD.pitch_at(b), 80), b)
    assert render_tab(Assignment(notes, []), BOARD).splitlines()[:3] == ["e|-3---|", "B|-----|", "G|---2-|"]


def test_same_string_in_one_column_moves_to_the_next_column() -> None:
    first, second = Position(1, 3), Position(1, 5)
    notes = [
        TabNote(NoteEvent(0.0, 0.5, BOARD.pitch_at(first), 80), first),
        TabNote(NoteEvent(0.01, 0.5, BOARD.pitch_at(second), 80), second),
    ]
    assert render_tab(Assignment(notes, []), BOARD).splitlines()[0] == "e|-3-5-|"


def test_long_tabs_wrap_without_splitting_columns() -> None:
    text = render_tab(tab(*[{1: f} for f in range(10, 16)]), BOARD, width=14)
    assert text == golden("""
        e|-10-11-12-|
        B|----------|
        G|----------|
        D|----------|
        A|----------|
        E|----------|

        e|-13-14-15-|
        B|----------|
        G|----------|
        D|----------|
        A|----------|
        E|----------|
    """)
    assert all(len(line) <= 14 for line in text.splitlines())


def test_unplayable_notes_are_listed_under_the_tab() -> None:
    lost = NoteEvent(1.234, 1.5, 36, 80)
    text = render_tab(Assignment(tab({1: 0}).notes, [lost]), BOARD)
    assert text.splitlines()[-1] == "# unplayable: C2 at 1.23 s"


def test_labels_follow_the_tuning() -> None:
    drop_d = Fretboard(tuning=(64, 59, 55, 50, 45, 38))
    assert string_labels(drop_d) == ["e", "B", "G", "D", "A", "D"]
    # Sharps widen the label column for every string.
    half_down = Fretboard(tuning=(63, 58, 54, 49, 44, 39))
    assert string_labels(half_down) == ["d#", "A#", "F#", "C#", "G#", "D#"]
    open_g = Fretboard(tuning=(62, 59, 55, 50, 43, 38))
    assert string_labels(open_g) == ["d", "B", "G", "D", "G", "D"]
    seven = Fretboard(tuning=(64, 59, 55, 50, 45, 40, 35))
    assert len(render_tab(Assignment([], []), seven).splitlines()) == 7


def test_empty_input() -> None:
    assert render_tab(Assignment([], []), BOARD) == golden("""
        e|-|
        B|-|
        G|-|
        D|-|
        A|-|
        E|-|
    """)
