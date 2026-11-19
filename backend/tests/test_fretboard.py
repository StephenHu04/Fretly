from __future__ import annotations

import pytest

from tab.fretboard import STANDARD_TUNING, Fretboard, Position

BOARD = Fretboard()


def test_standard_tuning_matches_spec_table() -> None:
    # Strings 1..6: E4 B3 G3 D3 A2 E2.
    assert [BOARD.open_pitch(s) for s in range(1, 7)] == [64, 59, 55, 50, 45, 40]
    assert BOARD.num_strings == 6
    assert (BOARD.lowest_pitch, BOARD.highest_pitch) == (40, 88)


def test_open_strings_include_fret_zero() -> None:
    for string, pitch in enumerate(STANDARD_TUNING, start=1):
        assert Position(string, 0) in BOARD.candidates(pitch)
    assert Position(6, 0).is_open and not Position(6, 1).is_open


@pytest.mark.parametrize("pitch", [40, 41, 44])  # E2, F2, G#2
def test_low_notes_have_exactly_one_position(pitch: int) -> None:
    assert BOARD.candidates(pitch) == [Position(6, pitch - 40)]


def test_e4_has_six_positions_lowest_fret_first() -> None:
    assert BOARD.candidates(64) == [
        Position(1, 0),
        Position(2, 5),
        Position(3, 9),
        Position(4, 14),
        Position(5, 19),
        Position(6, 24),
    ]


def test_top_of_range_is_fret_24_on_high_e() -> None:
    assert BOARD.candidates(88) == [Position(1, 24)]


@pytest.mark.parametrize("pitch", [0, 39, 89, 127])
def test_out_of_range_pitches_have_no_candidates(pitch: int) -> None:
    assert BOARD.candidates(pitch) == []
    assert not BOARD.is_playable(pitch)


def test_every_candidate_sounds_the_requested_pitch() -> None:
    for pitch in range(BOARD.lowest_pitch, BOARD.highest_pitch + 1):
        candidates = BOARD.candidates(pitch)
        assert candidates, pitch
        assert all(BOARD.pitch_at(p) == pitch for p in candidates)
        assert len({p.string for p in candidates}) == len(candidates)


def test_max_fret_is_configurable() -> None:
    board = Fretboard(max_fret=12)
    assert board.highest_pitch == 76
    assert board.candidates(64) == [Position(1, 0), Position(2, 5), Position(3, 9)]
    assert board.candidates(77) == []


def test_custom_tuning() -> None:
    drop_d = Fretboard(tuning=(64, 59, 55, 50, 45, 38))
    assert drop_d.candidates(38) == [Position(6, 0)]
    assert drop_d.lowest_pitch == 38


@pytest.mark.parametrize(
    "kwargs",
    [{"tuning": ()}, {"max_fret": -1}],
)
def test_invalid_fretboards_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Fretboard(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("position", [Position(0, 0), Position(7, 0), Position(1, -1), Position(1, 25)])
def test_pitch_at_rejects_positions_off_the_board(position: Position) -> None:
    with pytest.raises(ValueError):
        BOARD.pitch_at(position)
