from __future__ import annotations

import itertools
import random

import pytest

from pipeline.midi_parser import NoteEvent
from tab.assign import assign_lowest_fret
from tab.fretboard import Fretboard, Position
from tab.metrics import measure
from tab.optimizer import OptimizerSettings, assign_optimized, sequence_cost

BOARD = Fretboard()


def melody(pitches: list[int], step_s: float = 0.25) -> list[NoteEvent]:
    return [NoteEvent(i * step_s, (i + 1) * step_s, p, 100) for i, p in enumerate(pitches)]


def positions(pitches: list[int], settings: OptimizerSettings | None = None) -> list[Position]:
    return [t.position for t in assign_optimized(melody(pitches), BOARD, settings).notes]


def test_every_position_sounds_the_note() -> None:
    notes = melody(list(range(40, 89)))
    result = assign_optimized(notes, BOARD)
    assert result.unplayable == []
    assert [t.note for t in result.notes] == notes
    assert all(BOARD.pitch_at(t.position) == t.note.pitch for t in result.notes)


@pytest.mark.parametrize("seed", range(20))
def test_matches_brute_force_minimum(seed: int) -> None:
    rng = random.Random(seed)
    pitches = [rng.randint(52, 76) for _ in range(5)]  # mid-neck: 3-5 candidates each
    best = min(
        sequence_cost(combo) for combo in itertools.product(*(BOARD.candidates(p) for p in pitches))
    )
    assert sequence_cost(positions(pitches)) == pytest.approx(best)


@pytest.mark.parametrize("seed", range(10))
def test_never_costs_more_than_the_baseline(seed: int) -> None:
    rng = random.Random(seed)
    notes = melody([rng.randint(40, 88) for _ in range(40)])
    optimized = [t.position for t in assign_optimized(notes, BOARD).notes]
    baseline = [t.position for t in assign_lowest_fret(notes, BOARD).notes]
    assert sequence_cost(optimized) <= sequence_cost(baseline) + 1e-9


def test_high_run_stays_in_one_position() -> None:
    # The baseline slides up the high E string (3 shifts); one box covers it.
    notes = melody([69, 72, 74, 76, 79, 81, 84])
    assert measure(assign_lowest_fret(notes, BOARD)).position_shifts == 3
    assert measure(assign_optimized(notes, BOARD)).position_shifts == 0


def test_repeated_notes_keep_their_position() -> None:
    path = positions([69, 72, 74, 74, 74, 72, 69])
    assert path[2] == path[3] == path[4]
    assert measure(assign_optimized(melody([69, 72, 74, 74, 74, 72, 69]), BOARD)).position_shifts == 0


def test_large_jump_takes_a_single_shift() -> None:
    notes = melody([40, 43, 45, 76, 79, 81])
    assert measure(assign_optimized(notes, BOARD)).position_shifts == 1


def test_open_strings_do_not_let_the_hand_jump_for_free() -> None:
    # F#2 (fret 2, low E) -> open B -> F#4. Played at fret 2 on the high E, the
    # hand doesn't move; the cost has to see past the open string to know that.
    path = positions([42, 59, 66])
    assert path == [Position(6, 2), Position(2, 0), Position(1, 2)]
    far = [Position(6, 2), Position(2, 0), Position(2, 7)]
    assert sequence_cost(far) - sequence_cost(path) >= OptimizerSettings().move_weight * 5


def test_single_position_notes_are_forced() -> None:
    assert positions([40, 41, 42]) == [Position(6, 0), Position(6, 1), Position(6, 2)]


def test_unplayable_notes_are_reported_and_skipped() -> None:
    notes = melody([69, 30, 72, 100])
    result = assign_optimized(notes, BOARD)
    assert [t.note.pitch for t in result.notes] == [69, 72]
    assert result.unplayable == [notes[1], notes[3]]


def test_deterministic() -> None:
    pitches = [64, 67, 69, 71, 74, 76, 79]
    assert positions(pitches) == positions(pitches)


def test_with_only_the_low_fret_preference_it_matches_the_baseline() -> None:
    notes = melody([69, 72, 74, 76, 79, 81, 84])
    height_only = OptimizerSettings(move_weight=0, shift_weight=0, string_weight=0, height_weight=1)
    baseline = [t.position for t in assign_lowest_fret(notes, BOARD).notes]
    assert positions([n.pitch for n in notes], settings=height_only) == baseline
    assert positions([n.pitch for n in notes]) != baseline


def test_respects_the_fretboard_configuration() -> None:
    short = Fretboard(max_fret=5)
    result = assign_optimized(melody([64, 70]), short)
    assert all(t.position.fret <= 5 for t in result.notes)
    assert [n.pitch for n in result.unplayable] == [70]


def test_empty_and_all_unplayable_input() -> None:
    assert assign_optimized([], BOARD).notes == []
    notes = melody([10, 20])
    result = assign_optimized(notes, BOARD)
    assert result.notes == [] and result.unplayable == notes


# --- Open strings and hand position (decision 012) -----------------------------


def test_open_strings_are_used_in_open_position() -> None:
    # no_chord1: the hand stays within the first 4 frets, so E4 and B3 go open.
    path = positions([64, 63, 59, 64, 63, 59])
    assert path == [Position(1, 0), Position(2, 4), Position(2, 0)] * 2


def test_mid_neck_phrase_matches_the_player_more_closely() -> None:
    # no_chord5: starts on B4 (fret 7 on e, its lowest spot) and stays around
    # frets 2-7. Free open strings made the optimizer jump to the open e for
    # E4; the player fretted it at B fret 5.
    phrase = [71, 64, 61, 61, 63, 64, 66, 59, 59, 68, 66, 68, 66, 66, 64, 61]
    played = [(1, 7), (2, 5), (3, 6), (3, 6), (2, 4), (2, 5), (2, 7), (3, 4), (3, 4),
              (1, 4), (1, 2), (1, 4), (1, 2), (1, 2), (1, 0), (2, 2)]  # fmt: skip
    played_positions = [Position(*p) for p in played]

    def matches(settings: OptimizerSettings | None) -> int:
        return sum(a == b for a, b in zip(positions(phrase, settings), played_positions, strict=True))

    assert positions(phrase)[1] == Position(2, 5)
    assert matches(None) == 10
    assert matches(OptimizerSettings(open_string_weight=0, height_weight=0.1)) == 5  # before decision 012


def test_open_string_cost_grows_with_hand_height() -> None:
    near, far = Position(4, 5), Position(3, 9)  # hand reaching fret 5 vs fret 9
    open_b = Position(2, 0)
    settings = OptimizerSettings(height_weight=0, string_weight=0)
    extra_near = sequence_cost([near, open_b], settings) - sequence_cost([near], settings)
    extra_far = sequence_cost([far, open_b], settings) - sequence_cost([far], settings)
    assert extra_near == pytest.approx(settings.open_string_weight * 1)
    assert extra_far == pytest.approx(settings.open_string_weight * 5)
