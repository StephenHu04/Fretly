from __future__ import annotations

import pytest

from pipeline.midi_parser import NoteEvent
from tab.cleanup import (
    CleanupResult,
    GhostFilterSettings,
    RingingFilterSettings,
    ThumpFilterSettings,
    clean_notes,
    remove_low_thumps,
    remove_octave_ghosts,
    remove_ringing_open_strings,
)
from tab.fretboard import Fretboard


def note(start: float, pitch: int, velocity: int = 100, length: float = 0.4) -> NoteEvent:
    return NoteEvent(start, start + length, pitch, velocity)


def test_ghost_an_octave_up_at_half_velocity_is_removed() -> None:
    real, ghost = note(0.0, 40, 100), note(0.01, 52, 50)
    result = remove_octave_ghosts([real, ghost])
    assert result.kept == [real]
    assert result.removed == [ghost]


def test_loud_real_octave_is_kept() -> None:
    notes = [note(0.0, 40, 100), note(0.0, 52, 90)]
    assert remove_octave_ghosts(notes).kept == notes


def test_octave_with_a_later_onset_is_kept() -> None:
    notes = [note(0.0, 40, 100), note(0.2, 52, 40)]
    assert remove_octave_ghosts(notes).removed == []


@pytest.mark.parametrize("interval", [7, 11, 13, 24])  # fifth, near-octaves, two octaves
def test_only_exact_octaves_count(interval: int) -> None:
    notes = [note(0.0, 40, 100), note(0.0, 40 + interval, 30)]
    assert remove_octave_ghosts(notes).removed == []


def test_quiet_note_an_octave_below_a_loud_one_is_a_ghost() -> None:
    # no_chord4: a quiet G#2 under each loud G#3 (G string, fret 1).
    real, ghost = note(3.566, 56, 90), note(3.566, 44, 47)
    assert remove_octave_ghosts([real, ghost]).removed == [ghost]
    assert remove_octave_ghosts([real, ghost], GhostFilterSettings(below=False)).removed == []


def test_loud_note_is_never_the_ghost() -> None:
    # Whichever octave the quiet note is in, only the quiet one goes.
    for loud, quiet in ((40, 52), (52, 40)):
        notes = [note(0.0, loud, 100), note(0.0, quiet, 30)]
        assert remove_octave_ghosts(notes).kept == [notes[0]]


def test_stacked_ghosts_are_all_removed() -> None:
    real, ghost1, ghost2 = note(0.0, 40, 100), note(0.0, 52, 50), note(0.0, 64, 25)
    result = remove_octave_ghosts([ghost2, real, ghost1])
    assert result.kept == [real]
    assert result.removed == [ghost1, ghost2]


def test_phase_2_synthetic_phrase_loses_exactly_its_ghosts() -> None:
    # Shape of the Phase 2 smoke test: an 11-note E minor pentatonic run with
    # octave ghosts at about half velocity on the 7 lowest notes.
    run = [note(0.4 * i, p, 100) for i, p in enumerate([40, 43, 45, 47, 50, 52, 55, 57, 59, 62, 64])]
    ghosts = [note(n.start_s + 0.01, n.pitch + 12, 48) for n in run[:7]]
    result = remove_octave_ghosts(run + ghosts)
    assert result.kept == run
    assert sorted(result.removed) == sorted(ghosts)


def test_thresholds_are_configurable() -> None:
    notes = [note(0.0, 40, 100), note(0.08, 52, 70)]
    assert remove_octave_ghosts(notes).removed == []
    loose = GhostFilterSettings(max_onset_gap_s=0.1, max_velocity_ratio=0.75)
    assert remove_octave_ghosts(notes, loose).removed == [notes[1]]


def test_empty_input() -> None:
    result = remove_octave_ghosts([])
    assert result.kept == [] and result.removed == []


@pytest.mark.parametrize("kwargs", [{"max_onset_gap_s": -0.01}, {"max_velocity_ratio": 0}, {"max_velocity_ratio": 1.5}])
def test_invalid_settings_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        GhostFilterSettings(**kwargs)


# --- Ringing open strings ------------------------------------------------------

BOARD = Fretboard()


def test_open_string_ringing_after_a_finger_lifts_is_removed() -> None:
    # G#3 (G string, fret 1) ends; the open G rings on, quieter (no_chord4).
    fretted, ringing = NoteEvent(1.67, 1.92, 56, 93), NoteEvent(1.93, 2.17, 55, 56)
    result = remove_ringing_open_strings([fretted, ringing], BOARD)
    assert result.kept == [fretted]
    assert result.removed == [ringing]


def test_open_note_may_start_just_before_the_fretted_note_ends() -> None:
    # no_chord1: B3 starts 12 ms before E4 (B string, fret 5) ends.
    notes = [NoteEvent(3.81, 4.04, 64, 92), NoteEvent(4.03, 4.46, 59, 55)]
    assert len(remove_ringing_open_strings(notes, BOARD).removed) == 1


def test_loud_open_note_is_kept() -> None:
    # no_chord3: a real B3 right after E4, much louder than it.
    notes = [NoteEvent(4.90, 5.17, 64, 63), NoteEvent(5.22, 6.37, 59, 89)]
    assert remove_ringing_open_strings(notes, BOARD).removed == []


def test_open_note_played_later_is_kept() -> None:
    # no_chord1: D#4 (B fret 4) then B3 picked 116 ms later; a deliberate note.
    notes = [NoteEvent(1.27, 1.50, 63, 70), NoteEvent(1.61, 1.92, 59, 50)]
    assert remove_ringing_open_strings(notes, BOARD).removed == []


def test_previous_note_must_fit_on_that_string() -> None:
    # G#3 is below the open B, so it can't have been fretted on the B string.
    notes = [NoteEvent(1.67, 1.92, 56, 93), NoteEvent(1.93, 2.17, 59, 40)]
    assert remove_ringing_open_strings(notes, BOARD).removed == []


def test_only_open_string_pitches_are_candidates() -> None:
    notes = [NoteEvent(0.0, 0.5, 64, 100), NoteEvent(0.5, 1.0, 60, 40)]  # C4 isn't an open string
    assert remove_ringing_open_strings(notes, BOARD).removed == []


def test_fretted_note_must_start_first() -> None:
    notes = [NoteEvent(0.5, 0.52, 64, 100), NoteEvent(0.48, 1.0, 59, 40)]
    assert remove_ringing_open_strings(notes, BOARD).removed == []


def test_uses_the_tuning() -> None:
    # In drop D the low string is D2 (38); in standard tuning D2 isn't open.
    notes = [NoteEvent(0.0, 0.5, 41, 100), NoteEvent(0.51, 1.0, 38, 40)]
    drop_d = Fretboard(tuning=(64, 59, 55, 50, 45, 38))
    assert len(remove_ringing_open_strings(notes, drop_d).removed) == 1
    assert remove_ringing_open_strings(notes, BOARD).removed == []


def test_ringing_thresholds_are_configurable() -> None:
    notes = [NoteEvent(0.0, 0.5, 64, 100), NoteEvent(0.6, 1.0, 59, 40)]  # 100 ms gap
    assert remove_ringing_open_strings(notes, BOARD).removed == []
    loose = RingingFilterSettings(max_lag_s=0.15)
    assert len(remove_ringing_open_strings(notes, BOARD, loose).removed) == 1


@pytest.mark.parametrize("kwargs", [{"max_lead_s": -1}, {"max_lag_s": -1}, {"max_velocity_ratio": 0}])
def test_invalid_ringing_settings_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        RingingFilterSettings(**kwargs)


# --- Low thumps ---------------------------------------------------------------


def test_quiet_low_note_under_a_louder_note_is_removed() -> None:
    # neck1: a G2 thump while D5 (e fret 10) sounds.
    melody, thump = NoteEvent(4.62, 5.25, 74, 79), NoteEvent(4.62, 4.85, 43, 42)
    result = remove_low_thumps([melody, thump])
    assert result.kept == [melody]
    assert result.removed == [thump]


def test_thump_may_start_just_before_the_note() -> None:
    # no_chord6: G2 starts 70 ms before D#4.
    notes = [NoteEvent(2.18, 2.44, 43, 47), NoteEvent(2.25, 2.52, 63, 87)]
    assert len(remove_low_thumps(notes).removed) == 1


def test_thump_during_a_held_note_is_removed() -> None:
    # pent5: A2 0.6 s into a held C5.
    notes = [NoteEvent(7.97, 8.93, 72, 83), NoteEvent(8.60, 8.86, 45, 40)]
    assert len(remove_low_thumps(notes).removed) == 1


def test_low_note_after_the_louder_one_ends_is_kept() -> None:
    notes = [NoteEvent(0.0, 0.3, 72, 100), NoteEvent(0.35, 0.6, 45, 40)]
    assert remove_low_thumps(notes).removed == []


@pytest.mark.parametrize(
    "low",
    [
        NoteEvent(0.0, 0.3, 64, 40),  # less than an octave below
        NoteEvent(0.0, 0.3, 45, 90),  # nearly as loud
        NoteEvent(0.0, 0.6, 45, 40),  # held too long to be a thump
    ],
)
def test_real_looking_low_notes_are_kept(low: NoteEvent) -> None:
    assert remove_low_thumps([NoteEvent(0.0, 0.5, 72, 100), low]).removed == []


def test_lone_low_note_is_kept() -> None:
    # pent5: handling noise before playing starts has nothing above it.
    notes = [NoteEvent(1.51, 1.65, 43, 45)]
    assert remove_low_thumps(notes).kept == notes


def test_thump_settings_are_configurable_and_checked() -> None:
    notes = [NoteEvent(0.0, 0.5, 72, 100), NoteEvent(0.0, 0.3, 64, 40)]
    assert len(remove_low_thumps(notes, ThumpFilterSettings(min_interval=7)).removed) == 1
    for kwargs in ({"min_interval": 0}, {"max_lead_s": -1}, {"max_duration_s": 0}, {"max_velocity_ratio": 2}):
        with pytest.raises(ValueError):
            ThumpFilterSettings(**kwargs)


def test_clean_notes_runs_each_enabled_filter() -> None:
    real, ghost = NoteEvent(0.0, 0.5, 40, 100), NoteEvent(0.0, 0.5, 52, 45)
    fretted, ringing = NoteEvent(1.0, 1.5, 56, 90), NoteEvent(1.5, 2.0, 55, 50)
    melody, thump = NoteEvent(3.0, 3.5, 74, 90), NoteEvent(3.0, 3.2, 43, 40)
    notes = [real, ghost, fretted, ringing, melody, thump]

    result = clean_notes(notes, BOARD)
    assert result.kept == [real, fretted, melody]
    assert result.removed == {"octave_ghost": [ghost], "ringing_open_string": [ringing], "low_thump": [thump]}

    assert clean_notes(notes, BOARD, octave_ghosts=False, low_thumps=False).removed == {
        "ringing_open_string": [ringing]
    }
    assert clean_notes(notes, BOARD, False, False, False) == CleanupResult(sorted(notes), {})
