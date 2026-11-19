from __future__ import annotations

import pytest

from pipeline.midi_parser import NoteEvent
from tab.cleanup import GhostFilterSettings, remove_octave_ghosts


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


def test_quiet_note_below_a_loud_one_is_not_a_ghost() -> None:
    # The ghost is always the upper note.
    notes = [note(0.0, 40, 30), note(0.0, 52, 100)]
    assert remove_octave_ghosts(notes).removed == []


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
