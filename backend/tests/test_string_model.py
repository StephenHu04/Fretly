from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.midi_parser import NoteEvent
from scripts.calibrate import tuning_warnings
from tab.alternatives import alternatives
from tab.fretboard import Fretboard, Position
from tab.optimizer import MIN_STRING_PROBABILITY, OptimizerSettings, assign_optimized, sound_cost
from tab.string_model import (
    FEATURE_COUNT,
    SAMPLE_RATE,
    Profile,
    StringModel,
    calibration_notes,
    cents_off,
    fit_profile,
    note_features,
)

BOARD = Fretboard(max_fret=19)
NOTE_S = 0.6


def pluck(pitch: int, string: int, cents: float = 0.0, seed: int = 0) -> np.ndarray:
    """A plucked-string-like tone: thicker strings lose their upper harmonics faster."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(NOTE_S * SAMPLE_RATE)) / SAMPLE_RATE
    f0 = 440 * 2 ** ((pitch - 69 + cents / 100) / 12)
    rolloff = 1.5 + 2.5 * string  # dB per harmonic
    tone = np.zeros_like(t)
    for k in range(1, 16):
        if k * f0 < SAMPLE_RATE / 2:
            gain = 10 ** ((-rolloff * (k - 1) + rng.normal(0, 1.0)) / 20)
            tone += gain * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 6.28))
    return 0.3 * tone * np.exp(-3 * t) / np.abs(tone).max()


def recording(notes: list[tuple[int, int]], cents: float = 0.0) -> tuple[np.ndarray, list[NoteEvent]]:
    """Notes (pitch, string) one after another, with a little silence before each."""
    gap = int(0.1 * SAMPLE_RATE)
    parts, events = [], []
    for i, (pitch, string) in enumerate(notes):
        start = (i * (NOTE_S + 0.1)) + 0.1
        parts += [np.zeros(gap), pluck(pitch, string, cents, seed=i)]
        events.append(NoteEvent(start, start + NOTE_S, pitch, 80))
    return np.concatenate(parts).astype(np.float32), events


def calibration_set(frets: int = 12) -> tuple[np.ndarray, list[int], list[int]]:
    feats, strings, fret_list = [], [], []
    for string in range(1, 7):
        pairs = [(BOARD.open_pitch(string) + f, string) for f in range(frets + 1)]
        samples, events = recording(pairs)
        for f, e in enumerate(events):
            feats.append(note_features(samples, e))
            strings.append(string)
            fret_list.append(f)
    return np.array(feats), strings, fret_list


@pytest.fixture(scope="module")
def fitted() -> tuple[StringModel, object]:
    feats, strings, frets = calibration_set()
    return fit_profile(feats, strings, frets, BOARD)


def test_features_have_fixed_size_and_mark_short_notes() -> None:
    samples, events = recording([(57, 3)])
    f = note_features(samples, events[0])
    assert f.shape == (FEATURE_COUNT,) and f[0] == 57 and np.isfinite(f).all()
    short = note_features(samples, NoteEvent(0.0, 0.01, 57, 80))
    assert short[0] == 57 and np.isnan(short[1:]).all()


def test_features_ignore_how_loud_the_note_is() -> None:
    samples, events = recording([(57, 3)])
    assert np.allclose(note_features(samples, events[0])[1:17], note_features(samples * 0.2, events[0])[1:17], atol=0.01)


def test_tells_strings_apart_on_frets_it_did_not_see(fitted) -> None:
    _, check = fitted
    assert check.accuracy > 0.8 > check.chance
    assert 0 < check.temperature <= 1


def test_probabilities_cover_only_strings_that_can_play_the_pitch(fitted) -> None:
    model, _ = fitted
    samples, events = recording([(64, 1), (64, 5)])  # E4: open e, or A string fret 19
    p = model.probabilities(np.array([note_features(samples, e) for e in events]), BOARD)
    legal = {pos.string for pos in BOARD.candidates(64)}
    for row in p:
        assert row.sum() == pytest.approx(1.0)
        assert all(row[s - 1] == 0 for s in range(1, 7) if s not in legal)
    assert p[0].argmax() == 0  # bright, so the e string
    assert p[1].argmax() == 4  # dull, so the A string


def test_profile_round_trips_through_json(fitted, tmp_path: Path) -> None:
    model, _ = fitted
    path = tmp_path / "p" / "profile.json"
    Profile(model, BOARD.tuning, {"notes": 78}).save(path)
    loaded = Profile.load(path)
    samples, events = recording([(59, 2), (59, 4), (40, 6)])
    assert loaded.info["notes"] == 78
    assert loaded.string_probabilities(samples, events, BOARD) == Profile(model, BOARD.tuning, {}).string_probabilities(samples, events, BOARD)


def test_profile_rejects_another_tuning_and_other_files(fitted, tmp_path: Path) -> None:
    model, _ = fitted
    samples, events = recording([(59, 2)])
    drop_d = Fretboard(tuning=(64, 59, 55, 50, 45, 38))
    with pytest.raises(ValueError, match="tuning"):
        Profile(model, BOARD.tuning, {}).string_probabilities(samples, events, drop_d)
    bad = tmp_path / "x.json"
    bad.write_text('{"version": 99}', encoding="utf-8")
    with pytest.raises(ValueError):
        Profile.load(bad)


def test_unplayable_notes_get_no_probabilities(fitted) -> None:
    model, _ = fitted
    samples, events = recording([(30, 6), (59, 2)])
    probs = Profile(model, BOARD.tuning, {}).string_probabilities(samples, events, BOARD)
    assert list(probs) == [events[1]]
    assert set(probs[events[1]]) == {p.string for p in BOARD.candidates(59)}


def test_fit_needs_notes_on_every_string() -> None:
    feats, strings, _ = calibration_set(frets=1)
    with pytest.raises(ValueError, match="string"):
        StringModel.fit(feats, strings)


def test_cents_off_measures_detuning() -> None:
    for cents in (0, 30, -25):
        samples, events = recording([(52, 4)], cents=cents)
        assert cents_off(samples, events[0]) == pytest.approx(cents, abs=3)


def test_calibration_notes_take_the_loudest_at_each_fret() -> None:
    notes = [NoteEvent(0, 1, 40, 50), NoteEvent(1, 2, 40, 90), NoteEvent(2, 3, 41, 70), NoteEvent(3, 4, 52, 60)]
    found, missing = calibration_notes(notes, 40, 3)
    assert found == {0: notes[1], 1: notes[2]}
    assert missing == [2, 3]


def test_tuning_warnings() -> None:
    assert tuning_warnings("lowE", {0: 5, 1: 20, 2: 30}) == []
    warn = tuning_warnings("lowE", {0: 39, 1: 60, 2: 64})
    assert len(warn) == 2 and "tune it" in warn[0] and "next note" in warn[1]


# --- the optimizer with sound


RIFF = [NoteEvent(i * 0.3, (i + 1) * 0.3, p, 80) for i, p in enumerate([60, 62, 64, 65, 67])]


def test_sound_can_move_notes_to_the_string_it_heard() -> None:
    plain = assign_optimized(RIFF, BOARD)
    target = {n: {p.string: (0.99 if p.string == 4 else 0.01) for p in BOARD.candidates(n.pitch)} for n in RIFF}
    heard = assign_optimized(RIFF, BOARD, sound=target)
    assert any(t.position.string != 4 for t in plain.notes)
    assert all(t.position.string == 4 for t in heard.notes)


def test_no_sound_or_zero_weight_changes_nothing() -> None:
    plain = assign_optimized(RIFF, BOARD)
    target = {n: {p.string: (0.99 if p.string == 4 else 0.01) for p in BOARD.candidates(n.pitch)} for n in RIFF}
    assert assign_optimized(RIFF, BOARD, sound={}) == plain
    assert assign_optimized(RIFF, BOARD, OptimizerSettings(sound_weight=0), sound=target) == plain


def test_sound_cost_has_a_floor() -> None:
    note = RIFF[0]
    cost = sound_cost([note], [Position(2, 1)], {note: {2: 0.0}})
    assert cost == pytest.approx(-np.log(MIN_STRING_PROBABILITY))


def test_pins_beat_the_sound_and_alternatives_use_it() -> None:
    target = {n: {p.string: (0.99 if p.string == 4 else 0.01) for p in BOARD.candidates(n.pitch)} for n in RIFF}
    pinned = assign_optimized(RIFF, BOARD, pins={0: Position(3, 5)}, sound=target)
    assert pinned.notes[0].position == Position(3, 5)
    first = alternatives(RIFF, BOARD, 2, sound=target)[0]
    assert first == assign_optimized(RIFF, BOARD, sound=target)
