from __future__ import annotations

from pathlib import Path

import pretty_midi
import pytest

from pipeline.midi_parser import (
    MidiParseError,
    NoteEvent,
    midi_to_note_name,
    notes_from_basic_pitch,
    notes_from_pretty_midi,
    parse_midi,
)


def _midi(notes: list[tuple[float, float, int, int]], is_drum: bool = False) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=25, is_drum=is_drum)  # steel-string guitar
    for start, end, pitch, velocity in notes:
        inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))
    pm.instruments.append(inst)
    return pm


@pytest.mark.parametrize(
    ("pitch", "name"),
    [(40, "E2"), (45, "A2"), (50, "D3"), (55, "G3"), (59, "B3"), (64, "E4"), (60, "C4"), (61, "C#4")],
)
def test_note_names_match_standard_tuning(pitch: int, name: str) -> None:
    assert midi_to_note_name(pitch) == name


def test_parse_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "melody.mid"
    _midi([(0.0, 0.5, 64, 90), (0.5, 1.0, 67, 80)]).write(str(path))

    notes = parse_midi(path)

    assert [n.pitch for n in notes] == [64, 67]
    assert notes[0].start_s == pytest.approx(0.0)
    assert notes[1].end_s == pytest.approx(1.0, abs=1e-3)
    assert notes[0].velocity == 90
    assert notes[0].name == "E4"


def test_notes_sorted_by_onset_then_pitch() -> None:
    pm = _midi([(1.0, 1.5, 60, 80), (0.0, 0.5, 67, 80), (0.0, 0.5, 64, 80)])
    notes = notes_from_pretty_midi(pm)
    assert [(n.start_s, n.pitch) for n in notes] == [(0.0, 64), (0.0, 67), (1.0, 60)]


def test_drum_tracks_are_ignored() -> None:
    pm = _midi([(0.0, 0.5, 64, 90)])
    pm.instruments.append(_midi([(0.0, 0.1, 36, 100)], is_drum=True).instruments[0])
    assert [n.pitch for n in notes_from_pretty_midi(pm)] == [64]


def test_zero_length_notes_are_dropped() -> None:
    notes = notes_from_pretty_midi(_midi([(0.5, 0.5, 64, 90), (1.0, 1.2, 65, 90)]))
    assert [n.pitch for n in notes] == [65]


def test_empty_midi(tmp_path: Path) -> None:
    path = tmp_path / "empty.mid"
    pretty_midi.PrettyMIDI().write(str(path))
    assert parse_midi(path) == []


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(MidiParseError) as exc:
        parse_midi(tmp_path / "nope.mid")
    assert exc.value.code == "file_not_found"


def test_malformed_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.mid"
    path.write_bytes(b"not a midi file")
    with pytest.raises(MidiParseError) as exc:
        parse_midi(path)
    assert exc.value.code == "unreadable_midi"


def test_basic_pitch_events_conversion() -> None:
    events = [
        (0.5, 1.0, 67, 0.5, None),
        (0.0, 0.4, 64, 1.0, [0, 1]),
        (0.2, 0.2, 60, 0.8, None),  # zero length -> dropped
        (0.3, 0.6, 62, 0.0, None),  # zero amplitude -> clamped to velocity 1
    ]
    notes = notes_from_basic_pitch(events)
    assert notes == [
        NoteEvent(0.0, 0.4, 64, 127),
        NoteEvent(0.3, 0.6, 62, 1),
        NoteEvent(0.5, 1.0, 67, 64),
    ]


def test_to_dict_includes_name() -> None:
    assert NoteEvent(0.0, 1.0, 40, 100).to_dict() == {
        "start_s": 0.0, "end_s": 1.0, "pitch": 40, "velocity": 100, "name": "E2",
    }
