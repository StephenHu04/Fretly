from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pretty_midi
import pytest

from pipeline.midi_parser import NoteEvent, parse_midi
from pipeline.transcribe import (
    GUITAR_MAX_HZ,
    GUITAR_MIN_HZ,
    TranscriptionError,
    TranscriptionSettings,
    transcribe,
)
from tests.conftest import WriteTone

# What Basic Pitch would return for a two-note E4 -> G4 phrase.
FAKE_EVENTS = [(0.0, 0.5, 64, 0.8, None), (0.5, 1.0, 67, 0.6, [0, 1])]


class FakePredict:
    def __init__(self, events: list[Any] = FAKE_EVENTS, error: Exception | None = None) -> None:
        self.events = events
        self.error = error
        self.message: str | None = None  # printed to stdout, like Basic Pitch
        self.kwargs: dict[str, Any] = {}

    def __call__(self, audio_path: Path, **kwargs: Any) -> tuple[Any, pretty_midi.PrettyMIDI, list[Any]]:
        self.kwargs = kwargs
        if self.message:
            print(self.message)
        if self.error:
            raise self.error
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=25)
        for start, end, pitch, amp, _ in self.events:
            inst.notes.append(pretty_midi.Note(round(127 * amp), pitch, start, end))
        pm.instruments.append(inst)
        return {}, pm, self.events


def test_transcribe_writes_midi_and_returns_notes(write_tone: WriteTone, tmp_path: Path) -> None:
    result = transcribe(write_tone(), tmp_path / "out" / "clip.mid", predict_fn=FakePredict())

    assert result.midi_path.is_file()
    assert [n.pitch for n in result.notes] == [64, 67]
    assert result.notes[0] == NoteEvent(0.0, 0.5, 64, 102)
    # The saved MIDI round-trips to the same pitches.
    assert [n.pitch for n in parse_midi(result.midi_path)] == [64, 67]


def test_default_settings_bound_to_guitar_range(write_tone: WriteTone, tmp_path: Path) -> None:
    fake = FakePredict()
    transcribe(write_tone(), tmp_path / "clip.mid", predict_fn=fake)
    assert fake.kwargs["minimum_frequency"] == GUITAR_MIN_HZ
    assert fake.kwargs["maximum_frequency"] == GUITAR_MAX_HZ
    # E2 and E6 (fret 24, high E) must sit inside the bounds.
    assert GUITAR_MIN_HZ < 82.41 and 1318.51 < GUITAR_MAX_HZ


def test_custom_settings_passed_through(write_tone: WriteTone, tmp_path: Path) -> None:
    fake = FakePredict()
    settings = TranscriptionSettings(onset_threshold=0.7, minimum_note_length_ms=50.0)
    transcribe(write_tone(), tmp_path / "clip.mid", settings=settings, predict_fn=fake)
    assert fake.kwargs["onset_threshold"] == 0.7
    assert fake.kwargs["minimum_note_length"] == 50.0


def test_silence_gives_empty_result(write_tone: WriteTone, tmp_path: Path) -> None:
    result = transcribe(write_tone(), tmp_path / "clip.mid", predict_fn=FakePredict(events=[]))
    assert result.notes == []
    assert parse_midi(result.midi_path) == []


def test_missing_audio(tmp_path: Path) -> None:
    with pytest.raises(TranscriptionError) as exc:
        transcribe(tmp_path / "nope.wav", tmp_path / "x.mid", predict_fn=FakePredict())
    assert exc.value.code == "file_not_found"


def test_inference_failure_is_wrapped(write_tone: WriteTone, tmp_path: Path) -> None:
    fake = FakePredict(error=RuntimeError("onnx exploded at C:/internal"))
    with pytest.raises(TranscriptionError) as exc:
        transcribe(write_tone(), tmp_path / "x.mid", predict_fn=fake)
    assert exc.value.code == "inference_failed"
    assert "internal" not in exc.value.message


def test_missing_model_reports_clearly(
    write_tone: WriteTone, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A None entry in sys.modules makes the import fail even when installed.
    monkeypatch.setitem(sys.modules, "basic_pitch.inference", None)
    with pytest.raises(TranscriptionError) as exc:
        transcribe(write_tone(), tmp_path / "x.mid")
    assert exc.value.code == "model_unavailable"
