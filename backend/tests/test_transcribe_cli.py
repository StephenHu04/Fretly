from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pipeline import transcribe as transcribe_module
from scripts.transcribe_audio import main
from tests.conftest import WriteTone
from tests.test_transcribe import FakePredict


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> FakePredict:
    fake = FakePredict()
    monkeypatch.setattr(transcribe_module, "_load_predict", lambda: fake)
    return fake


def test_cli_json_end_to_end_with_fake_model(
    write_tone: WriteTone, tmp_path: Path, fake_model: FakePredict, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "out"
    code = main([str(write_tone(name="riff.wav")), "--out-dir", str(out_dir), "--skip-preprocess", "--json"])

    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert [n["name"] for n in report["notes"]] == ["E4", "G4"]
    assert (out_dir / "riff.mid").is_file()
    assert "transcribe_s" in report["timings"]


def test_cli_table_output(
    write_tone: WriteTone, tmp_path: Path, fake_model: FakePredict, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(write_tone()), "--out-dir", str(tmp_path), "--skip-preprocess"]) == 0
    out = capsys.readouterr().out
    assert "Notes: 2" in out
    assert "E4" in out and "G4" in out


def test_cli_thresholds_reach_model(
    write_tone: WriteTone, tmp_path: Path, fake_model: FakePredict
) -> None:
    main([str(write_tone()), "--out-dir", str(tmp_path), "--skip-preprocess", "--onset-threshold", "0.65"])
    assert fake_model.kwargs["onset_threshold"] == 0.65


def test_cli_validation_error_exit_code(
    tmp_path: Path, fake_model: FakePredict, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(tmp_path / "missing.wav"), "--out-dir", str(tmp_path)]) == 1
    assert "file_not_found" in capsys.readouterr().err


# --- Real-model test: opt-in with `pytest -m model` ---------------------------


@pytest.mark.model
@pytest.mark.skipif(importlib.util.find_spec("basic_pitch") is None, reason="basic-pitch not installed")
def test_real_basic_pitch_detects_sequence(tmp_path: Path) -> None:
    """Synthesize a plucked-like A2 -> E3 -> A3 sequence and check detection.

    This checks that the real model runs and finds the right pitches on clean
    synthetic input. It is not an accuracy benchmark for real guitar.
    """
    from pipeline.transcribe import transcribe

    sr = 22050
    pitches = [45, 52, 57]
    note_s = 0.6
    t = np.arange(int(sr * note_s)) / sr
    env = np.exp(-3.0 * t)
    audio = np.concatenate([
        env * sum((0.5 / k) * np.sin(2 * np.pi * k * 440.0 * 2 ** ((p - 69) / 12) * t) for k in (1, 2, 3))
        for p in pitches
    ])
    wav = tmp_path / "seq.wav"
    sf.write(str(wav), 0.5 * audio / np.abs(audio).max(), sr)

    result = transcribe(wav, tmp_path / "seq.mid")
    detected = {n.pitch for n in result.notes}
    assert set(pitches) <= detected
