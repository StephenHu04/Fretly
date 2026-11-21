from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline import audio_validation
from pipeline.audio_validation import (
    AudioValidationError,
    ValidationLimits,
    ffprobe_available,
    inspect_audio,
    probe_with_ffprobe,
    validate_audio,
)
from tests.conftest import WriteTone


def test_inspect_reports_metadata(write_tone: WriteTone) -> None:
    path = write_tone(duration_s=2.0, sample_rate=22050, channels=2)
    info = inspect_audio(path)
    assert info.readable
    assert info.duration_s == pytest.approx(2.0)
    assert info.sample_rate == 22050
    assert info.channels == 2
    assert info.format == "WAV"
    assert info.file_size_bytes == path.stat().st_size


def test_inspect_flac(write_tone: WriteTone) -> None:
    info = inspect_audio(write_tone(name="tone.flac", duration_s=1.5))
    assert info.format == "FLAC"
    assert info.duration_s == pytest.approx(1.5)


def test_validate_accepts_typical_clip(write_tone: WriteTone) -> None:
    info = validate_audio(write_tone(duration_s=3.0))
    assert info.channels == 1


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AudioValidationError) as exc:
        validate_audio(tmp_path / "nope.wav")
    assert exc.value.code == "file_not_found"


def test_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")
    with pytest.raises(AudioValidationError) as exc:
        validate_audio(path)
    assert exc.value.code == "empty_file"


def test_garbage_bytes_are_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "fake.wav"
    path.write_bytes(b"this is not audio" * 100)
    with pytest.raises(AudioValidationError) as exc:
        validate_audio(path)
    assert exc.value.code == "unreadable_audio"


def test_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(AudioValidationError) as exc:
        validate_audio(path)
    assert exc.value.code == "unsupported_extension"


def test_extension_check_is_case_insensitive(write_tone: WriteTone) -> None:
    assert validate_audio(write_tone(name="TONE.WAV", fmt="WAV")).format == "WAV"


def test_too_short(write_tone: WriteTone) -> None:
    with pytest.raises(AudioValidationError) as exc:
        validate_audio(write_tone(duration_s=0.1))
    assert exc.value.code == "too_short"


def test_too_long(write_tone: WriteTone) -> None:
    limits = ValidationLimits(max_duration_s=1.0)
    with pytest.raises(AudioValidationError) as exc:
        validate_audio(write_tone(duration_s=2.0), limits)
    assert exc.value.code == "too_long"


def test_too_large(write_tone: WriteTone) -> None:
    limits = ValidationLimits(max_file_bytes=1000)
    with pytest.raises(AudioValidationError) as exc:
        validate_audio(write_tone(duration_s=1.0), limits)
    assert exc.value.code == "file_too_large"


def test_error_messages_do_not_leak_paths(tmp_path: Path) -> None:
    path = tmp_path / "secret_dir" / "fake.wav"
    path.parent.mkdir()
    path.write_bytes(b"garbage" * 100)
    with pytest.raises(AudioValidationError) as exc:
        validate_audio(path)
    assert str(tmp_path) not in exc.value.message
    assert "secret_dir" not in exc.value.message


def test_cli_json_output(write_tone: WriteTone, capsys: pytest.CaptureFixture[str]) -> None:
    from scripts.inspect_audio import main

    assert main([str(write_tone(duration_s=1.0)), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["readable"] is True
    assert report["sample_rate"] == 44100


def test_cli_reports_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from scripts.inspect_audio import main

    assert main([str(tmp_path / "missing.wav"), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["readable"] is False
    assert report["error_code"] == "file_not_found"


@pytest.mark.ffmpeg
@pytest.mark.skipif(not ffprobe_available(), reason="ffprobe not on PATH")
def test_ffprobe_agrees_with_soundfile(write_tone: WriteTone) -> None:
    path = write_tone(duration_s=2.0, sample_rate=48000, channels=2)
    probe = probe_with_ffprobe(path)
    assert probe["sample_rate"] == 48000
    assert probe["channels"] == 2
    assert probe["duration_s"] == pytest.approx(2.0, abs=0.01)


# --- M4A/AAC (phone recordings), inspected with ffprobe -----------------------


@pytest.mark.ffmpeg
@pytest.mark.skipif(not ffprobe_available(), reason="ffprobe not on PATH")
def test_m4a_is_validated(write_m4a: WriteTone) -> None:
    info = validate_audio(write_m4a(duration_s=2.0, sample_rate=44100, channels=2))
    assert info.readable
    assert info.duration_s == pytest.approx(2.0, abs=0.05)  # AAC adds a little padding
    assert (info.sample_rate, info.channels) == (44100, 2)
    assert (info.format, info.subtype) == ("M4A", "AAC")


@pytest.mark.ffmpeg
@pytest.mark.skipif(not ffprobe_available(), reason="ffprobe not on PATH")
def test_m4a_duration_limits_apply(write_m4a: WriteTone) -> None:
    with pytest.raises(AudioValidationError) as exc:
        validate_audio(write_m4a(duration_s=2.0), ValidationLimits(max_duration_s=1.0))
    assert exc.value.code == "too_long"


@pytest.mark.ffmpeg
@pytest.mark.skipif(not ffprobe_available(), reason="ffprobe not on PATH")
def test_corrupt_m4a_is_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "broken.m4a"
    path.write_bytes(b"not really audio" * 100)
    with pytest.raises(AudioValidationError) as exc:
        validate_audio(path)
    assert exc.value.code == "unreadable_audio"
    assert str(tmp_path) not in exc.value.message


def test_m4a_without_ffprobe_says_ffmpeg_is_needed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "phone.m4a"
    path.write_bytes(b"\x00" * 100)
    monkeypatch.setattr(audio_validation.shutil, "which", lambda _: None)
    with pytest.raises(AudioValidationError) as exc:
        validate_audio(path)
    assert exc.value.code == "ffprobe_missing"
    assert "M4A" in exc.value.message
