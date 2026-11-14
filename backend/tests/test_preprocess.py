from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pipeline import preprocess
from pipeline.audio_validation import inspect_audio
from pipeline.preprocess import (
    FFMPEG_ENV_VAR,
    TARGET_SAMPLE_RATE,
    PreprocessError,
    build_ffmpeg_command,
    find_ffmpeg,
    preprocess_audio,
)
from tests.conftest import WriteTone


def _fake_run(returncode: int = 0, write_output: bool = True):
    """Stand-in for subprocess.run that mimics ffmpeg writing its output file."""
    calls: list[list[str]] = []

    def run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if write_output:
            Path(cmd[-1]).write_bytes(b"RIFF fake wav")
        return subprocess.CompletedProcess(cmd, returncode, "", "error at C:/secret/path")

    run.calls = calls  # type: ignore[attr-defined]
    return run


@pytest.fixture
def fake_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    exe = tmp_path / "ffmpeg.exe"
    exe.write_bytes(b"")
    monkeypatch.setenv(FFMPEG_ENV_VAR, str(exe))
    return exe


def test_command_targets_mono_22050_pcm() -> None:
    cmd = build_ffmpeg_command("ffmpeg", Path("in.m4a"), Path("out.wav"))
    assert cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[cmd.index("-ar") + 1] == str(TARGET_SAMPLE_RATE) == "22050"
    assert cmd[cmd.index("-c:a") + 1] == "pcm_s16le"
    assert cmd[cmd.index("-i") + 1] == "in.m4a"
    assert cmd[-1] == "out.wav"
    assert "-nostdin" in cmd


def test_env_override_must_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FFMPEG_ENV_VAR, str(tmp_path / "missing.exe"))
    assert find_ffmpeg() is None


def test_missing_input(tmp_path: Path, fake_ffmpeg: Path) -> None:
    with pytest.raises(PreprocessError) as exc:
        preprocess_audio(tmp_path / "nope.wav", tmp_path / "out.wav")
    assert exc.value.code == "file_not_found"


def test_missing_ffmpeg(
    write_tone: WriteTone, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(FFMPEG_ENV_VAR, raising=False)
    monkeypatch.setattr(preprocess.shutil, "which", lambda _: None)
    with pytest.raises(PreprocessError) as exc:
        preprocess_audio(write_tone(), tmp_path / "out.wav")
    assert exc.value.code == "ffmpeg_missing"


def test_success_creates_output_dir(
    write_tone: WriteTone, tmp_path: Path, fake_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _fake_run()
    monkeypatch.setattr(preprocess.subprocess, "run", run)
    out = preprocess_audio(write_tone(), tmp_path / "nested" / "out.wav")
    assert out.is_file()
    assert run.calls[0][0] == str(fake_ffmpeg)  # type: ignore[attr-defined]


def test_nonzero_exit_hides_stderr(
    write_tone: WriteTone, tmp_path: Path, fake_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preprocess.subprocess, "run", _fake_run(returncode=1))
    with pytest.raises(PreprocessError) as exc:
        preprocess_audio(write_tone(), tmp_path / "out.wav")
    assert exc.value.code == "ffmpeg_failed"
    assert "secret" not in exc.value.message


def test_missing_output_is_failure(
    write_tone: WriteTone, tmp_path: Path, fake_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preprocess.subprocess, "run", _fake_run(write_output=False))
    with pytest.raises(PreprocessError) as exc:
        preprocess_audio(write_tone(), tmp_path / "out.wav")
    assert exc.value.code == "ffmpeg_failed"


def test_timeout(
    write_tone: WriteTone, tmp_path: Path, fake_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(cmd: list[str], **_: object) -> None:
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(preprocess.subprocess, "run", run)
    with pytest.raises(PreprocessError) as exc:
        preprocess_audio(write_tone(), tmp_path / "out.wav")
    assert exc.value.code == "ffmpeg_timeout"


@pytest.mark.ffmpeg
@pytest.mark.skipif(find_ffmpeg() is None, reason="ffmpeg not on PATH")
def test_real_ffmpeg_converts_stereo_44k_to_mono_22k(
    write_tone: WriteTone, tmp_path: Path
) -> None:
    src = write_tone(duration_s=2.0, sample_rate=44100, channels=2)
    info = inspect_audio(preprocess_audio(src, tmp_path / "out.wav"))
    assert info.sample_rate == 22050
    assert info.channels == 1
    assert info.duration_s == pytest.approx(2.0, abs=0.01)
