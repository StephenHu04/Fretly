from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pipeline.preprocess import find_ffmpeg

WriteTone = Callable[..., Path]


@pytest.fixture
def write_tone(tmp_path: Path) -> WriteTone:
    """Write a deterministic sine tone (A3 = 220 Hz) and return its path."""

    def _write(
        name: str = "tone.wav",
        duration_s: float = 1.0,
        sample_rate: int = 44100,
        channels: int = 1,
        fmt: str | None = None,
    ) -> Path:
        t = np.arange(int(duration_s * sample_rate)) / sample_rate
        mono = 0.3 * np.sin(2 * np.pi * 220.0 * t)
        data = np.column_stack([mono] * channels) if channels > 1 else mono
        path = tmp_path / name
        sf.write(str(path), data, sample_rate, format=fmt)
        return path

    return _write


@pytest.fixture
def write_m4a(write_tone: WriteTone, tmp_path: Path) -> WriteTone:
    """Encode a tone as AAC in an M4A container, like a phone recording."""
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")

    def _write(name: str = "phone.m4a", **tone_kwargs: object) -> Path:
        wav = write_tone(name=f"{Path(name).stem}-src.wav", **tone_kwargs)
        out = tmp_path / name
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), "-c:a", "aac", str(out)],
            check=True,
        )
        return out

    return _write
