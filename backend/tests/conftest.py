from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

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
