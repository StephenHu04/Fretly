"""Normalize audio with FFmpeg into the format the transcription model expects.

Output is mono 16-bit PCM WAV at Basic Pitch's native rate (22,050 Hz, from
``basic_pitch.constants.AUDIO_SAMPLE_RATE``). Basic Pitch can resample on its
own, but converting up front gives one predictable format for every stage and
lets FFmpeg decode containers libsndfile cannot (e.g. M4A/AAC).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from pipeline.errors import PipelineError

TARGET_SAMPLE_RATE = 22050
TARGET_CHANNELS = 1
FFMPEG_ENV_VAR = "FRETLY_FFMPEG"


class PreprocessError(PipelineError):
    """FFmpeg is missing or failed."""


def find_ffmpeg() -> str | None:
    """Locate ffmpeg via $FRETLY_FFMPEG, falling back to PATH."""
    override = os.environ.get(FFMPEG_ENV_VAR)
    if override:
        return override if Path(override).is_file() else None
    return shutil.which("ffmpeg")


def build_ffmpeg_command(
    ffmpeg: str,
    input_path: Path,
    output_path: Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS,
) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner", "-loglevel", "error", "-nostdin",
        "-y",
        "-i", str(input_path),
        "-vn",  # drop cover art / video streams
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(output_path),
    ]


def preprocess_audio(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
    timeout_s: float = 120.0,
) -> Path:
    """Convert ``input_path`` to mono PCM WAV at ``sample_rate``.

    Returns the output path. Raises PreprocessError on failure.
    """
    src, dst = Path(input_path), Path(output_path)
    if not src.is_file():
        raise PreprocessError("file_not_found", "Audio file was not found.")

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise PreprocessError(
            "ffmpeg_missing",
            f"FFmpeg is not installed (or ${FFMPEG_ENV_VAR} is not a valid path).",
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_ffmpeg_command(ffmpeg, src, dst, sample_rate=sample_rate)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise PreprocessError("ffmpeg_timeout", "Audio conversion timed out.") from exc

    if result.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
        # FFmpeg's stderr can contain local paths, so keep it out of the message.
        raise PreprocessError("ffmpeg_failed", "Audio conversion failed.")
    return dst
