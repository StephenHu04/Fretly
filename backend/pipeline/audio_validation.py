"""Inspect and validate local audio files before they enter the pipeline.

Decoding uses libsndfile (via the ``soundfile`` package), which handles WAV,
FLAC, OGG/Vorbis, AIFF and, with libsndfile >= 1.1, MP3. Containers such as
M4A/AAC need FFmpeg and are handled by the ffprobe helpers instead.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import soundfile as sf

from pipeline.errors import PipelineError

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}
)

# Limits sized for the MVP's short (~15-60 s) clips, with headroom.
DEFAULT_MAX_FILE_BYTES = 50 * 1024 * 1024
DEFAULT_MIN_DURATION_S = 0.5
DEFAULT_MAX_DURATION_S = 120.0


class AudioValidationError(PipelineError):
    """The file is missing, unreadable or outside the configured limits."""


@dataclass(frozen=True)
class AudioInfo:
    readable: bool
    duration_s: float
    sample_rate: int
    channels: int
    frames: int
    format: str
    subtype: str
    format_description: str
    file_size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationLimits:
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    min_duration_s: float = DEFAULT_MIN_DURATION_S
    max_duration_s: float = DEFAULT_MAX_DURATION_S
    allowed_extensions: frozenset[str] = field(default=SUPPORTED_EXTENSIONS)


def inspect_audio(path: str | Path) -> AudioInfo:
    """Read audio metadata and confirm the file actually decodes.

    Raises AudioValidationError if the file is missing or unreadable.
    """
    p = Path(path)
    if not p.is_file():
        raise AudioValidationError("file_not_found", "Audio file was not found.")

    size = p.stat().st_size
    if size == 0:
        raise AudioValidationError("empty_file", "Audio file is empty.")

    try:
        meta = sf.info(str(p))
        # Header parsing alone can succeed on truncated files, so decode a
        # small block to prove the audio data is readable too.
        with sf.SoundFile(str(p)) as f:
            f.read(frames=min(4096, max(f.frames, 1)))
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise AudioValidationError(
            "unreadable_audio", "Audio file could not be decoded."
        ) from exc

    if meta.samplerate <= 0 or meta.channels <= 0:
        raise AudioValidationError(
            "invalid_audio", "Audio file has an invalid sample rate or channel count."
        )

    return AudioInfo(
        readable=True,
        duration_s=meta.frames / meta.samplerate,
        sample_rate=meta.samplerate,
        channels=meta.channels,
        frames=meta.frames,
        format=meta.format,
        subtype=meta.subtype,
        format_description=meta.format_info,
        file_size_bytes=size,
    )


def validate_audio(
    path: str | Path, limits: ValidationLimits | None = None
) -> AudioInfo:
    """Inspect a file and enforce extension, size and duration limits."""
    limits = limits or ValidationLimits()
    p = Path(path)

    if p.suffix.lower() not in limits.allowed_extensions:
        allowed = ", ".join(sorted(limits.allowed_extensions))
        raise AudioValidationError(
            "unsupported_extension", f"Unsupported file type. Allowed: {allowed}."
        )

    if p.is_file() and p.stat().st_size > limits.max_file_bytes:
        mb = limits.max_file_bytes / (1024 * 1024)
        raise AudioValidationError(
            "file_too_large", f"Audio file exceeds the {mb:g} MB limit."
        )

    info = inspect_audio(p)

    if info.duration_s < limits.min_duration_s:
        raise AudioValidationError(
            "too_short",
            f"Recording is too short (minimum {limits.min_duration_s:g} s).",
        )
    if info.duration_s > limits.max_duration_s:
        raise AudioValidationError(
            "too_long",
            f"Recording is too long (maximum {limits.max_duration_s:g} s).",
        )
    return info


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def probe_with_ffprobe(path: str | Path, timeout_s: float = 30.0) -> dict[str, object]:
    """Return format/stream metadata from ffprobe for cross-checking.

    Raises AudioValidationError if ffprobe is missing or rejects the file.
    """
    exe = shutil.which("ffprobe")
    if exe is None:
        raise AudioValidationError("ffprobe_missing", "ffprobe is not installed.")

    cmd = [
        exe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", "-select_streams", "a:0",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioValidationError("ffprobe_timeout", "ffprobe timed out.") from exc

    if result.returncode != 0:
        raise AudioValidationError("ffprobe_failed", "ffprobe could not read the file.")

    data = json.loads(result.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise AudioValidationError("no_audio_stream", "File has no audio stream.")

    stream = streams[0]
    fmt = data.get("format", {})
    return {
        "format": fmt.get("format_name"),
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate", 0)),
        "channels": int(stream.get("channels", 0)),
        "duration_s": float(fmt.get("duration") or stream.get("duration") or 0.0),
    }
