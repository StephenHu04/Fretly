"""Run the audio-to-notes stages in order: validate -> preprocess -> transcribe."""

from __future__ import annotations

import time
from pathlib import Path

from pipeline.audio_validation import AudioInfo, validate_audio
from pipeline.preprocess import preprocess_audio
from pipeline.transcribe import TranscriptionResult, TranscriptionSettings, transcribe


def transcribe_file(
    src: Path,
    out_dir: Path,
    settings: TranscriptionSettings | None = None,
    skip_preprocess: bool = False,
) -> tuple[AudioInfo, TranscriptionResult, dict[str, float]]:
    """Validate, preprocess and transcribe ``src``; also return stage timings.

    Raises PipelineError from whichever stage fails.
    """
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    info = validate_audio(src)
    timings["validate_s"] = time.perf_counter() - t0

    model_input = src
    if not skip_preprocess:
        t0 = time.perf_counter()
        model_input = preprocess_audio(src, out_dir / f"{src.stem}.22k.wav")
        timings["preprocess_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = transcribe(model_input, out_dir / f"{src.stem}.mid", settings)
    timings["transcribe_s"] = time.perf_counter() - t0
    return info, result, timings
