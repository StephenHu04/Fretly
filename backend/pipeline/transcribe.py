"""Audio-to-MIDI transcription with Spotify's Basic Pitch.

Basic Pitch is imported lazily so the rest of the pipeline (and the unit
tests) work without the model installed. Tests inject a fake ``predict``.
The call matches ``basic_pitch.inference.predict`` in basic-pitch 0.4.0:

    predict(audio_path, model_or_model_path, onset_threshold, frame_threshold,
            minimum_note_length, minimum_frequency, maximum_frequency,
            multiple_pitch_bends, melodia_trick, debug_file, midi_tempo)
        -> (model_output, pretty_midi.PrettyMIDI, note_events)
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pipeline.errors import PipelineError
from pipeline.midi_parser import NoteEvent, notes_from_basic_pitch

PredictFn = Callable[..., tuple[Any, Any, list[Any]]]

# Standard-tuning guitar spans E2 (82.4 Hz) to E6 (fret 24 on high E,
# 1318.5 Hz). Bounding the search a little outside that range removes
# sub-bass rumble and high harmonics the fretboard could never produce.
GUITAR_MIN_HZ = 75.0
GUITAR_MAX_HZ = 1400.0


class TranscriptionError(PipelineError):
    """The model is unavailable or inference failed."""


@dataclass(frozen=True)
class TranscriptionSettings:
    # Basic Pitch's own defaults, except the guitar frequency bounds.
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    minimum_note_length_ms: float = 127.70
    minimum_frequency_hz: float | None = GUITAR_MIN_HZ
    maximum_frequency_hz: float | None = GUITAR_MAX_HZ
    melodia_trick: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TranscriptionResult:
    notes: list[NoteEvent]
    midi_path: Path
    settings: TranscriptionSettings


def _load_predict() -> PredictFn:
    try:
        from basic_pitch.inference import predict
    except ImportError as exc:
        raise TranscriptionError(
            "model_unavailable",
            "Basic Pitch is not installed; see README for setup.",
        ) from exc
    return predict


def transcribe(
    audio_path: str | Path,
    midi_out: str | Path,
    settings: TranscriptionSettings | None = None,
    predict_fn: PredictFn | None = None,
) -> TranscriptionResult:
    """Run Basic Pitch on ``audio_path``, write MIDI to ``midi_out``."""
    settings = settings or TranscriptionSettings()
    src, dst = Path(audio_path), Path(midi_out)
    if not src.is_file():
        raise TranscriptionError("file_not_found", "Audio file was not found.")

    predict = predict_fn or _load_predict()
    try:
        # Basic Pitch prints "Predicting MIDI for <path>..." to stdout, which
        # would corrupt callers' own output (e.g. the CLI's --json).
        with contextlib.redirect_stdout(io.StringIO()):
            _, midi_data, note_events = predict(
                src,
                onset_threshold=settings.onset_threshold,
                frame_threshold=settings.frame_threshold,
                minimum_note_length=settings.minimum_note_length_ms,
                minimum_frequency=settings.minimum_frequency_hz,
                maximum_frequency=settings.maximum_frequency_hz,
                melodia_trick=settings.melodia_trick,
            )
    except TranscriptionError:
        raise
    except Exception as exc:  # model backends raise a wide variety of types
        raise TranscriptionError("inference_failed", "Transcription failed.") from exc

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        midi_data.write(str(dst))
    except Exception as exc:
        raise TranscriptionError("midi_write_failed", "Could not save MIDI output.") from exc

    return TranscriptionResult(
        notes=notes_from_basic_pitch(note_events),
        midi_path=dst,
        settings=settings,
    )
