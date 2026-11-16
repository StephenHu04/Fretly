"""Parse MIDI into plain note events for downstream cleanup and tab optimization.

Everything after transcription works with ``NoteEvent`` rather than a MIDI
library's types, so the optimizer stays independent of how notes were produced.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pretty_midi

from pipeline.errors import PipelineError

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


class MidiParseError(PipelineError):
    """The MIDI file is missing or cannot be read."""


@dataclass(frozen=True, order=True)
class NoteEvent:
    start_s: float
    end_s: float
    pitch: int
    velocity: int

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def name(self) -> str:
        return midi_to_note_name(self.pitch)

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "name": self.name}


def midi_to_note_name(pitch: int) -> str:
    """MIDI 60 -> 'C4', 40 -> 'E2' (scientific pitch notation)."""
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def _is_valid(start: float, end: float, pitch: int, velocity: int) -> bool:
    return 0 <= pitch <= 127 and 0 < velocity <= 127 and 0 <= start < end


def notes_from_pretty_midi(midi: pretty_midi.PrettyMIDI) -> list[NoteEvent]:
    """Collect notes from all non-drum instruments, sorted by onset then pitch.

    Zero-length and out-of-range notes are dropped rather than passed on.
    """
    notes = [
        NoteEvent(float(n.start), float(n.end), int(n.pitch), int(n.velocity))
        for inst in midi.instruments
        if not inst.is_drum
        for n in inst.notes
        if _is_valid(n.start, n.end, n.pitch, n.velocity)
    ]
    return sorted(notes)


def parse_midi(path: str | Path) -> list[NoteEvent]:
    p = Path(path)
    if not p.is_file():
        raise MidiParseError("file_not_found", "MIDI file was not found.")
    try:
        midi = pretty_midi.PrettyMIDI(str(p))
    except Exception as exc:  # pretty_midi/mido raise many unrelated types
        raise MidiParseError("unreadable_midi", "MIDI file could not be read.") from exc
    return notes_from_pretty_midi(midi)


def notes_from_basic_pitch(
    events: Iterable[Sequence[object]],
) -> list[NoteEvent]:
    """Convert Basic Pitch note events to NoteEvents.

    Basic Pitch yields ``(start_s, end_s, pitch, amplitude, pitch_bends)`` with
    amplitude in [0, 1]; it is mapped to MIDI velocity the same way Basic
    Pitch does when writing its own MIDI (``round(127 * amplitude)``).
    """
    notes: list[NoteEvent] = []
    for event in events:
        start, end, pitch, amplitude = event[:4]
        velocity = max(1, min(127, round(127 * float(amplitude))))  # type: ignore[arg-type]
        start_f, end_f, pitch_i = float(start), float(end), int(pitch)  # type: ignore[arg-type]
        if _is_valid(start_f, end_f, pitch_i, velocity):
            notes.append(NoteEvent(start_f, end_f, pitch_i, velocity))
    return sorted(notes)
