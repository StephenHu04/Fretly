"""Read single-note passages from GuitarSet annotations.

GuitarSet (Xi et al., ISMIR 2018, CC BY 4.0) labels every note with the string
it was played on, so it can score fret choice on real players. Only the
annotations are needed; no audio is read.

Fretly handles single notes so far, so only ``solo`` files are used (``comp``
files are chords) and passages are cut at double stops and chords.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.midi_parser import NoteEvent
from tab.fretboard import Fretboard, Position

# Set before any scores were seen (decision 017).
HOLDOUT_PLAYERS = frozenset({"04", "05"})
MIN_ONSET_GAP_S = 0.05  # notes starting closer together than this are a chord
MAX_OVERLAP_S = 0.1  # longer overlaps are treated as two notes held together
MIN_PASSAGE_NOTES = 8

_SOLO_FILE = re.compile(r"(?:^|/)(\d\d)_([^/]+)_solo\.jams$")
_VELOCITY = 90  # GuitarSet has no velocities; the assigners don't use them


@dataclass(frozen=True)
class PlayedNote:
    note: NoteEvent
    position: Position


@dataclass(frozen=True)
class Passage:
    player: str
    piece: str
    played: tuple[PlayedNote, ...]


def read_notes(jams: dict[str, Any], fretboard: Fretboard) -> list[PlayedNote]:
    """Notes of one annotation file, in order of onset.

    There is one ``note_midi`` annotation per string, low E first. A note that
    can't be on its labelled string raises ValueError.
    """
    strings = [a for a in jams["annotations"] if a["namespace"] == "note_midi"]
    if len(strings) != fretboard.num_strings:
        raise ValueError(f"expected {fretboard.num_strings} note_midi annotations, found {len(strings)}")
    played = []
    for annotation in strings:
        string = fretboard.num_strings - int(annotation["annotation_metadata"]["data_source"])
        for item in annotation["data"]:
            pitch = round(item["value"])
            position = Position(string, pitch - fretboard.open_pitch(string))
            if not 0 <= position.fret <= fretboard.max_fret:
                raise ValueError(f"MIDI {pitch} isn't on string {string}")
            start = item["time"]
            played.append(PlayedNote(NoteEvent(start, start + item["duration"], pitch, _VELOCITY), position))
    return sorted(played, key=lambda p: (p.note.start_s, p.note.pitch))


def _sound_together(a: NoteEvent, b: NoteEvent) -> bool:
    """Whether two notes (``a`` starting first) are played together, not one after the other."""
    return b.start_s - a.start_s < MIN_ONSET_GAP_S or min(a.end_s, b.end_s) - b.start_s > MAX_OVERLAP_S


def single_note_passages(played: Iterable[PlayedNote]) -> list[list[PlayedNote]]:
    """Runs of at least MIN_PASSAGE_NOTES notes with nothing sounding together.

    Every note that sounds together with another is dropped and ends the run.
    """
    ordered = list(played)
    chordal = [False] * len(ordered)
    for i, a in enumerate(ordered):
        for j in range(i + 1, len(ordered)):
            b = ordered[j].note
            if b.start_s >= max(a.note.end_s, a.note.start_s + MIN_ONSET_GAP_S):
                break
            if _sound_together(a.note, b):
                chordal[i] = chordal[j] = True
    runs: list[list[PlayedNote]] = [[]]
    for item, skip in zip(ordered, chordal, strict=True):
        if skip:
            runs.append([])
        else:
            runs[-1].append(item)
    return [run for run in runs if len(run) >= MIN_PASSAGE_NOTES]


def load_passages(annotations: Path, fretboard: Fretboard, holdout: bool = False) -> Iterator[Passage]:
    """Single-note passages from ``annotation.zip``, for the development or the held-out players."""
    with zipfile.ZipFile(annotations) as archive:
        for name in sorted(archive.namelist()):
            match = _SOLO_FILE.search(name)
            if not match or (match.group(1) in HOLDOUT_PLAYERS) != holdout:
                continue
            notes = read_notes(json.loads(archive.read(name)), fretboard)
            for run in single_note_passages(notes):
                yield Passage(match.group(1), match.group(2), tuple(run))
