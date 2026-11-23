"""Score pipeline output against a reference tab of what was actually played.

Two separate scores, so transcription errors and fret-choice errors don't mix
(spec principle 4):

- detection: which played notes the model found (by pitch, in order);
- positions: given the *reference* notes, how often an assigner picks the
  same string and fret as the player.

Reference files are the ``*.tab.txt`` files in ``samples/``: a tab for people
plus one ``# <n> <string> <fret> <name> <pitch>`` comment line per note.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from pipeline.midi_parser import NoteEvent
from tab.fretboard import Fretboard, Position
from tab.render import string_labels

_NOTE_LINE = re.compile(r"^#\s*\d+\s+(\S+)\s+(\d+)\s+\S+\s+(\d+)\s*$")


def parse_reference(text: str, fretboard: Fretboard) -> list[Position]:
    """Read the per-note lines of a reference tab, in order.

    Raises ValueError for an unknown string label or a pitch that doesn't
    match its string and fret, so typos in a reference can't skew a score.
    """
    strings = {label.strip(): i for i, label in enumerate(string_labels(fretboard), start=1)}
    positions = []
    for line in text.splitlines():
        match = _NOTE_LINE.match(line.strip())
        if not match:
            continue
        label, fret, pitch = match.group(1), int(match.group(2)), int(match.group(3))
        if label not in strings:
            raise ValueError(f"unknown string {label!r} in: {line.strip()}")
        position = Position(strings[label], fret)
        if fretboard.pitch_at(position) != pitch:
            raise ValueError(f"string {label} fret {fret} isn't MIDI {pitch}: {line.strip()}")
        positions.append(position)
    return positions


def match_pitches(reference: Sequence[int], detected: Sequence[int]) -> list[tuple[int, int]]:
    """Pair up reference and detected notes by pitch, keeping both in order.

    Longest common subsequence: the most pairs possible without reordering
    either list. Returns (reference index, detected index) pairs.
    """
    n, m = len(reference), len(detected)
    # best[i][j] = most pairs between reference[i:] and detected[j:]
    best = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if reference[i] == detected[j]:
                best[i][j] = best[i + 1][j + 1] + 1
            else:
                best[i][j] = max(best[i + 1][j], best[i][j + 1])
    pairs = []
    i = j = 0
    while i < n and j < m:
        if reference[i] == detected[j]:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif best[i + 1][j] >= best[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


@dataclass(frozen=True)
class DetectionScore:
    played: int
    detected: int
    missed: list[int]  # reference pitches with no match
    extra: list[NoteEvent]  # detected notes with no match

    @property
    def correct(self) -> int:
        return self.played - len(self.missed)

    @property
    def recall(self) -> float:
        return self.correct / self.played if self.played else 1.0

    @property
    def precision(self) -> float:
        return self.correct / self.detected if self.detected else 1.0


def score_detection(reference: Sequence[int], detected: Sequence[NoteEvent]) -> DetectionScore:
    ordered = sorted(detected, key=lambda n: (n.start_s, n.pitch))
    pairs = match_pitches(reference, [n.pitch for n in ordered])
    matched_ref = {i for i, _ in pairs}
    matched_det = {j for _, j in pairs}
    return DetectionScore(
        played=len(reference),
        detected=len(ordered),
        missed=[p for i, p in enumerate(reference) if i not in matched_ref],
        extra=[n for j, n in enumerate(ordered) if j not in matched_det],
    )


def position_agreement(reference: Sequence[Position], assigned: Sequence[Position]) -> int:
    """How many notes got the same string and fret as the reference."""
    if len(reference) != len(assigned):
        raise ValueError("reference and assigned positions differ in length")
    return sum(r == a for r, a in zip(reference, assigned, strict=True))
