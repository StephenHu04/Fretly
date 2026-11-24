"""Fixed test phrases for comparing assigners (milestone 3.8).

Each one exercises a different fret-choice situation. They're note sequences
only: the player's fingering isn't part of them, so they're scored on the
playability metrics. Agreement with real players is measured separately on
recordings with reference tabs (scripts/evaluate.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.midi_parser import NoteEvent

NOTE_SPACING_S = 0.2


@dataclass(frozen=True)
class Phrase:
    name: str
    description: str
    pitches: tuple[int, ...]

    def notes(self) -> list[NoteEvent]:
        """One note every 0.2 s: quick enough that consecutive notes count toward stretch."""
        return [
            NoteEvent(i * NOTE_SPACING_S, (i + 1) * NOTE_SPACING_S, pitch, 90)
            for i, pitch in enumerate(self.pitches)
        ]


PHRASES: tuple[Phrase, ...] = (
    Phrase("open_riff", "open-position riff (no_chord1)", (64, 63, 59, 64, 63, 59, 64, 63, 59, 64, 66, 68)),
    Phrase("mid_neck", "phrase around frets 2-7 (no_chord5)",
           (71, 64, 61, 61, 63, 64, 66, 59, 59, 68, 66, 68, 66, 66, 64, 61)),
    Phrase("high_run", "run from A4 to C6 on the top strings", (69, 72, 74, 76, 79, 81, 84)),
    Phrase("repeated", "repeated notes in a high phrase", (69, 72, 74, 74, 74, 72, 69)),
    Phrase("jump", "low riff that jumps up the neck", (40, 43, 45, 76, 79, 81)),
    Phrase("pentatonic_5", "A minor pentatonic, 5th position", (55, 57, 60, 62, 64, 67, 69, 72)),
    Phrase("melody_7_10", "melody around frets 7-10", (67, 69, 71, 72, 74, 72, 71, 69, 67, 64)),
    Phrase("low_riff", "riff on the low E and A strings", (40, 43, 45, 40, 43, 46, 45, 43, 40)),
    Phrase("wide_range", "arpeggio across the whole neck and back",
           (45, 52, 57, 64, 69, 76, 81, 76, 69, 64, 57, 52, 45)),
)  # fmt: skip
