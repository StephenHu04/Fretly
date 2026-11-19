"""Fretboard model and candidate string/fret positions for a pitch.

Strings are numbered the way guitarists number them: 1 is the high E, 6 the
low E. A pitch is playable on a string when ``0 <= pitch - open_pitch <= max_fret``.
"""

from __future__ import annotations

from dataclasses import dataclass

# Open-string MIDI pitches for strings 1..6: E4 B3 G3 D3 A2 E2.
STANDARD_TUNING: tuple[int, ...] = (64, 59, 55, 50, 45, 40)
DEFAULT_MAX_FRET = 24


@dataclass(frozen=True, order=True)
class Position:
    string: int
    fret: int

    @property
    def is_open(self) -> bool:
        return self.fret == 0


@dataclass(frozen=True)
class Fretboard:
    tuning: tuple[int, ...] = STANDARD_TUNING
    max_fret: int = DEFAULT_MAX_FRET

    def __post_init__(self) -> None:
        if not self.tuning:
            raise ValueError("tuning needs at least one string")
        if self.max_fret < 0:
            raise ValueError("max_fret must be non-negative")

    @property
    def num_strings(self) -> int:
        return len(self.tuning)

    @property
    def lowest_pitch(self) -> int:
        return min(self.tuning)

    @property
    def highest_pitch(self) -> int:
        return max(self.tuning) + self.max_fret

    def open_pitch(self, string: int) -> int:
        if not 1 <= string <= self.num_strings:
            raise ValueError(f"string must be 1-{self.num_strings}, got {string}")
        return self.tuning[string - 1]

    def pitch_at(self, position: Position) -> int:
        if not 0 <= position.fret <= self.max_fret:
            raise ValueError(f"fret must be 0-{self.max_fret}, got {position.fret}")
        return self.open_pitch(position.string) + position.fret

    def candidates(self, pitch: int) -> list[Position]:
        """Every position that sounds ``pitch``, lowest fret first.

        An empty list means the pitch is outside the fretboard's range.
        """
        positions = [
            Position(string, pitch - open_pitch)
            for string, open_pitch in enumerate(self.tuning, start=1)
            if 0 <= pitch - open_pitch <= self.max_fret
        ]
        return sorted(positions, key=lambda p: (p.fret, p.string))

    def is_playable(self, pitch: int) -> bool:
        return bool(self.candidates(pitch))
