"""Plain-text tablature in the usual guitar-tab layout.

    e|-0-----0---|
    B|---4-0---0-|
    ...

One line per string (top string first), one column per onset. Rhythm isn't
shown: columns are evenly spaced. See docs/phase-3-plan.md, milestone 3.6.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pipeline.midi_parser import NOTE_NAMES
from tab.assign import Assignment
from tab.fretboard import Fretboard

# Notes starting this close together share a column (double stops, chords).
SAME_COLUMN_S = 0.03
DEFAULT_WIDTH = 80


@dataclass
class _Column:
    start_s: float
    frets: dict[int, int] = field(default_factory=dict)  # string -> fret

    @property
    def width(self) -> int:
        return max(len(str(f)) for f in self.frets.values())


def string_labels(fretboard: Fretboard) -> list[str]:
    """Open-string note names, top string first, padded to equal width.

    The top string is lower-case (``e``) so it's never confused with low E.
    """
    names = [NOTE_NAMES[pitch % 12] for pitch in fretboard.tuning]
    names[0] = names[0].lower()
    width = max(len(n) for n in names)
    return [n.ljust(width) for n in names]


def _columns(assignment: Assignment) -> list[_Column]:
    columns: list[_Column] = []
    for tab_note in assignment.notes:  # already in time order
        string, fret = tab_note.position.string, tab_note.position.fret
        current = columns[-1] if columns else None
        if (
            current is not None
            and tab_note.note.start_s - current.start_s <= SAME_COLUMN_S
            and string not in current.frets
        ):
            current.frets[string] = fret
        else:
            # A new onset, or a second note on a string already used in this
            # column: start a new column rather than overwrite it.
            columns.append(_Column(tab_note.note.start_s, {string: fret}))
    return columns


def _block(columns: list[_Column], labels: list[str]) -> str:
    lines = []
    for string, label in enumerate(labels, start=1):
        cells = "".join(
            "-" + str(col.frets.get(string, "")).ljust(col.width, "-") for col in columns
        )
        lines.append(f"{label}|{cells}-|")
    return "\n".join(lines)


def render_tab(assignment: Assignment, fretboard: Fretboard, width: int = DEFAULT_WIDTH) -> str:
    """Render ``assignment`` as tab, wrapping lines longer than ``width``.

    Wrapped blocks repeat the string labels and never split a column.
    Unplayable notes are listed underneath so nothing disappears silently.
    """
    labels = string_labels(fretboard)
    columns = _columns(assignment)

    overhead = len(labels[0]) + len("|") + len("-|")
    blocks: list[list[_Column]] = [[]]
    used = overhead
    for col in columns:
        needed = 1 + col.width
        if blocks[-1] and used + needed > width:
            blocks.append([])
            used = overhead
        blocks[-1].append(col)
        used += needed

    text = "\n\n".join(_block(cols, labels) for cols in blocks)
    for note in assignment.unplayable:
        text += f"\n# unplayable: {note.name} at {note.start_s:.2f} s"
    return text
