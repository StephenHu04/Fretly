"""Dynamic programming (Viterbi) assigner for monophonic note sequences.

The baseline picks each note's position on its own. This picks the whole
sequence at once, minimising a transparent cost: hand movement, position
shifts, string changes and a mild preference for low frets.

State is (position, hand) rather than just position. The hand remembers the
fret window used since the last shift and the last fretted note, so open
strings don't let it jump around the neck for free (fret 2 -> open -> fret 12
still costs a 10-fret move). The search is exact: it returns a minimum-cost
sequence under ``sequence_cost``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pipeline.midi_parser import NoteEvent
from tab.assign import Assignment, TabNote
from tab.fretboard import Fretboard, Position
from tab.metrics import HAND_SPAN_FRETS


@dataclass(frozen=True)
class OptimizerSettings:
    # Chosen against six reference clips with scripts/evaluate.py (decision 012);
    # move/finger split in decision 015.
    move_weight: float = 1.0  # per fret the hand window moves on a shift
    finger_weight: float = 0.25  # per fret the fingers move inside the window
    shift_weight: float = 5.0  # per position shift (leaving the 4-fret window)
    string_weight: float = 0.5  # per string crossed between consecutive notes
    height_weight: float = 0.05  # per fret, so ties prefer lower positions
    open_string_weight: float = 0.5  # per fret the hand reaches above open position, for an open string
    # Silence before a note makes a shift cheaper: full price up to
    # full_cost_rest_s, free from free_rest_s, linear between (decision 016).
    full_cost_rest_s: float = 0.25
    free_rest_s: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.full_cost_rest_s < self.free_rest_s:
            raise ValueError("need 0 <= full_cost_rest_s < free_rest_s")

    def shift_scale(self, rest_s: float) -> float:
        """How much of a shift's cost applies after ``rest_s`` of silence (1 = all)."""
        ramp = (self.free_rest_s - rest_s) / (self.free_rest_s - self.full_cost_rest_s)
        return min(1.0, max(0.0, ramp))


@dataclass(frozen=True)
class _Hand:
    low: int  # fret window used since the last shift
    high: int
    last: int  # fret of the most recent fretted note


_State = tuple[Position, _Hand | None]


def _step(
    prev: Position | None, hand: _Hand | None, pos: Position, settings: OptimizerSettings, rest_s: float = 0.0
) -> tuple[float, _Hand | None]:
    """Cost of playing ``pos`` next, and where the hand ends up."""
    cost = settings.height_weight * pos.fret
    if prev is not None:
        cost += settings.string_weight * abs(pos.string - prev.string)
    if pos.is_open:
        # The fretting hand stays where it was. In open position (the hand
        # within the first 4 frets) open strings are the natural choice;
        # higher up, players keep to fretted notes.
        if hand is not None:
            cost += settings.open_string_weight * max(0, hand.high - HAND_SPAN_FRETS)
        return cost, hand
    fret = pos.fret
    if hand is None:
        return cost, _Hand(fret, fret, fret)
    low, high = min(hand.low, fret), max(hand.high, fret)
    # Same shift rule as tab.metrics, so the optimizer targets what's measured.
    if high - low >= HAND_SPAN_FRETS:
        # The hand moves just far enough for the new fret to fit in a 4-fret
        # window that still overlaps where it was.
        if fret > hand.high:
            distance = fret - (hand.low + HAND_SPAN_FRETS - 1)
        else:
            distance = (hand.high - HAND_SPAN_FRETS + 1) - fret
        cost += settings.shift_scale(rest_s) * (settings.shift_weight + settings.move_weight * distance)
        return cost, _Hand(fret, fret, fret)
    # Inside the window only the fingers move, not the hand.
    return cost + settings.finger_weight * abs(fret - hand.last), _Hand(low, high, fret)


def rests_before(notes: Sequence[NoteEvent]) -> list[float]:
    """Silence before each note: time since every earlier note ended (0 for the first)."""
    rests: list[float] = []
    sounding_until: float | None = None
    for note in notes:
        rests.append(0.0 if sounding_until is None else max(0.0, note.start_s - sounding_until))
        sounding_until = note.end_s if sounding_until is None else max(sounding_until, note.end_s)
    return rests


def sequence_cost(
    positions: Sequence[Position],
    settings: OptimizerSettings | None = None,
    rests: Sequence[float] | None = None,
) -> float:
    """Total cost of playing ``positions`` in order (used by tests and comparisons).

    ``rests`` is the silence before each position; without it there is none.
    """
    settings = settings or OptimizerSettings()
    rests = [0.0] * len(positions) if rests is None else rests
    total, prev, hand = 0.0, None, None
    for pos, rest_s in zip(positions, rests, strict=True):
        cost, hand = _step(prev, hand, pos, settings, rest_s)
        total, prev = total + cost, pos
    return total


def assign_optimized(
    notes: Iterable[NoteEvent], fretboard: Fretboard, settings: OptimizerSettings | None = None
) -> Assignment:
    """Assign every playable note so the whole sequence has minimum cost.

    Unplayable notes are reported and skipped; their neighbours connect
    directly. Ties go to the earliest candidate found, which is the lowest
    fret, so results are deterministic.
    """
    settings = settings or OptimizerSettings()
    playable: list[tuple[NoteEvent, list[Position]]] = []
    unplayable: list[NoteEvent] = []
    for note in sorted(notes):
        candidates = fretboard.candidates(note.pitch)
        if candidates:
            playable.append((note, candidates))
        else:
            unplayable.append(note)
    if not playable:
        return Assignment([], unplayable)

    # Each column maps state -> (best total cost so far, previous state).
    columns: list[dict[_State, tuple[float, _State | None]]] = []
    column: dict[_State, tuple[float, _State | None]] = {}
    for pos in playable[0][1]:
        cost, hand = _step(None, None, pos, settings)
        column[(pos, hand)] = (cost, None)
    columns.append(column)

    rests = rests_before([note for note, _ in playable])
    for (_, candidates), rest_s in zip(playable[1:], rests[1:], strict=True):
        nxt: dict[_State, tuple[float, _State | None]] = {}
        for state, (total, _) in column.items():
            prev, hand = state
            for pos in candidates:
                cost, new_hand = _step(prev, hand, pos, settings, rest_s)
                key = (pos, new_hand)
                if key not in nxt or total + cost < nxt[key][0]:
                    nxt[key] = (total + cost, state)
        column = nxt
        columns.append(column)

    # Walk the back-pointers from the cheapest final state.
    state: _State | None = min(column, key=lambda s: column[s][0])
    path: list[Position] = []
    for col in reversed(columns):
        assert state is not None
        path.append(state[0])
        state = col[state][1]
    path.reverse()

    placed = [TabNote(note, pos) for (note, _), pos in zip(playable, path, strict=True)]
    return Assignment(placed, unplayable)
