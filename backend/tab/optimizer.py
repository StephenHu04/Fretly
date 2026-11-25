"""Dynamic programming (Viterbi) assigner for monophonic note sequences.

The baseline picks each note's position on its own. This picks the whole
sequence at once, minimising a transparent cost: hand movement, position
shifts, string changes and a mild preference for a home position.

State is (position, hand) rather than just position. The hand remembers the
fret window used since the last shift and the last fretted note, so open
strings don't let it jump around the neck for free (fret 2 -> open -> fret 12
still costs a 10-fret move). The search is exact: it returns a minimum-cost
sequence under ``sequence_cost``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from pipeline.midi_parser import NoteEvent
from tab.assign import Assignment, TabNote
from tab.fretboard import Fretboard, Position
from tab.metrics import HAND_SPAN_FRETS


@dataclass(frozen=True)
class OptimizerSettings:
    # Tuned on GuitarSet's development players, guarded by our own clips and the
    # fixed phrases (decision 018). move and shift are unchanged since 015.
    move_weight: float = 1.0  # per fret the hand window moves on a shift
    finger_weight: float = 0.5  # per fret the fingers move inside the window
    shift_weight: float = 5.0  # per position shift (leaving the 4-fret window)
    string_weight: float = 0.5  # per string crossed between consecutive notes
    height_weight: float = 0.15  # per fret away from home_fret, for fretted notes
    open_string_weight: float = 2.0  # per fret the hand reaches above open position, for an open string
    home_fret: int = 6  # fret the height cost pulls fretted notes toward
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


# Playing styles (decision 019). "open" is the pre-018 weights; "lead" was tuned
# on GuitarSet like "balanced" but without our own clips as a guardrail.
STYLES: Mapping[str, OptimizerSettings] = {
    "balanced": OptimizerSettings(),
    "open": OptimizerSettings(finger_weight=0.25, height_weight=0.05, open_string_weight=0.5, home_fret=0),
    "lead": OptimizerSettings(move_weight=1.5, finger_weight=0.05, shift_weight=2.0, height_weight=0.15, home_fret=7),
}


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
    cost = 0.0 if pos.is_open else settings.height_weight * abs(pos.fret - settings.home_fret)
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


def _start_hand(start_fret: int | None) -> _Hand | None:
    if start_fret is None:
        return None
    if start_fret < 1:
        raise ValueError("start_fret must be at least 1")
    return _Hand(start_fret, start_fret, start_fret)


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
    start_fret: int | None = None,
) -> float:
    """Total cost of playing ``positions`` in order (used by tests and comparisons).

    ``rests`` is the silence before each position; without it there is none.
    """
    settings = settings or OptimizerSettings()
    rests = [0.0] * len(positions) if rests is None else rests
    total, prev, hand = 0.0, None, _start_hand(start_fret)
    for pos, rest_s in zip(positions, rests, strict=True):
        cost, hand = _step(prev, hand, pos, settings, rest_s)
        total, prev = total + cost, pos
    return total


def assign_optimized(
    notes: Iterable[NoteEvent],
    fretboard: Fretboard,
    settings: OptimizerSettings | None = None,
    pins: Mapping[int, Position] | None = None,
    start_fret: int | None = None,
) -> Assignment:
    """Assign every playable note so the whole sequence has minimum cost.

    ``pins`` fixes notes the user has placed (index in time order -> position)
    and the rest is solved around them; ``start_fret`` says where the hand is
    before the first note. A pin that doesn't sound its note, or an index out
    of range, raises ValueError.

    Unplayable notes are reported and skipped; their neighbours connect
    directly. Ties go to the earliest candidate found, which is the lowest
    fret, so results are deterministic.
    """
    settings = settings or OptimizerSettings()
    playable: list[tuple[NoteEvent, list[Position]]] = []
    unplayable: list[NoteEvent] = []
    ordered = sorted(notes)
    pins = pins or {}
    for index, pos in pins.items():
        if not 0 <= index < len(ordered):
            raise ValueError(f"there is no note {index + 1}; there are {len(ordered)}")
        if pos not in fretboard.candidates(ordered[index].pitch):
            raise ValueError(f"string {pos.string} fret {pos.fret} doesn't play note {index + 1} ({ordered[index].name})")
    start_hand = _start_hand(start_fret)
    for i, note in enumerate(ordered):
        candidates = [pins[i]] if i in pins else fretboard.candidates(note.pitch)
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
        cost, hand = _step(None, start_hand, pos, settings)
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
