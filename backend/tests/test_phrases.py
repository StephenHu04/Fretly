from __future__ import annotations

import pytest

from scripts.compare_assigners import main
from tab.assign import assign_lowest_fret
from tab.fretboard import Fretboard
from tab.metrics import measure
from tab.optimizer import assign_optimized
from tab.phrases import PHRASES, Phrase

BOARD = Fretboard()


def test_phrases_are_playable_and_named_uniquely() -> None:
    assert len({p.name for p in PHRASES}) == len(PHRASES)
    for phrase in PHRASES:
        assert all(BOARD.is_playable(pitch) for pitch in phrase.pitches), phrase.name


@pytest.mark.parametrize("phrase", PHRASES, ids=lambda p: p.name)
def test_optimizer_never_shifts_more_than_the_baseline(phrase: Phrase) -> None:
    optimized = measure(assign_optimized(phrase.notes(), BOARD))
    baseline = measure(assign_lowest_fret(phrase.notes(), BOARD))
    assert optimized.position_shifts <= baseline.position_shifts


def test_optimizer_is_better_overall() -> None:
    # Milestone 3.8 acceptance: fewer shifts and less hand travel in total.
    opt = [measure(assign_optimized(p.notes(), BOARD)) for p in PHRASES]
    base = [measure(assign_lowest_fret(p.notes(), BOARD)) for p in PHRASES]
    assert sum(m.position_shifts for m in opt) < sum(m.position_shifts for m in base)
    assert sum(m.fret_travel for m in opt) < sum(m.fret_travel for m in base)


def test_compare_assigners_prints_every_phrase(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["-v"]) == 0
    out = capsys.readouterr().out
    assert all(p.name in out for p in PHRASES)
    assert "total" in out and "e|" in out
