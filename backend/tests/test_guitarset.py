from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from pipeline.midi_parser import NoteEvent
from scripts.score_guitarset import main
from tab.fretboard import Fretboard, Position
from tab.guitarset import PlayedNote, load_passages, read_notes, single_note_passages

BOARD = Fretboard()


def jams(per_string: dict[int, list[tuple[float, float, float]]]) -> dict:
    """GuitarSet-shaped annotation: data_source (0 = low E) -> (time, duration, pitch)."""
    strings = [
        {
            "namespace": "note_midi",
            "annotation_metadata": {"data_source": str(i)},
            "data": [{"time": t, "duration": d, "value": v} for t, d, v in per_string.get(i, [])],
        }
        for i in range(6)
    ]
    return {"annotations": [{"namespace": "tempo", "data": []}, *strings]}


def line(count: int, start: float = 0.0, duration: float = 0.2) -> list[PlayedNote]:
    onsets = [start + i * 0.25 for i in range(count)]
    return [PlayedNote(NoteEvent(t, t + duration, 60, 90), Position(2, 1)) for t in onsets]


def test_strings_frets_and_order() -> None:
    notes = read_notes(jams({5: [(1.0, 0.2, 69.1)], 0: [(0.0, 0.2, 42.9)]}), BOARD)
    assert [p.position for p in notes] == [Position(6, 3), Position(1, 5)]
    assert [p.note.pitch for p in notes] == [43, 69]


def test_bad_annotations_are_rejected() -> None:
    with pytest.raises(ValueError, match="isn't on string"):
        read_notes(jams({5: [(0.0, 0.2, 40.0)]}), BOARD)
    with pytest.raises(ValueError, match="expected 6"):
        read_notes({"annotations": []}, BOARD)


def test_a_chord_splits_a_line_and_is_dropped() -> None:
    chord = [PlayedNote(NoteEvent(2.0, 2.2, p, 90), Position(1, 0)) for p in (64, 67)]
    runs = single_note_passages(line(8) + chord + line(9, start=2.25))
    assert [len(r) for r in runs] == [8, 9]


def test_short_runs_are_dropped() -> None:
    assert single_note_passages(line(7)) == []


def test_small_overlaps_are_still_single_notes() -> None:
    assert len(single_note_passages(line(8, duration=0.33))) == 1  # 80 ms overlap
    assert single_note_passages(line(8, duration=0.40)) == []  # 150 ms


def test_only_solos_of_the_requested_players_are_loaded(tmp_path: Path) -> None:
    data = json.dumps(jams({3: [(i * 0.25, 0.2, 57.0) for i in range(8)]}))
    path = tmp_path / "annotation.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name in ("00_A_solo", "00_A_comp", "05_B_solo"):
            z.writestr(f"{name}.jams", data)
    assert [(p.player, p.piece, len(p.played)) for p in load_passages(path, BOARD)] == [("00", "A", 8)]
    assert [p.player for p in load_passages(path, BOARD, holdout=True)] == ["05"]
    assert main(["--annotations", str(path)]) == 0


def test_missing_download_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--annotations", str(tmp_path / "nope.zip")]) == 1
    assert "not found" in capsys.readouterr().err
