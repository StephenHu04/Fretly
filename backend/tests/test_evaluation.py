from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline import transcribe as transcribe_module
from pipeline.midi_parser import NoteEvent
from scripts.evaluate import find_audio, find_references, main
from tab.assign import Assignment, TabNote
from tab.evaluation import exact_in_tab, match_pitches, parse_reference, position_agreement, score_detection
from tab.fretboard import Fretboard, Position
from tests.conftest import WriteTone
from tests.test_transcribe import FakePredict

BOARD = Fretboard()
SAMPLES = Path(__file__).resolve().parents[2] / "samples"

REFERENCE = """\
# Reference tab for riff.wav
e|-0---|
B|---5-|

# The same 2 notes in order: string, fret, pitch (MIDI)
# 1  e 0  E4  64
# 2  B 8  G4  67
"""


def notes(*pitches: int) -> list[NoteEvent]:
    return [NoteEvent(i * 0.5, i * 0.5 + 0.4, p, 80) for i, p in enumerate(pitches)]


def test_parse_reference() -> None:
    assert parse_reference(REFERENCE, BOARD) == [Position(1, 0), Position(2, 8)]


def test_parse_reference_rejects_mismatched_pitch() -> None:
    with pytest.raises(ValueError, match="isn't MIDI"):
        parse_reference("# 1  e 3  E4  64\n", BOARD)


def test_parse_reference_rejects_unknown_string() -> None:
    with pytest.raises(ValueError, match="unknown string"):
        parse_reference("# 1  X 0  E4  64\n", BOARD)


def test_committed_references_parse() -> None:
    references = sorted(SAMPLES.rglob("*.tab.txt"))
    assert references, "expected reference tabs under samples/"
    for path in references:
        assert parse_reference(path.read_text(encoding="utf-8"), BOARD), path.name


def test_match_pitches_keeps_order() -> None:
    assert match_pitches([64, 63, 59], [64, 63, 59]) == [(0, 0), (1, 1), (2, 2)]
    # An extra note in the middle and a missing note at the end.
    assert match_pitches([64, 63, 59], [64, 70, 63]) == [(0, 0), (1, 2)]
    # A swap can only keep one of the two notes.
    assert len(match_pitches([1, 2], [2, 1])) == 1
    assert match_pitches([], [1]) == [] and match_pitches([1], []) == []


def test_score_detection() -> None:
    detected = notes(64, 52, 63, 59)  # 52 is extra
    score = score_detection([64, 63, 59, 66], detected)
    assert (score.played, score.detected, score.correct) == (4, 4, 3)
    assert score.missed == [66]
    assert [n.pitch for n in score.extra] == [52]
    assert score.recall == pytest.approx(0.75) and score.precision == pytest.approx(0.75)


def test_score_detection_sorts_by_time() -> None:
    detected = list(reversed(notes(64, 63)))
    assert score_detection([64, 63], detected).correct == 2


def test_empty_scores() -> None:
    score = score_detection([], [])
    assert (score.recall, score.precision) == (1.0, 1.0)


def test_position_agreement() -> None:
    ref = [Position(1, 0), Position(2, 5)]
    assert position_agreement(ref, [Position(1, 0), Position(1, 0)]) == 1
    with pytest.raises(ValueError):
        position_agreement(ref, [Position(1, 0)])


def test_holdout_folders_are_skipped_unless_asked_for(tmp_path: Path) -> None:
    for rel in ("clips/a.tab.txt", "holdout/b.tab.txt", "clips/holdout/c.tab.txt"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(REFERENCE)
    assert [p.name for p in find_references([tmp_path])] == ["a.tab.txt"]
    assert [p.name for p in find_references([tmp_path / "holdout"])] == ["b.tab.txt"]
    assert [p.name for p in find_references([tmp_path / "holdout" / "b.tab.txt"])] == ["b.tab.txt"]


def test_holdout_flag_cannot_take_paths(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--holdout", str(tmp_path)])


def test_find_audio_matches_the_reference_name(tmp_path: Path) -> None:
    (tmp_path / "riff.tab.txt").write_text(REFERENCE)
    assert find_audio(tmp_path / "riff.tab.txt") is None
    (tmp_path / "riff.m4a").write_bytes(b"x")
    assert find_audio(tmp_path / "riff.tab.txt") == tmp_path / "riff.m4a"


def test_evaluate_end_to_end_with_fake_model(
    write_tone: WriteTone, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # FakePredict "hears" E4 then G4; the reference says E4 (open e) then G4 on B fret 8.
    monkeypatch.setattr(transcribe_module, "_load_predict", lambda: FakePredict())
    clips = tmp_path / "clips"
    clips.mkdir()
    write_tone(name="clips/riff.wav")
    (clips / "riff.tab.txt").write_text(REFERENCE)

    assert main([str(clips), "--out-dir", str(tmp_path / "out"), "--json"]) == 0
    clip = json.loads(capsys.readouterr().out)["clips"][0]
    assert (clip["played"], clip["correct"], clip["missed"], clip["extra"]) == (2, 2, [], [])
    # Both assigners put G4 on open-position e fret 3, not B fret 8.
    assert clip["positions_matching"] == {"optimized": 1, "baseline": 1}
    assert clip["exact_in_tab"] == {"optimized": 1, "baseline": 1}

    assert main([str(clips), "--out-dir", str(tmp_path / "out"), "-v"]) == 0
    assert "Recall      100%" in capsys.readouterr().out


def test_evaluate_without_references(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path)]) == 1
    assert "no reference tabs" in capsys.readouterr().err


def test_exact_in_tab_needs_the_right_note_and_position() -> None:
    reference = [Position(1, 0), Position(2, 5), Position(2, 0)]  # E4 E4 B3
    tabbed = [
        TabNote(NoteEvent(0.0, 0.4, 64, 80), Position(1, 0)),  # right
        TabNote(NoteEvent(0.5, 0.9, 64, 80), Position(1, 0)),  # right note, wrong string
        TabNote(NoteEvent(0.7, 0.9, 52, 80), Position(4, 2)),  # extra note
    ]  # B3 missed
    assert exact_in_tab(reference, Assignment(tabbed, []), BOARD) == 1
