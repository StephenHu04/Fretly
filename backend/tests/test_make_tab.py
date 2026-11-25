from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pretty_midi
import pytest

from pipeline import transcribe as transcribe_module
from scripts.make_tab import main
from tests.conftest import WriteTone
from tests.test_transcribe import FakePredict

BACKEND = Path(__file__).resolve().parents[1]


def write_midi(path: Path, notes: list[tuple[float, float, int, int]]) -> Path:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=25)
    inst.notes = [pretty_midi.Note(vel, pitch, start, end) for start, end, pitch, vel in notes]
    pm.instruments.append(inst)
    pm.write(str(path))
    return path


# A high run the baseline plays by sliding up the e string.
HIGH_RUN = [(i * 0.25, (i + 1) * 0.25, p, 100) for i, p in enumerate([69, 72, 74, 76, 79, 81, 84])]


def test_midi_to_tab(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    midi = write_midi(tmp_path / "riff.mid", [(0.0, 0.5, 64, 100), (0.5, 1.0, 63, 100), (1.0, 1.5, 59, 100)])
    assert main([str(midi)]) == 0
    out = capsys.readouterr().out
    assert "Input: riff.mid (MIDI)" in out
    assert "e|-0-----|\nB|---4-0-|" in out
    assert "Baseline" not in out


def test_both_methods_side_by_side(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    midi = write_midi(tmp_path / "run.mid", HIGH_RUN)
    assert main([str(midi), "--method", "both"]) == 0
    out = capsys.readouterr().out
    assert "Optimized:" in out and "Baseline:" in out
    shifts = next(line for line in out.splitlines() if line.startswith("position shifts"))
    assert shifts.split()[-2:] == ["0", "3"]  # optimized, baseline


def test_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    midi = write_midi(tmp_path / "run.mid", HIGH_RUN)
    assert main([str(midi), "--method", "both", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["source"]["type"] == "midi"
    assert set(report["results"]) == {"optimized", "baseline"}
    optimized = report["results"]["optimized"]
    assert optimized["metrics"]["position_shifts"] == 0
    assert optimized["tab"].startswith("e|")
    assert {"string", "fret", "pitch", "name"} <= set(optimized["notes"][0])


def test_octave_ghosts_are_removed_and_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    midi = write_midi(tmp_path / "ghost.mid", [(0.0, 0.5, 40, 100), (0.0, 0.5, 52, 45)])
    assert main([str(midi)]) == 0
    out = capsys.readouterr().out
    assert "removed 1 octave ghost" in out and "octave ghost: E3 at 0.00 s" in out

    assert main([str(midi), "--no-ghost-filter", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert "octave_ghost" not in report["removed"]
    assert len(report["results"]["optimized"]["notes"]) == 2


def test_unplayable_notes_are_listed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    midi = write_midi(tmp_path / "low.mid", [(0.0, 0.5, 36, 100), (0.5, 1.0, 40, 100)])
    assert main([str(midi)]) == 0
    out = capsys.readouterr().out
    assert "# unplayable: C2 at 0.00 s" in out


def test_audio_runs_transcription_first(
    write_tone: WriteTone, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(transcribe_module, "_load_predict", lambda: FakePredict())
    clip = write_tone(name="riff.wav")
    assert main([str(clip), "--skip-preprocess", "--out-dir", str(tmp_path / "out"), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["source"]["type"] == "audio"
    assert "transcribe_s" in report["source"]["timings"]
    # FakePredict returns E4 then G4.
    assert [n["name"] for n in report["results"]["optimized"]["notes"]] == ["E4", "G4"]


def test_errors_exit_with_code_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path / "missing.wav")]) == 1
    assert "error [file_not_found]" in capsys.readouterr().err
    bad = tmp_path / "bad.mid"
    bad.write_bytes(b"not midi")
    assert main([str(bad)]) == 1
    assert "error [unreadable_midi]" in capsys.readouterr().err


def test_max_fret_limits_the_neck(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    midi = write_midi(tmp_path / "high.mid", [(0.0, 0.5, 83, 100), (0.5, 1.0, 84, 100)])  # e string frets 19, 20
    assert main([str(midi), "--max-fret", "19", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)["results"]["optimized"]
    assert [n["fret"] for n in result["notes"]] == [19]
    assert [n["pitch"] for n in result["unplayable"]] == [84]


def test_hints_and_styles(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    midi = write_midi(tmp_path / "riff.mid", [(0.0, 0.5, 64, 100), (0.5, 1.0, 63, 100), (1.0, 1.5, 59, 100)])
    assert main([str(midi), "--pin", "1:B5", "--list-notes"]) == 0
    out = capsys.readouterr().out
    assert "note 1: E4 at 0.00 s" in out and "B|-5" in out
    assert main([str(midi), "--style", "lead", "--start-fret", "7"]) == 0
    assert "Optimized:" in capsys.readouterr().out
    assert main([str(midi), "--pin", "1:B6"]) == 1
    assert "doesn't play note 1 (E4)" in capsys.readouterr().err
    assert main([str(midi), "--start-fret", "0"]) == 1
    capsys.readouterr()
    assert main([str(midi), "--method", "both", "--alternatives", "3", "--json"]) == 0
    assert {"optimized", "option 2", "baseline"} <= set(json.loads(capsys.readouterr().out)["results"])


@pytest.mark.parametrize(
    "extra",
    [["--max-fret", "0"], ["--max-fret", "30"], ["--pin", "1-B5"], ["--pin", "1:H5"], ["--pin", "1:B5", "--pin", "1:G9"], ["--method", "baseline", "--style", "lead"]],
)
def test_bad_options_are_usage_errors(extra: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["riff.mid", *extra])
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize(
    "script",
    ["make_tab.py", "transcribe_audio.py", "inspect_audio.py", "benchmark_pipeline.py", "evaluate.py", "compare_assigners.py",
     "score_guitarset.py"],
)
def test_scripts_run_as_commands(script: str) -> None:
    # Run the way a user does, so import problems that pytest's path setup
    # hides (e.g. importing from scripts/ or an uninstalled package) show up.
    result = subprocess.run(
        [sys.executable, f"scripts/{script}", "--help"], cwd=BACKEND, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
