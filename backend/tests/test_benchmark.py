from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from scripts.benchmark_pipeline import (
    Backend,
    current_rss_mb,
    main,
    make_synthetic_clips,
    peak_rss_mb,
    percentile,
    summarize,
    synth_phrase,
)
from tests.test_transcribe import FakePredict


class FakeBackend:
    """Records which model each predict call received."""

    def __init__(self) -> None:
        self.models_loaded = 0
        self.fake = FakePredict()
        self.models_seen: list[Any] = []

    def predict(self, audio_path: Path, model_or_model_path: Any = "default-path", **kwargs: Any) -> Any:
        self.models_seen.append(model_or_model_path)
        return self.fake(audio_path, **kwargs)

    def load(self) -> Backend:
        def load_model() -> str:
            self.models_loaded += 1
            return "loaded-model"

        return Backend(self.predict, load_model, {"basic-pitch": "fake", "tensorflow": "fake"})


def test_percentile_nearest_rank() -> None:
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0
    assert percentile([1.0, 2.0, 3.0], 95) == 3.0  # small n: p95 is the max
    assert percentile([5.0], 95) == 5.0
    assert percentile(list(map(float, range(1, 101))), 95) == 95.0
    with pytest.raises(ValueError):
        percentile([], 50)


def test_summarize_reports_real_time_factor() -> None:
    s = summarize([2.0, 1.0, 3.0], audio_duration_s=10.0)
    assert s["runs"] == 3
    assert s["median_s"] == 2.0
    assert s["p95_s"] == 3.0
    assert s["min_s"] == 1.0 and s["max_s"] == 3.0
    assert s["rtf"] == pytest.approx(0.2)


def test_synth_phrase_is_deterministic_and_bounded() -> None:
    a, b = synth_phrase(2.0), synth_phrase(2.0)
    assert len(a) == 2 * 22050
    assert np.array_equal(a, b)
    assert np.max(np.abs(a)) <= 0.5 + 1e-6


def test_synthetic_clips_have_requested_durations(tmp_path: Path) -> None:
    clips = make_synthetic_clips([1.0, 2.5], tmp_path)
    for clip, expected in zip(clips, [1.0, 2.5], strict=True):
        info = sf.info(str(clip.path))
        assert info.samplerate == 22050 and info.channels == 1
        assert info.duration == pytest.approx(expected, abs=1e-3)
        assert clip.duration_s == expected


def test_memory_readings_are_plausible() -> None:
    peak = peak_rss_mb()
    if peak is None:
        pytest.skip("peak memory not supported on this platform")
    assert peak > 1.0
    now = current_rss_mb()
    if now is not None:
        assert 1.0 < now <= peak + 1.0


def test_main_runs_both_modes_and_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    backend = FakeBackend()
    json_out = tmp_path / "bench.json"
    code = main(
        ["--durations", "1", "2", "--repeats", "2", "--out-dir", str(tmp_path / "out"),
         "--json-out", str(json_out)],
        backend_loader=backend.load,
    )
    assert code == 0

    # One model load for the whole run; warm-up + 2 clips x 2 repeats reuse it,
    # and the reload mode passes no model (so Basic Pitch would load its own).
    assert backend.models_loaded == 1
    assert backend.models_seen.count("loaded-model") == 1 + 2 * 2
    assert backend.models_seen.count("default-path") == 2 * 2

    report = json.loads(json_out.read_text())
    assert [c["name"] for c in report["clips"]] == ["synth-1s", "synth-2s"]
    assert report["cold_start"]["first_inference_clip"] == "synth-1s"
    for c in report["clips"]:
        assert c["model_kept_loaded"]["runs"] == 2
        assert c["model_reloaded_per_job"]["runs"] == 2

    out = capsys.readouterr().out
    assert "Cold start" in out and "model kept loaded" in out and "RTF" in out


def test_no_reload_skips_reload_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    backend = FakeBackend()
    json_out = tmp_path / "bench.json"
    main(["--durations", "1", "--repeats", "1", "--no-reload", "--out-dir", str(tmp_path),
          "--json-out", str(json_out)], backend_loader=backend.load)
    assert "default-path" not in backend.models_seen
    assert "model_reloaded_per_job" not in json.loads(json_out.read_text())["clips"][0]


def test_real_audio_is_validated(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--audio", str(tmp_path / "missing.wav"), "--out-dir", str(tmp_path)],
                backend_loader=FakeBackend().load)
    assert code == 1
    assert "error [" in capsys.readouterr().err


def test_rejects_zero_repeats(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--repeats", "0", "--out-dir", str(tmp_path)], backend_loader=FakeBackend().load)


@pytest.mark.model
@pytest.mark.skipif(importlib.util.find_spec("basic_pitch") is None, reason="basic-pitch not installed")
def test_benchmark_with_real_model(tmp_path: Path) -> None:
    json_out = tmp_path / "bench.json"
    assert main(["--durations", "2", "--repeats", "1", "--out-dir", str(tmp_path),
                 "--json-out", str(json_out)]) == 0
    report = json.loads(json_out.read_text())
    assert report["cold_start"]["model_load_s"] > 0
    assert report["clips"][0]["model_kept_loaded"]["rtf"] > 0
