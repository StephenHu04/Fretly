"""Benchmark Basic Pitch transcription: cold start, warm inference, memory.

Measures, in one fresh process:

- cold start: importing TensorFlow + Basic Pitch, loading the model, and the
  first inference (which also builds TensorFlow's graph);
- per-clip time with the model kept loaded vs reloaded for every job
  (``basic_pitch.inference.predict`` reloads it unless given a loaded Model);
- real-time factor (RTF = processing time / audio duration; below 1 is faster
  than real time);
- process memory: baseline, after model load, and peak.

Clips are synthetic guitar-like phrases unless real recordings are given.

Usage:
    python scripts/benchmark_pipeline.py                         # 15/30/60 s synthetic
    python scripts/benchmark_pipeline.py --durations 15 --repeats 5
    python scripts/benchmark_pipeline.py --audio ../samples/riff.wav
    python scripts/benchmark_pipeline.py --json-out outputs/benchmark.json
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import io
import json
import math
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from pipeline.audio_validation import validate_audio
from pipeline.errors import PipelineError
from pipeline.preprocess import TARGET_SAMPLE_RATE, preprocess_audio
from pipeline.transcribe import PredictFn, TranscriptionError, transcribe

DEFAULT_DURATIONS_S = (15.0, 30.0, 60.0)

# E minor pentatonic, E2 to E4: the same phrase used for the Phase 2 smoke test.
PHRASE_MIDI = (40, 43, 45, 47, 50, 52, 55, 57, 59, 62, 64)


# --- Memory -----------------------------------------------------------------
# Standard library only. tracemalloc can't see TensorFlow's native
# allocations, so these read the OS's resident-set figures instead.

def _windows_memory() -> tuple[float, float]:
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32")
    psapi = ctypes.WinDLL("psapi")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD,
    ]
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return counters.WorkingSetSize / 2**20, counters.PeakWorkingSetSize / 2**20


def current_rss_mb() -> float | None:
    """Resident memory of this process now, or None where unsupported (macOS)."""
    try:
        if sys.platform == "win32":
            return _windows_memory()[0]
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (OSError, ValueError, AttributeError):
        return None


def peak_rss_mb() -> float | None:
    """Highest resident memory this process has reached."""
    try:
        if sys.platform == "win32":
            return _windows_memory()[1]
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS bytes.
        return peak / 2**20 if sys.platform == "darwin" else peak / 2**10
    except (OSError, ImportError, AttributeError):
        return None


# --- Statistics -------------------------------------------------------------

def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. With few samples, p95 is simply the maximum."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def summarize(times_s: list[float], audio_duration_s: float) -> dict[str, float]:
    median = statistics.median(times_s)
    return {
        "runs": len(times_s),
        "median_s": median,
        "p95_s": percentile(times_s, 95),
        "min_s": min(times_s),
        "max_s": max(times_s),
        "rtf": median / audio_duration_s,
    }


# --- Clips ------------------------------------------------------------------

def synth_phrase(duration_s: float, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Deterministic plucked-string-like phrase, looped to ``duration_s``.

    Each note has a few decaying harmonics so the model sees something closer
    to a guitar than a pure sine. Not a stand-in for real recordings.
    """
    note_s = 0.4
    t = np.arange(int(note_s * sample_rate)) / sample_rate
    envelope = np.exp(-4.0 * t)
    notes = []
    for midi in PHRASE_MIDI:
        f0 = 440.0 * 2 ** ((midi - 69) / 12)
        tone = sum((0.6 / k) * np.sin(2 * np.pi * k * f0 * t) for k in (1, 2, 3, 4))
        notes.append(tone * envelope)
    one_pass = np.concatenate(notes)
    n = int(duration_s * sample_rate)
    looped = np.tile(one_pass, n // len(one_pass) + 1)[:n]
    return (0.5 * looped / np.max(np.abs(looped))).astype(np.float32)


@dataclass(frozen=True)
class Clip:
    name: str
    path: Path
    duration_s: float
    preprocess_s: float | None = None


def make_synthetic_clips(durations_s: list[float], out_dir: Path) -> list[Clip]:
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for d in durations_s:
        path = out_dir / f"synth-{d:g}s.wav"
        sf.write(str(path), synth_phrase(d), TARGET_SAMPLE_RATE, subtype="PCM_16")
        clips.append(Clip(path.stem, path, d))
    return clips


def prepare_real_clips(paths: list[Path], out_dir: Path, skip_preprocess: bool) -> list[Clip]:
    clips = []
    for src in paths:
        info = validate_audio(src)
        if skip_preprocess:
            clips.append(Clip(src.stem, src, info.duration_s))
            continue
        t0 = time.perf_counter()
        wav = preprocess_audio(src, out_dir / f"{src.stem}.22k.wav")
        clips.append(Clip(src.stem, wav, info.duration_s, time.perf_counter() - t0))
    return clips


# --- Model backend ----------------------------------------------------------

@dataclass(frozen=True)
class Backend:
    predict: PredictFn
    load_model: Callable[[], Any]
    versions: dict[str, str]


def import_backend() -> Backend:
    """Import Basic Pitch (and with it TensorFlow). This is the slow part of cold start."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's info logs
    try:
        from importlib.metadata import version

        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model, predict
    except ImportError as exc:
        raise TranscriptionError(
            "model_unavailable",
            "Basic Pitch is not installed; see README for setup.",
        ) from exc

    versions = {}
    for pkg in ("basic-pitch", "tensorflow"):
        try:
            versions[pkg] = version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    return Backend(predict, lambda: Model(ICASSP_2022_MODEL_PATH), versions)


# --- Benchmark --------------------------------------------------------------

def _time_transcribe(clip: Clip, predict_fn: PredictFn, out_dir: Path) -> float:
    # Basic Pitch prints "Predicting MIDI for ..." on every call.
    with contextlib.redirect_stdout(io.StringIO()):
        t0 = time.perf_counter()
        transcribe(clip.path, out_dir / f"{clip.name}.mid", predict_fn=predict_fn)
        return time.perf_counter() - t0


def run_benchmark(
    clips: list[Clip],
    repeats: int,
    out_dir: Path,
    include_reload: bool = True,
    backend_loader: Callable[[], Backend] = import_backend,
) -> dict[str, Any]:
    baseline_rss = current_rss_mb()

    t0 = time.perf_counter()
    backend = backend_loader()
    import_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = backend.load_model()
    model_load_s = time.perf_counter() - t0
    rss_after_load = current_rss_mb()

    kept = partial(backend.predict, model_or_model_path=model)

    # The first call pays one-off graph construction; it's part of cold
    # start, so it's reported there and excluded from the warm numbers.
    first_clip = min(clips, key=lambda c: c.duration_s)
    first_inference_s = _time_transcribe(first_clip, kept, out_dir)

    results = []
    for clip in clips:
        entry: dict[str, Any] = {
            "name": clip.name,
            "audio_duration_s": clip.duration_s,
            "preprocess_s": clip.preprocess_s,
            "model_kept_loaded": summarize(
                [_time_transcribe(clip, kept, out_dir) for _ in range(repeats)], clip.duration_s
            ),
        }
        if include_reload:
            # Default predict() builds a new Model each call, as the pipeline
            # does today. TensorFlow is already imported, so this is the cost
            # a long-lived worker would pay per job.
            entry["model_reloaded_per_job"] = summarize(
                [_time_transcribe(clip, backend.predict, out_dir) for _ in range(repeats)],
                clip.duration_s,
            )
        results.append(entry)

    return {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            **backend.versions,
        },
        "settings": {"repeats": repeats},
        "cold_start": {
            "import_s": import_s,
            "model_load_s": model_load_s,
            "first_inference_s": first_inference_s,
            "first_inference_clip": first_clip.name,
            "total_s": import_s + model_load_s + first_inference_s,
        },
        "clips": results,
        "memory_mb": {
            "baseline": baseline_rss,
            "after_model_load": rss_after_load,
            "peak": peak_rss_mb(),
        },
    }


def format_report(report: dict[str, Any]) -> str:
    m, cold, mem = report["machine"], report["cold_start"], report["memory_mb"]
    lines = [
        f"Machine: {m['platform']}, {m['cpu_count']} CPUs, Python {m['python']}, "
        f"basic-pitch {m.get('basic-pitch', '?')}, TensorFlow {m.get('tensorflow', '?')}",
        "",
        "Cold start (fresh process)",
        *(
            f"  {label:<36}{value:7.2f} s"
            for label, value in (
                ("import TensorFlow + Basic Pitch", cold["import_s"]),
                ("load model", cold["model_load_s"]),
                (f"first inference ({cold['first_inference_clip']})", cold["first_inference_s"]),
                ("total", cold["total_s"]),
            )
        ),
        "",
        f"Per clip ({report['settings']['repeats']} runs each; median / p95, RTF from median)",
    ]

    def cell(s: dict[str, float] | None) -> str:
        if s is None:
            return "-"
        return f"{s['median_s']:6.2f} / {s['p95_s']:6.2f} s  RTF {s['rtf']:.3f}"

    lines.append(f"  {'clip':<16}{'audio':>8}   {'model kept loaded':<30}{'model reloaded per job':<30}")
    for c in report["clips"]:
        lines.append(
            f"  {c['name']:<16}{c['audio_duration_s']:7.1f}s   "
            f"{cell(c['model_kept_loaded']):<30}{cell(c.get('model_reloaded_per_job')):<30}"
        )
        if c["preprocess_s"] is not None:
            lines.append(f"  {'':<16}{'':>8}   FFmpeg preprocessing {c['preprocess_s']:.2f} s")

    def mb(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.0f} MB"

    lines += [
        "",
        f"Memory: baseline {mb(mem['baseline'])}, after model load {mb(mem['after_model_load'])}, "
        f"peak {mb(mem['peak'])}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None, backend_loader: Callable[[], Backend] = import_backend) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--durations", type=float, nargs="+", default=list(DEFAULT_DURATIONS_S),
        help="Synthetic clip lengths in seconds (ignored with --audio)",
    )
    parser.add_argument("--audio", type=Path, nargs="+", help="Benchmark these recordings instead")
    parser.add_argument("--repeats", type=int, default=3, help="Timed runs per clip and mode")
    parser.add_argument("--no-reload", action="store_true", help="Skip the reload-per-job mode")
    parser.add_argument("--skip-preprocess", action="store_true", help="Don't run FFmpeg on --audio files")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--json-out", type=Path, help="Also write the results as JSON here")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    try:
        if args.audio:
            clips = prepare_real_clips(args.audio, args.out_dir, args.skip_preprocess)
        else:
            clips = make_synthetic_clips(args.durations, args.out_dir)
        report = run_benchmark(
            clips, args.repeats, args.out_dir,
            include_reload=not args.no_reload, backend_loader=backend_loader,
        )
    except PipelineError as exc:
        print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
        return 1

    print(format_report(report))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"\nJSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
