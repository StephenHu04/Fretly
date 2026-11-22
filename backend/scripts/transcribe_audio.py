"""Transcribe a guitar recording to MIDI and list the detected notes.

Pipeline: validate -> FFmpeg preprocess -> Basic Pitch -> MIDI + note events.

Usage:
    python scripts/transcribe_audio.py path/to/clip.wav [--out-dir outputs] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.errors import PipelineError
from pipeline.run import transcribe_file
from pipeline.transcribe import TranscriptionSettings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="Path to a local audio file")
    parser.add_argument("--out-dir", default="outputs", help="Where to write artifacts")
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Feed the file to Basic Pitch directly (no FFmpeg needed)",
    )
    parser.add_argument("--onset-threshold", type=float, default=0.5)
    parser.add_argument("--frame-threshold", type=float, default=0.3)
    parser.add_argument("--min-note-ms", type=float, default=127.70)
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args(argv)

    src = Path(args.path)
    out_dir = Path(args.out_dir)
    settings = TranscriptionSettings(
        onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold,
        minimum_note_length_ms=args.min_note_ms,
    )

    try:
        info, result, timings = transcribe_file(src, out_dir, settings, args.skip_preprocess)
    except PipelineError as exc:
        print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({
            "input": info.to_dict(),
            "midi_path": str(result.midi_path),
            "settings": settings.to_dict(),
            "timings": timings,
            "notes": [n.to_dict() for n in result.notes],
        }, indent=2))
        return 0

    print(f"Input: {info.duration_s:.2f} s, {info.sample_rate} Hz, {info.channels} ch ({info.format})")
    print(f"MIDI:  {result.midi_path}")
    print("Time:  " + ", ".join(f"{k}={v:.2f}" for k, v in timings.items()))
    print(f"Notes: {len(result.notes)}\n")
    print(f"{'#':>4}  {'start':>7}  {'end':>7}  {'dur':>6}  {'note':<4} {'midi':>4}  {'vel':>3}")
    for i, n in enumerate(result.notes, 1):
        print(
            f"{i:>4}  {n.start_s:7.3f}  {n.end_s:7.3f}  {n.duration_s:6.3f}  "
            f"{n.name:<4} {n.pitch:>4}  {n.velocity:>3}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
