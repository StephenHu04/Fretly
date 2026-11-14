"""Inspect a local audio file and report whether Fretly can use it.

Usage:
    python scripts/inspect_audio.py path/to/clip.wav [--json] [--ffprobe]
"""

from __future__ import annotations

import argparse
import json
import sys

from pipeline.audio_validation import (
    AudioValidationError,
    ffprobe_available,
    probe_with_ffprobe,
    validate_audio,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="Path to a local audio file")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    parser.add_argument(
        "--ffprobe",
        action="store_true",
        help="Also cross-check the file with ffprobe (requires FFmpeg)",
    )
    args = parser.parse_args(argv)

    report: dict[str, object] = {"path": args.path}
    exit_code = 0

    try:
        report.update(validate_audio(args.path).to_dict())
    except AudioValidationError as exc:
        report.update(readable=False, error_code=exc.code, error=exc.message)
        exit_code = 1

    if args.ffprobe:
        if not ffprobe_available():
            report["ffprobe"] = {"error": "ffprobe is not installed."}
        else:
            try:
                report["ffprobe"] = probe_with_ffprobe(args.path)
            except AudioValidationError as exc:
                report["ffprobe"] = {"error_code": exc.code, "error": exc.message}

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for key, value in report.items():
            if isinstance(value, float):
                value = f"{value:.3f}"
            print(f"{key:>20}: {value}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
