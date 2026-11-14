# Fretly
An end-to-end audio-to-tablature transcription engine combining polyphonic pitch detection, and dynamic programming fretboard pathfinding.

See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full specification and roadmap.

## Status

Phase 1 (repository and local environment) — audio inspection and validation only.
No transcription or tab generation yet.

## Setup

Requirements:

- Python 3.11 or 3.12
- FFmpeg (optional for now; used to cross-check files and needed from Phase 2)

From the repository root:

```bash
cd backend
python -m venv .venv          # on Windows, `py -3.12 -m venv .venv` also works
```

Activate the environment:

| Shell | Command |
|---|---|
| PowerShell | `.venv\Scripts\Activate.ps1` |
| cmd | `.venv\Scripts\activate.bat` |
| Git Bash | `source .venv/Scripts/activate` |
| macOS / Linux | `source .venv/bin/activate` |

Install dependencies (exact versions from the lock file, then the package itself):

```bash
pip install -r requirements.lock
pip install -e . --no-deps
```

Or install the latest compatible versions instead: `pip install -e ".[dev]"`.

### Installing FFmpeg

- Windows: `winget install Gyan.FFmpeg` (then open a new terminal)
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`

Verify with `ffprobe -version`.

## Inspecting an audio file

From `backend/` with the environment active:

```bash
python scripts/inspect_audio.py path/to/clip.wav
python scripts/inspect_audio.py path/to/clip.wav --json      # machine-readable
python scripts/inspect_audio.py path/to/clip.wav --ffprobe   # cross-check with FFmpeg
```

It reports readability, duration, sample rate, channel count and detected
format, and exits with code 1 if the file fails validation.

Current validation rules (see `backend/pipeline/audio_validation.py`):

- Extensions: `.wav .flac .ogg .mp3 .aiff .aif`
- Max file size: 50 MB
- Duration: 0.5–120 s
- The audio data must actually decode, not just the header.

M4A/AAC isn't accepted yet. It needs FFmpeg conversion, which arrives in Phase 2.

## Running tests

```bash
cd backend
pytest
```

The ffprobe test is skipped automatically when FFmpeg isn't installed.

## Repository layout

```text
backend/
  pipeline/audio_validation.py   audio inspection and validation
  scripts/inspect_audio.py       command-line inspection tool
  tests/                         pytest suite (fixtures generated at runtime)
samples/                         local recordings (audio files are git-ignored)
PROJECT_SPEC.md                  specification and roadmap
```
