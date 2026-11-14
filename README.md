# Fretly

[![tests](https://github.com/StephenHu04/Fretly/actions/workflows/tests.yml/badge.svg)](https://github.com/StephenHu04/Fretly/actions/workflows/tests.yml)

An end-to-end audio-to-tablature transcription engine combining polyphonic pitch detection, and dynamic programming fretboard pathfinding.

- [PROJECT_SPEC.md](PROJECT_SPEC.md): full specification and roadmap
- [docs/decisions.md](docs/decisions.md): technical decisions and their trade-offs

## Status

- Phase 1 (environment, audio inspection): done.
- Phase 2 (audio → MIDI): pipeline works end to end with the real model on
  synthetic audio; still to do is checking it on real guitar recordings.
- No tab generation yet.

## Setup

Requirements:

- **Python 3.11** (not 3.12; see [Basic Pitch setup](#basic-pitch-setup))
- FFmpeg

From the repository root:

```bash
cd backend
py -3.11 -m venv .venv        # Windows; elsewhere: python3.11 -m venv .venv
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

Or install the latest compatible versions instead:

```bash
pip install -e ".[dev,transcription]"   # full pipeline (includes TensorFlow, large)
pip install -e ".[dev]"                 # fast tests only; the model is mocked
```

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

## Transcribing audio to MIDI

From `backend/` with the environment active:

```bash
python scripts/transcribe_audio.py path/to/clip.wav            # note table
python scripts/transcribe_audio.py path/to/clip.wav --json     # machine-readable
python scripts/transcribe_audio.py clip.wav --skip-preprocess  # without FFmpeg
```

The pipeline validates the file, converts it with FFmpeg to mono 22,050 Hz WAV
(Basic Pitch's native rate), runs Basic Pitch, and writes these to `outputs/`
(git-ignored):

- `<name>.22k.wav`: the preprocessed audio
- `<name>.mid`: the transcribed MIDI

It also prints every detected note (start, end, name, MIDI pitch, velocity) with
the time each stage took. Model thresholds can be tuned with `--onset-threshold`,
`--frame-threshold` and `--min-note-ms`.

Detection is limited to 75–1400 Hz, which covers a standard-tuned guitar from
E2 up to fret 24 on the high E string.

### Basic Pitch setup

`basic-pitch` 0.4.0 (the latest release) declares `tensorflow<2.15.1` with
Python ≥ 3.11, and TensorFlow 2.15 has no Python 3.12 build, so the project
targets **Python 3.11**. Both the lock file and the `transcription` extra install
it with TensorFlow 2.15.0. Without it, `transcribe_audio.py` exits with
`error [model_unavailable]`.

The package also ships an ONNX copy of its model. Running that on Python 3.12 with
`onnxruntime` is possible, but only by installing with `--no-deps`, which bypasses
the package's declared constraints. It's not used here.

### Observed on synthetic audio

On an 11-note E minor pentatonic run (E2 to E4, synthesised with harmonics,
4.4 s), measured 2026-11-13:

- All 11 notes were detected with correct pitch, and onsets were within ~35 ms.
- 7 extra notes appeared exactly one octave above low notes, at the same
  onset and about half the velocity (second-harmonic ghosts).
- Transcription took ~16 s, including TensorFlow start-up (not yet measured
  separately).

This is a smoke test, not an accuracy measurement. Real guitar recordings come next.

### Known limitations

Apart from the octave ghosts above, these come from Basic Pitch's documented
behaviour and haven't been measured on real guitar yet. Phase 6 adds measurements.

- Basic Pitch is a general instrument model, not guitar-specific. It reports
  pitches and timing, not string/fret positions (that's the Phase 3 optimizer).
- It works best on a single isolated instrument. Backing tracks, distortion
  and heavy reverb add spurious notes.
- Bends, slides and vibrato may be split into several notes or quantized to a
  semitone. Pitch-bend data is currently discarded.
- Guitar string harmonics can show up as extra notes an octave or a fifth
  above the played note. Phase 3 cleanup will need to address this.
- Notes shorter than `--min-note-ms` (default ~128 ms) are dropped, which can
  lose fast passages.

## Running tests

```bash
cd backend
pytest              # fast suite; model inference is mocked
pytest -m model     # real Basic Pitch inference (requires basic-pitch installed)
```

The FFmpeg tests are skipped automatically when FFmpeg isn't installed.

## Repository layout

```text
backend/
  pipeline/audio_validation.py   audio inspection and validation
  scripts/inspect_audio.py       command-line inspection tool
  tests/                         pytest suite (fixtures generated at runtime)
samples/                         local recordings (audio files are git-ignored)
PROJECT_SPEC.md                  specification and roadmap
```
