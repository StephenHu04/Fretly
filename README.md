# Fretly

[![tests](https://github.com/StephenHu04/Fretly/actions/workflows/tests.yml/badge.svg)](https://github.com/StephenHu04/Fretly/actions/workflows/tests.yml)

An end-to-end audio-to-tablature transcription engine combining polyphonic pitch detection, and dynamic programming fretboard pathfinding.

- [PROJECT_SPEC.md](PROJECT_SPEC.md): full specification and roadmap
- [docs/decisions.md](docs/decisions.md): technical decisions and their trade-offs
- [docs/phase-3-plan.md](docs/phase-3-plan.md): plan for the tablature optimizer

## Status

- Phase 1 (environment, audio inspection): done.
- Phase 2 (audio → MIDI): pipeline works end to end, is
  [benchmarked](#benchmarking), and has been checked against
  [six real recordings](#observed-on-real-recordings) (97% of played notes found, 80% of
  detected notes correct).
- Phase 3 (tablature optimizer): in progress. The fretboard model, octave-ghost
  filter, lowest-fret baseline, playability metrics, dynamic programming
  optimizer, plain-text tab output and an [audio-to-tab command](#making-tabs)
  are done; comparing the optimizer with the baseline is next (see the
  [plan](docs/phase-3-plan.md)).

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

- Extensions: `.wav .flac .ogg .mp3 .aiff .aif .m4a .aac`
- Max file size: 50 MB
- Duration: 0.5–120 s
- The audio data must actually decode, not just the header.

`.m4a` and `.aac`, the formats most phone recorders (including iPhone Voice
Memos) use, can't be read by libsndfile. They're inspected with ffprobe instead,
so they need FFmpeg installed. ffprobe reads only the header, so a file that's
corrupt further in is caught when FFmpeg converts it (`ffmpeg_failed`).

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
- Transcription took ~16 s, almost all of it one-off TensorFlow start-up and
  model loading (see [Benchmarking](#benchmarking)).

This is a smoke test, not an accuracy measurement.

### Observed on real recordings

Six single-note phrases recorded on a phone (M4A, 48 kHz, 5–14 s each) in
`samples/audio_clips/`, each checked against a tab of what was actually played
(`no_chord*.tab.txt`) with `scripts/evaluate.py`. `no_chord2` repeats the
`no_chord1` phrase and `no_chord6` repeats `no_chord5`. Detected notes are
matched to played notes in order (longest common subsequence of pitches).
Measured 2026-11-22 at default settings:

| Clip | Played | Found | Missed | Extra | Frets: optimizer / baseline | In tab: optimizer / baseline |
|---|---|---|---|---|---|---|
| no_chord1 | 12 | 11 | 1 | 1 | 11 / 11 | 10 / 10 |
| no_chord2 | 12 | 12 | 0 | 4 | 11 / 11 | 11 / 11 |
| no_chord3 | 12 | 11 | 1 | 3 | 12 / 12 | 9 / 11 |
| no_chord4 | 25 | 25 | 0 | 3 | 25 / 25 | 25 / 25 |
| no_chord5 | 16 | 16 | 0 | 6 | 10 / 9 | 12 / 9 |
| no_chord6 | 16 | 15 | 1 | 5 | 10 / 9 | 8 / 8 |
| **Total** | **93** | **90** | **3** | **22** | **79 / 77 (85% / 83%)** | **75 / 74 (81% / 80%)** |

*Frets*: given the correct notes, how often the assigner picks the player's
string and fret. *In tab*: end to end, played notes that appear in the final
tab exactly as played (right note, string and fret).

**Detection: recall 97%, precision 80%.** Nearly every played note is found;
about one detected note in five is extra. On the first four clips the extras
were:

| Cause | Extras | Removed by filters | Example |
|---|---|---|---|
| Open string ringing when a finger lifts off it | 12 | 6 | B3 after B fret 5 → e fret 2; G3 after G#3 on G fret 1 |
| Octave-below ghost of a loud note | 2 | 2 | quiet G#2 under G#3 |
| One held note split in two | 2 | 0 | E3 |
| Noise before playing starts | 2 | 0 | quiet A2 and D2 (below the guitar's range, so reported as unplayable) |
| Wrong octave | 1 | 0 | E5 instead of E4 |

The ringing open-string filter (decision 011) was tuned on those four clips.
On `no_chord5` and `no_chord6`, recorded afterwards, it removed 2 more ringing
notes and no real ones, a first sign it isn't just fitted to the clips it was
built on. The new clips' remaining extras are mostly low notes (G2, E2, E3)
that look like sympathetic resonance of the lower strings.

**String/fret choices.** On the four open-position clips the optimizer and
the baseline tie, because the lowest fret is already the natural choice there.
The difference is the phrase in `no_chord5/6`, which sits around frets 2–7 and
frets notes the player could have played open. The first version of the
optimizer treated open strings as free and lost to the baseline there (69/93
vs 77/93 on frets, 68/93 vs 74/93 in the tab). It now charges for an open
string once the hand has moved past the first four frets (decision 012):

```text
Played                                Optimizer before                      Optimizer now
e|-7-----------------4-2-4-2-2-0---|  e|-7-0-------0-----------------0---|  e|-7---------0-2-----4-2-4-2-2-0---|
B|---5-----4-5-7-----------------2-|  B|-------------7-0-0-9-7-9-7-7-----|  B|---5-2-2-4-----0-0-------------2-|
G|-----6-6-------4-4---------------|  G|-----6-6-8---------------------6-|  G|---------------------------------|
```

It now beats the baseline on fret choice (79/93 vs 77/93) and is one note
ahead end to end (75/93 vs 74/93), too small a margin to call. Most of the
remaining end-to-end errors aren't fret-choice problems. In `no_chord3` the model hears E5 instead of E4, and the
optimizer follows that wrong note up the neck while the baseline ignores it.
In `no_chord1/2` the player frets E4 at B5, a choice neither assigner makes.

**Transcription errors, not the optimizer, cause some odd fingerings.** On the
model's output for `no_chord3`, the optimizer played C#4 at fret 11 on the D
string because an extra E5 needs fret 12. On the correct notes it matched all
12 of the player's positions. Scoring fret choices on reference notes as well
as model output keeps the two kinds of error apart (spec principle 4).

Six short phrases from one player aren't an accuracy benchmark; Phase 6 adds a
proper evaluation set.

### Known limitations

Apart from the octave ghosts and ringing open strings above, these come
from Basic Pitch's documented behaviour and haven't been measured on real
guitar yet. Phase 6 adds measurements.

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

## Making tabs

From `backend/` with the environment active:

```bash
python scripts/make_tab.py ../samples/audio_clips/no_chord1.m4a       # audio -> tab
python scripts/make_tab.py clip.mid                                   # MIDI -> tab
python scripts/make_tab.py clip.m4a --method both                     # optimizer vs baseline
python scripts/make_tab.py clip.m4a --json                            # machine-readable
```

Audio goes through transcription first (as in the previous section); MIDI
files skip it. Then two cleanup filters remove notes nobody played: octave
ghosts (`--no-ghost-filter` keeps them) and open strings ringing after a
finger lifts (`--no-ring-filter`). Each remaining note gets a string and fret,
and the result prints as tab with playability metrics:

```text
Input: no_chord1.m4a (5.61 s, M4A), transcribed in 19.4 s
Notes: 13 detected, removed 1 ringing open string
  ringing open string: B3 at 4.03 s

Optimized:
e|-0-----0-----0-----0-4-|
B|---4-0---4-0---4-0---0-|
G|-----------------------|
D|-----------------------|
A|-----------------------|
E|-----------------------|

                         optimized
position shifts                  0
...
```

Rhythm isn't shown yet: columns are evenly spaced. Simultaneous notes share a
column, but chord fingering isn't optimized yet (Phase 3, milestone 3.9).

If `make_tab.py` fails with `No module named 'tab'`, the environment was
installed before the `tab` package existed; rerun `pip install -e . --no-deps`.

## Evaluating against reference tabs

```bash
python scripts/evaluate.py          # every samples/**/<name>.tab.txt with matching audio
python scripts/evaluate.py -v       # also list missed, extra and filtered notes
python scripts/evaluate.py --json
```

For each reference tab it transcribes the matching recording, runs the
cleanup filters and reports two separate scores (decision 005): **detection**
(played notes found, missed and extra) and **string/fret agreement** (given
the *reference* notes, how often each assigner picks the player's string and
fret), plus **in tab**: played notes that come out exactly right in the final
tab. `--no-ghost-filter` and `--no-ring-filter` show what each filter
contributes. Results are in [Observed on real recordings](#observed-on-real-recordings).

Clips in `samples/holdout/` are skipped by default. They're kept back so the
final score comes from recordings nothing was tuned on; score them with
`python scripts/evaluate.py --holdout`.

### Comparing the assigners on fixed phrases

```bash
python scripts/compare_assigners.py       # metrics for both assigners on each phrase
python scripts/compare_assigners.py -v    # also print both tabs
```

Nine phrases in `tab/phrases.py` cover open position, mid-neck, high runs,
repeated notes, jumps, a pentatonic box, low riffs and a full-neck arpeggio.
Totals (optimizer / baseline, lower is better):

| Position shifts | Fret travel | Max stretch | String changes |
|---|---|---|---|
| 5 / 15 | 97 / 116 | 23 / 27 | 40 / 25 |

The optimizer never shifts more than the baseline on any phrase, and in total
shifts a third as often, travels less and stretches less. The cost is more
string changes, which is what staying in one position takes. A test keeps
these guarantees.

## Benchmarking

From `backend/` with the environment active:

```bash
python scripts/benchmark_pipeline.py                              # 15/30/60 s synthetic clips
python scripts/benchmark_pipeline.py --repeats 5 --json-out outputs/benchmark.json
python scripts/benchmark_pipeline.py --audio ../samples/riff.wav  # real recordings
```

In one fresh process it measures cold start (TensorFlow import, model load and
first inference), then times each clip with the model **kept loaded** and
**reloaded per job**. Basic Pitch's `predict()` reloads the model on every call
unless it's given a loaded one, which is what the pipeline currently does. It
also reports the real-time factor (RTF = processing time ÷ audio duration) and
process memory.

### Results (2026-11-14)

Windows 11, 16 logical CPUs, CPU only, Python 3.11.9, TensorFlow 2.15.0,
basic-pitch 0.4.0. Synthetic guitar-like clips, 5 runs each; times are the
median, from the Basic Pitch call to the MIDI file being written (FFmpeg
preprocessing not included).

| Cold start | Time |
|---|---|
| Import TensorFlow + Basic Pitch | 7.9 s |
| Load model | 1.1 s |
| First inference (15 s clip) | 2.4 s |
| **Total** | **11.4 s** |

| Clip | Model kept loaded | RTF | Model reloaded per job | RTF |
|---|---|---|---|---|
| 15 s | 0.18 s | 0.012 | 1.42 s | 0.095 |
| 30 s | 0.36 s | 0.012 | 1.72 s | 0.057 |
| 60 s | 0.73 s | 0.012 | 2.04 s | 0.034 |

Memory: 37 MB baseline, 412 MB after model load, 710 MB peak.

What this means:

- With the model kept loaded, inference runs at ~1.2% of real time and scales
  linearly with clip length: a 60 s clip takes under a second.
- Reloading costs a fixed ~1.3 s per job. For a typical 15 s clip that's ~8×
  slower than keeping the model loaded.
- A fresh worker needs ~11 s before its first result. Workers should start
  once and stay up; this rules out starting a process per job.
- At ~0.7 GB per worker, memory (not CPU time) is likely to cap how many
  workers fit on one machine.

These are single-machine numbers on synthetic audio. Real recordings may
produce more notes and slightly different timings. See
[decision 008](docs/decisions.md#008--keep-the-model-loaded-in-each-worker-process).

## Running tests

```bash
cd backend
pytest              # fast suite; model inference is mocked
pytest -m model     # real Basic Pitch inference (requires basic-pitch installed)
```

The FFmpeg tests are skipped automatically when FFmpeg isn't installed.

Lint with [ruff](https://docs.astral.sh/ruff/) (configured in `pyproject.toml`; CI runs it on every push):

```bash
ruff check .        # add --fix to apply safe fixes
```

## Repository layout

```text
backend/
  pipeline/errors.py             PipelineError base (stable code + user-safe message)
  pipeline/audio_validation.py   audio inspection and validation
  pipeline/preprocess.py         FFmpeg conversion to mono 22.05 kHz WAV
  pipeline/transcribe.py         Basic Pitch wrapper (audio -> MIDI)
  pipeline/midi_parser.py        MIDI / model output -> NoteEvent
  pipeline/run.py                validate -> preprocess -> transcribe in one call
  tab/fretboard.py               tuning, positions, candidate string/fret positions
  tab/cleanup.py                 cleanup filters: octave ghosts, ringing open strings
  tab/assign.py                  assignment types and the lowest-fret baseline
  tab/metrics.py                 playability metrics (shifts, stretch, travel)
  tab/optimizer.py               dynamic programming (Viterbi) string/fret optimizer
  tab/render.py                  plain-text tablature
  tab/evaluation.py              scoring against reference tabs
  tab/phrases.py                 fixed test phrases for comparing assigners
  scripts/inspect_audio.py       command-line inspection tool
  scripts/transcribe_audio.py    command-line audio -> MIDI pipeline
  scripts/make_tab.py            command-line audio or MIDI -> tab
  scripts/evaluate.py            score the pipeline on clips with reference tabs
  scripts/compare_assigners.py   optimizer vs baseline on the fixed phrases
  scripts/benchmark_pipeline.py  cold start, inference speed and memory benchmark
  tests/                         pytest suite (fixtures generated at runtime)
samples/                         local recordings (audio files are git-ignored)
PROJECT_SPEC.md                  specification and roadmap
```
