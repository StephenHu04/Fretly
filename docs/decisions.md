# Decision log

Short records of technical decisions: the context, what was chosen, and what it
costs. Newest entries go at the bottom. When a decision is reversed, add a new
entry that points back to the old one; don't edit history.

---

## 001 — Target Python 3.11 so Basic Pitch runs on TensorFlow

**Date:** 2026-11-13 · **Status:** Accepted

**Context.** `basic-pitch` 0.4.0, the latest release, declares
`tensorflow<2.15.1` on Python ≥ 3.11, and TensorFlow 2.15 has no Python 3.12
build, so `pip install basic-pitch` fails on 3.12. The package also ships an ONNX
copy of its model that runs on `onnxruntime`.

**Options.**
1. Python 3.11 + TensorFlow 2.15: the configuration the package supports.
2. Python 3.12 + ONNX, installed with `--no-deps`: a much lighter install, but it
   bypasses the package's declared constraints, so it's an unsupported setup.

**Decision.** Option 1. Reproducibility matters more than install size during the
proof of concept, and a supported setup means bugs are ours, not artefacts of an
unusual install.

**Consequences.** Large install (TensorFlow) and slow cold start. The ONNX route
stays available as an optimisation if benchmarks show TensorFlow start-up or
memory dominates worker cost.

---

## 002 — Basic Pitch is an optional dependency; tests mock it

**Date:** 2026-11-13 · **Status:** Accepted

**Context.** Real inference needs TensorFlow (hundreds of MB) and takes seconds
per clip. Most code (validation, preprocessing, parsing, and soon the optimizer)
doesn't need the model.

**Decision.** `basic-pitch` lives in the `transcription` extra and is imported
lazily inside `pipeline/transcribe.py`. `transcribe()` accepts an injected
`predict_fn`, so unit tests use a fake. Real-model tests carry the `model`
marker and run only with `pytest -m model`.

**Consequences.** The default test suite runs in under a second and CI installs
without TensorFlow. The cost is that CI never exercises the real model, so model
tests must be run locally before changes to `transcribe.py` are merged.

---

## 003 — Normalise audio with FFmpeg to mono 22,050 Hz before inference

**Date:** 2026-11-13 · **Status:** Accepted

**Context.** Basic Pitch resamples input itself, so preprocessing isn't strictly
required.

**Decision.** Convert every input to mono 16-bit PCM WAV at 22,050 Hz (Basic
Pitch's `AUDIO_SAMPLE_RATE`) with FFmpeg first.

**Consequences.** Every later stage sees one predictable format. FFmpeg also
decodes containers libsndfile can't (M4A/AAC). The normalised file is a stored
artifact, so a failed transcription can be retried without re-decoding. FFmpeg
becomes a system dependency, though `--skip-preprocess` remains as an escape
hatch.

---

## 004 — Limit detection to the guitar's frequency range (75–1400 Hz)

**Date:** 2026-11-13 · **Status:** Accepted

**Context.** Basic Pitch is a general instrument model. A standard-tuned
guitar spans E2 (82.4 Hz) to E6 (fret 24 on high E, 1318.5 Hz).

**Decision.** Pass `minimum_frequency=75`, `maximum_frequency=1400` by default.

**Consequences.** Removes rumble and high-harmonic detections the fretboard
could never produce. This doesn't remove octave ghosts *inside* the range;
those need a cleanup step (planned for Phase 3). The bounds must be revisited
for alternate tunings, 7-string guitars or bass.

---

## 005 — `NoteEvent` is the boundary between transcription and tablature

**Date:** 2026-11-13 · **Status:** Accepted

**Context.** Notes can come from Basic Pitch's tuples, a MIDI file, or (later)
reference annotations from an evaluation dataset.

**Decision.** All sources convert to a small frozen dataclass,
`pipeline.midi_parser.NoteEvent(start_s, end_s, pitch, velocity)`. Cleanup and
the optimizer only accept `NoteEvent`s.

**Consequences.** The optimizer can be tested, benchmarked and evaluated with
*ground-truth* notes, independent of the model. That's what makes it possible to
separate transcription errors from fret-assignment errors (spec principle 4).

---

## 006 — Errors carry a stable code and a user-safe message

**Date:** 2026-11-13 · **Status:** Accepted

**Context.** Errors will eventually reach a browser. FFmpeg stderr and exception
text can contain local paths.

**Decision.** Each pipeline module raises its own error type with a stable
`code` (e.g. `unreadable_audio`, `ffmpeg_failed`) and a `message` that never
includes paths, stderr or stack traces. The original exception is chained for
server-side logging. Validation also decodes a block of audio rather than
trusting the header, so truncated files fail early.

**Consequences.** The future API can map codes to HTTP responses directly. Tests
check that messages don't leak paths.

---

## 007 — Defer Redis, PostgreSQL and S3 until a milestone needs them

**Date:** 2026-11-13 · **Status:** Accepted

**Context.** The architecture diagram shows the full production stack. The spec
says not to add infrastructure before it's useful.

**Decision.** Phases 1–3 are a local CLI pipeline with files on disk. A queue,
database and object storage come in Phase 4, introduced one at a time when a
concrete need appears (e.g. jobs must survive restarts → PostgreSQL).

**Consequences.** Faster iteration on the hard parts (transcription quality, the
optimizer). The pipeline stages already read and write files, so moving them
into queue workers later should be mostly a wiring change.

---

## 008 — Keep the model loaded in each worker process

**Date:** 2026-11-14 · **Status:** Accepted (implemented in Phase 4)

**Context.** `basic_pitch.inference.predict()` builds a new `Model` on every
call unless it's passed one that's already loaded, and `transcribe()` currently
relies on that default. `scripts/benchmark_pipeline.py` measured on a 16-thread
Windows laptop, CPU only (full numbers in the README):

- cold start in a fresh process: 11.4 s (7.9 s importing TensorFlow, 1.1 s
  loading the model, 2.4 s for the first inference);
- per job with the model kept loaded: 0.18 s for 15 s of audio, 0.73 s for 60 s
  (RTF ≈ 0.012, linear in length);
- per job with the model reloaded: a further ~1.3 s fixed cost, so 1.42 s for
  15 s of audio (~8× slower);
- memory: ~0.4 GB after loading, ~0.7 GB peak per process.

**Options.**
1. Start a process per job: pays the 11 s cold start every time.
2. Long-lived workers that reload the model per job: simple, but most of the
   time for a typical clip goes to reloading the model.
3. Long-lived workers that load the model once and pass it to `predict()`.

**Decision.** Option 3. In Phase 4, workers load the model at start-up and
`transcribe()` accepts the loaded model, through the existing `predict_fn`
injection point or an explicit argument. No change now: the CLI runs one job
per process, where the difference doesn't apply.

**Consequences.** Inference stops being the bottleneck for 15–60 s clips, so
queueing and I/O will dominate job latency. Worker count per machine is bounded
by memory (~0.7 GB each) rather than CPU time, and workers need a warm-up
period before they report ready. If memory becomes the limit, the ONNX model
(decision 001) is the next thing to measure. These numbers come from synthetic
audio on one machine, so they should be rerun on real recordings and on the
deployment hardware.

---

## 009 — Filter octave ghosts by pitch, onset and velocity

**Date:** 2026-11-18 · **Status:** Accepted

**Context.** On the Phase 2 synthetic phrase, Basic Pitch added 7 notes exactly
one octave above real notes, at the same onset and about half the velocity
(second harmonics). In tab these would become notes nobody played.

**Decision.** `tab.cleanup.remove_octave_ghosts` drops a note when another note
sits exactly 12 semitones below it, starts within 50 ms, and is at least
1/0.6 times as loud. Every note is checked against the original input, and
removed notes are returned so the CLI can show them.

**Consequences.** Deliberately quiet octaves (e.g. an octave played softly over
a bass note) are removed too. That's acceptable for the monophonic MVP. Both
thresholds are in `GhostFilterSettings` and were set from synthetic audio, so
they need retuning on real recordings. Other harmonics (the fifth above the
octave) aren't handled yet.

---

## 010 — The optimizer's state includes the hand, not just the position

**Date:** 2026-11-20 · **Status:** Accepted

**Context.** A Viterbi search whose state is only the current string/fret
position can't tell where the fretting hand is after an open string. The plan
said open strings would cost nothing to reach or leave, but then fret 2 → open
→ fret 12 would look free, which is wrong.

**Options.**
1. Position-only state, open strings free: smallest search, but it misjudges
   any phrase with open strings in it.
2. Position-only state, measuring moves from the previous note even when it's
   open: also simple, but it charges hand moves that never happen.
3. State = (position, hand), where the hand is the fret window used since the
   last shift plus the last fretted fret.

**Decision.** Option 3. The window uses the same 4-fret rule as
`tab.metrics`, so the optimizer minimises what is measured. The search stays
exact: tests check it against brute force over every combination on random
phrases.

**Consequences.** More states per note: up to about 10 hand windows per
fretted candidate instead of 1, plus the hands carried across open strings.
That's still tiny for 15–60 s clips. The starting weights (move 1.0, shift 5.0,
string 0.5, low-fret preference 0.1) are hand-picked and get checked against the
baseline in milestone 3.8.

---

## 011 — Remove ringing open strings using timing and loudness, before assignment

**Date:** 2026-11-22 · **Status:** Accepted

**Context.** Four real clips with reference tabs (`scripts/evaluate.py`) had 19
extra notes, 12 of them open strings ringing when a finger lifted off a fretted
note on that string (B3 and G3). The plan expected a filter that runs after
string assignment. But in `no_chord1` the player fretted E4 on the B string
while the assigner put it on the open e string, so checking the assigned string
would miss that case.

**Decision.** `tab.cleanup.remove_ringing_open_strings` runs before assignment.
It drops a note when (1) its pitch is an open string's, (2) a note that *could*
be fretted on that string started earlier and ended at most 50 ms after, or 70
ms before, this note's onset, and (3) this note is at most 0.75 times as loud.
The thresholds come from every open-string note in the four clips: the ringing
ones that fit start within ±60 ms of the fretted note ending, at ≤ 0.70 times
its velocity. The closest real note starts 46 ms after but at 1.41 times the
velocity, and the next closest 81 ms after.

**Consequences.** Precision 76% → 82%, recall unchanged at 97% (6 ringing notes
removed, 0 real notes). 6 ringing notes remain because they look like real
notes on timing and loudness alone. A deliberate quiet pull-off to an open
string would be removed too, though none occurs in these clips. The thresholds
were fitted on the same four clips they're evaluated on, so the gain is
optimistic until new clips confirm it. `--no-ring-filter` turns it off in
`make_tab.py` and `evaluate.py`.
