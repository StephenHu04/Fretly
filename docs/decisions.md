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
