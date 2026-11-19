# Phase 3 plan: tablature optimizer

Turns `NoteEvent`s (decision 005) into string/fret positions and plain-text tab.
This is the part of Fretly that isn't an off-the-shelf model, so it gets its
own package, tests and metrics. The checklist and acceptance criteria are in
[PROJECT_SPEC.md](../PROJECT_SPEC.md#phase-3--custom-tablature-optimizer-weeks-23);
this document is the order of work and the design behind it.

**Status:** 3.1 and 3.2 done (2026-11-18); 3.3 next. Each milestone below is
roughly one commit, and each leaves the test suite green.

## Scope

In: standard tuning, six strings, 24 frets (configurable), monophonic lines
first, then simple chords if the monophonic version is stable.

Out (for now): rhythm quantization, bends/slides/hammer-ons, alternate tunings,
learned cost weights (Phase 6 stretch goal).

## Where the code goes

A new `backend/tab/` package, separate from `pipeline/`. It's pure Python with
no audio, TensorFlow or FFmpeg imports, so it can be tested and benchmarked on
ground-truth notes as well as model output.

```text
backend/tab/
  fretboard.py   Tuning, Position, candidate generation
  cleanup.py     note filters (octave ghosts first)
  assign.py      TabNote, Assignment, the baseline assigner
  optimizer.py   dynamic programming (Viterbi) assigner
  metrics.py     playability metrics
  render.py      plain-text tab
backend/scripts/tab_from_midi.py   MIDI or audio -> tab + metrics
```

Core types:

```python
Position(string: int, fret: int)            # string 1 = high E ... 6 = low E
TabNote(note: NoteEvent, position: Position | None)   # None = unplayable
Assignment(notes: list[TabNote], unplayable: list[NoteEvent])
```

Both assigners have the same signature, so the metrics and renderer don't care
which one ran:

```python
def assign(notes: list[NoteEvent], fretboard: Fretboard) -> Assignment: ...
```

## Milestones

### 3.1 Fretboard model and candidates ✅
- `Tuning` (open-string MIDI pitches; standard = 64, 59, 55, 50, 45, 40) and
  `Fretboard(tuning, max_fret=24)`.
- `candidates(pitch)` returns every `Position` with `0 <= pitch - open <= max_fret`,
  ordered by fret.
- Tests: open strings, a pitch with exactly one position (E2, F2), a pitch with
  six positions (E4), below range (< 40), above range (> 88), custom `max_fret`.

### 3.2 Octave-ghost filter ✅
Basic Pitch emitted second-harmonic notes in Phase 2: an octave above a real
note, same onset, about half the velocity.
- Rule: drop note B when there's a note A with `B.pitch == A.pitch + 12`,
  onsets within 50 ms, and `B.velocity <= 0.6 * A.velocity`. Thresholds live in
  a settings dataclass so they can be tuned on real recordings.
- Returns the kept notes *and* the removed ones, so the CLI can show what was
  filtered.
- Tests: ghost removed, a loud real octave kept, octave with different onset
  kept, the Phase 2 synthetic phrase loses exactly its ghosts.
- Known risk: deliberate quiet octaves get removed. Acceptable for monophonic
  MVP clips; recorded in decision 009.

### 3.3 Baseline assigner
- "Lowest fret wins" for each note independently (ties go to the higher string).
  This is what a naive tool would do and is the yardstick for the optimizer.
- Notes with no candidates go to `unplayable`, never silently dropped or
  transposed (spec acceptance criterion).
- Tests: every assigned position reproduces the note's pitch, unplayable
  notes are reported, empty input gives an empty assignment.

### 3.4 Playability metrics
Defined before the optimizer so it's judged on numbers fixed in advance:

| Metric | Definition |
|---|---|
| Position shifts / min | the hand sits on a 4-fret window; a fretted note outside it moves the window and counts as a shift |
| Max stretch | largest fret distance between consecutive fretted notes less than 250 ms apart |
| Fret travel | total absolute fret change between consecutive fretted notes |
| String changes | consecutive notes on different strings |
| Unplayable notes | count, reported separately |

Open strings don't move the hand, so they're skipped when measuring shifts,
stretch and travel.

### 3.5 Dynamic programming optimizer
Viterbi over the candidate lattice: one column per note, one state per
candidate position (at most six), so it runs in O(notes × 6²).

Starting cost function, with hand-picked weights in a settings dataclass:

```text
transition(a -> b) = w_move   * |fret_b - fret_a|          (fretted -> fretted)
                   + w_shift  * [leaves the 4-fret window]
                   + w_string * |string_b - string_a|
state(b)           = w_height * fret_b                      (mild preference for low positions)
```

- Open strings cost nothing to reach or leave (the hand stays put).
- Unplayable notes are taken out of the chain; their neighbours connect directly.
- Tests: a scale run stays in one position instead of sliding along one string,
  a repeated note keeps its position, a large jump takes one shift instead of
  several, ties resolve deterministically, it never picks an invalid candidate.

### 3.6 Plain-text tab
- Six lines (high E at the top), one column per distinct onset, fret numbers
  padded to equal width. Rhythm isn't shown yet.
- Golden-file tests for a short riff, two-digit frets and an empty input.

### 3.7 Command-line tool
`python scripts/tab_from_midi.py clip.mid [--method baseline|optimized] [--json]`
prints the tab, the metrics, filtered ghosts and unplayable notes. An `--audio`
option runs the Phase 2 pipeline first.

### 3.8 Baseline vs optimizer
- A small set of fixed test phrases (scale runs, a riff with position shifts,
  repeated notes, a wide-range melody) written as `NoteEvent` lists in the tests.
- A script prints both assigners' metrics side by side. The results go in the
  README, and the chosen weights go in a decision-log entry.
- Acceptance: the optimizer is no worse on every phrase and better on shifts
  or travel overall.

### 3.9 Simple chords (if 3.1–3.8 are stable)
- Group notes whose onsets fall within 30 ms.
- Chord states are combinations of candidates on distinct strings with a fret
  span of at most 4 (open strings excluded from the span). Six strings keep this
  small enough to enumerate.
- The same Viterbi runs over chord states, with movement measured from the
  shape's lowest fretted note.
- Tests: open E major, a barre shape, an unplayable cluster reported, a chord
  followed by a single note.

## Open questions
- **Hand position as state:** moving across open strings loses track of where
  the hand is. If tests show this causes bad choices, extend the state to
  (position, hand window) at a cost of more states per note.
- **Weights:** hand-tuned in 3.5 and checked in 3.8. Learning them from
  GuitarSet fingerings is a Phase 6 stretch goal.
- **Ghost thresholds:** set from synthetic audio. Retune them once real
  recordings are in `samples/`.
