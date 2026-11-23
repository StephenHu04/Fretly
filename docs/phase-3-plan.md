# Phase 3 plan: tablature optimizer

Turns `NoteEvent`s (decision 005) into string/fret positions and plain-text tab.
This is the part of Fretly that isn't an off-the-shelf model, so it gets its
own package, tests and metrics. The checklist and acceptance criteria are in
[PROJECT_SPEC.md](../PROJECT_SPEC.md#phase-3--custom-tablature-optimizer-weeks-23);
this document is the order of work and the design behind it.

**Status:** 3.1–3.7 done (2026-11-21); 3.8 next. Each milestone below is
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
backend/scripts/make_tab.py        audio or MIDI -> tab + metrics
```

Core types:

```python
Position(string: int, fret: int)            # string 1 = high E ... 6 = low E
TabNote(note: NoteEvent, position: Position)
Assignment(notes: list[TabNote], unplayable: list[NoteEvent])   # unplayable kept separately
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

### 3.3 Baseline assigner ✅
- "Lowest fret wins" for each note independently (ties go to the higher string).
  This is what a naive tool would do and is the yardstick for the optimizer.
- Notes with no candidates go to `unplayable`, never silently dropped or
  transposed (spec acceptance criterion).
- Tests: every assigned position reproduces the note's pitch, unplayable
  notes are reported, empty input gives an empty assignment.

### 3.4 Playability metrics ✅
Defined before the optimizer so it's judged on numbers fixed in advance:

| Metric | Definition |
|---|---|
| Position shifts / min | a shift happens when a fretted note can't fit in one 4-fret window with every fretted note played since the last shift |
| Max stretch | largest fret distance between consecutive fretted notes less than 250 ms apart |
| Fret travel | total absolute fret change between consecutive fretted notes |
| String changes | consecutive notes on different strings |
| Unplayable notes | count, reported separately |

Open strings don't move the hand, so they're skipped when measuring shifts,
stretch and travel.

### 3.5 Dynamic programming optimizer ✅
Viterbi over the candidate lattice: one column per note, one state per
(candidate position, hand) pair. There are at most six positions per note and a
handful of hand windows each, so it stays linear in the number of notes.

Starting cost function, with hand-picked weights in a settings dataclass:

```text
transition(a -> b) = w_move   * |fret_b - fret_a|          (fretted -> fretted)
                   + w_shift  * [leaves the 4-fret window]
                   + w_string * |string_b - string_a|
state(b)           = w_height * fret_b                      (mild preference for low positions)
```

- Open strings don't move the hand. The state is (position, hand), where the
  hand remembers its fret window and last fretted note, so an open string
  can't hide a jump (decision 010).
- Unplayable notes are taken out of the chain; their neighbours connect directly.
- Tests: a scale run stays in one position instead of sliding along one string,
  a repeated note keeps its position, a large jump takes one shift instead of
  several, ties resolve deterministically, it never picks an invalid candidate.

### 3.6 Plain-text tab ✅
`tab/render.py`: `render_tab(assignment, fretboard, width=80) -> str`. It takes
an `Assignment` from either assigner and returns plain text in the usual
guitar-tab layout, the same style as the reference tabs in `samples/`.

**What `no_chord1` will look like** (optimizer output from the model's notes):

```text
e|-0-----0-----0-----0---4-|
B|---4-0---4-0---4-0---0-0-|
G|-------------------------|
D|-------------------------|
A|-------------------------|
E|-------------------------|
```

and, for comparison, the reference tab in the same layout:

```text
e|-0-----0-----0-------2-4-|
B|---4-0---4-0---4-0-5-----|
G|-------------------------|
D|-------------------------|
A|-------------------------|
E|-------------------------|
```

The differences are the ones found in the real-recording check: the extra
open-B notes and the missed F#4, and E4 placed on the open e string rather
than fret 5 on B.

Layout rules:

| Rule | Detail |
|---|---|
| Lines | One per string, high e at the top. Each starts with its label and `\|` and ends with `-\|` |
| Labels | Note name of each open string, from the tuning (`e B G D A E`; drop D gives `e B G D A D`). The top string is always lower-case, the usual tab convention, so high `e` is never confused with low `E`. Sharps widen every label (`d# A# F# ...`) |
| Columns | One per onset. Notes starting within 30 ms of each other share a column (double stops now, chords in 3.9) |
| Same-string clash | If two notes in one column land on the same string (possible before 3.9), the later one moves to the next column, so nothing is overwritten |
| Spacing | Each column is `-` plus the fret number, padded with `-` to the widest number in that column, so two-digit frets stay aligned (example below) |
| Rhythm | Not shown: columns are evenly spaced. Rhythm needs tempo and quantization, which are out of scope for Phase 3 |
| Wrapping | Lines longer than `width` (default 80) wrap into blocks separated by a blank line; each block repeats the labels. Columns are never split |
| Unplayable notes | Listed under the tab as `# unplayable: C2 at 1.23 s`, so nothing disappears silently |
| Empty input | Six lines like `e\|-\|`, with no columns |

Two-digit frets and a double stop:

```text
e|-12-10----8--|
B|-------13-10-|
G|-------------|
D|-------------|
A|-------------|
E|-------------|
```

Tests compare the output with exact expected strings ("golden" tests):
the `no_chord1` example above, two-digit alignment, a double stop in one
column, a same-string clash, wrapping at a small width, unplayable notes,
drop-D labels and empty input.

Later, not in 3.6: spacing proportional to time between notes, bar lines once
tempo is known, and technique marks (`h`, `p`, `/`).

### 3.7 Command-line tool ✅
`python scripts/make_tab.py <audio or .mid> [--method optimized|baseline|both] [--json]`
prints the tab, the metrics, filtered ghosts and unplayable notes. Audio files
run the Phase 2 pipeline first (shared with `transcribe_audio.py` through
`pipeline/run.py`); `.mid` files skip it. Renamed from `tab_from_midi.py`,
since it takes audio too and detects the input type from the extension.

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
- ~~**Hand position as state.**~~ Resolved in 3.5: the state includes the hand
  (decision 010).
- **Weights:** hand-tuned in 3.5 and checked in 3.8. Learning them from
  GuitarSet fingerings is a Phase 6 stretch goal.
- **Ghost thresholds:** set from synthetic audio. On four real clips the filter
  removed 3 real ghosts (G#4 above G#3) and no real notes. Ghosts also appear
  an octave *below* loud notes (2 quiet G#2s in `no_chord4`); the filter
  doesn't handle those yet.
- **Ringing open strings:** filter added 2026-11-22
  (`tab.cleanup.remove_ringing_open_strings`, decision 011). It runs before
  assignment after all, using every string a note *could* be on. It removes 6
  of the 12 ringing notes in the four clips and no real notes. The other 6
  are as loud as the note before them or don't start where it ends; telling
  those apart from real notes may need the audio itself, not just note events.
- **Octave-below ghosts:** 2 quiet G#2s under G#3 in `no_chord4`. Extending
  the ghost filter downwards is a small change; measure with `evaluate.py`.
