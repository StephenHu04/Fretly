# Sample recordings

Local guitar recordings for manual testing. Audio files in this folder are
git-ignored to keep the repository small; only this README is committed.

## Recording a test clip

For the MVP, a good sample is:

- A single clean (undistorted) guitar, ideally DI or a close mic.
- Standard tuning (E A D G B E).
- 15–60 seconds long.
- A simple single-note melody first; a few simple chords later.
- WAV or FLAC (lossless) at 44.1 or 48 kHz.

Suggested name: `clean_melody_01.wav`. When a recording has a known reference
(a MIDI file or a tab you played from), keep it next to the audio with the same
base name, e.g. `clean_melody_01.tab.txt`. Phase 6 evaluation will use these.

Check a clip with:

```bash
python scripts/inspect_audio.py ../samples/clean_melody_01.wav
```

(run from `backend/` with the virtual environment active).
