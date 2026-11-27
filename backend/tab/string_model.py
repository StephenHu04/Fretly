"""Tell which string a note was played on from its sound (decision 023).

The same pitch sounds different on different strings: a thick string played
high up the neck has weaker upper harmonics and more inharmonicity than a thin
one played low. Those differences depend on the guitar, strings and microphone,
so the model is fitted to a short calibration recording of the player's own
guitar (every string, open to fret 12), saved as a small JSON profile.

Per note it measures level-independent timbre features, fits each string's
features as a straight line in pitch, and scores a new note by Gaussian
likelihood under each string that can play its pitch. The optimizer adds
``-log p(string)`` to its cost, so the sound settles positions the notes leave
open and hand movement still decides the rest.

numpy only; audio is mono at ``SAMPLE_RATE`` (what the FFmpeg stage writes).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.midi_parser import NoteEvent
from tab.fretboard import Fretboard

SAMPLE_RATE = 22050
HARMONICS = 15
FEATURE_COUNT = 1 + HARMONICS + 6  # pitch, harmonic levels, 6 summary features
PROFILE_VERSION = 1

_FFT_SIZE = 16384
_STIFFNESS = np.concatenate([[0.0], np.logspace(-5, -3, 18)])  # inharmonicity coefficients tried
_DETUNE_CENTS = np.linspace(-35, 35, 15)
_K = np.arange(1, HARMONICS + 1)

# Model settings, fixed in the trial before scoring (decision 023).
RIDGE = 0.1  # on each string's pitch line
SHRINK = 0.5  # pull of the shared covariance toward its diagonal
JITTER = 0.05  # added to the covariance diagonal
MIN_NOTES_PER_STRING = 3


def _hz(pitch: float) -> float:
    return 440.0 * 2 ** ((pitch - 69) / 12)


def note_features(samples: np.ndarray, note: NoteEvent) -> np.ndarray:
    """Pitch plus timbre features of one note; NaN where the note is too short to measure.

    Uses 60-250 ms of sound starting 20 ms after the onset. All levels are
    relative, so how hard the note was played matters little.
    """
    out = np.full(FEATURE_COUNT, np.nan)
    out[0] = note.pitch
    start = int((note.start_s + 0.02) * SAMPLE_RATE)
    length = int(np.clip(note.end_s - note.start_s - 0.02, 0.06, 0.25) * SAMPLE_RATE)
    seg = samples[max(0, start) : max(0, start) + length]
    if len(seg) < 600 or not seg.any():
        return out
    mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), _FFT_SIZE)) + 1e-9
    level = 20 * np.log10(mag)
    # Find the harmonic series (allowing slight detuning and string stiffness)
    # that best fits the spectrum.
    f0s = _hz(note.pitch) * 2 ** (_DETUNE_CENTS / 1200)
    freqs = f0s[:, None, None] * _K * np.sqrt(1 + _STIFFNESS[None, :, None] * _K**2)
    bins = np.round(freqs * _FFT_SIZE / SAMPLE_RATE).astype(int)
    inside = bins < len(mag)
    clipped = np.minimum(bins, len(mag) - 1)
    score = np.where(inside, level[clipped], level.min()).sum(-1)
    i, j = np.unravel_index(score.argmax(), score.shape)
    harm = np.where(inside[i, j], level[clipped[i, j]], np.nan)
    rel = harm - np.nanmax(harm)
    hz = np.arange(len(mag)) * SAMPLE_RATE / _FFT_SIZE
    power = mag**2
    centroid = (hz * power).sum() / power.sum() / _hz(note.pitch)
    half = len(seg) // 2
    decay = _db((seg[half:] ** 2).mean()) - _db((seg[:half] ** 2).mean())
    before = samples[max(0, start - int(0.04 * SAMPLE_RATE)) : max(0, start) + 1]
    attack = _db((seg[: SAMPLE_RATE // 50] ** 2).mean()) - _db((before**2).mean() if len(before) else 0.0)
    odd_even = np.nanmean(rel[::2]) - np.nanmean(rel[1::2])
    upper = np.nanmean(rel[5:]) if np.isfinite(rel[5:]).any() else np.nan
    out[1:] = np.concatenate([rel, [np.log10(_STIFFNESS[j] + 1e-6), centroid, decay, attack, odd_even, upper]])
    return out


def _db(power: float) -> float:
    return float(10 * np.log10(power + 1e-12))


def cents_off(samples: np.ndarray, note: NoteEvent) -> float | None:
    """How far the note is from its equal-tempered pitch, in cents (None if too short)."""
    seg = samples[int((note.start_s + 0.05) * SAMPLE_RATE) : int(min(note.end_s, note.start_s + 0.8) * SAMPLE_RATE)]
    if len(seg) < 1500:
        return None
    size = 1 << 18
    mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), size))
    cents = np.arange(-80, 81)
    f0 = _hz(note.pitch) * 2 ** (cents / 1200)
    score = sum(np.log(mag[np.minimum(np.round(k * f0 * size / SAMPLE_RATE).astype(int), len(mag) - 1)] + 1e-9)
                for k in (1, 2, 3, 4))
    return float(cents[int(np.argmax(score))])


def _log_softmax(x: np.ndarray) -> np.ndarray:
    top = x.max(axis=1, keepdims=True)
    return x - top - np.log(np.exp(x - top).sum(axis=1, keepdims=True))


@dataclass(frozen=True)
class StringModel:
    """Per-string Gaussian model of note features, fitted to one guitar."""

    mean: np.ndarray  # (FEATURE_COUNT,) for standardising
    scale: np.ndarray
    coef: np.ndarray  # (strings, 2, FEATURE_COUNT - 1): intercept and slope in pitch
    inv_cov: np.ndarray  # (FEATURE_COUNT - 1, FEATURE_COUNT - 1), shared by the strings
    temperature: float = 1.0  # scales log-likelihoods so probabilities are calibrated

    @property
    def num_strings(self) -> int:
        return len(self.coef)

    @classmethod
    def fit(cls, features: np.ndarray, strings: Sequence[int], num_strings: int = 6) -> StringModel:
        """Fit to labelled notes (``strings`` numbered 1..num_strings). Temperature stays 1."""
        x, y = np.asarray(features, float), np.asarray(strings)
        if x.ndim != 2 or x.shape[1] != FEATURE_COUNT or len(x) != len(y):
            raise ValueError("features must be (notes, FEATURE_COUNT) with one string per note")
        for s in range(1, num_strings + 1):
            if (y == s).sum() < MIN_NOTES_PER_STRING:
                raise ValueError(f"string {s} needs at least {MIN_NOTES_PER_STRING} notes")
        mean, scale = np.nanmean(x, 0), np.nanstd(x, 0) + 1e-6
        z = _standardise(x, mean, scale)
        coef, residuals = [], []
        for s in range(1, num_strings + 1):
            rows = y == s
            a = _design(x[rows])
            c = np.linalg.solve(a.T @ a + RIDGE * np.eye(2), a.T @ z[rows])
            coef.append(c)
            residuals.append(z[rows] - a @ c)
        r = np.vstack(residuals)
        cov = r.T @ r / len(r)
        cov = (1 - SHRINK) * cov + SHRINK * np.diag(np.diag(cov)) + JITTER * np.eye(len(cov))
        return cls(mean, scale, np.array(coef), np.linalg.inv(cov))

    def log_likelihood(self, features: np.ndarray, fretboard: Fretboard) -> np.ndarray:
        """(notes, strings) log-likelihoods, -inf on strings that can't play the pitch."""
        x = np.atleast_2d(np.asarray(features, float))
        z, a = _standardise(x, self.mean, self.scale), _design(x)
        out = np.full((len(x), self.num_strings), -np.inf)
        for s in range(self.num_strings):
            d = z - a @ self.coef[s]
            out[:, s] = -0.5 * np.einsum("ij,jk,ik->i", d, self.inv_cov, d)
        for i, pitch in enumerate(x[:, 0]):
            legal = {p.string for p in fretboard.candidates(int(pitch))}
            out[i, [s - 1 for s in range(1, self.num_strings + 1) if s not in legal]] = -np.inf
        return out

    def probabilities(self, features: np.ndarray, fretboard: Fretboard) -> np.ndarray:
        """(notes, strings) probabilities; column 0 is string 1. Rows of unplayable notes are NaN."""
        scaled = self.log_likelihood(features, fretboard) * self.temperature
        with np.errstate(invalid="ignore"):
            top = scaled.max(axis=1, keepdims=True)
            p = np.exp(scaled - top)
            return p / p.sum(axis=1, keepdims=True)

    def with_temperature(self, temperature: float) -> StringModel:
        return StringModel(self.mean, self.scale, self.coef, self.inv_cov, temperature)

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": _list(self.mean), "scale": _list(self.scale), "coef": self.coef.tolist(),
            "inv_cov": self.inv_cov.tolist(), "temperature": self.temperature,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> StringModel:
        mean = np.array([np.nan if v is None else v for v in d["mean"]], float)  # type: ignore[union-attr]
        scale = np.array([np.nan if v is None else v for v in d["scale"]], float)  # type: ignore[union-attr]
        return cls(mean, scale, np.array(d["coef"], float), np.array(d["inv_cov"], float), float(d["temperature"]))  # type: ignore[arg-type]


def _list(a: np.ndarray) -> list[float | None]:
    return [None if not np.isfinite(v) else float(v) for v in a]


def _standardise(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.nan_to_num((x - mean) / scale)[:, 1:]


def _design(x: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(x)), (x[:, 0] - 57) / 12]


@dataclass(frozen=True)
class CrossCheck:
    temperature: float
    accuracy: float  # notes whose most likely string was right, each fret left out in turn
    chance: float  # the same for guessing among the legal strings
    per_string: dict[int, float]


def fit_profile(
    features: np.ndarray, strings: Sequence[int], frets: Sequence[int], fretboard: Fretboard
) -> tuple[StringModel, CrossCheck]:
    """Fit the model and choose its temperature by leaving out one fret at a time.

    Returns the model fitted to every note and how well it did on left-out frets.
    """
    x, y, f = np.asarray(features, float), np.asarray(strings), np.asarray(frets)
    held = np.full((len(y), fretboard.num_strings), -np.inf)
    for fret in np.unique(f):
        out = f == fret
        held[out] = StringModel.fit(x[~out], y[~out], fretboard.num_strings).log_likelihood(x[out], fretboard)
    rows = np.arange(len(y))

    def loss(t: float) -> float:
        return float(-_log_softmax(np.where(np.isfinite(held), held * t, -1e300))[rows, y - 1].mean())

    temps = np.linspace(0.01, 1.0, 991)
    temperature = float(temps[int(np.argmin([loss(t) for t in temps]))])
    hit = held.argmax(1) == y - 1
    chance = float(np.mean([1 / len(fretboard.candidates(int(p))) for p in x[:, 0]]))
    per_string = {int(s): float(hit[y == s].mean()) for s in np.unique(y)}
    model = StringModel.fit(x, y, fretboard.num_strings).with_temperature(temperature)
    return model, CrossCheck(temperature, float(hit.mean()), chance, per_string)


@dataclass(frozen=True)
class Profile:
    """A fitted model plus where it came from, saved as JSON."""

    model: StringModel
    tuning: tuple[int, ...]
    info: Mapping[str, object]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": PROFILE_VERSION, "tuning": list(self.tuning), "info": dict(self.info), "model": self.model.to_dict()}
        path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Profile:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != PROFILE_VERSION:
            raise ValueError(f"{path.name} is not a version {PROFILE_VERSION} Fretly profile")
        return cls(StringModel.from_dict(data["model"]), tuple(data["tuning"]), data.get("info", {}))

    def string_probabilities(
        self, samples: np.ndarray, notes: Iterable[NoteEvent], fretboard: Fretboard
    ) -> dict[NoteEvent, dict[int, float]]:
        """For each playable note, the probability of each string that can play it."""
        if tuple(fretboard.tuning) != self.tuning:
            raise ValueError("the profile was recorded in a different tuning")
        playable = [n for n in notes if fretboard.is_playable(n.pitch)]
        if not playable:
            return {}
        p = self.model.probabilities(np.array([note_features(samples, n) for n in playable]), fretboard)
        return {n: {pos.string: float(p[i, pos.string - 1]) for pos in fretboard.candidates(n.pitch)}
                for i, n in enumerate(playable)}


def read_samples(path: Path) -> np.ndarray:
    """Mono float samples of a WAV written by the FFmpeg stage (``SAMPLE_RATE``)."""
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {rate}")
    return data.mean(axis=1)


def calibration_notes(notes: Iterable[NoteEvent], open_pitch: int, frets: int) -> tuple[dict[int, NoteEvent], list[int]]:
    """The loudest detected note at each expected pitch (open to ``frets``), and frets not found."""
    found: dict[int, NoteEvent] = {}
    by_pitch: dict[int, NoteEvent] = {}
    for n in notes:
        if n.pitch not in by_pitch or n.velocity > by_pitch[n.pitch].velocity:
            by_pitch[n.pitch] = n
    for fret in range(frets + 1):
        if open_pitch + fret in by_pitch:
            found[fret] = by_pitch[open_pitch + fret]
    return found, [f for f in range(frets + 1) if f not in found]
