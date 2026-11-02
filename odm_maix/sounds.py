"""Synthesized placeholder sounds for expressions (numpy only).

A "quack" is a buzzy sawtooth with a falling pitch through a nasal formant;
chirps are pitch sweeps. Good enough to hear which mood fired.
"""

from __future__ import annotations

import numpy as np

RATE = 48000

# name -> list of (start_hz, end_hz, seconds, kind); a 30 ms gap separates notes
SOUNDS = {
    "happy": [(700, 900, 0.09, "quack"), (800, 1000, 0.09, "quack")],
    "curious": [(500, 1100, 0.25, "chirp")],
    "startled": [(1200, 1500, 0.12, "quack")],
    "sleepy": [(500, 250, 0.6, "chirp")],
    "hello": [(600, 750, 0.12, "quack"), (650, 800, 0.12, "quack"), (700, 900, 0.15, "quack")],
    "ok": [(700, 600, 0.1, "quack")],
}


def _note(f0, f1, dur, kind):
    n = int(RATE * dur)
    t = np.arange(n) / RATE
    freq = np.linspace(f0, f1, n) if kind == "chirp" else np.geomspace(f0, f0 * 0.75, n) * (f1 / f0) ** 0.1
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    if kind == "quack":
        saw = 2 * ((phase / (2 * np.pi)) % 1.0) - 1
        # crude nasal formant: mix in a resonance near 1.2 kHz
        x = 0.6 * saw + 0.4 * np.sin(2 * np.pi * 1200 * t) * (saw > 0)
    else:
        x = np.sin(phase)
    env = np.minimum(1.0, t / 0.01) * np.minimum(1.0, (dur - t) / 0.03)
    return x * np.clip(env, 0, 1)


def synth(name: str, volume: float = 0.6) -> np.ndarray:
    """int16 PCM at RATE Hz."""
    gap = np.zeros(int(RATE * 0.03))
    parts = []
    for f0, f1, dur, kind in SOUNDS[name]:
        parts += [_note(f0, f1, dur, kind), gap]
    x = np.concatenate(parts)
    return (np.clip(x * volume, -1, 1) * 32767).astype(np.int16)
