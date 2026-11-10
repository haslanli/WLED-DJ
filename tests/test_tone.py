"""Tonal colour: chords in, an angle on the circle of fifths out."""
import math

import numpy as np

from wled_dj.analysis.tone import Tone, ToneTracker

RATE, HOP = 44100, 441
C4, E4, G4 = 261.63, 329.63, 392.00
G3, B3, D4 = 196.00, 246.94, 293.66
C_MAJOR, G_MAJOR = 45.0, 75.0           # by hand: C, E, G sit at 0, 120 and 30 degrees; a fifth up is 30 degrees on


def chord(freqs, seconds, amp=.2):
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp / len(freqs) * sum(np.sin(2 * np.pi * f * t) for f in freqs)).astype(np.float32)


def noise(seconds, like):
    rng = np.random.default_rng(11)
    x = rng.standard_normal(int(seconds * RATE)).astype(np.float32)
    return x * float(np.sqrt(np.mean(like ** 2)) / np.sqrt(np.mean(x ** 2)))


def listen(tracker, signal):
    """Feed the signal hop by hop; returns (Tone, chord_step) after every hop."""
    out = []
    for i in range(len(signal) // HOP):
        tone = tracker.step(signal[i * HOP:(i + 1) * HOP])
        out.append((tone, tracker.chord_step))
    return out


def apart(a, b):
    return abs((a - b + 180) % 360 - 180)


def test_a_major_chord_has_its_place():
    c = listen(ToneTracker(RATE), chord((C4, E4, G4), 3.0))[-1][0]
    g = listen(ToneTracker(RATE), chord((G3, B3, D4), 3.0))[-1][0]
    assert isinstance(c, Tone) and apart(c.chord, C_MAJOR) < 10 and apart(c.angle, C_MAJOR) < 10
    assert apart(g.chord, G_MAJOR) < 10 and c.confidence > .3 and g.confidence > .3


def test_octaves_do_not_matter():
    up = listen(ToneTracker(RATE), chord((2 * C4, 2 * E4, 2 * G4), 3.0))[-1][0]
    assert apart(up.chord, C_MAJOR) < 10


def test_the_song_colour_moves_slowly():
    heard = listen(ToneTracker(RATE), np.concatenate((chord((C4, E4, G4), 6.0), chord((G3, B3, D4), 2.0))))
    tone, step = heard[-1]
    assert apart(tone.chord, G_MAJOR) < 10 and tone.angle < 60 and apart(tone.angle, C_MAJOR) < 15
    assert max(s for _, s in heard[600:]) > 20            # the chord colour moved while the chord changed
    assert heard[590][1] < 2                              # and stood still while it was held


def test_noise_holds_the_colour():
    c = chord((C4, E4, G4), 3.0)
    tracker = ToneTracker(RATE)
    before = listen(tracker, c)[-1][0]
    for filler in (noise(3.0, like=c), np.zeros(3 * RATE, np.float32)):
        heard = listen(tracker, filler)
        after = heard[-1][0]
        assert after.confidence < .1 and all(t.confidence < .1 for t, _ in heard[100:])
        assert apart(after.chord, before.chord) < 1 and apart(after.angle, before.angle) < 1
        assert all(math.isfinite(v) for t, _ in heard for v in (t.angle, t.chord, t.confidence))
    assert listen(ToneTracker(RATE), np.zeros(RATE, np.float32))[-1][0] == Tone()


def test_a_vague_colour_does_not_count_as_a_chord_change():
    semitones = [C4 * 2 ** (k / 12) for k in range(12)]

    def cluster(heavy, seconds=2.0):
        """All twelve notes at once, one of them louder: a profile that barely points anywhere."""
        t = np.arange(int(seconds * RATE)) / RATE
        return (.02 * sum((1.8 if k == heavy else 1.0) * np.sin(2 * np.pi * f * t) for k, f in enumerate(semitones))).astype(np.float32)

    tracker = ToneTracker(RATE)
    heard = listen(tracker, np.concatenate([cluster(0), cluster(6), cluster(0), cluster(6)]))
    sure = [tone.confidence for tone, _ in heard[100:]]
    assert .05 < max(sure) < .2 and max(step for _, step in heard) == 0      # the colour flips about; nobody should act on it
