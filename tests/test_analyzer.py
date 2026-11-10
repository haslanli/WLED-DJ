import numpy as np

from signals import RATE, music_pattern, noise_floor
from wled_dj.analysis.analyzer import Analyzer
from wled_dj.analysis.features import LEVELS, Features
from wled_dj.capture import AudioBlock


def blocks(signal, t0, size=512):
    for start in range(0, len(signal), size):
        yield AudioBlock(signal[start:start + size], t0 + start / RATE, 0)


def analyse(signal, t0=100.0, size=512):
    analyzer = Analyzer()
    out = []
    for block in blocks(signal, t0, size):
        out.extend(analyzer.feed(block))
    return out


def test_one_features_per_hop():
    fs = analyse(noise_floor(.1, -30), t0=100.0)
    assert len(fs) == 10
    assert np.allclose(np.diff([f.t for f in fs]), .01)
    assert abs(fs[-1].t - (100.0 + 4799 / 48000)) < 1e-6
    assert all(isinstance(f, Features) and set(f.levels) == set(LEVELS) for f in fs)


def test_gap_resets_state():
    pattern = music_pattern(2.0)
    analyzer = Analyzer()
    for block in blocks(pattern, 10.0):
        analyzer.feed(block)
    after = []
    for block in blocks(pattern, 13.0):
        after.extend(analyzer.feed(block))
    assert after[0].kick is None and after[0].snare is None and after[0].beat.bpm == 0
    assert abs(after[0].t - (13.0 + 479 / 48000)) < 1e-6


def test_autogain_quiet_and_loud_similar():
    pattern = music_pattern(6.0)
    quiet = analyse(pattern * 10 ** (-24 / 20))
    loud = analyse(pattern)
    tail = slice(-100, None)
    q = np.mean([f.levels["energy"] for f in quiet[tail]])
    l = np.mean([f.levels["energy"] for f in loud[tail]])
    assert q > .1 and abs(q - l) < .1  # sparse pattern: near-silent between hits


def test_silent_flag():
    assert all(f.silent for f in analyse(np.zeros((RATE // 2, 2), np.float32)))
    assert not any(f.silent for f in analyse(noise_floor(.5, -20)))


def test_end_to_end_128bpm_audio():
    beat = 60 / 128
    fs = analyse(music_pattern(12.0))
    assert abs(fs[-1].beat.bpm - 128) <= 2
    kicks = np.arange(0, 12.0, beat)
    kicks = kicks[kicks >= 1.0]
    found = np.array([f.t - 100.0 for f in fs if f.kick is not None])
    hit = [k for k in kicks if np.any((found >= k) & (found <= k + .03))]
    assert len(hit) / len(kicks) >= .9


def test_silence_features():
    f = Features.silence(5.0)
    assert f.t == 5.0 and f.silent and f.kick is None and all(v == 0 for v in f.levels.values())


# ---- with layer separation (the fake stem helper) ----------------------------------------------

from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402

from wled_dj.analysis.analyzer import WARM_HOPS  # noqa: E402
from wled_dj.stems.link import StemLink  # noqa: E402

FAKE_STEMS = str(Path(__file__).parent / "fakes" / "fake_stems.py")
KICK_EVERY = 60 / 128


@pytest.fixture
def stems():
    made = []

    def make(*options):
        link = StemLink([sys.executable, FAKE_STEMS, *options])
        made.append(link)
        link.start()
        deadline = time.monotonic() + 5
        while link.state != "on" and time.monotonic() < deadline:
            time.sleep(.01)
        assert link.state == "on"
        return link

    yield make
    for link in made:
        link.stop()


def feed_all(analyzer, signal, t0=100.0, wait_s=.2, size=480):
    out = []
    for block in blocks(signal, t0, size):
        out.extend(analyzer.feed(block, wait_s))
    return out


def kick_times(features, after):
    return [f.t for f in features if f.kick is not None and f.t >= after]


def test_without_stems_nothing_changes():
    fs = analyse(music_pattern(2.0))
    assert all(f.source == "bands" and f.layers is not None and f.layers.source == "bands" for f in fs)


def test_stems_take_over_after_warm_up(stems):
    link = stems()
    fs = feed_all(Analyzer(link), music_pattern(3.0))
    assert len(fs) == 300
    assert all(f.source == "bands" and f.layers.source == "bands" for f in fs[:WARM_HOPS - 1])
    assert all(f.source == "stems" and f.layers.source == "stems" for f in fs[WARM_HOPS:])
    mix_time = 100.0 + (480 * 201 - 1) / 48000                  # when hop 200 of the mix ends
    assert abs(fs[200].t - (mix_time - link.latency_s)) < 1e-6
    assert all(b.t > a.t for a, b in zip(fs, fs[1:]))


def test_kicks_are_reported_once(stems):
    fs = feed_all(Analyzer(stems()), music_pattern(8.0))
    kicks = kick_times(fs, 100.5)
    expected = len([k for k in np.arange(0, 8.0, KICK_EVERY) if k >= .5])
    assert abs(len(kicks) - expected) <= 1
    assert min(np.diff(kicks)) > .3


def test_late_hops_fall_back_for_events_and_hold_layers(stems):
    # The helper stalls for 200 ms from about 1.83 s, across the kick at 1.875 s.
    link = stems("--stall-after", "630", "--stall-ms", "200")
    fs = feed_all(Analyzer(link), music_pattern(4.0), wait_s=.01)
    assert len(fs) == 400 and link.state == "on"
    assert all(f.source == "stems" for f in fs[WARM_HOPS:])
    held = [i for i in range(WARM_HOPS, 400) if fs[i].layers is fs[i - 1].layers]
    assert len(held) >= 10 and 180 <= held[0] <= 190            # the same layers, hop after hop, while stems are late
    kicks = kick_times(fs, 100.5)
    expected = len([k for k in np.arange(0, 4.0, KICK_EVERY) if k >= .5])
    assert abs(len(kicks) - expected) <= 1 and min(np.diff(kicks)) > .3
    assert any(abs(k - (100.0 + 4 * KICK_EVERY)) < .04 for k in kicks)      # the kick inside the stall was still seen


def test_gap_realigns(stems):
    analyzer = Analyzer(stems())
    signal = music_pattern(6.0)
    before = feed_all(analyzer, signal[:480 * 250])
    after = feed_all(analyzer, signal[480 * 250:], t0=100.0 + 2.5 + .4)     # the host clock jumped 0.4 s
    assert before[-1].source == "stems"
    assert after[0].source == "bands" and after[0].t > before[-1].t
    assert all(f.source == "stems" for f in after[150:])
    assert all(b.t > a.t for a, b in zip(after, after[1:]))


def test_silence_idles_the_helper(stems):
    link = stems()
    analyzer = Analyzer(link)
    music = music_pattern(3.0)
    assert feed_all(analyzer, music)[-1].source == "stems"
    silence = np.zeros((480 * 200, 2), np.float32)
    feed_all(analyzer, silence[:480 * 120], t0=103.0)            # the first second of silence is still sent
    sent = link.frames_sent
    quiet = feed_all(analyzer, silence[480 * 120:], t0=104.2)
    assert link.frames_sent == sent and all(f.source == "bands" for f in quiet)
    back = feed_all(analyzer, music, t0=105.0)
    assert link.frames_sent > sent
    assert all(f.source == "bands" for f in back[:WARM_HOPS - 1]) and all(f.source == "stems" for f in back[WARM_HOPS + 5:])


class RecordingLink:
    """A link that only records what the analyzer asks of it: one index per push, every wait it is given."""
    state, latency_s = "on", .006

    def __init__(self):
        self.pushed, self.waits = 0, []

    def push(self, hop, t_end):
        self.pushed += 1
        return self.pushed - 1

    def collect(self, wait_s):
        self.waits.append(wait_s)
        return []


def test_each_further_hop_in_a_block_may_wait_ten_ms_longer():
    link = RecordingLink()
    analyzer = Analyzer(link)
    signal = music_pattern(1.0)
    analyzer.feed(AudioBlock(signal[:960], 100.0, 0), .010)             # two hops in one block
    analyzer.feed(AudioBlock(signal[960:1440], 100.02, 0), .010)        # one hop
    analyzer.feed(AudioBlock(signal[1440:1700], 100.03, 0), .010)       # no complete hop: nothing to wait for
    assert link.pushed == 3 and link.waits == [pytest.approx(.020), pytest.approx(.010)]


def test_an_overdue_block_waits_only_for_what_is_left():
    link = RecordingLink()
    analyzer = Analyzer(link)
    signal = music_pattern(1.0)
    analyzer.feed(AudioBlock(signal[:960], 100.0, 0), -.004)            # 4 ms overdue, but its second hop has 6 ms left
    analyzer.feed(AudioBlock(signal[960:1440], 100.02, 0), -.004)       # 4 ms overdue: no wait at all
    assert link.waits == [pytest.approx(.006), 0.0]


class ChordsInOther:
    """A link whose stems are silent except for chord stabs in `other`. Hops listed in `withheld` come back
    one collect late, as when the helper stalls."""
    state, latency_s = "on", .006

    def __init__(self, stab_times, seconds, withheld=()):
        from signals import stab
        from wled_dj.stems.link import StemHop
        self.StemHop = StemHop
        self.other = stab(stab_times, seconds, rate=44100).mean(axis=1).astype(np.float32)
        self.withheld, self.times, self.waiting = set(withheld), {}, []

    def push(self, hop, t_end):
        index = len(self.times)
        self.times[index] = t_end
        self.waiting.append(index)
        return index

    def collect(self, wait_s):
        ready = [i for i in self.waiting if i not in self.withheld]
        self.waiting = [i for i in self.waiting if i in self.withheld]
        self.withheld.difference_update(self.waiting)             # released at the next collect
        silent = np.zeros(441, np.float32)
        return [self.StemHop(i, self.times[i] - self.latency_s, silent, silent, silent,
                             self.other[i * 441:(i + 1) * 441], self.other[i * 441:(i + 1) * 441]) for i in sorted(ready)]


def stab_times(features):
    return [round(f.t - 100.0, 2) for f in features if f.stab is not None]


def test_stabs_come_from_the_other_stem():
    link = ChordsInOther([2.0, 2.5, 3.0], 4.0)
    fs = feed_all(Analyzer(link), noise_floor(4.0, -30))
    found = stab_times(fs)
    assert len(found) == 3 and all(0 <= t - s <= .06 for t, s in zip(found, [2.0, 2.5, 3.0]))
    assert all(f.stab.name == "stab" and f.source == "stems" for f in fs if f.stab is not None)


def test_without_stems_there_are_no_stabs():
    assert stab_times(analyse(music_pattern(3.0) + noise_floor(3.0, -30))) == []
    assert Features(t=0.0).stab is None


def test_a_stab_whose_stems_are_late_is_not_reported():
    late = range(198, 212)                                        # the helper stalls across the stab at 2.0 s
    fs = feed_all(Analyzer(ChordsInOther([2.0, 2.5, 3.0], 4.0, withheld=late)), noise_floor(4.0, -30))
    found = stab_times(fs)
    assert len(found) == 2 and all(0 <= t - s <= .06 for t, s in zip(found, [2.5, 3.0]))


# ---- structure, tone and voice on every hop --------------------------------------------------------------

from signals import hats, sine  # noqa: E402
from wled_dj.analysis.structure import SECTIONS  # noqa: E402


class ArrayLink:
    """A link whose stems are whole arrays at 44.1 kHz (missing ones silent). While the newest hop is one of
    `withheld`, nothing comes back, as when the helper stalls; after the stall everything comes at once.
    Setting `state` to "paused" stops it; back to "on" it starts again from hop 0, as a restarted helper does."""
    latency_s = .006

    def __init__(self, seconds, withheld=(), **stems):
        from wled_dj.stems.link import StemHop
        self.StemHop, self.state = StemHop, "on"
        silent = np.zeros(int(seconds * 44100) + 441, np.float32)
        self.stems = {name: silent for name in ("drums", "bass", "vocals", "other")}
        for name, signal in stems.items():
            self.stems[name] = np.concatenate((signal.mean(axis=1).astype(np.float32), silent))[:len(silent)]
        self.withheld, self.times, self.waiting, self.offset, self.total = set(withheld), {}, [], 0, 0

    def push(self, hop, t_end):
        self.total += 1
        if self.state != "on":
            self.times, self.waiting, self.offset = {}, [], self.total       # the next start counts from 0
            return None
        index = self.total - 1 - self.offset
        self.times[index] = t_end
        self.waiting.append(index)
        return index

    def collect(self, wait_s):
        if self.state != "on":
            return []
        if self.waiting and self.waiting[-1] + self.offset in self.withheld:
            return []
        ready, self.waiting = self.waiting, []
        hops = []
        for i in sorted(ready):
            a = (i + self.offset) * 441
            part = {name: x[a:a + 441] for name, x in self.stems.items()}
            hops.append(self.StemHop(i, self.times[i] - self.latency_s, part["drums"], part["bass"], part["vocals"],
                                     part["other"], sum(part.values())))
        return hops


def chord44(seconds, amp=.2):
    return sum(sine(f, seconds, amp / 3, rate=44100) for f in (261.63, 329.63, 392.0))


def events_at(features):
    return [i for i, f in enumerate(features) if f.structure.boundary or f.structure.drop
            or f.structure.vocal_entry or f.structure.vocal_exit]


def test_features_carry_structure_tone_and_voice():
    mix = noise_floor(4.0, -30)
    alone = analyse(mix)
    assert all(f.structure.section in SECTIONS and f.tone.confidence == 0 and f.voice is not None for f in alone)
    fs = feed_all(Analyzer(ArrayLink(4.0, other=chord44(4.0))), mix)
    assert all(f.source == "bands" and f.tone.confidence == 0 for f in fs[:WARM_HOPS - 1])        # the mix: noise has no colour
    assert all(f.source == "stems" and f.tone.confidence > .3 and f.voice.level > .5 for f in fs[WARM_HOPS + 100:])
    assert abs(fs[-1].tone.chord - 45) < 10 and fs[-1].structure.section == "breakdown"         # C major, no drums


def test_late_stems_hold_the_reading():
    late = range(250, 265)
    fs = feed_all(Analyzer(ArrayLink(4.0, withheld=late, other=chord44(4.0))), noise_floor(4.0, -30))
    for i in late:
        assert fs[i].source == "stems" and fs[i].tone is fs[249].tone and fs[i].voice is fs[249].voice
        assert fs[i].structure.section == fs[249].structure.section and fs[i].layers is fs[249].layers


def test_an_event_from_late_stems_arrives_once():
    drums = hats(np.arange(8.0, 12.0, .125), 12.0, rate=44100)                 # drums come in at 8 s
    late = range(800, 845)                                                    # and their stems stall across it
    fs = feed_all(Analyzer(ArrayLink(12.0, withheld=late, other=chord44(12.0), drums=drums)), noise_floor(12.0, -30))
    assert [i for i, f in enumerate(fs) if f.structure.boundary > 0] == [845]
    assert fs[845].structure.boundary == 1.0 and fs[900].layers.drums.present


def test_a_change_of_source_is_not_an_event():
    link = ArrayLink(14.0, other=chord44(14.0))
    analyzer = Analyzer(link)
    mix = np.concatenate((noise_floor(6.5, -40), noise_floor(7.5, -30, seed=2)))      # the mix steps up 10 dB at 6.5 s
    fs = feed_all(analyzer, mix[:480 * 600])
    link.state = "paused"                                                            # at 6.0 s, half a second before it
    fs += feed_all(analyzer, mix[480 * 600:480 * 900], t0=106.0)
    link.state = "on"                                                                # back at 9.0 s
    fs += feed_all(analyzer, mix[480 * 900:], t0=109.0)
    sources = [f.source for f in fs]
    changes = [i for i in range(1, len(fs)) if sources[i] != sources[i - 1]]
    assert changes == [99, 600, 999]             # warmed up; paused; warmed up again after coming back
    for change in changes:
        assert not [i for i in events_at(fs) if change <= i < change + 200]
        assert fs[change].structure.section == fs[change - 1].structure.section       # no jump on the change itself
    assert all(f.structure.section in SECTIONS for f in fs)
    assert events_at(feed_all(Analyzer(), mix))                                      # that step is a boundary to the mix alone


def test_a_gap_starts_the_trackers_again():
    analyzer = Analyzer()
    pattern = music_pattern(14.0)
    before = feed_all(analyzer, pattern[:480 * 800])
    assert before[-1].structure.since_s > 4
    after = feed_all(analyzer, pattern[480 * 800:], t0=100.0 + 8.0 + .4)        # the host clock jumped 0.4 s
    assert after[5].structure.since_s < .2 and all(f.structure.since_s < 2.6 for f in after[:250])     # started again: a groove needs 2 s
    assert not any(f.structure.drop or f.structure.boundary for f in after[:400])
