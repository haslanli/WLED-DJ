"""An example look: the whole strip glows with the song's energy and jumps on every kick.

Copy this file into your looks folder (~/Library/Application Support/WLED DJ/looks/) and restart WLED DJ, or
start the engine with --looks examples/looks. It then appears in the studio's Look picker as "Glow"."""
import numpy as np

from wled_dj.looks.canvas import Canvas, mass, pulse
from wled_dj.looks.colour import RING


class Glow:
    id = "glow"          # saved in the settings: keep it stable
    name = "Glow"        # shown in the studio

    def __init__(self, pixels: int):
        self.canvas = Canvas(pixels)
        self.kick_at = -1e9

    def render(self, f, lead: float) -> np.ndarray:
        """One frame for the Features `f` (one arrives every 10 ms). Returns (pixels, 3) float32 in 0-1."""
        if f.structure is None or f.tone is None or f.silent:
            self.canvas.clear()                     # nothing known yet, or silence: dark
            return self.canvas.values()
        if f.kick is not None:
            self.kick_at = f.t
        place = f.tone.angle / 360                  # the song's colour, as a place on the palette ring
        self.canvas.fill(*RING.colour(place, .1 + .3 * f.structure.energy))
        hit = pulse(f.t - self.kick_at, .05, .15)   # 1 on a kick, held 50 ms, then dying away
        self.canvas.paint(mass(self.canvas.r, .2, .1) * hit, *RING.colour(place, .8, chroma=.5))
        return self.canvas.values()
