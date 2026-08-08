"""Animation.

Pure maths -- no pygame, no drawing. Values here are driven by a delta time and
read by whatever is painting the frame.

The distinction that matters: the old "animation" moved a text label from one
state to another between frames. Nothing was interpolated, so nothing appeared
to move; the label was simply somewhere else. Everything in this module carries
a *current* value and a *target*, and closes the gap over time along a curve.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional, Tuple

Point = Tuple[float, float]
Easing = Callable[[float], float]


# ----------------------------------------------------------------------
# Easing curves
# ----------------------------------------------------------------------

def linear(t: float) -> float:
    return t


def ease_out(t: float) -> float:
    """Fast then settling. The default for anything responding to the user."""
    return 1.0 - (1.0 - t) ** 3


def ease_in_out(t: float) -> float:
    """Symmetric. Used where a move has no obvious cause, such as the camera."""
    if t < 0.5:
        return 4 * t * t * t
    return 1.0 - ((-2 * t + 2) ** 3) / 2


def ease_out_back(t: float) -> float:
    """Overshoots slightly and settles. Gives a state a sense of weight."""
    c1 = 1.70158
    c3 = c1 + 1.0
    return 1.0 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def ease_out_elastic(t: float) -> float:
    """A short spring. Reserved for the moment a run is accepted."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    c4 = (2 * math.pi) / 3
    return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * c4) + 1


def pulse(t: float) -> float:
    """Rises to one and falls back, for a highlight that flashes and fades."""
    return math.sin(math.pi * max(0.0, min(1.0, t)))


# ----------------------------------------------------------------------
# Animated values
# ----------------------------------------------------------------------

@dataclass
class Animated:
    """A float that moves towards a target over a fixed duration.

    Retargeting mid-flight restarts the curve from wherever the value currently
    is, rather than snapping. That is what lets a user step forward while
    playback is running without the display jumping.
    """

    value: float = 0.0
    target: float = 0.0
    duration: float = 200.0
    easing: Easing = ease_out
    _from: float = field(default=0.0, init=False)
    _elapsed: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._from = self.value
        self._elapsed = self.duration

    @property
    def is_settled(self) -> bool:
        return self._elapsed >= self.duration

    def set(self, target: float, duration: Optional[float] = None,
            easing: Optional[Easing] = None) -> None:
        """Move towards a new target, starting from the current value."""
        if abs(target - self.target) < 1e-6 and not self.is_settled:
            return
        self._from = self.value
        self.target = target
        self._elapsed = 0.0
        if duration is not None:
            self.duration = duration
        if easing is not None:
            self.easing = easing

    def jump_to(self, value: float) -> None:
        """Set the value with no animation."""
        self.value = self._from = self.target = value
        self._elapsed = self.duration

    def update(self, dt: float) -> None:
        if self.is_settled:
            self.value = self.target
            return
        self._elapsed = min(self.duration, self._elapsed + dt)
        if self.duration <= 0:
            self.value = self.target
            return
        t = self.easing(self._elapsed / self.duration)
        self.value = self._from + (self.target - self._from) * t


@dataclass
class AnimatedPoint:
    """Two :class:`Animated` values travelling together."""

    x: Animated = field(default_factory=Animated)
    y: Animated = field(default_factory=Animated)

    @property
    def value(self) -> Point:
        return (self.x.value, self.y.value)

    @property
    def is_settled(self) -> bool:
        return self.x.is_settled and self.y.is_settled

    def set(self, target: Point, duration: Optional[float] = None,
            easing: Optional[Easing] = None) -> None:
        self.x.set(target[0], duration, easing)
        self.y.set(target[1], duration, easing)

    def jump_to(self, target: Point) -> None:
        self.x.jump_to(target[0])
        self.y.jump_to(target[1])

    def update(self, dt: float) -> None:
        self.x.update(dt)
        self.y.update(dt)


class Track:
    """A named collection of animated values.

    Nodes come and go as the user edits, so their animations are created on
    demand and dropped when the state disappears. This keeps that bookkeeping in
    one place instead of scattering ``if key in dict`` through the renderer.
    """

    def __init__(self, duration: float = 180.0, easing: Easing = ease_out):
        self._values: Dict[str, Animated] = {}
        self._duration = duration
        self._easing = easing

    def get(self, key: str) -> float:
        entry = self._values.get(key)
        return entry.value if entry else 0.0

    def set(self, key: str, target: float,
            duration: Optional[float] = None,
            easing: Optional[Easing] = None) -> None:
        entry = self._values.get(key)
        if entry is None:
            entry = Animated(value=0.0, target=0.0,
                             duration=self._duration, easing=self._easing)
            self._values[key] = entry
        entry.set(target, duration, easing)

    def update(self, dt: float) -> None:
        for entry in self._values.values():
            entry.update(dt)

    def drop_missing(self, live_keys: Iterable[str]) -> None:
        """Forget animations for keys that no longer exist."""
        live = set(live_keys)
        for key in [k for k in self._values if k not in live]:
            del self._values[key]

    def clear(self) -> None:
        self._values.clear()


@dataclass
class Timer:
    """A one-shot progress value from 0 to 1 over a duration.

    Used for effects that play once and stop: the flash along an edge as it is
    traversed, or the ring that expands when a run is accepted.
    """

    duration: float = 400.0
    elapsed: float = field(default=0.0, init=False)
    running: bool = field(default=False, init=False)

    @property
    def progress(self) -> float:
        if self.duration <= 0:
            return 1.0
        return max(0.0, min(1.0, self.elapsed / self.duration))

    @property
    def finished(self) -> bool:
        return not self.running or self.progress >= 1.0

    def start(self, duration: Optional[float] = None) -> None:
        if duration is not None:
            self.duration = duration
        self.elapsed = 0.0
        self.running = True

    def stop(self) -> None:
        self.running = False
        self.elapsed = self.duration

    def update(self, dt: float) -> None:
        if not self.running:
            return
        self.elapsed += dt
        if self.elapsed >= self.duration:
            self.elapsed = self.duration
            self.running = False
