"""Where things are drawn.

Positions and curve offsets, kept entirely separate from the automaton. Moving a
state cannot change the language, because the language is computed from a value
that has no coordinates in it.

The previous model stored a position on every ``State`` and a pixel arc offset
inside the transition table. That is what let rendering data and the transition
function fall out of step with each other, and it is why the same automaton drawn
in two places could disagree.

Immutable, like everything else in the engine, with one deliberate exception
noted on :meth:`Layout.with_position`.
"""

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from fsa.symbols import StateId

Point = Tuple[float, float]
Edge = Tuple[StateId, StateId]

_EMPTY_POSITIONS: Mapping[StateId, Point] = MappingProxyType({})
_EMPTY_ARCS: Mapping[Edge, float] = MappingProxyType({})

#: Default spacing when placing a state automatically.
PLACEMENT_STEP = 78.0

#: Decimal places kept for coordinates.
#:
#: Positions are pixels, so anything past a thousandth is noise. Rounding here,
#: rather than only when writing a file, means the in-memory value is always at
#: storage precision -- so ``loads(dumps(d)) == d`` holds exactly instead of
#: nearly, and a document that has been saved is equal to the one that has not.
PRECISION = 3


@dataclass(frozen=True, slots=True, eq=False)
class Layout:
    """Coordinates for an automaton's states, and bows for its edges."""

    positions: Mapping[StateId, Point] = field(default=_EMPTY_POSITIONS)
    arc_offsets: Mapping[Edge, float] = field(default=_EMPTY_ARCS)

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", MappingProxyType({
            state: (round(float(point[0]), PRECISION),
                    round(float(point[1]), PRECISION))
            for state, point in dict(self.positions).items()
        }))
        object.__setattr__(self, "arc_offsets", MappingProxyType({
            (source, target): round(float(offset), PRECISION)
            for (source, target), offset in dict(self.arc_offsets).items()
            if offset
        }))

    # ------------------------------------------------------------------
    # Value semantics
    # ------------------------------------------------------------------

    def _key(self) -> Tuple[Any, ...]:
        return (tuple(sorted(self.positions.items())),
                tuple(sorted(self.arc_offsets.items())))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Layout):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return f"Layout({len(self.positions)} positions, {len(self.arc_offsets)} arcs)"

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def position_of(self, state: StateId) -> Point:
        """Where a state sits. The origin if it has never been placed."""
        return self.positions.get(state, (0.0, 0.0))

    def arc_of(self, source: StateId, target: StateId) -> float:
        """How far an edge bows. Zero means draw it straight."""
        return self.arc_offsets.get((source, target), 0.0)

    def bounds(self, radius: float = 0.0) -> Optional[Tuple[float, float, float, float]]:
        """The rectangle containing every placed state, or ``None`` if empty."""
        if not self.positions:
            return None
        xs = [point[0] for point in self.positions.values()]
        ys = [point[1] for point in self.positions.values()]
        return (min(xs) - radius, min(ys) - radius,
                max(xs) + radius, max(ys) + radius)

    def centre(self) -> Point:
        """The midpoint of the placed states."""
        box = self.bounds()
        if box is None:
            return (0.0, 0.0)
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    # ------------------------------------------------------------------
    # Derivation
    # ------------------------------------------------------------------

    def with_position(self, state: StateId, point: Point) -> "Layout":
        """Place a state.

        Allocating a new Layout per mouse-motion event would be wasteful, so
        the editor keeps a scratch position while a drag is in flight and calls
        this once, on release. Nothing else should need to care.
        """
        return Layout({**dict(self.positions), state: point}, self.arc_offsets)

    def with_positions(self, points: Mapping[StateId, Point]) -> "Layout":
        """Place several states at once."""
        return Layout({**dict(self.positions), **dict(points)}, self.arc_offsets)

    def without_state(self, state: StateId) -> "Layout":
        """Forget a state, and any edge bows that mention it."""
        return Layout(
            {s: p for s, p in self.positions.items() if s != state},
            {edge: offset for edge, offset in self.arc_offsets.items()
             if state not in edge},
        )

    def with_arc(self, source: StateId, target: StateId, offset: float) -> "Layout":
        """Bow an edge. Zero removes the bow."""
        arcs = dict(self.arc_offsets)
        if offset:
            arcs[(source, target)] = offset
        else:
            arcs.pop((source, target), None)
        return Layout(self.positions, arcs)

    def restricted_to(self, states: Iterable[StateId]) -> "Layout":
        """Drop everything that is not about one of ``states``.

        Used after an operation that removes states, so the layout cannot
        accumulate coordinates for things that no longer exist.
        """
        live = set(states)
        return Layout(
            {s: p for s, p in self.positions.items() if s in live},
            {edge: offset for edge, offset in self.arc_offsets.items()
             if edge[0] in live and edge[1] in live},
        )

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    def free_position(self, near: Point, minimum_gap: float = PLACEMENT_STEP) -> Point:
        """A point close to ``near`` that does not sit on an existing state.

        Adding states used to put every one of them at the exact centre of the
        view, stacked on the same pixel -- and because hit-testing returns the
        first match, only the oldest was ever clickable again.

        Spirals outward until it finds room.
        """
        if not self._collides(near, minimum_gap):
            return near

        step = minimum_gap
        for ring in range(1, 12):
            count = 6 * ring
            for i in range(count):
                angle = 2 * math.pi * i / count
                candidate = (near[0] + math.cos(angle) * step * ring,
                             near[1] + math.sin(angle) * step * ring)
                if not self._collides(candidate, minimum_gap):
                    return candidate
        return (near[0] + step * 12, near[1])

    def _collides(self, point: Point, gap: float) -> bool:
        return any(
            (point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2 < gap * gap
            for other in self.positions.values()
        )

    @staticmethod
    def grid(states: Iterable[StateId], origin: Point = (160.0, 160.0),
             columns: int = 4, step: float = 150.0) -> "Layout":
        """Lay states out in rows.

        A fallback for automata that arrive without coordinates -- a file
        written by hand, or the output of an algorithm that produced new states.
        """
        placed: Dict[StateId, Point] = {}
        for index, state in enumerate(states):
            row, column = divmod(index, max(1, columns))
            placed[state] = (origin[0] + column * step, origin[1] + row * step)
        return Layout(placed)
