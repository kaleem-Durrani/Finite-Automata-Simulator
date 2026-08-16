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

Because coordinates live outside the automaton, an automaton can exist without
any -- which is the normal case for anything an algorithm builds rather than a
user draws. :meth:`Layout.auto` is where those machines get somewhere to be.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from fsa.automaton import DFA
from fsa.nfa import NFA
from fsa.symbols import StateId

Point = Tuple[float, float]
Edge = Tuple[StateId, StateId]

#: Either machine, for the few places that read only what both expose:
#: ``states``, ``initial``, ``accept``, ``alphabet``, ``labels`` and
#: ``grouped_transitions``. Placement is one of them -- where a state is drawn
#: cannot depend on how many targets its moves have -- and
#: :class:`fsa.document.Document` is the other, which is why the alias lives
#: here: layout is the lowest module that needs it, and the document already
#: depends on layout, so putting it the other way round would be a cycle.
AnyAutomaton = Union[DFA, NFA]

_EMPTY_POSITIONS: Mapping[StateId, Point] = MappingProxyType({})
_EMPTY_ARCS: Mapping[Edge, float] = MappingProxyType({})

#: Default spacing when placing a state automatically.
PLACEMENT_STEP = 78.0

#: The closest two automatically placed states are ever put, centre to centre.
#:
#: The GUI draws a state as a circle of radius 30 (``rendering.renderer``'s
#: ``STATE_RADIUS``), so 60 is exactly where two of them touch. The number is
#: written out here instead of imported because nothing under ``fsa`` may
#: import the renderer -- the engine has no display dependency, and CI enforces
#: it -- and the remaining 30 is what stops touching circles: it leaves the
#: arrow between two neighbours somewhere to draw its symbols.
AUTO_SEPARATION = 90.0

#: How much wider than tall an automatic layout is.
#:
#: Columns get more room than rows because the horizontal gaps are where the
#: arrows and their labels are drawn, while the vertical gaps hold nothing.
#: Any value >= 1 keeps the separation promise intact, since two states in
#: different columns are already a full column step apart in x alone.
LAYER_ASPECT = 1.5

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

    @staticmethod
    def auto(automaton: AnyAutomaton,
             algorithm: str = "bfs_layers",
             origin: Point = (160.0, 160.0),
             minimum_separation: float = AUTO_SEPARATION) -> "Layout":
        """Place every state of an automaton, from scratch.

        Nothing else in this program can position a state the user did not
        position by hand, so an operation that *invents* states -- completion,
        a product construction, the subset construction -- has nowhere to put
        them: they all land on the origin in one pile, where hit-testing finds
        only the topmost and the rest are lost. Every algorithm that returns a
        new automaton needs this first.

        ``"bfs_layers"`` gives each state a column chosen by its distance from
        the initial state. An edge then normally points from one column to the
        next, so the drawing reads left to right in the direction the
        transitions actually go, and the initial state is where a reader looks
        for it. Within a column, states are ordered by id and centred on a
        common line.

        States the initial state cannot reach have no distance to sort them
        by, and neither has anything at all when there is no initial state.
        They are not dropped -- a state with no coordinates is a state that
        cannot be drawn, found or deleted -- but placed in a block after the
        last layer, separated from it by one empty column so the picture says
        what it means: these are not part of the machine's flow.

        Args:
            automaton: What to place. Only its states and transitions are
                read, and nothing about it is changed.
            algorithm: Which placement to use. ``"bfs_layers"`` is the only
                one so far; anything else raises :class:`ValueError` rather
                than quietly falling back, because a caller that named a
                layout wants that layout and not a surprise.
            origin: The top-left corner of the drawing -- not the position of
                the first state. Columns are centred against each other, so
                which state sits highest depends on the tallest column, and
                pinning the initial state instead would push everything else
                off the top of the view.
            minimum_separation: The centre-to-centre distance no two states
                are placed closer than; see :data:`AUTO_SEPARATION`. Rows are
                spaced by exactly this, columns by ``LAYER_ASPECT`` times it.
                Coordinates are rounded to :data:`PRECISION` when the layout
                is built, so the guarantee is exact to within a thousandth of
                a pixel.

        Returns:
            A layout holding a position for every state of ``automaton`` and
            for nothing else, and no arc offsets: a drawing nobody has touched
            yet has no bowed edges to remember. Identical automata give
            identical layouts, in this run and in the next -- everything
            iterated here comes out of a ``sorted`` call, because set order in
            Python is not stable between processes.
        """
        if algorithm != "bfs_layers":
            raise ValueError(
                f"unknown layout algorithm {algorithm!r}; expected 'bfs_layers'")

        columns: List[Tuple[StateId, ...]] = list(_bfs_layers(automaton))

        on_the_spine = {state for column in columns for state in column}
        orphans = tuple(sorted(
            state for state in automaton.states if state not in on_the_spine))
        if orphans:
            if columns:
                # An empty column places nothing but still costs a column of
                # width, which is the gap that reads as "and now something
                # else".
                columns.append(())
            columns.extend(_wrapped_columns(orphans))

        column_step = minimum_separation * LAYER_ASPECT
        placed: Dict[StateId, Point] = {}
        for index, column in enumerate(columns):
            x = index * column_step
            top = -(len(column) - 1) * minimum_separation / 2
            for row, state in enumerate(column):
                placed[state] = (x, top + row * minimum_separation)

        if not placed:
            return Layout()

        # Shift the finished drawing so its bounding box starts at the origin.
        # Centring the columns puts half of a tall one above the line it was
        # centred on, and a caller handed negative coordinates would have to
        # scroll to find the machine it just built.
        left = min(point[0] for point in placed.values())
        highest = min(point[1] for point in placed.values())
        return Layout({
            state: (point[0] + origin[0] - left, point[1] + origin[1] - highest)
            for state, point in placed.items()
        })


# ----------------------------------------------------------------------
# Automatic layout, internals
# ----------------------------------------------------------------------


def _bfs_layers(automaton: AnyAutomaton) -> Tuple[Tuple[StateId, ...], ...]:
    """The reachable states, grouped by their BFS distance from the initial one.

    Empty when there is no initial state: then nothing has a distance, and the
    caller has to find a home for the whole state set.

    Distances are what make the drawing readable, so they are measured the same
    way :func:`fsa.analysis.reachable` measures reachability -- following delta
    forwards, one edge at a time.
    """
    if automaton.initial is None:
        return ()

    initial: StateId = automaton.initial
    depth: Dict[StateId, int] = {initial: 0}
    queue = deque([initial])
    while queue:
        state = queue.popleft()
        # Read through grouped_transitions rather than delta directly. Both
        # DFA and NFA expose it, and it answers the only question layout has --
        # which states are one edge apart -- without caring how many targets a
        # symbol has or whether an edge is an epsilon move. Asking for
        # `target(state, symbol)` tied placement to determinism, so an NFA
        # could not be drawn at all.
        #
        # Sorted, so the traversal does not depend on set order. The distances
        # would survive that, but a future change that used visit order for
        # anything would silently draw a different picture on every run.
        for source, target in sorted(automaton.grouped_transitions()):
            if source != state or target in depth:
                continue
            depth[target] = depth[state] + 1
            queue.append(target)

    layers: List[List[StateId]] = [[] for _ in range(max(depth.values()) + 1)]
    for state, distance in sorted(depth.items()):
        layers[distance].append(state)
    return tuple(tuple(layer) for layer in layers)


def _wrapped_columns(states: Sequence[StateId]) -> Tuple[Tuple[StateId, ...], ...]:
    """``states`` cut into roughly square columns.

    One column would do, but an automaton whose initial state was deleted has
    *every* state in here, and a dozen of them in a single column runs off the
    bottom of any view. A square block stays on screen.
    """
    if not states:
        return ()
    rows = math.isqrt(len(states) - 1) + 1  # ceil(sqrt(n)), without floats
    return tuple(tuple(states[start:start + rows])
                 for start in range(0, len(states), rows))
