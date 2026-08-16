"""What to draw, described without saying how.

A :class:`Scene` is a flat list of visual facts: this circle is here, at this
radius, lit this much; this edge runs along this path and carries this label.

The renderer consumes a Scene and knows nothing about automata. The application
produces a Scene and knows nothing about pixels. That boundary is why the
rendering work survives the model rewrite: swapping the legacy DFA for the
engine changes only the code that *builds* the scene, not the several hundred
lines that draw it.

Everything here is in world coordinates. The renderer applies the camera.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]


class NodeKind(Enum):
    """How a state should read at a glance."""

    NORMAL = "normal"
    DEAD = "dead"
    """No accepting state is reachable from here. Derived from the transition
    function, never from a flag the user set."""
    UNREACHABLE = "unreachable"
    """No word can reach this state, so it cannot affect the language."""


@dataclass(frozen=True, slots=True)
class NodeVisual:
    """One state, as it should appear this frame."""

    id: str
    position: Point
    radius: float
    label: str
    kind: NodeKind = NodeKind.NORMAL
    is_accept: bool = False

    # Continuous, because they are animated. 0 is off, 1 is fully applied.
    selected: float = 0.0
    hover: float = 0.0
    active: float = 0.0
    """How lit this state is as the current state of a run."""
    settle: float = 0.0
    """A brief scale-up when the state is first entered."""


@dataclass(frozen=True, slots=True)
class EdgeVisual:
    """One transition group, as a drawn path."""

    key: Tuple[str, str]
    path: Sequence[Point]
    label: str
    label_at: Optional[Point] = None
    """Where the label goes, when the midpoint of the path is a bad place --
    inside a self-loop, for instance."""
    color_index: int = 0
    active: float = 0.0
    """How lit this edge is, while a run is traversing it."""
    muted: float = 0.0
    """How far this edge is faded back, when something else has focus."""
    show_arrowhead: bool = True


@dataclass(frozen=True, slots=True)
class GhostEdge:
    """The edge the user is currently dragging out, drawn dashed."""

    path: Sequence[Point]
    label: str
    valid: bool = True


@dataclass(frozen=True, slots=True)
class TokenVisual:
    """One read head travelling along one edge during execution.

    This is the thing that makes the machine look like it is running. It moves
    continuously along :attr:`EdgeVisual.path` rather than appearing at each
    state in turn.

    There may be several in a frame -- see :attr:`Scene.tokens`. A token is a
    *branch* being taken, not "the position of the machine", which is why the
    plural was the right shape to reach for rather than a second kind of
    marker for the nondeterministic case.
    """

    position: Point
    radius: float
    trail: Sequence[Point] = ()
    intensity: float = 1.0


@dataclass(frozen=True, slots=True)
class StartMarker:
    """The arrow identifying the initial state."""

    path: Sequence[Point]


@dataclass
class Scene:
    """Everything on the canvas for one frame."""

    nodes: List[NodeVisual] = field(default_factory=list)
    edges: List[EdgeVisual] = field(default_factory=list)
    start_marker: Optional[StartMarker] = None
    ghost_edge: Optional[GhostEdge] = None

    tokens: List[TokenVisual] = field(default_factory=list)
    """The read heads in flight, one per branch the machine is taking.

    A list rather than the single optional token this was, because a
    nondeterministic move is several moves happening at once. Empty at rest,
    one entry while a deterministic machine crosses an edge, and one per
    surviving branch while a nondeterministic one does -- which is the same
    rule stated once, not a special case bolted onto the side.
    """

    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """The world rectangle containing every node, or None if empty.

        Used by fit-to-content, which is what rescues a user who has panned
        away and lost their automaton.
        """
        if not self.nodes:
            return None
        xs_min = min(node.position[0] - node.radius for node in self.nodes)
        xs_max = max(node.position[0] + node.radius for node in self.nodes)
        ys_min = min(node.position[1] - node.radius for node in self.nodes)
        ys_max = max(node.position[1] + node.radius for node in self.nodes)
        return (xs_min, ys_min, xs_max, ys_max)
