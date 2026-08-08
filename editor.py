"""Editing state.

Holds the document being edited plus everything the *editor* knows that the
document does not: what is selected, what the mouse is over, what is being
dragged, whether there are unsaved changes, and where the file lives.

That separation is the point. Three separate crashes came from the app keeping
pointers into a mutable model and the model deleting the thing they pointed at.
Here the document is immutable, so replacing it can never invalidate a reference
by surprise -- and every replacement goes through one method, which drops any
pointer that no longer names a live state.

No pygame. This is testable without a display.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import fsa
from fsa import Document, Layout, Point, StateId


@dataclass
class Drag:
    """A state being moved.

    The position lives here, not in the layout, until the mouse is released.
    Layouts are immutable, so committing one per motion event would allocate a
    new mapping every frame of a drag; committing on release allocates one.
    """

    state: StateId
    offset: Point
    position: Point


class EditorModel:
    """The document being edited, and the editor's own state."""

    def __init__(self, document: Optional[Document] = None):
        self.document: Document = document or Document()
        self.selection: Optional[StateId] = None
        self.hover: Optional[StateId] = None
        self.drag: Optional[Drag] = None
        self.dirty: bool = False
        self.path: Optional[str] = None

        # Half-drawn transition, if any.
        self.pending_source: Optional[StateId] = None
        self.pending_arc: float = 0.0

        # (automaton, result) for the last analysis run. Keyed on the value
        # itself, which is immutable, so there is no flag to forget to clear.
        self._cached_analysis: Optional[Tuple[Any, Any]] = None

    # ------------------------------------------------------------------
    # Document replacement
    # ------------------------------------------------------------------

    def apply(self, document: Document, *, dirty: bool = True) -> None:
        """Adopt a new document and drop any pointer it invalidates.

        Every edit goes through here, so there is exactly one place that has to
        get reference cleanup right.
        """
        self.document = document
        self.forget_missing()
        if dirty:
            self.dirty = True

    def replace(self, document: Document, path: Optional[str]) -> None:
        """Load a whole new document, discarding all editing state."""
        self.document = document
        self.selection = None
        self.hover = None
        self.drag = None
        self.pending_source = None
        self.pending_arc = 0.0
        self.path = path
        self.dirty = False

    def forget_missing(self) -> None:
        """Drop references to states the document no longer contains."""
        states = self.document.automaton.states
        if self.selection is not None and self.selection not in states:
            self.selection = None
        if self.hover is not None and self.hover not in states:
            self.hover = None
        if self.drag is not None and self.drag.state not in states:
            self.drag = None
        if self.pending_source is not None and self.pending_source not in states:
            self.cancel_transition()

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    @property
    def automaton(self) -> fsa.DFA:
        return self.document.automaton

    @property
    def layout(self) -> Layout:
        return self.document.layout

    def position_of(self, state: StateId) -> Point:
        """Where a state is right now, including mid-drag."""
        if self.drag is not None and self.drag.state == state:
            return self.drag.position
        return self.document.layout.position_of(state)

    def positions(self) -> Dict[StateId, Point]:
        """Every state's current position, including the one being dragged."""
        return {state: self.position_of(state)
                for state in self.document.automaton.states}

    def state_at(self, point: Point, radius: float) -> Optional[StateId]:
        """The state under a world point, topmost first.

        Sorted so that the most recently added state wins when two overlap.
        Returning the first match in insertion order meant only the oldest of a
        stack was ever clickable.
        """
        best: Optional[StateId] = None
        best_distance = radius * radius
        for state in self.document.automaton.states:
            position = self.position_of(state)
            distance = ((point[0] - position[0]) ** 2
                        + (point[1] - position[1]) ** 2)
            if distance <= best_distance:
                best, best_distance = state, distance
        return best

    # ------------------------------------------------------------------
    # Selection and hover
    # ------------------------------------------------------------------

    def select(self, state: Optional[StateId]) -> None:
        self.selection = state if state in self.document.automaton.states else None

    def set_hover(self, state: Optional[StateId]) -> None:
        self.hover = state if state in self.document.automaton.states else None

    # ------------------------------------------------------------------
    # Dragging
    # ------------------------------------------------------------------

    def begin_drag(self, state: StateId, grab: Point) -> None:
        if state not in self.document.automaton.states:
            return
        position = self.document.layout.position_of(state)
        self.drag = Drag(state=state,
                         offset=(grab[0] - position[0], grab[1] - position[1]),
                         position=position)

    def update_drag(self, point: Point) -> None:
        if self.drag is None:
            return
        self.drag.position = (point[0] - self.drag.offset[0],
                              point[1] - self.drag.offset[1])

    def end_drag(self) -> bool:
        """Commit the drag to the layout. True if anything actually moved."""
        if self.drag is None:
            return False
        drag, self.drag = self.drag, None
        if drag.position == self.document.layout.position_of(drag.state):
            return False
        self.apply(self.document.move_state(drag.state, drag.position))
        return True

    # ------------------------------------------------------------------
    # Half-drawn transitions
    # ------------------------------------------------------------------

    def begin_transition(self, source: StateId) -> None:
        if source in self.document.automaton.states:
            self.pending_source = source
            self.pending_arc = 0.0

    def bend_pending(self, delta: float, limit: float = 110.0) -> None:
        self.pending_arc = max(-limit, min(limit, self.pending_arc + delta))

    def cancel_transition(self) -> None:
        self.pending_source = None
        self.pending_arc = 0.0

    # ------------------------------------------------------------------
    # Edits
    # ------------------------------------------------------------------

    def add_state(self, position: Point, minimum_gap: Optional[float] = None) -> StateId:
        document, state = self.document.add_state(position, minimum_gap=minimum_gap)
        self.apply(document)
        return state

    def remove_state(self, state: StateId) -> bool:
        if state not in self.document.automaton.states:
            return False
        self.apply(self.document.remove_state(state))
        return True

    def toggle_accept(self, state: StateId) -> bool:
        """Returns whether the state accepts afterwards."""
        self.apply(self.document.toggle_accept(state))
        return state in self.document.automaton.accept

    def set_initial(self, state: Optional[StateId]) -> None:
        self.apply(self.document.set_initial(state))

    def add_transition(self, source: StateId, symbol: str, target: StateId,
                       arc: float = 0.0) -> bool:
        """Define a transition. False if either state is gone.

        Reported rather than raised: the states can disappear between a gesture
        starting and finishing, and that is the user deleting something, not a
        programming error.
        """
        states = self.document.automaton.states
        if source not in states or target not in states:
            return False
        self.apply(self.document.add_transition(source, symbol, target, arc))
        return True

    def remove_transition(self, source: StateId, symbol: str) -> None:
        self.apply(self.document.remove_transition(source, symbol))

    def make_trap(self, state: StateId) -> Tuple[bool, int]:
        """Loop every symbol back to a state. Returns success and how many
        existing transitions were replaced."""
        automaton = self.document.automaton
        if state not in automaton.states or not automaton.alphabet:
            return False, 0
        replaced = sum(
            1 for symbol in automaton.alphabet
            if automaton.target(state, symbol) not in (None, state))
        self.apply(self.document.make_trap(state))
        return True, replaced

    def add_symbol(self, symbol: str) -> bool:
        if symbol in self.document.automaton.alphabet:
            return False
        try:
            self.apply(self.document.add_symbol(symbol))
        except fsa.IllegalSymbolError:
            return False
        return True

    # ------------------------------------------------------------------
    # Analysis, cached until the automaton changes
    # ------------------------------------------------------------------

    def analysis(self) -> Tuple[frozenset, frozenset, bool]:
        """Dead states, unreachable states, and whether anything accepts.

        Recomputed only when the automaton value changes -- which, because it is
        immutable, is a cheap identity check rather than a dirty flag someone
        has to remember to set.
        """
        automaton = self.document.automaton
        cached = getattr(self, "_cached_analysis", None)
        if cached is not None and cached[0] is automaton:
            return cached[1]

        unreachable = fsa.unreachable_states(automaton)
        # With no accepting state every state is technically a trap. True, and
        # useless: it greys the whole canvas out while the user is still
        # drawing. The absence of accepting states is reported on its own.
        dead = fsa.dead_states(automaton) if automaton.accept else frozenset()
        result = (dead, unreachable, bool(automaton.accept))

        self._cached_analysis = (automaton, result)
        return result
