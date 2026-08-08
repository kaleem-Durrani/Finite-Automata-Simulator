"""An automaton together with its layout.

This is what the editor edits and what a file contains. Its whole job is to
keep the two halves in step: adding a state gives it coordinates, removing one
forgets them, and there is no way to do either operation to only one half.

That was the previous model's central flaw. Positions lived on state objects and
curve offsets inside the transition table, so "the automaton" and "the picture of
the automaton" were the same mutable structure updated in several places -- and
when one place was missed, the drawn graph and the simulated graph disagreed
without anything noticing.

Every operation returns a new Document. Undo is therefore ``stack.append(doc)``.
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

from fsa.automaton import DFA
from fsa.layout import PLACEMENT_STEP, Layout, Point
from fsa.symbols import StateId, Symbol

#: Room required when the user chose the spot: enough to not overlap, no more.
OVERLAP_GAP = 62.0


@dataclass(frozen=True, slots=True)
class Document:
    """An automaton, its layout, and the counter used to name new states."""

    automaton: DFA = field(default_factory=DFA)
    layout: Layout = field(default_factory=Layout)
    next_id: int = 0

    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------

    def fresh_id(self) -> StateId:
        """The name the next state would get, without taking it."""
        candidate = self.next_id
        while f"q{candidate}" in self.automaton.states:
            candidate += 1
        return f"q{candidate}"

    # ------------------------------------------------------------------
    # States
    # ------------------------------------------------------------------

    def add_state(self, position: Point,
                  state_id: Optional[StateId] = None,
                  minimum_gap: Optional[float] = None) -> Tuple["Document", StateId]:
        """Add a state at a position, avoiding any state already there.

        ``minimum_gap`` controls how much room it insists on. The default is
        comfortable spacing, which is what automatic placement wants -- pressing
        Space repeatedly used to stack every state on one pixel. When the user
        picked the spot themselves, pass the tighter :data:`OVERLAP_GAP` so the
        state lands where they clicked unless it would genuinely overlap.

        Returns the new document and the id used, because the caller almost
        always wants to select what it just created.
        """
        state_id = state_id or self.fresh_id()
        placed = self.layout.free_position(
            position, minimum_gap if minimum_gap is not None else PLACEMENT_STEP)
        automaton = self.automaton.with_state(state_id)
        number = int(state_id[1:]) if state_id[1:].isdigit() else self.next_id
        return (
            Document(
                automaton=automaton,
                layout=self.layout.with_position(state_id, placed),
                next_id=max(self.next_id, number + 1),
            ),
            state_id,
        )

    def remove_state(self, state_id: StateId) -> "Document":
        """Remove a state from both halves at once."""
        automaton = self.automaton.without_state(state_id)
        return Document(
            automaton=automaton,
            layout=self.layout.without_state(state_id).restricted_to(automaton.states),
            next_id=self.next_id,
        )

    def move_state(self, state_id: StateId, position: Point) -> "Document":
        """Reposition a state. The automaton is untouched, by construction."""
        return Document(self.automaton,
                        self.layout.with_position(state_id, position),
                        self.next_id)

    def toggle_accept(self, state_id: StateId) -> "Document":
        return Document(self.automaton.with_accept_toggled(state_id),
                        self.layout, self.next_id)

    def set_initial(self, state_id: Optional[StateId]) -> "Document":
        return Document(self.automaton.with_initial(state_id),
                        self.layout, self.next_id)

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def add_transition(self, source: StateId, symbol: Symbol, target: StateId,
                       arc: float = 0.0) -> "Document":
        """Define delta(source, symbol) = target, optionally bowing the edge."""
        return Document(
            self.automaton.with_transition(source, symbol, target),
            self.layout.with_arc(source, target, arc) if arc else self.layout,
            self.next_id,
        )

    def remove_transition(self, source: StateId, symbol: Symbol) -> "Document":
        """Make delta undefined at a pair, dropping the bow if that was the last
        symbol on the edge."""
        target = self.automaton.target(source, symbol)
        automaton = self.automaton.without_transition(source, symbol)
        layout = self.layout
        if target is not None and (source, target) not in automaton.grouped_transitions():
            layout = layout.with_arc(source, target, 0.0)
        return Document(automaton, layout, self.next_id)

    def make_trap(self, state_id: StateId) -> "Document":
        """Turn a state into a genuine trap: no acceptance, every symbol loops.

        An operation rather than a flag. The old "dead end" marker changed what
        the simulator accepted without changing a single transition, so the tool
        computed a different language than the diagram it drew.
        """
        automaton = self.automaton.without_accept(state_id)
        for symbol in sorted(automaton.alphabet):
            automaton = automaton.with_transition(state_id, symbol, state_id)
        return Document(automaton, self.layout, self.next_id)

    # ------------------------------------------------------------------
    # Alphabet
    # ------------------------------------------------------------------

    def add_symbol(self, symbol: Symbol) -> "Document":
        return Document(self.automaton.with_symbol(symbol), self.layout, self.next_id)

    def remove_symbol(self, symbol: Symbol) -> "Document":
        return Document(self.automaton.without_symbol(symbol), self.layout,
                        self.next_id)

    # ------------------------------------------------------------------
    # Wholesale replacement
    # ------------------------------------------------------------------

    def with_automaton(self, automaton: DFA) -> "Document":
        """Swap the automaton, keeping coordinates for states that survive.

        Anything new is placed on a grid below the existing layout, so the
        result of an algorithm that invents states is at least visible.
        """
        layout = self.layout.restricted_to(automaton.states)
        missing = [s for s in sorted(automaton.states) if s not in layout.positions]
        if missing:
            box = layout.bounds()
            origin = (160.0, 160.0) if box is None else (box[0], box[3] + 150.0)
            layout = layout.with_positions(Layout.grid(missing, origin).positions)
        return Document(automaton, layout, self.next_id)

    @staticmethod
    def of(automaton: DFA, layout: Optional[Layout] = None) -> "Document":
        """Build a document, laying out any state that has no position."""
        document = Document(DFA(), layout or Layout(), 0)
        document = document.with_automaton(automaton)
        highest = -1
        for state in automaton.states:
            if state.startswith("q") and state[1:].isdigit():
                highest = max(highest, int(state[1:]))
        return Document(document.automaton, document.layout, highest + 1)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def states(self) -> Iterable[StateId]:
        return self.automaton.states

    def position_of(self, state_id: StateId) -> Point:
        return self.layout.position_of(state_id)
