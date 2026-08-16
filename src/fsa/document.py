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

**The automaton is always an** :class:`~fsa.nfa.NFA`. A deterministic document
is not a different kind of document; it is an NFA that happens to satisfy
:meth:`fsa.nfa.NFA.is_deterministic`. The alternative was ``Union[DFA, NFA]``,
and the choice was made by changing the type and reading what broke: widening to
NFA breaks thirty call sites, because NFA was built to mirror DFA's surface --
``states``, ``alphabet``, ``accept``, ``initial``, ``labels``, ``label_of`` and
``grouped_transitions`` are the same on both, so almost every *read* keeps
working untouched. A union would have needed an ``isinstance`` at every one of
those thirty instead, and one forgotten check is a crash in the frame loop.

What the union would have bought is a type error when a DFA-only algorithm is
handed a nondeterministic machine, and that is bought back here by one method:
:meth:`as_dfa`. Every DFA-only operation goes through it, so the failure happens
at one named boundary, with a message that says which state and symbol made the
machine nondeterministic and what to do about it -- instead of an
``isinstance`` scattered through the callers, or a silent determinisation that
would hand back a machine with different states from the one on screen.

A DFA is still accepted everywhere a document is built, and converted on the way
in with :func:`fsa.nfa.from_dfa`, so a caller holding the output of
:func:`fsa.minimize.minimize` does not have to spell the lift out.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from fsa import ops
from fsa.automaton import DFA
from fsa.layout import PLACEMENT_STEP, AnyAutomaton, Layout, Point
from fsa.nfa import NFA, from_dfa, to_dfa
from fsa.symbols import StateId, Symbol

#: Room required when the user chose the spot: enough to not overlap, no more.
OVERLAP_GAP = 62.0


def next_id_for(states: Iterable[StateId]) -> int:
    """The first ``qN`` counter that cannot collide with ``states``.

    One past the highest ``qN`` present, so a document opened from a file --
    hand-written, or written by a build that counted differently -- cannot name
    its next state over one it already has. Public because the serializer needs
    exactly this rule for a machine it has not yet made a document of, and two
    copies of a rule about naming is how two halves of a program come to
    disagree about which names are free.
    """
    highest = -1
    for state in states:
        if state.startswith("q") and state[1:].isdigit():
            highest = max(highest, int(state[1:]))
    return highest + 1


@dataclass(frozen=True, slots=True, init=False)
class Document:
    """An automaton, its layout, and the counter used to name new states."""

    automaton: NFA
    layout: Layout
    next_id: int

    def __init__(self, automaton: Optional[AnyAutomaton] = None,
                 layout: Optional[Layout] = None, next_id: int = 0) -> None:
        """Hold ``automaton``, converting a DFA to its NFA reading.

        The constructor is written out rather than generated so that the
        parameter is wider than the field: callers hand over whichever machine
        they are holding, and what is stored is always an NFA. Doing the
        conversion here rather than at thirty call sites is the whole reason
        the widening was affordable.
        """
        stored = NFA() if automaton is None else (
            from_dfa(automaton) if isinstance(automaton, DFA) else automaton)
        object.__setattr__(self, "automaton", stored)
        object.__setattr__(self, "layout", Layout() if layout is None else layout)
        object.__setattr__(self, "next_id", next_id)

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------

    @property
    def is_deterministic(self) -> bool:
        """Whether this document's machine is a DFA written in NFA form.

        A *fact* about the machine, not a verdict on it: a partial delta is
        deterministic, and a nondeterministic machine is a legal thing to draw,
        save and run. Nothing here or in :func:`fsa.analysis.defects` treats
        the answer as a defect -- see docs/LESSONS.md on the complete/trim
        cycle, where a legal design choice was labelled a fault with a Fix
        button beside it.
        """
        return self.automaton.is_deterministic()

    def as_dfa(self) -> DFA:
        """The deterministic view of this document's machine.

        The one boundary at which a DFA-only algorithm meets a document. Every
        such call site goes through here, so the type error the old ``DFA``
        field gave for free becomes exactly one runtime error, raised in one
        place, carrying the state and symbol that caused it.

        Not the subset construction: this converts a machine that is *already*
        deterministic and refuses anything else, because determinising returns
        a machine with different state ids and no operation should do that to a
        document behind the user's back. :func:`fsa.subset.determinize` is the
        one that builds a new machine.

        Raises:
            NondeterministicError: If the machine has an epsilon move or two
                targets on one symbol.
        """
        return to_dfa(self.automaton)

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

    def add_transition(self, source: StateId, symbol: Optional[Symbol],
                       target: StateId, arc: float = 0.0) -> "Document":
        """Add ``target`` to delta(source, symbol), optionally bowing the edge.

        **Adds a branch, where this used to replace one.** Drawing a second
        edge from one state on one symbol is now what it looks like -- two
        arrows -- rather than the first silently disappearing. That silent
        overwrite is precisely what made nondeterminism undrawable, so the
        change is the point of the phase and not a side effect of it.

        ``symbol`` may be :data:`fsa.nfa.EPSILON` (``None``), for a move that
        reads nothing. It is not added to the alphabet, because it is not a
        letter of it.
        """
        return Document(
            self.automaton.with_transition(source, symbol, target),
            self.layout.with_arc(source, target, arc) if arc else self.layout,
            self.next_id,
        )

    def remove_transition(self, source: StateId, symbol: Optional[Symbol],
                          target: Optional[StateId] = None) -> "Document":
        """Remove one branch of a move, or the whole move.

        With ``target`` given, only that arrow goes and any other branch on the
        same symbol stays -- what a user deleting one drawn edge means. Without
        it, delta becomes undefined at ``(source, symbol)`` entirely, which is
        what a caller holding only a ``(state, symbol)`` pair can ask for.

        A bow belongs to the *edge*, not to the symbol, so it is dropped only
        when the removal leaves no arrow at all between the two states.
        """
        losing = self.automaton.targets(source, symbol)
        if target is not None:
            losing = losing & {target}

        automaton = self.automaton.without_transition(source, symbol, target)
        layout = self.layout
        surviving = automaton.grouped_transitions()
        for gone in sorted(losing):
            if (source, gone) not in surviving:
                layout = layout.with_arc(source, gone, 0.0)
        return Document(automaton, layout, self.next_id)

    def make_trap(self, state_id: StateId) -> "Document":
        """Turn a state into a genuine trap: no acceptance, every symbol loops.

        An operation rather than a flag. The old "dead end" marker changed what
        the simulator accepted without changing a single transition, so the tool
        computed a different language than the diagram it drew.

        Every move out of the state is removed first, epsilon moves included.
        Adding the self-loops alone would leave the old branches beside them --
        :meth:`fsa.nfa.NFA.with_transition` adds rather than replaces -- and a
        state you can still leave is not a trap however many loops it has.
        """
        automaton = self.automaton.without_accept(state_id)
        for symbol in automaton.outgoing(state_id):
            automaton = automaton.without_transition(state_id, symbol)
        for symbol in sorted(automaton.alphabet):
            automaton = automaton.with_transition(state_id, symbol, state_id)
        return Document(automaton, self.layout, self.next_id)

    def complete(self, trap_id: Optional[StateId] = None) -> Tuple["Document", Optional[StateId]]:
        """Make delta total; see :func:`fsa.ops.complete`.

        Goes through :meth:`with_automaton` so an invented trap state gets
        grid coordinates instead of landing at the origin. The trap's id
        comes back with the document because the caller almost always wants
        to point at what the fix created; ``(self, None)`` means there was
        nothing to fix.

        Raises:
            NondeterministicError: Via :meth:`as_dfa`. Completion is defined on
                a function, and delta here may be a relation.
        """
        automaton, trap = ops.complete(self.as_dfa(), trap_id)
        if trap is None:
            return self, None
        return self.with_automaton(automaton), trap

    # ------------------------------------------------------------------
    # Alphabet
    # ------------------------------------------------------------------

    def add_symbol(self, symbol: Symbol) -> "Document":
        return Document(self.automaton.with_symbol(symbol), self.layout, self.next_id)

    # ------------------------------------------------------------------
    # Wholesale replacement
    # ------------------------------------------------------------------

    def with_automaton(self, automaton: AnyAutomaton) -> "Document":
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
    def of(automaton: AnyAutomaton, layout: Optional[Layout] = None) -> "Document":
        """Build a document, laying out any state that has no position."""
        document = Document(NFA(), layout or Layout(), 0)
        document = document.with_automaton(automaton)
        return Document(document.automaton, document.layout,
                        next_id_for(automaton.states))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def states(self) -> Iterable[StateId]:
        return self.automaton.states

    def position_of(self, state_id: StateId) -> Point:
        return self.layout.position_of(state_id)
