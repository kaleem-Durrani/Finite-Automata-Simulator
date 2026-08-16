"""Editing state.

Holds the document being edited plus everything the *editor* knows that the
document does not: what is selected, what the mouse is over, what is being
dragged, whether there are unsaved changes, where the file lives, and what
the document used to be (undo).

That separation is the point. Three separate crashes came from the app keeping
pointers into a mutable model and the model deleting the thing they pointed at.
Here the document is immutable, so replacing it can never invalidate a reference
by surprise -- and every replacement goes through one method, which drops any
pointer that no longer names a live state.

No pygame. This is testable without a display.

The document holds an ``NFA``, always -- see :mod:`fsa.document`. Nothing here
may assume one target per ``(state, symbol)``, and everything that needs a DFA
asks the document for :meth:`~fsa.document.Document.as_dfa` behind a check.
This file runs inside a 60fps loop, so an exception raised here is a window
that closes while someone is drawing; the DFA-only readings degrade to "no
answer" instead.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import fsa
from fsa import Document, Layout, Point, StateId

#: How much history undo keeps. Documents share almost all of their structure,
#: so 200 entries is cheap -- the cap exists to bound a marathon session, not
#: because entries are expensive.
UNDO_DEPTH = 200

#: How an epsilon move is named in the phrases undo reports. The engine spells
#: it ``None`` and a file spells it ``null``; neither is a thing to show a
#: person in "undo transition q0 -None-> q1".
EPSILON_LABEL = "ε"


def _shown(symbol: Optional[str]) -> str:
    """A transition symbol as it appears in a phrase shown to the user."""
    return EPSILON_LABEL if symbol is None else symbol


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

        # History is a list of previous document values -- the payoff of the
        # document being immutable. Each entry pairs the value to restore with
        # the phrase naming the edit that replaced it, so the GUI can say what
        # an undo undid.
        self._undo: List[Tuple[Document, str]] = []
        self._redo: List[Tuple[Document, str]] = []

        # Half-drawn transition, if any.
        self.pending_source: Optional[StateId] = None
        self.pending_arc: float = 0.0

        # (automaton, result) for the last analysis/defects run. Keyed on the
        # value itself, which is immutable, so there is no flag to forget to
        # clear.
        self._cached_analysis: Optional[Tuple[Any, Any]] = None
        self._cached_defects: Optional[Tuple[Any, Any]] = None

    # ------------------------------------------------------------------
    # Document replacement
    # ------------------------------------------------------------------

    def apply(self, document: Document, *, dirty: bool = True,
              action: str = "edit") -> None:
        """Adopt a new document and drop any pointer it invalidates.

        Every edit goes through here, so there is exactly one place that has to
        get reference cleanup right -- and, now, exactly one place that records
        history. ``action`` is the short phrase undo will report ("add q3").

        A document equal to the current one is adopted by doing nothing: a
        no-op edit must not eat an undo slot or mark the file dirty.
        """
        if document == self.document:
            return
        self._undo.append((self.document, action))
        if len(self._undo) > UNDO_DEPTH:
            del self._undo[0]
        # A new edit invalidates the redone future.
        self._redo.clear()
        self.document = document
        self.forget_missing()
        if dirty:
            self.dirty = True

    def replace(self, document: Document, path: Optional[str]) -> None:
        """Load a whole new document, discarding all editing state.

        Including history: undo must not carry one file's past into another.
        """
        self.document = document
        self.selection = None
        self.hover = None
        self.drag = None
        self.pending_source = None
        self.pending_arc = 0.0
        self.path = path
        self.dirty = False
        self._undo.clear()
        self._redo.clear()

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
    # Undo and redo
    # ------------------------------------------------------------------

    def undo(self) -> Optional[str]:
        """Restore the previous document. Returns the phrase naming the edit
        that was undone, or ``None`` if there was nothing to undo.

        The restored value goes through the same pointer cleanup as an edit:
        bringing a document back must not resurrect a selection or drag that
        was dropped when its state disappeared.
        """
        if not self._undo:
            return None
        document, action = self._undo.pop()
        self._redo.append((self.document, action))
        self.document = document
        self.forget_missing()
        self.dirty = True
        return action

    def redo(self) -> Optional[str]:
        """The mirror image of :meth:`undo`."""
        if not self._redo:
            return None
        document, action = self._redo.pop()
        self._undo.append((self.document, action))
        self.document = document
        self.forget_missing()
        self.dirty = True
        return action

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    @property
    def automaton(self) -> fsa.NFA:
        return self.document.automaton

    @property
    def is_deterministic(self) -> bool:
        """Whether the machine on the canvas is a DFA.

        A fact the interface reports, never a defect it flags. Exposed here so
        that a panel or a menu item can ask without reaching through two
        objects for it, and so there is one place to look when wondering what
        the editor does differently for a nondeterministic machine.
        """
        return self.document.is_deterministic

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
        # One history entry per drag: motion accumulated in self.drag, so this
        # is the first time the document changes.
        self.apply(self.document.move_state(drag.state, drag.position),
                   action=f"move {drag.state}")
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
        self.apply(document, action=f"add {state}")
        return state

    def remove_state(self, state: StateId) -> bool:
        if state not in self.document.automaton.states:
            return False
        self.apply(self.document.remove_state(state), action=f"delete {state}")
        return True

    def rename(self, state: StateId, label: str) -> bool:
        """Set a state's display label. False if the state does not exist.

        A blank label means "back to the id" -- there is no way to give a state
        an invisible name. So does typing the id itself: both drop the label
        rather than storing it, so that clearing a name returns the document to
        the value it had before the name was given. Storing the id instead
        would leave a document that renders identically but compares unequal,
        which is enough to mark the file dirty and spend an undo slot on a
        change with nothing to show for it.

        The label is stripped, so it cannot differ from the name reported back
        to the user by invisible padding.
        """
        if state not in self.document.automaton.states:
            return False
        text = label.strip()
        automaton = (self.document.automaton.with_label(state, text)
                     if text and text != state
                     else self.document.automaton.with_label_removed(state))
        self.apply(Document(automaton, self.document.layout, self.document.next_id),
                   action=f"rename {state}")
        return True

    def toggle_accept(self, state: StateId) -> bool:
        """Returns whether the state accepts afterwards."""
        document = self.document.toggle_accept(state)
        accepting = state in document.automaton.accept
        self.apply(document, action=(f"accept {state}" if accepting
                                     else f"unaccept {state}"))
        return accepting

    def set_initial(self, state: Optional[StateId]) -> None:
        self.apply(self.document.set_initial(state), action=f"start at {state}")

    def add_transition(self, source: StateId, symbol: Optional[str],
                       target: StateId, arc: float = 0.0) -> bool:
        """Add a transition. False if either state is gone.

        Reported rather than raised: the states can disappear between a gesture
        starting and finishing, and that is the user deleting something, not a
        programming error.

        An existing move on the same symbol is **kept**: this adds a branch
        rather than replacing one, so drawing a second arrow leaves two arrows.
        ``symbol`` may be ``None``, for a move that reads nothing.
        """
        states = self.document.automaton.states
        if source not in states or target not in states:
            return False
        self.apply(self.document.add_transition(source, symbol, target, arc),
                   action=f"transition {source} -{_shown(symbol)}-> {target}")
        return True

    def remove_transition(self, source: StateId, symbol: Optional[str],
                          target: Optional[StateId] = None) -> None:
        """Remove one branch of a move, or every branch of it.

        ``target`` says which arrow the user pointed at. Without it every
        target on that symbol goes, which is the only thing a caller holding
        just a ``(state, symbol)`` pair can mean -- and was the only possible
        meaning while delta had one target.
        """
        self.apply(self.document.remove_transition(source, symbol, target),
                   action=f"remove {source} -{_shown(symbol)}->")

    def make_trap(self, state: StateId) -> Tuple[bool, int]:
        """Loop every symbol back to a state. Returns success and how many
        existing moves out of it were replaced.

        Counted per symbol, epsilon included, because that is what the user
        sees disappear: a symbol whose only target was already this state is
        not a change, and one that led anywhere else is.
        """
        automaton = self.document.automaton
        if state not in automaton.states or not automaton.alphabet:
            return False, 0
        replaced = sum(1 for targets in automaton.outgoing(state).values()
                       if targets - {state})
        self.apply(self.document.make_trap(state), action=f"trap {state}")
        return True, replaced

    def add_symbol(self, symbol: str) -> bool:
        if symbol in self.document.automaton.alphabet:
            return False
        try:
            self.apply(self.document.add_symbol(symbol),
                       action=f"symbol '{symbol}'")
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

        The first two are DFA facts: :func:`fsa.analysis.reachable` and
        :func:`fsa.analysis.co_reachable` both walk a delta with one target per
        pair. On a nondeterministic machine they are reported as empty --
        nothing is greyed out and nothing is hatched -- rather than guessed at
        or raised. An empty answer says "not known here", which is true; a
        wrong one would put a hatch on a state that is perfectly alive, and
        this program has already learned what happens when the drawing and the
        analysis disagree.
        """
        automaton = self.document.automaton
        cached = getattr(self, "_cached_analysis", None)
        if cached is not None and cached[0] is automaton:
            return cached[1]

        if self.document.is_deterministic:
            deterministic = self.document.as_dfa()
            unreachable = fsa.unreachable_states(deterministic)
            # With no accepting state every state is technically a trap. True,
            # and useless: it greys the whole canvas out while the user is
            # still drawing. The absence of accepting states is reported on its
            # own.
            dead = (fsa.dead_states(deterministic) if automaton.accept
                    else frozenset())
        else:
            dead = unreachable = frozenset()
        result = (dead, unreachable, bool(automaton.accept))

        self._cached_analysis = (automaton, result)
        return result

    def defects(self) -> Tuple[fsa.Defect, ...]:
        """Everything worth telling the user about the automaton, ranked.

        Cached on the automaton value like :meth:`analysis`, because the
        diagnostics panel reads this every frame and reverse-reachability every
        16ms would be paying for the same answer over and over.

        Nothing is listed for a nondeterministic machine. Every defect
        :func:`fsa.analysis.defects` knows about -- an incomplete delta, dead
        states, unreachable states -- is a statement about a transition
        *function*, and none of them can be computed here without first
        determinizing, which would describe a machine that is not the one on
        screen. Nondeterminism itself is emphatically not added to the list:
        it is a legal design choice, and this codebase has been burned once
        already by putting a Fix button next to one (docs/LESSONS.md, the
        complete/trim cycle). The panel simply has nothing to say, which is
        honest; the status panel still warns when no state accepts.
        """
        automaton = self.document.automaton
        cached = getattr(self, "_cached_defects", None)
        if cached is not None and cached[0] is automaton:
            result: Tuple[fsa.Defect, ...] = cached[1]
            return result

        found = (fsa.defects(self.document.as_dfa())
                 if self.document.is_deterministic else ())
        self._cached_defects = (automaton, found)
        return found
