"""The automaton itself: an immutable value.

Two design decisions here carry most of the weight.

**The transition function is flat.** ``{(state, symbol): target}`` rather than
``{state: {symbol: target}}``. There is no per-state sub-dictionary, so the
class of bug where a state legitimately has no outgoing edges and every lookup
against it raises ``KeyError`` cannot be written.

**Every operation returns a new automaton.** Nothing mutates. Undo becomes
"keep the previous value", equality is structural, and a snapshot taken by one
part of the program cannot be changed underneath it by another -- all three of
which were real defects in the mutable model this replaces.

Rendering data lives elsewhere. The automaton knows nothing about positions,
radii, colours or curves; mixing those into the model is what allowed the drawn
graph and the simulated graph to disagree.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Set, Tuple

from fsa.errors import DuplicateStateError, UnknownStateError
from fsa.symbols import StateId, Symbol, check_symbol, normalize_alphabet

Transitions = Mapping[Tuple[StateId, Symbol], StateId]

_EMPTY_MAP: Mapping[Any, Any] = MappingProxyType({})


@dataclass(frozen=True, slots=True, eq=False)
class DFA:
    """A deterministic finite automaton, possibly partial.

    Args:
        states: The state set Q.
        alphabet: The input alphabet, Sigma.
        transitions: The transition function delta, as a flat mapping. May be
            partial: a missing ``(state, symbol)`` key means delta is undefined
            there, which is a rejection with a distinct reason, not a crash.
        initial: The start state, or ``None``. ``None`` is legal and means the
            recognised language is undefined rather than empty -- the state an
            editor is in before a start state has been chosen, and the state it
            falls back to when the start state is deleted.
        accept: The accepting states, F.
        labels: Optional display names. Purely cosmetic; the engine never reads
            them, so renaming a state cannot change the language.
    """

    states: FrozenSet[StateId] = frozenset()
    alphabet: FrozenSet[Symbol] = frozenset()
    transitions: Transitions = _EMPTY_MAP
    initial: Optional[StateId] = None
    accept: FrozenSet[StateId] = frozenset()
    labels: Mapping[StateId, str] = field(default=_EMPTY_MAP)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Normalise every field to an immutable type and validate."""
        states = frozenset(self.states)
        alphabet = normalize_alphabet(self.alphabet)

        transitions: Dict[Tuple[StateId, Symbol], StateId] = {}
        for (source, symbol), target in dict(self.transitions).items():
            if source not in states:
                raise UnknownStateError(f"transition from unknown state {source!r}")
            if target not in states:
                raise UnknownStateError(f"transition to unknown state {target!r}")
            check_symbol(symbol)
            if symbol not in alphabet:
                raise UnknownStateError(
                    f"transition on {symbol!r}, which is not in the alphabet")
            transitions[(source, symbol)] = target

        accept = frozenset(self.accept)
        unknown = accept - states
        if unknown:
            raise UnknownStateError(f"accepting states not in Q: {sorted(unknown)}")

        if self.initial is not None and self.initial not in states:
            raise UnknownStateError(f"initial state {self.initial!r} is not in Q")

        labels = {
            state: str(text) for state, text in dict(self.labels).items()
            if state in states
        }

        object.__setattr__(self, "states", states)
        object.__setattr__(self, "alphabet", alphabet)
        object.__setattr__(self, "transitions", MappingProxyType(transitions))
        object.__setattr__(self, "accept", accept)
        object.__setattr__(self, "labels", MappingProxyType(labels))

    # ------------------------------------------------------------------
    # Value semantics
    # ------------------------------------------------------------------

    def _key(self) -> Tuple[Any, ...]:
        """A canonical, hashable summary of this automaton."""
        return (
            tuple(sorted(self.states)),
            tuple(sorted(self.alphabet)),
            tuple(sorted(self.transitions.items())),
            self.initial,
            tuple(sorted(self.accept)),
            tuple(sorted(self.labels.items())),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DFA):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return (f"DFA(|Q|={len(self.states)}, |S|={len(self.alphabet)}, "
                f"|d|={len(self.transitions)}, initial={self.initial!r}, "
                f"|F|={len(self.accept)})")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def target(self, state: StateId, symbol: Symbol) -> Optional[StateId]:
        """Where delta sends ``state`` on ``symbol``, or ``None`` if undefined."""
        return self.transitions.get((state, symbol))

    def outgoing(self, state: StateId) -> Mapping[Symbol, StateId]:
        """The transitions leaving ``state``, keyed by symbol."""
        return MappingProxyType({
            symbol: target
            for (source, symbol), target in self.transitions.items()
            if source == state
        })

    def grouped_transitions(self) -> Mapping[Tuple[StateId, StateId], FrozenSet[Symbol]]:
        """
        Edges as (from, to) -> symbols, for drawing.

        Computed on demand rather than stored. The old model kept this as a
        second, hand-maintained copy of the transition function; the two fell
        out of step, and the renderer drew one automaton while the simulator ran
        another. Deriving it means they cannot disagree.
        """
        grouped: Dict[Tuple[StateId, StateId], Set[Symbol]] = {}
        for (source, symbol), target in self.transitions.items():
            grouped.setdefault((source, target), set()).add(symbol)
        return MappingProxyType({
            edge: frozenset(symbols) for edge, symbols in grouped.items()
        })

    def label_of(self, state: StateId) -> str:
        """The display name for ``state``, defaulting to its id."""
        return self.labels.get(state, state)

    # ------------------------------------------------------------------
    # Derivation -- each returns a new automaton
    # ------------------------------------------------------------------

    def _replace(self, **changes: Any) -> "DFA":
        current: Dict[str, Any] = {
            "states": self.states,
            "alphabet": self.alphabet,
            "transitions": self.transitions,
            "initial": self.initial,
            "accept": self.accept,
            "labels": self.labels,
        }
        current.update(changes)
        return DFA(**current)

    def with_state(self, state: StateId) -> "DFA":
        """Add a state. The first state added becomes the initial state."""
        if state in self.states:
            raise DuplicateStateError(f"state {state!r} already exists")
        return self._replace(
            states=self.states | {state},
            initial=self.initial if self.initial is not None else state,
        )

    def without_state(self, state: StateId) -> "DFA":
        """
        Remove a state, and every transition into or out of it.

        If it was the initial state, the automaton is left with no initial
        state rather than having one silently chosen for it. Picking an
        arbitrary replacement changes the recognised language without telling
        anyone, which is exactly what the previous model did.
        """
        if state not in self.states:
            raise UnknownStateError(f"no such state: {state!r}")
        return self._replace(
            states=self.states - {state},
            transitions={
                (source, symbol): target
                for (source, symbol), target in self.transitions.items()
                if source != state and target != state
            },
            initial=None if self.initial == state else self.initial,
            accept=self.accept - {state},
            labels={s: t for s, t in self.labels.items() if s != state},
        )

    def with_symbol(self, symbol: Symbol) -> "DFA":
        """Extend the alphabet."""
        return self._replace(alphabet=self.alphabet | {check_symbol(symbol)})

    def without_symbol(self, symbol: Symbol) -> "DFA":
        """Remove a symbol from the alphabet, and every transition on it."""
        return self._replace(
            alphabet=self.alphabet - {symbol},
            transitions={
                key: target for key, target in self.transitions.items()
                if key[1] != symbol
            },
        )

    def with_transition(self, source: StateId, symbol: Symbol,
                        target: StateId) -> "DFA":
        """
        Define delta(source, symbol) = target.

        Replaces any existing transition on that symbol, because the machine is
        deterministic. The symbol is added to the alphabet if it is new.
        """
        for state in (source, target):
            if state not in self.states:
                raise UnknownStateError(f"no such state: {state!r}")
        symbol = check_symbol(symbol)
        return self._replace(
            alphabet=self.alphabet | {symbol},
            transitions={**dict(self.transitions), (source, symbol): target},
        )

    def without_transition(self, source: StateId, symbol: Symbol) -> "DFA":
        """Make delta undefined at ``(source, symbol)``."""
        remaining = dict(self.transitions)
        remaining.pop((source, symbol), None)
        return self._replace(transitions=remaining)

    def with_initial(self, state: Optional[StateId]) -> "DFA":
        """Set, or clear, the start state."""
        return self._replace(initial=state)

    def with_accept(self, state: StateId) -> "DFA":
        """Mark a state accepting."""
        if state not in self.states:
            raise UnknownStateError(f"no such state: {state!r}")
        return self._replace(accept=self.accept | {state})

    def without_accept(self, state: StateId) -> "DFA":
        """Mark a state non-accepting."""
        return self._replace(accept=self.accept - {state})

    def with_accept_toggled(self, state: StateId) -> "DFA":
        """Flip whether a state is accepting."""
        if state in self.accept:
            return self.without_accept(state)
        return self.with_accept(state)

    def with_label(self, state: StateId, text: str) -> "DFA":
        """Set a state's display name."""
        if state not in self.states:
            raise UnknownStateError(f"no such state: {state!r}")
        return self._replace(labels={**dict(self.labels), state: text})

    def with_states(self, states: Iterable[StateId]) -> "DFA":
        """Add several states at once."""
        automaton = self
        for state in states:
            automaton = automaton.with_state(state)
        return automaton
