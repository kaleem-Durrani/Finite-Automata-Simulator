"""Nondeterminism: delta as a relation rather than a function.

A DFA's delta answers "where does this take me?" with one state. An NFA's
answers with a *set*, possibly empty, and it may also move without reading
anything at all. That is one line of difference in the type and a different
answer to nearly every question underneath it, which is why this is a separate
value type and not a flag on :class:`~fsa.automaton.DFA`. Keeping them apart
keeps the determinism a DFA promises inside its type: ``DFA.target`` returns
one state because it always can, the DFA simulator holds one current state
because it always can, and neither has to ask what kind of machine today's is.

Three decisions carry the rest of the module.

**A missing key is the empty set.** delta here is partial *and*
nondeterministic, so ``transitions`` never stores an empty target set:
:meth:`NFA.targets` synthesises one for any key it does not hold, and the
constructor drops any empty set handed to it. "No move from here" therefore has
exactly one spelling, and the bug where two machines that behave identically
compare unequal -- because one of them wrote its dead ends down -- cannot be
written.

**Epsilon is ``None``, and it is a key, not a symbol.** ``(state, EPSILON)`` is
an epsilon move, and ``EPSILON`` is never in the alphabet, so no character has
to be reserved and an epsilon move can never be confused with a transition on
some literal ``'e'``. The price is that ``sorted(transitions.items())`` raises
``TypeError`` -- ``None`` does not compare with ``str`` -- which is exactly
what :meth:`NFA.sorted_transitions` exists to stop every caller rediscovering.

**A configuration is a set of states.** The simulator's whole state is a set,
and :class:`NfaRun` keeps the entire sequence of them rather than a verdict,
because the interesting thing about a nondeterministic run is watching the set
spread out and collapse. A run is still finite and still linear in the length
of the word: the machine tracks all branches at once instead of backtracking.

Rendering data lives elsewhere, as with :class:`~fsa.automaton.DFA`. Nothing
here knows about positions, colours or curves.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    Final,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)

from fsa.automaton import DFA
from fsa.errors import DuplicateStateError, NondeterministicError, UnknownStateError
from fsa.simulate import Verdict
from fsa.symbols import StateId, Symbol, check_symbol, normalize_alphabet

#: The key for a move that reads nothing. ``None`` rather than a sentinel
#: object or a reserved character, so that ``Optional[Symbol]`` says the whole
#: truth about a transition key in the type system, and no alphabet loses a
#: letter to it.
EPSILON: Final[None] = None

#: delta, flat: ``{(state, symbol_or_EPSILON): targets}``. Flat for the same
#: reason DFA's is -- a state with no outgoing edges needs no sub-dictionary to
#: be missing, so the KeyError class of bug has nowhere to live.
NfaTransitions = Mapping[Tuple[StateId, Optional[Symbol]], FrozenSet[StateId]]

#: One entry of delta, flattened and put in order: ``(source, symbol, targets)``
#: with the targets sorted. See :meth:`NFA.sorted_transitions`.
TransitionRow = Tuple[StateId, Optional[Symbol], Tuple[StateId, ...]]

_EMPTY_MAP: Mapping[Any, Any] = MappingProxyType({})
_NO_TARGETS: FrozenSet[StateId] = frozenset()


def _symbol_order(symbol: Optional[Symbol]) -> Tuple[int, str]:
    """A sort key for a transition symbol, epsilon first.

    ``None`` has no ordering against ``str``, so anything that sorts transition
    keys needs this. Epsilon sorts before every real symbol because that is how
    it reads on screen and in a saved file: the move that costs nothing first.
    """
    return (0, "") if symbol is None else (1, symbol)


@dataclass(frozen=True, slots=True, eq=False)
class NFA:
    """A nondeterministic finite automaton, with epsilon moves.

    Args:
        states: The state set Q.
        alphabet: The input alphabet, Sigma. Never contains ``EPSILON``.
        transitions: The transition relation delta, as a flat mapping from
            ``(state, symbol)`` to the set of states that pair can lead to.
            ``symbol`` is ``EPSILON`` for a move that reads nothing. Partial: a
            missing key means the empty set -- no move, which is a rejection
            with a reason rather than a crash -- and an empty set passed in is
            stored as no key at all, so the two are indistinguishable
            afterwards. Any iterable of state ids is accepted as the value.
        initial: The start state, or ``None``. As with
            :class:`~fsa.automaton.DFA`, ``None`` means "no language defined
            yet" rather than "the empty language": it is the state an editor is
            in before a start state is chosen, and the one it falls back to
            when that state is deleted.
        accept: The accepting states, F.
        labels: Optional display names. Cosmetic; the engine never reads them.
    """

    states: FrozenSet[StateId] = frozenset()
    alphabet: FrozenSet[Symbol] = frozenset()
    transitions: NfaTransitions = _EMPTY_MAP
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

        transitions: Dict[Tuple[StateId, Optional[Symbol]], FrozenSet[StateId]] = {}
        for key, targets in dict(self.transitions).items():
            source, symbol = key
            if source not in states:
                raise UnknownStateError(f"transition from unknown state {source!r}")
            if symbol is not None:
                check_symbol(symbol)
                if symbol not in alphabet:
                    raise UnknownStateError(
                        f"transition on {symbol!r}, which is not in the alphabet")

            if isinstance(targets, str):
                # frozenset("q10") is {"q", "1", "0"}: passing a bare id where a
                # set of them was meant would shred it into one target per
                # character and validate cleanly if those happened to be states.
                # A state id *is* a string and a string *is* iterable, so this
                # is the one mistake the rest of the checks cannot catch.
                raise TypeError(
                    f"targets for {key!r} must be a set of states, not the bare "
                    f"state {targets!r}")

            frozen = frozenset(targets)
            unknown = frozen - states
            if unknown:
                raise UnknownStateError(
                    f"transition to states not in Q: {sorted(unknown)}")
            if frozen:
                # The empty set is spelled by leaving the key out entirely. Two
                # spellings of one fact is how a machine ends up unequal to an
                # identically-behaving copy of itself.
                transitions[(source, symbol)] = frozen

        accept = frozenset(self.accept)
        unknown_accept = accept - states
        if unknown_accept:
            raise UnknownStateError(
                f"accepting states not in Q: {sorted(unknown_accept)}")

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
            self.sorted_transitions(),
            self.initial,
            tuple(sorted(self.accept)),
            tuple(sorted(self.labels.items())),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NFA):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        epsilons = sum(1 for _, symbol in self.transitions if symbol is None)
        return (f"NFA(|Q|={len(self.states)}, |S|={len(self.alphabet)}, "
                f"|d|={len(self.transitions)}, eps={epsilons}, "
                f"initial={self.initial!r}, |F|={len(self.accept)})")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def sorted_transitions(self) -> Tuple[TransitionRow, ...]:
        """delta as ``(source, symbol, targets)`` triples in one agreed order.

        Public because everything that has to write delta out -- equality, the
        serializer, an exporter, a diagnostics panel -- needs the same order,
        and the obvious ``sorted(transitions.items())`` does not work here:
        keys hold ``Optional[Symbol]`` and ``None`` raises against ``str``.
        Getting that wrong once, in one place, is better than nine callers each
        inventing a key function; and the targets come out sorted too, because
        a set's iteration order changes between processes (see docs/LESSONS.md).
        """
        return tuple(
            (source, symbol, tuple(sorted(targets)))
            for (source, symbol), targets in sorted(
                self.transitions.items(),
                key=lambda item: (item[0][0], _symbol_order(item[0][1]))))

    def targets(self, state: StateId, symbol: Optional[Symbol]) -> FrozenSet[StateId]:
        """The states ``(state, symbol)`` can lead to; empty if it leads nowhere.

        A key delta does not hold means the empty set, not an error -- the
        machine is allowed to have no move, and every algorithm here would
        otherwise have to guard its lookups. ``symbol`` may be ``EPSILON``, and
        an unknown state answers with the empty set for the same reason.
        """
        return self.transitions.get((state, symbol), _NO_TARGETS)

    def epsilon_closure(self, states: Iterable[StateId]) -> FrozenSet[StateId]:
        """Every state reachable from ``states`` by epsilon moves alone.

        Reflexive: the states given are always in the result, so the closure of
        a machine with no epsilon moves is just what it was handed.

        The ``not in closure`` test is the whole algorithm. An epsilon *cycle*
        -- q0 to q1 and back to q0, or a state with an epsilon self-loop -- is
        legal, common in constructions like Thompson's, and the case a naive
        closure spins on forever. Membership is checked before enqueueing, so
        every state joins the frontier at most once and the walk is linear in
        the number of epsilon edges.
        """
        closure: Set[StateId] = set(states)
        frontier: List[StateId] = list(closure)
        while frontier:
            state = frontier.pop()
            for target in self.targets(state, EPSILON):
                if target not in closure:
                    closure.add(target)
                    frontier.append(target)
        return frozenset(closure)

    def is_deterministic(self) -> bool:
        """Whether this machine is a DFA written in NFA form.

        Two conditions, and only two: no epsilon moves, and no ``(state,
        symbol)`` with more than one target. A *partial* delta is deterministic
        -- a state with no move on some symbol has nothing to choose between,
        so nothing is nondeterministic about it. Conflating the two would make
        :func:`to_dfa` refuse most of the machines it exists to convert.
        """
        return all(
            symbol is not None and len(targets) == 1
            for (_, symbol), targets in self.transitions.items())

    def outgoing(self, state: StateId) -> Mapping[Optional[Symbol], FrozenSet[StateId]]:
        """The moves leaving ``state``, keyed by symbol, epsilon included."""
        return MappingProxyType({
            symbol: frozenset(targets)
            for source, symbol, targets in self.sorted_transitions()
            if source == state
        })

    def grouped_transitions(self) -> Mapping[Tuple[StateId, StateId], FrozenSet[Optional[Symbol]]]:
        """Edges as ``(from, to) -> symbols``, for drawing.

        ``EPSILON`` appears in the symbol set of an edge that carries an
        epsilon move, so one edge between two states can be labelled with both
        real symbols and epsilon -- which is what the renderer must draw.

        Derived on demand, never stored, for the reason
        :meth:`DFA.grouped_transitions` gives: the old model kept a second
        hand-maintained copy of the edges, the two fell out of step, and the
        renderer drew one automaton while the simulator ran another.
        """
        grouped: Dict[Tuple[StateId, StateId], Set[Optional[Symbol]]] = {}
        for source, symbol, targets in self.sorted_transitions():
            for target in targets:
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

    def _replace(self, **changes: Any) -> "NFA":
        current: Dict[str, Any] = {
            "states": self.states,
            "alphabet": self.alphabet,
            "transitions": self.transitions,
            "initial": self.initial,
            "accept": self.accept,
            "labels": self.labels,
        }
        current.update(changes)
        return NFA(**current)

    def with_state(self, state: StateId) -> "NFA":
        """Add a state. The first state added becomes the initial state."""
        if state in self.states:
            raise DuplicateStateError(f"state {state!r} already exists")
        return self._replace(
            states=self.states | {state},
            initial=self.initial if self.initial is not None else state,
        )

    def without_state(self, state: StateId) -> "NFA":
        """Remove a state, every move out of it, and every move into it.

        If it was the initial state the machine is left without one, rather
        than having a replacement chosen for it behind the user's back -- the
        same rule, for the same reason, as :meth:`DFA.without_state`.
        """
        if state not in self.states:
            raise UnknownStateError(f"no such state: {state!r}")

        surviving: Dict[Tuple[StateId, Optional[Symbol]], FrozenSet[StateId]] = {}
        for (source, symbol), targets in self.transitions.items():
            if source == state:
                continue
            remaining = targets - {state}
            if remaining:
                # An entry whose only target was the removed state vanishes
                # rather than becoming an empty set: nondeterminism means a
                # move can lose one branch and keep the others.
                surviving[(source, symbol)] = remaining

        return self._replace(
            states=self.states - {state},
            transitions=surviving,
            initial=None if self.initial == state else self.initial,
            accept=self.accept - {state},
            labels={s: t for s, t in self.labels.items() if s != state},
        )

    def with_symbol(self, symbol: Symbol) -> "NFA":
        """Extend the alphabet. ``EPSILON`` is refused: it is not a symbol."""
        return self._replace(alphabet=self.alphabet | {check_symbol(symbol)})

    def without_symbol(self, symbol: Optional[Symbol]) -> "NFA":
        """Remove a symbol from the alphabet, and every move on it.

        Passing ``EPSILON`` removes every epsilon move and leaves the alphabet
        alone, which is the right answer rather than a coincidence: epsilon was
        never in the alphabet to remove.
        """
        return self._replace(
            alphabet=self.alphabet - {symbol},
            transitions={
                key: targets for key, targets in self.transitions.items()
                if key[1] != symbol
            },
        )

    def with_transition(self, source: StateId, symbol: Optional[Symbol],
                        target: StateId) -> "NFA":
        """Add ``target`` to the set delta(source, symbol) leads to.

        **Adds, where :meth:`DFA.with_transition` replaces.** That single
        difference is the whole point of the type: a second edge from one state
        on one symbol is legal here, and silently overwriting the first is
        precisely the behaviour that made nondeterminism undrawable before.

        ``symbol`` may be ``EPSILON``, in which case nothing is added to the
        alphabet -- epsilon is not a letter of it. Any other symbol is added if
        it is new, as on a DFA.
        """
        for state in (source, target):
            if state not in self.states:
                raise UnknownStateError(f"no such state: {state!r}")

        alphabet = self.alphabet
        if symbol is not None:
            symbol = check_symbol(symbol)
            alphabet = alphabet | {symbol}

        return self._replace(
            alphabet=alphabet,
            transitions={
                **dict(self.transitions),
                (source, symbol): self.targets(source, symbol) | {target},
            },
        )

    def without_transition(self, source: StateId, symbol: Optional[Symbol],
                           target: Optional[StateId] = None) -> "NFA":
        """Remove one branch of a move, or the whole move.

        With ``target`` given, only that one branch goes and any others on the
        same symbol stay -- what a user deleting one drawn edge means. With
        ``target`` left out, delta becomes undefined at ``(source, symbol)``
        entirely, which is what :meth:`DFA.without_transition` does and what a
        caller holding only a ``(state, symbol)`` pair can ask for.
        """
        remaining = dict(self.transitions)
        if target is None:
            remaining.pop((source, symbol), None)
        else:
            survivors = self.targets(source, symbol) - {target}
            if survivors:
                remaining[(source, symbol)] = survivors
            else:
                remaining.pop((source, symbol), None)
        return self._replace(transitions=remaining)

    def with_initial(self, state: Optional[StateId]) -> "NFA":
        """Set, or clear, the start state."""
        return self._replace(initial=state)

    def with_accept(self, state: StateId) -> "NFA":
        """Mark a state accepting."""
        if state not in self.states:
            raise UnknownStateError(f"no such state: {state!r}")
        return self._replace(accept=self.accept | {state})

    def without_accept(self, state: StateId) -> "NFA":
        """Mark a state non-accepting."""
        return self._replace(accept=self.accept - {state})

    def with_accept_toggled(self, state: StateId) -> "NFA":
        """Flip whether a state is accepting."""
        if state in self.accept:
            return self.without_accept(state)
        return self.with_accept(state)

    def with_label(self, state: StateId, text: str) -> "NFA":
        """Set a state's display name."""
        if state not in self.states:
            raise UnknownStateError(f"no such state: {state!r}")
        return self._replace(labels={**dict(self.labels), state: text})

    def with_label_removed(self, state: StateId) -> "NFA":
        """Drop a state's display name, so it shows as its own id again.

        Not the same as setting the label to the id, even though
        :meth:`label_of` cannot tell them apart -- equality can, so writing the
        id back would count as a real edit for a change nobody can see. See
        :meth:`DFA.with_label_removed`.
        """
        if state not in self.labels:
            return self
        return self._replace(
            labels={s: t for s, t in self.labels.items() if s != state})

    def with_states(self, states: Iterable[StateId]) -> "NFA":
        """Add several states at once."""
        automaton = self
        for state in states:
            automaton = automaton.with_state(state)
        return automaton


# ---------------------------------------------------------------------------
# Between the two machines
# ---------------------------------------------------------------------------


def from_dfa(automaton: DFA) -> NFA:
    """The same machine, read as an NFA. Never fails.

    Every DFA is an NFA whose target sets are all singletons, so this is the
    trivial direction of the correspondence and the one that lets the algorithms
    layer take either kind of machine. The language is untouched: for every
    word, this NFA and ``automaton`` reach the same states, halt at the same
    place, and give the same verdict -- including the *reason* for a rejection,
    since a singleton that has no move becomes an empty configuration.
    """
    return NFA(
        states=automaton.states,
        alphabet=automaton.alphabet,
        transitions={
            key: frozenset({target})
            for key, target in automaton.transitions.items()
        },
        initial=automaton.initial,
        accept=automaton.accept,
        labels=automaton.labels,
    )


def to_dfa(automaton: NFA) -> DFA:
    """The same machine as a DFA, when it already is one.

    The exact inverse of :func:`from_dfa`, and deliberately *not* the subset
    construction: this converts a machine that is already deterministic, and
    refuses anything else rather than quietly determinising it. Turning an NFA
    into a DFA changes the state set, and a function that sometimes does that
    and sometimes does not is a function whose result you cannot reason about.
    :func:`fsa.subset.determinize` is the one that builds a new machine.

    Partiality is preserved and is not an obstacle: a missing move stays
    missing, and the DFA rejects with
    :attr:`~fsa.simulate.Verdict.REJECT_NO_TRANSITION` exactly where the NFA
    ran out of branches.

    Raises:
        NondeterministicError: If any state has an epsilon move or two targets
            on one symbol. The message names the offending state and symbol,
            because "this machine is nondeterministic" is not actionable and
            "q0 has 2 targets on 'a'" is.
    """
    transitions: Dict[Tuple[StateId, Symbol], StateId] = {}
    for source, symbol, targets in automaton.sorted_transitions():
        if symbol is None:
            raise NondeterministicError(
                f"cannot convert to a DFA: {source} has an epsilon move (to "
                f"{', '.join(targets)}); determinize it instead")
        if len(targets) > 1:
            raise NondeterministicError(
                f"cannot convert to a DFA: {source} has {len(targets)} targets "
                f"on {symbol!r} ({', '.join(targets)}); determinize it instead")
        transitions[(source, symbol)] = targets[0]

    return DFA(
        states=automaton.states,
        alphabet=automaton.alphabet,
        transitions=transitions,
        initial=automaton.initial,
        accept=automaton.accept,
        labels=automaton.labels,
    )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def _show(states: Iterable[StateId]) -> str:
    """A configuration as text: ``{q0, q1}``, sorted.

    Sorted because a frozenset iterates in an order that changes between
    processes, and a message that reads differently on every run cannot be
    compared, tested, or pasted into a bug report.
    """
    return "{" + ", ".join(sorted(states)) + "}"


@dataclass(frozen=True, slots=True)
class NfaStep:
    """One symbol read, moving the machine from one configuration to the next.

    The counterpart of :class:`fsa.simulate.Step`, with sets where that has
    single states. ``target`` is epsilon-closed, so it is genuinely where the
    machine stands after reading ``symbol`` -- not an intermediate value the
    caller has to finish computing.
    """

    index: int
    """Position in the word of the symbol that was read."""

    source: FrozenSet[StateId]
    symbol: Symbol
    target: FrozenSet[StateId]


@dataclass(frozen=True, slots=True)
class NfaRun:
    """The complete record of one word being simulated on an NFA.

    Mirrors :class:`fsa.simulate.Run` field for field, and reuses its
    :class:`~fsa.simulate.Verdict` rather than inventing a second vocabulary
    for the same five outcomes -- a UI that can already explain a DFA run
    should not need a second code path to explain this one.
    """

    word: str
    start: Optional[FrozenSet[StateId]]
    """The configuration before anything was read, or ``None`` if there was no
    start state. Already epsilon-closed, so a machine that reaches an accepting
    state by epsilon moves alone shows that in its very first frame."""

    steps: Tuple[NfaStep, ...]
    verdict: Verdict
    stopped_at: int
    """Index into ``word`` where the run stopped.

    ``len(word)`` when the whole word was consumed; otherwise the index of the
    symbol that could not be read, which is what a UI needs to point at the
    offending character."""

    offending_symbol: Optional[Symbol] = None
    """The symbol that stopped the run, if one did."""

    @property
    def configurations(self) -> Tuple[FrozenSet[StateId], ...]:
        """Where the machine stood after each symbol read, starting from
        :attr:`start`.

        The NFA's answer to :attr:`fsa.simulate.Run.path`, and named
        differently on purpose: there is no single path through a
        nondeterministic machine, so calling this one would invite a UI to draw
        a line through it. Same invariant, though --
        ``len(configurations) == len(steps) + 1`` whenever there is a start
        state -- and it holds for the same reason: this is derived from the
        steps rather than accumulated beside them, so the sequence and the word
        can be indexed by one counter without disagreeing.
        """
        if self.start is None:
            return ()
        return (self.start,) + tuple(step.target for step in self.steps)

    @property
    def final_states(self) -> FrozenSet[StateId]:
        """The configuration the run halted in; empty if it never began.

        For a run that died for want of a move this is the last *live*
        configuration -- the set of states that had nowhere to go -- not the
        empty set that followed it. That is the useful one: it is what the UI
        should highlight and what the explanation names.
        """
        configurations = self.configurations
        return configurations[-1] if configurations else frozenset()

    @property
    def accepted(self) -> bool:
        return self.verdict.is_accept

    @property
    def consumed(self) -> str:
        """The prefix of the word that was actually read."""
        return self.word[:self.stopped_at]

    @property
    def remaining(self) -> str:
        """The suffix that was never reached."""
        return self.word[self.stopped_at:]

    def explain(self) -> str:
        """One sentence saying what happened and why."""
        shown = f"'{self.word}'" if self.word else "the empty string"

        if self.verdict is Verdict.ACCEPT:
            return f"{shown} was accepted in {_show(self.final_states)}"

        if self.verdict is Verdict.NO_INITIAL_STATE:
            return (f"{shown} could not be run: this automaton has no initial "
                    f"state, so it does not define a language yet")

        if self.verdict is Verdict.REJECT_SYMBOL_NOT_IN_ALPHABET:
            return (f"{shown} was rejected: '{self.offending_symbol}' at position "
                    f"{self.stopped_at} is not in the alphabet")

        if self.verdict is Verdict.REJECT_NO_TRANSITION:
            # Deliberately not the DFA's "this automaton is incomplete". A
            # nondeterministic machine running out of branches is ordinary --
            # every alternative was explored and every one died -- and calling
            # that a defect would teach the wrong lesson.
            return (f"{shown} was rejected: no state in "
                    f"{_show(self.final_states)} has a transition on "
                    f"'{self.offending_symbol}' at position {self.stopped_at}, "
                    f"so every branch died there")

        return (f"{shown} was rejected: the whole string was read, but nothing "
                f"in {_show(self.final_states)} is an accepting state")


def run(automaton: NFA, word: str) -> NfaRun:
    """Simulate ``word`` and return the full record of what happened.

    All branches at once, rather than one at a time with backtracking: the
    machine's state *is* the set of states it could be in, so the cost is one
    pass over the word regardless of how much the alternatives multiply, and
    the sequence of sets falls out as a by-product. That sequence is the thing
    worth showing -- watching a configuration spread across the diagram and
    collapse again is the lesson nondeterminism has to teach.

    Rejections carry the same reasons a DFA's do, with one reading:
    :attr:`~fsa.simulate.Verdict.REJECT_NO_TRANSITION` means the configuration
    became empty, i.e. every branch died. The run stops there rather than
    reading on through an empty set -- the verdict would be the same either
    way, but stopping is what makes ``stopped_at`` point at the symbol nothing
    could read, exactly as the DFA simulator does.
    """
    if automaton.initial is None:
        return NfaRun(word=word, start=None, steps=(),
                      verdict=Verdict.NO_INITIAL_STATE, stopped_at=0)

    # Closed before the first symbol, so an epsilon move out of the start state
    # is already taken by the time position 0 is drawn.
    start = automaton.epsilon_closure([automaton.initial])
    current = start
    steps: List[NfaStep] = []

    for index, symbol in enumerate(word):
        if symbol not in automaton.alphabet:
            return NfaRun(word=word, start=start, steps=tuple(steps),
                          verdict=Verdict.REJECT_SYMBOL_NOT_IN_ALPHABET,
                          stopped_at=index, offending_symbol=symbol)

        moved: Set[StateId] = set()
        # Sorted only so the loop itself is deterministic; the union is a set,
        # so the result could not depend on the order anyway.
        for state in sorted(current):
            moved |= automaton.targets(state, symbol)
        target = automaton.epsilon_closure(moved)

        if not target:
            return NfaRun(word=word, start=start, steps=tuple(steps),
                          verdict=Verdict.REJECT_NO_TRANSITION,
                          stopped_at=index, offending_symbol=symbol)

        steps.append(NfaStep(index=index, source=current, symbol=symbol,
                             target=target))
        current = target

    verdict = (Verdict.ACCEPT if current & automaton.accept
               else Verdict.REJECT_NON_ACCEPTING)
    return NfaRun(word=word, start=start, steps=tuple(steps),
                  verdict=verdict, stopped_at=len(word))


def accepts(automaton: NFA, word: str) -> bool:
    """Whether ``word`` is in the language.

    Accepting means *some* branch survives to the end in an accepting state:
    one accepting state in the final configuration is enough, however many
    others are in there with it. Convenience over :func:`run`.
    """
    return run(automaton, word).accepted
