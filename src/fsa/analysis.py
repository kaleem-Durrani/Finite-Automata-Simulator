"""Structural facts about an automaton.

This module is what replaces ``StateType.DEAD_END``. A trap was a flag the user
set by hand, and simulation short-circuited on it -- so the tool could compute a
different language than the diagram it drew. Here trap-ness is *derived* from
the transition function: a state is dead when no accepting state is reachable
from it. It cannot disagree with the picture, because it is computed from the
picture.

Everything here is a pure function of the automaton, which makes all of it
unit-testable without a display.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Set, Tuple

from fsa.automaton import DFA
from fsa.symbols import StateId, Symbol


def missing_transitions(automaton: DFA) -> Tuple[Tuple[StateId, Symbol], ...]:
    """Every ``(state, symbol)`` pair for which delta is undefined.

    Sorted, so the result is stable to display and to compare.
    """
    return tuple(sorted(
        (state, symbol)
        for state in automaton.states
        for symbol in automaton.alphabet
        if (state, symbol) not in automaton.transitions
    ))


def is_complete(automaton: DFA) -> bool:
    """Whether delta is total: defined for every state and every symbol."""
    return not missing_transitions(automaton)


def reachable(automaton: DFA) -> FrozenSet[StateId]:
    """States reachable from the initial state by some word."""
    if automaton.initial is None:
        return frozenset()

    seen = {automaton.initial}
    queue = deque([automaton.initial])
    while queue:
        state = queue.popleft()
        for target in automaton.outgoing(state).values():
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return frozenset(seen)


def _incoming_map(automaton: DFA) -> Dict[StateId, Set[StateId]]:
    """Reverse edges: state -> the states with a transition into it."""
    incoming: Dict[StateId, Set[StateId]] = {state: set() for state in automaton.states}
    for (source, _symbol), target in automaton.transitions.items():
        incoming[target].add(source)
    return incoming


def co_reachable(automaton: DFA) -> FrozenSet[StateId]:
    """States from which some accepting state is reachable.

    Reverse breadth-first search from the accepting set.
    """
    incoming = _incoming_map(automaton)
    seen = set(automaton.accept)
    queue = deque(automaton.accept)
    while queue:
        state = queue.popleft()
        for source in incoming[state]:
            if source not in seen:
                seen.add(source)
                queue.append(source)
    return frozenset(seen)


def dead_states(automaton: DFA) -> FrozenSet[StateId]:
    """States from which no accepting state can ever be reached.

    Once the run enters one of these the answer is already decided, whatever
    the rest of the word says. This is the honest version of a "trap": derived,
    not declared.
    """
    return automaton.states - co_reachable(automaton)


def unreachable_states(automaton: DFA) -> FrozenSet[StateId]:
    """States no word can ever reach. They cannot affect the language."""
    return automaton.states - reachable(automaton)


def is_trap(automaton: DFA, state: StateId) -> bool:
    """Whether ``state`` is dead: no path from it reaches an accepting state."""
    return state in dead_states(automaton)


@dataclass(frozen=True, slots=True)
class Defect:
    """Something structurally wrong or suspicious about an automaton."""

    kind: str
    message: str
    states: Tuple[StateId, ...] = ()
    pairs: Tuple[Tuple[StateId, Symbol], ...] = ()

    @property
    def is_blocking(self) -> bool:
        """Whether this stops the automaton defining a language at all."""
        return self.kind in {"no_initial_state", "no_accepting_states"}


def defects(automaton: DFA) -> Tuple[Defect, ...]:
    """
    Everything worth telling the user about this automaton.

    Ordered most serious first. Each is a fact about the structure, not a
    judgement about the user's intent -- an unreachable state is worth pointing
    out, but it is not an error.
    """
    found: List[Defect] = []

    if automaton.initial is None:
        found.append(Defect(
            kind="no_initial_state",
            message="No initial state, so this automaton does not define a "
                    "language yet.",
        ))

    if not automaton.accept:
        found.append(Defect(
            kind="no_accepting_states",
            message="No accepting states, so no string can be accepted.",
        ))

    missing = missing_transitions(automaton)
    if missing:
        shown = ", ".join(f"{state} on '{symbol}'" for state, symbol in missing[:4])
        if len(missing) > 4:
            shown += f", and {len(missing) - 4} more"
        found.append(Defect(
            kind="incomplete",
            message=f"The transition function is incomplete ({len(missing)} "
                    f"missing): {shown}. Strings can be rejected for want of an "
                    f"arrow rather than by the language.",
            pairs=missing,
        ))

    dead = dead_states(automaton)
    reachable_dead = tuple(sorted(dead & reachable(automaton)))
    if reachable_dead:
        found.append(Defect(
            kind="dead_states",
            message=f"No accepting state can be reached from "
                    f"{', '.join(reachable_dead)}. Once a run arrives there the "
                    f"answer is already decided.",
            states=reachable_dead,
        ))

    unreachable = tuple(sorted(unreachable_states(automaton)))
    if unreachable:
        found.append(Defect(
            kind="unreachable_states",
            message=f"No string can reach {', '.join(unreachable)}; "
                    f"{'they have' if len(unreachable) > 1 else 'it has'} no "
                    f"effect on the language.",
            states=unreachable,
        ))

    return tuple(found)
