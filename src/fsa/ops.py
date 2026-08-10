"""Operations that transform whole automata.

The start of the algorithms layer (completion now; minimisation and product
constructions later). Everything is a pure function DFA -> DFA.
"""

from typing import Optional, Tuple

from fsa.analysis import is_complete, missing_transitions
from fsa.automaton import DFA
from fsa.symbols import StateId


def complete(automaton: DFA,
             trap_id: Optional[StateId] = None) -> Tuple[DFA, Optional[StateId]]:
    """Make delta total by routing every undefined pair to a trap state.

    Completion never changes the language: every previously-defined run is
    untouched, and every previously-undefined run now dies in the trap. The
    trap loops on every symbol and accepts nothing, so it is a genuine dead
    state -- derived from its edges by :func:`fsa.analysis.dead_states`, never
    flagged. (A flag is what let the old model compute a different language
    than the diagram it drew, so nothing flag-like is added here.)

    Args:
        automaton: The automaton to complete. When delta is already total --
            an empty alphabet has nothing missing -- it is returned unchanged,
            the very same object.
        trap_id: Where undefined pairs should go. Naming an existing state
            reuses it, which puts the language guarantee in the caller's
            hands: it holds only if that state is already dead. ``None``
            picks the first of ``trap``, ``trap1``, ``trap2``, ... that is
            not yet a state.

    Returns:
        The completed automaton and the trap's id, or ``(automaton, None)``
        when there was nothing to do.
    """
    if is_complete(automaton):
        return automaton, None

    if trap_id is None:
        trap_id = "trap"
        suffix = 1
        while trap_id in automaton.states:
            trap_id = f"trap{suffix}"
            suffix += 1

    trapped = automaton
    if trap_id not in trapped.states:
        # with_state promotes its state to initial when there is none. A
        # completion that invents a start state would turn "language not
        # defined yet" into "the empty language" behind the user's back, so
        # the original (possibly absent) initial state is restored.
        trapped = trapped.with_state(trap_id).with_initial(automaton.initial)

    # One constructor call rather than a with_transition() per pair: the
    # missing set can be |Q| x |Sigma| large, and every builder call
    # re-validates the whole automaton.
    transitions = dict(trapped.transitions)
    for pair in missing_transitions(trapped):
        transitions[pair] = trap_id
    completed = DFA(
        states=trapped.states,
        alphabet=trapped.alphabet,
        transitions=transitions,
        initial=trapped.initial,
        accept=trapped.accept,
        labels=trapped.labels,
    )
    return completed, trap_id
