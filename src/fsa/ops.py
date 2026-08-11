"""Operations that transform whole automata.

The start of the algorithms layer (completion and trimming now; minimisation
and product constructions later). Everything is a pure function DFA -> DFA.

The two operations here are opposites, and reading them together is the point:
:func:`complete` adds the one dead state that makes delta total, and
:func:`trim` removes every dead state there is. Neither changes the language --
they trade a total transition function against a diagram with nothing useless
in it, and which of the two a user wants depends on what they are about to do
with the machine.
"""

from typing import Dict, Optional, Tuple

from fsa.analysis import co_reachable, is_complete, missing_transitions, reachable
from fsa.automaton import DFA
from fsa.symbols import StateId, Symbol


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


def trim(automaton: DFA) -> DFA:
    """Keep only the states that can appear on an accepting run.

    A state earns its place by being *useful*: reachable from the initial
    state, and able to reach an accepting state. Anything else is scenery -- no
    word's verdict depends on it -- so deleting it cannot change the language.
    Both halves are already derived from the edges by :mod:`fsa.analysis`, and
    trimming is exactly their intersection: ``reachable & co_reachable``.

    Two consequences surprise people, and both are correct.

    **delta gets more partial.** An edge into a removed state is removed with
    it, so a run that used to walk into a trap and sit there now dies for want
    of an arrow. The word is rejected either way; only the *reason* changes,
    from :attr:`~fsa.simulate.Verdict.REJECT_NON_ACCEPTING` to
    :attr:`~fsa.simulate.Verdict.REJECT_NO_TRANSITION`. Since a trimmed machine
    is generally incomplete, operations that demand totality -- complement,
    most obviously -- want :func:`complete` after this, not before.

    **Nothing may survive.** If the initial state is itself useless the result
    has no states at all. That case is not as narrow as it looks: it covers an
    automaton with no accepting states, one whose start state is dead, and one
    with no start state to begin with. The result then keeps the alphabet and
    sets ``initial`` to ``None``, which deliberately weakens the reading from
    "this machine accepts nothing" to "this machine does not define a language
    yet" (what ``initial=None`` means -- see :class:`~fsa.automaton.DFA`).
    Keeping the start state alive purely to preserve the stronger reading would
    leave behind precisely one state that no accepting run can visit, which is
    the one thing this function exists to remove; :meth:`DFA.without_state`
    already sets the precedent that losing the start state clears it rather
    than electing a replacement. Nothing observable is lost either way --
    :func:`fsa.simulate.accepts` rejects every word under both readings -- but
    the front end should expect a diagram to empty out, and
    :func:`fsa.analysis.defects` to start reporting ``no_initial_state``.

    Args:
        automaton: The automaton to trim. When every state is already useful --
            an automaton with no states qualifies -- it is returned unchanged,
            the very same object.

    Returns:
        The trimmed automaton, in which no state is dead and none is
        unreachable. Idempotent, by that postcondition: a second trim finds
        nothing to remove and hands back the same object.
    """
    useful = reachable(automaton) & co_reachable(automaton)
    if useful == automaton.states:
        return automaton

    # An edge survives only if both of its endpoints do. Dropping it is the
    # whole reason delta comes back more partial than it went in.
    kept: Dict[Tuple[StateId, Symbol], StateId] = {
        (source, symbol): target
        for (source, symbol), target in automaton.transitions.items()
        if source in useful and target in useful
    }

    # One constructor call rather than a without_state() per casualty, for the
    # same reason completion builds its transitions in one go: every builder
    # call re-validates the entire automaton.
    #
    # The alphabet is passed through untouched. Trimming is a statement about
    # states, and the language is a set of words over Sigma; narrowing Sigma to
    # the symbols that still label an edge would leave the language alone but
    # change a rejection's reason from "no transition" to "not in the
    # alphabet", which is a lie about a machine the user drew.
    return DFA(
        states=useful,
        alphabet=automaton.alphabet,
        transitions=kept,
        # No special case is hiding in this line. A start state that is not
        # useful makes every state useless -- any state reachable from it would
        # have to reach an accepting state, which would make the start state
        # co-reachable too -- so `useful` is empty exactly when the initial
        # state is missing from it. The automaton keeps its start state or
        # keeps nothing.
        initial=automaton.initial if automaton.initial in useful else None,
        accept=automaton.accept & useful,
        # Labels for removed states are discarded by the constructor, which is
        # what we want: a display name for a state nobody can visit is residue,
        # and it would show up in equality and in the saved file.
        labels=automaton.labels,
    )
