"""Exceptions raised by the automata engine.

No module in :mod:`fsa` prints. Problems are raised or returned; deciding what
the user sees is the front end's job.
"""


class AutomatonError(Exception):
    """Base class for every error raised by the engine."""


class IllegalSymbolError(AutomatonError):
    """A symbol is not a single printable, non-whitespace character."""


class UnknownStateError(AutomatonError):
    """An operation named a state the automaton does not contain."""


class DuplicateStateError(AutomatonError):
    """An operation would add a state that already exists."""


class IncompleteAutomatonError(AutomatonError):
    """An operation requires a total transition function.

    Complementing a partial DFA is the motivating case: the complement of a
    machine that simply halts on some inputs is not the complement of its
    language. Completing it first is a real step, not a formality.
    """


class NondeterministicError(AutomatonError):
    """An operation requires a deterministic machine and was given a choice.

    Raised by :func:`fsa.nfa.to_dfa`. "Deterministic" here is a statement about
    delta alone: no epsilon moves, and no state with two targets on one symbol.
    A *partial* delta is still deterministic -- a state with no move at all has
    nothing to choose between -- so this is a different complaint from
    :class:`IncompleteAutomatonError`, and it has a different remedy. The fix
    for incompleteness is one trap state; the fix for this is the subset
    construction, which builds a machine with different states, so no operation
    performs it silently.
    """
