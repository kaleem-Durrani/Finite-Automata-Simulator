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
