"""Running a word through an automaton.

The important thing here is that a rejection has a *reason*. There are four
distinct ways a run can fail, they are distinguishable, and a student who is
told which one happened learns something; a student told only "REJECTED" does
not. Everything the user sees about a run -- the on-screen result, the CLI
output, the exported trace -- comes from :meth:`Run.explain`.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from fsa.automaton import DFA
from fsa.symbols import StateId, Symbol


class Verdict(str, Enum):
    """Why a run ended the way it did."""

    ACCEPT = "accept"
    """Consumed the whole word and halted in an accepting state."""

    REJECT_NON_ACCEPTING = "reject_non_accepting"
    """Consumed the whole word, but halted in a non-accepting state."""

    REJECT_NO_TRANSITION = "reject_no_transition"
    """delta is undefined for the current state and the next symbol.

    The machine is incomplete, not the word wrong. Distinguishing this from a
    plain rejection is the difference between "your string is not in the
    language" and "you forgot to draw an arrow".
    """

    REJECT_SYMBOL_NOT_IN_ALPHABET = "reject_symbol_not_in_alphabet"
    """The word contains a character the alphabet does not.

    Strictly the word is not over Sigma at all, so it is outside the language
    by definition rather than by computation.
    """

    NO_INITIAL_STATE = "no_initial_state"
    """There is no start state, so the language is undefined."""

    @property
    def is_accept(self) -> bool:
        return self is Verdict.ACCEPT


@dataclass(frozen=True, slots=True)
class Step:
    """One transition taken while reading a word."""

    index: int
    """Position in the word of the symbol that was read."""

    source: StateId
    symbol: Symbol
    target: StateId


@dataclass(frozen=True, slots=True)
class Run:
    """The complete record of one word being simulated."""

    word: str
    start: Optional[StateId]
    steps: Tuple[Step, ...]
    verdict: Verdict
    stopped_at: int
    """Index into ``word`` where the run stopped.

    Equal to ``len(word)`` when the whole word was consumed. For the two early
    stops it is the index of the symbol that could not be read -- which is what
    a UI needs in order to point at the offending character.
    """

    offending_symbol: Optional[Symbol] = None
    """The symbol that stopped the run, if one did."""

    @property
    def path(self) -> Tuple[StateId, ...]:
        """States visited, in order.

        Invariant: ``len(path) == len(steps) + 1`` whenever there is a start
        state. Deriving it from the steps rather than accumulating it separately
        is what keeps that true; a UI can index the path and the word by the
        same counter without them disagreeing.
        """
        if self.start is None:
            return ()
        return (self.start,) + tuple(step.target for step in self.steps)

    @property
    def final_state(self) -> Optional[StateId]:
        """The state the run halted in, if any."""
        path = self.path
        return path[-1] if path else None

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
            return f"{shown} was accepted in {self.final_state}"

        if self.verdict is Verdict.NO_INITIAL_STATE:
            return (f"{shown} could not be run: this automaton has no initial "
                    f"state, so it does not define a language yet")

        if self.verdict is Verdict.REJECT_SYMBOL_NOT_IN_ALPHABET:
            return (f"{shown} was rejected: '{self.offending_symbol}' at position "
                    f"{self.stopped_at} is not in the alphabet")

        if self.verdict is Verdict.REJECT_NO_TRANSITION:
            return (f"{shown} was rejected: no transition from "
                    f"{self.path[-1]} on '{self.offending_symbol}' at position "
                    f"{self.stopped_at} -- this automaton is incomplete")

        return (f"{shown} was rejected: the whole string was read, but "
                f"{self.final_state} is not an accepting state")


def run(automaton: DFA, word: str) -> Run:
    """
    Simulate ``word`` and return the full record of what happened.

    Reads left to right, stopping at the first symbol it cannot process.
    """
    if automaton.initial is None:
        return Run(word=word, start=None, steps=(),
                   verdict=Verdict.NO_INITIAL_STATE, stopped_at=0)

    current = automaton.initial
    steps: List[Step] = []

    for index, symbol in enumerate(word):
        if symbol not in automaton.alphabet:
            return Run(word=word, start=automaton.initial, steps=tuple(steps),
                       verdict=Verdict.REJECT_SYMBOL_NOT_IN_ALPHABET,
                       stopped_at=index, offending_symbol=symbol)

        target = automaton.target(current, symbol)
        if target is None:
            return Run(word=word, start=automaton.initial, steps=tuple(steps),
                       verdict=Verdict.REJECT_NO_TRANSITION,
                       stopped_at=index, offending_symbol=symbol)

        steps.append(Step(index=index, source=current, symbol=symbol, target=target))
        current = target

    verdict = (Verdict.ACCEPT if current in automaton.accept
               else Verdict.REJECT_NON_ACCEPTING)
    return Run(word=word, start=automaton.initial, steps=tuple(steps),
               verdict=verdict, stopped_at=len(word))


def accepts(automaton: DFA, word: str) -> bool:
    """Whether ``word`` is in the language. Convenience over :func:`run`."""
    return run(automaton, word).accepted
