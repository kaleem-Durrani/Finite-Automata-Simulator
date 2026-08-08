"""Enumerating the language an automaton recognises.

Breadth-first over words, so results come out shortest-first and in a stable
order. That matters more than it sounds: "here are the first ten strings your
machine accepts" is the fastest way to find out whether it recognises what you
meant, and a list that changes order between runs is useless for comparing.

The search is over *words*, not over states, so it works on a partial automaton
and terminates on an empty language given a length bound.
"""

from collections import deque
from typing import Deque, Iterable, Iterator, List, Optional, Tuple

from fsa.automaton import DFA
from fsa.simulate import Verdict, run
from fsa.symbols import StateId, Symbol

#: Refuse to search beyond this unless told otherwise, so an empty language on
#: a large alphabet cannot spin forever.
DEFAULT_MAX_LENGTH = 12


def _frontier(automaton: DFA, max_length: int) -> Iterator[Tuple[str, StateId]]:
    """Yield (word, state) in breadth-first order, shortest words first.

    Walks the reachable configurations rather than all strings over the
    alphabet, so the branching factor is the alphabet size but dead ends are
    pruned as soon as delta is undefined.
    """
    if automaton.initial is None:
        return

    alphabet = sorted(automaton.alphabet)
    queue: Deque[Tuple[str, StateId]] = deque([("", automaton.initial)])
    while queue:
        word, state = queue.popleft()
        yield word, state
        if len(word) >= max_length:
            continue
        for symbol in alphabet:
            target = automaton.target(state, symbol)
            if target is not None:
                queue.append((word + symbol, target))


def sample_language(automaton: DFA, limit: int = 10,
                    max_length: int = DEFAULT_MAX_LENGTH) -> List[str]:
    """The first ``limit`` accepted words, shortest first.

    Ties within a length are broken by alphabetical order of the symbols, so
    the result is deterministic and can be compared between runs -- which is
    what lets the README's examples be generated rather than hand-written and,
    as they once were, wrong.
    """
    found: List[str] = []
    for word, state in _frontier(automaton, max_length):
        if state in automaton.accept:
            found.append(word)
            if len(found) >= limit:
                break
    return found


def shortest_accepted(automaton: DFA,
                      max_length: int = DEFAULT_MAX_LENGTH) -> Optional[str]:
    """The shortest accepted word, or ``None`` if none was found in range."""
    words = sample_language(automaton, limit=1, max_length=max_length)
    return words[0] if words else None


def sample_rejected(automaton: DFA, limit: int = 10,
                    max_length: int = DEFAULT_MAX_LENGTH) -> List[str]:
    """The first ``limit`` words the automaton rejects, shortest first.

    Includes words that halt for want of a transition: an incomplete machine
    rejects those, and pretending otherwise would misrepresent what it does.
    """
    found: List[str] = []
    alphabet = sorted(automaton.alphabet)
    queue: Deque[str] = deque([""])
    seen = 0

    while queue and len(found) < limit and seen < 200_000:
        word = queue.popleft()
        seen += 1
        result = run(automaton, word)
        if result.verdict is not Verdict.ACCEPT:
            found.append(word)
        if len(word) < max_length:
            for symbol in alphabet:
                queue.append(word + symbol)
    return found


def is_empty(automaton: DFA, max_length: int = DEFAULT_MAX_LENGTH) -> bool:
    """Whether no word up to ``max_length`` is accepted.

    Not the same as "the language is empty" in general -- that needs
    reachability, which :func:`fsa.analysis.co_reachable` answers exactly. This
    is the cheap check the CLI uses when reporting a sample.
    """
    return not sample_language(automaton, limit=1, max_length=max_length)


def words_up_to(alphabet: Iterable[Symbol], length: int) -> Iterator[str]:
    """Every word over ``alphabet`` up to ``length``, shortest first."""
    symbols = sorted(alphabet)
    queue: Deque[str] = deque([""])
    while queue:
        word = queue.popleft()
        yield word
        if len(word) < length:
            for symbol in symbols:
                queue.append(word + symbol)
