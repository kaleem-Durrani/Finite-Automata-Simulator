"""Hypothesis strategies for the engine's values.

Replaces hand-rolled generators built on ``random.Random(seed)``. Those worked,
but a failure handed you a seed and whatever six-state machine it happened to
produce; hypothesis shrinks, so a failure arrives as the two-state machine that
actually reproduces it. On a codebase whose defects have repeatedly been about
*edge* cases -- a partial delta, an epsilon cycle, a state with no outgoing
transitions -- that difference is most of the value of property testing.

Nothing here is imported by ``src/fsa``. CI enforces that: the engine has no
required runtime dependencies and hypothesis is a test tool.
"""

from typing import Optional, Sequence

from hypothesis import strategies as st

import fsa

#: Small by default. Shrinking works best when the search space is not vast,
#: and every property here is about structure rather than scale -- the timing
#: properties live in the benchmarks instead.
DEFAULT_ALPHABET = ("a", "b")


def state_ids(count: int) -> Sequence[str]:
    """``q0 .. q{count-1}``. Fixed names, because the properties are about
    languages and shapes; a strategy over identifiers would shrink toward
    confusing ones without testing anything more."""
    return [f"q{index}" for index in range(count)]


@st.composite
def alphabets(draw: st.DrawFn, *, min_size: int = 1, max_size: int = 3):
    """A legal alphabet: one-character symbols, no duplicates."""
    return frozenset(draw(st.sets(
        st.sampled_from("abc01xyz"), min_size=min_size, max_size=max_size)))


@st.composite
def dfas(draw: st.DrawFn, *, min_states: int = 1, max_states: int = 6,
         alphabet: Optional[Sequence[str]] = None,
         complete: bool = False, with_initial: bool = True) -> fsa.DFA:
    """A deterministic automaton, partial unless ``complete`` is set.

    Partial delta is the default deliberately. A missing transition is a
    distinct rejection reason in this engine, and it is the case that naive
    implementations of every algorithm here get wrong -- so it has to be the
    common case in the generators, not a special one.
    """
    symbols = (frozenset(alphabet) if alphabet is not None
               else draw(alphabets()))
    count = draw(st.integers(min_value=min_states, max_value=max_states))
    ids = state_ids(count)

    automaton = fsa.DFA(states=frozenset(ids), alphabet=symbols)
    if with_initial:
        automaton = automaton.with_initial(ids[0])

    for state in ids:
        if draw(st.booleans()):
            automaton = automaton.with_accept(state)
        for symbol in sorted(symbols):
            if complete or draw(st.booleans()):
                target = draw(st.sampled_from(ids))
                automaton = automaton.with_transition(state, symbol, target)
    return automaton


@st.composite
def words(draw: st.DrawFn, *, alphabet: Sequence[str] = DEFAULT_ALPHABET,
          max_length: int = 8) -> str:
    """A word over ``alphabet``. The empty word is legal and included."""
    return "".join(draw(st.lists(st.sampled_from(sorted(alphabet)),
                                 max_size=max_length)))


@st.composite
def documents(draw: st.DrawFn, **kwargs) -> fsa.Document:
    """A document: an automaton plus coordinates for every one of its states."""
    automaton = draw(dfas(**kwargs))
    return fsa.Document(automaton, fsa.Layout.auto(automaton),
                        len(automaton.states))
