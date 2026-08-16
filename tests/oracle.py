"""An independent implementation to check ours against.

`automata-lib` is a mature, widely used automata package. It is **not** used to
implement anything in this project -- implementing the subset construction and
Moore's algorithm is the point of a teaching tool, and handing them to a library
would leave a GUI wrapped around somebody else's engine.

It is used here as a *differential oracle*. We compute an answer, it computes
the same answer independently, and the test asserts they agree over thousands of
generated machines. Two implementations agreeing is far stronger evidence than
either one agreeing with its own tests, which is the failure mode this file
exists to close: a module that only tests itself can be consistently wrong.

Imported by tests only. CI enforces that ``src/`` never mentions it.
"""

from typing import Dict

from automata.fa.dfa import DFA as OracleDFA

import fsa


class Unsupported(Exception):
    """The oracle cannot represent this automaton.

    Raised rather than silently skipped, so a differential test says what it
    declined to check instead of quietly checking nothing.
    """


def to_oracle(automaton: fsa.DFA) -> OracleDFA:
    """Convert one of our automata into the oracle's representation.

    Two differences to bridge. The oracle nests its transition table
    (``{state: {symbol: target}}``) where ours is flat, and it needs to be told
    explicitly that delta may be partial -- ours is partial by default, and a
    partial machine is exactly the case worth checking.
    """
    if automaton.initial is None:
        raise Unsupported("the oracle requires an initial state")
    if not automaton.alphabet:
        raise Unsupported("the oracle requires a non-empty alphabet")

    nested: Dict[str, Dict[str, str]] = {
        state: {} for state in automaton.states}
    for (source, symbol), target in automaton.transitions.items():
        nested[source][symbol] = target

    return OracleDFA(
        states=set(automaton.states),
        input_symbols=set(automaton.alphabet),
        transitions=nested,
        initial_state=automaton.initial,
        final_states=set(automaton.accept),
        allow_partial=True,
    )


def widen(automaton: fsa.DFA, alphabet: frozenset) -> fsa.DFA:
    """The same language, over a larger alphabet.

    Adding a symbol with no transitions leaves delta partial on it, so every
    word using that symbol still dies exactly where it did before. The language
    is untouched; only the declared alphabet grows.
    """
    for symbol in sorted(alphabet - automaton.alphabet):
        automaton = automaton.with_symbol(symbol)
    return automaton


def same_language(left: fsa.DFA, right: fsa.DFA) -> bool:
    """Whether the oracle considers the two automata language-equal.

    Both are widened to the union of their alphabets first. The oracle's ``==``
    is alphabet-*sensitive*: it calls the empty language over ``{a}`` different
    from the empty language over ``{b}``, though both accept exactly nothing.
    Language equality is a question about sets of strings and does not care
    which symbols were declared, so comparing without widening reports a
    disagreement that is really a difference of definition.

    Hypothesis found that on the first run and shrank it to exactly those two
    one-state machines -- which is what a differential oracle is for; it just
    caught the bridge rather than the engine.
    """
    union = frozenset(left.alphabet) | frozenset(right.alphabet)
    return bool(to_oracle(widen(left, union)) == to_oracle(widen(right, union)))


def oracle_accepts(automaton: fsa.DFA, word: str) -> bool:
    """Whether the oracle's simulator accepts the word.

    The oracle raises on a word containing a symbol outside the alphabet, where
    ours rejects it with a reason. That difference is deliberate on our side, so
    the caller decides -- here we report the rejection the engine would.
    """
    if any(symbol not in automaton.alphabet for symbol in word):
        return False
    return bool(to_oracle(automaton).accepts_input(word))
