"""Language equivalence, and the counterexample that explains a "no".

Two claims carry this module, and nearly every test below is one of them:

* ``counterexample(a, b)`` is ``None`` exactly when the languages are equal;
* when it is not ``None``, the word really is read differently by the two
  machines -- and no shorter word is.

The hard cases are all the same case: delta is partial, so a word one machine
can read and the other cannot *does* distinguish them, and a missing arrow is a
sink rather than a self-loop or a shrug. Imports no pygame and touches no
display.
"""

import itertools
import random
from typing import Iterator, List, Optional, Sequence

from fsa import DFA, accepts, complete
from fsa.equivalence import counterexample, equivalent

# ---------------------------------------------------------------------------
# Fixtures by hand
# ---------------------------------------------------------------------------


def a_star() -> DFA:
    """One state, accepting, looping on 'a'. Recognises a*."""
    return (DFA()
            .with_states(["q0"])
            .with_transition("q0", "a", "q0")
            .with_accept("q0"))


def a_star_in_three_states() -> DFA:
    """a* again, spread over a 3-cycle so every state is accepting."""
    return (DFA()
            .with_states(["s0", "s1", "s2"])
            .with_transition("s0", "a", "s1")
            .with_transition("s1", "a", "s2")
            .with_transition("s2", "a", "s0")
            .with_accept("s0").with_accept("s1").with_accept("s2"))


def multiples_of(n: int, prefix: str) -> DFA:
    """Unary counter mod ``n``, accepting when the count is a multiple of n."""
    ids = [f"{prefix}{i}" for i in range(n)]
    automaton = DFA().with_states(ids)
    for i, state in enumerate(ids):
        automaton = automaton.with_transition(state, "a", ids[(i + 1) % n])
    return automaton.with_accept(ids[0])


def empty_language() -> DFA:
    """Two states over {a, b}, complete, accepting nothing."""
    return (DFA()
            .with_states(["d0", "d1"])
            .with_transition("d0", "a", "d1")
            .with_transition("d0", "b", "d1")
            .with_transition("d1", "a", "d1")
            .with_transition("d1", "b", "d1"))


# ---------------------------------------------------------------------------
# Brute force, used as an independent oracle
# ---------------------------------------------------------------------------


def words_up_to(symbols: Sequence[str], max_length: int) -> Iterator[str]:
    """Every word over ``symbols``, shortest first, alphabetical within a length."""
    for length in range(max_length + 1):
        for letters in itertools.product(sorted(symbols), repeat=length):
            yield "".join(letters)


def first_disagreement(left: DFA, right: DFA, max_length: int) -> Optional[str]:
    """The first word (by length, then alphabet) the two verdicts differ on.

    Deliberately built out of :func:`fsa.accepts` alone, so it shares no code
    with the pair search it is checking.
    """
    alphabet = sorted(left.alphabet | right.alphabet)
    for word in words_up_to(alphabet, max_length):
        if accepts(left, word) != accepts(right, word):
            return word
    return None


# ---------------------------------------------------------------------------
# The basic contract
# ---------------------------------------------------------------------------


def test_a_machine_is_equivalent_to_itself():
    assert equivalent(a_star(), a_star())
    assert counterexample(a_star(), a_star()) is None


def test_renaming_states_does_not_change_the_language():
    renamed = (DFA()
               .with_states(["start"])
               .with_transition("start", "a", "start")
               .with_accept("start"))
    assert equivalent(a_star(), renamed)


def test_the_same_language_in_a_different_number_of_states():
    assert equivalent(a_star(), a_star_in_three_states())
    assert counterexample(a_star_in_three_states(), a_star()) is None


def test_unreachable_states_cannot_affect_the_answer():
    """The search walks reachable pairs, so a spare component is invisible."""
    padded = (a_star()
              .with_state("spare")
              .with_transition("spare", "a", "spare")
              .with_accept("spare")
              .with_initial("q0"))
    assert padded.states == {"q0", "spare"}
    assert equivalent(a_star(), padded)


def test_differing_on_a_single_word_is_reported():
    accepts_a = (DFA()
                 .with_states(["q0", "q1"])
                 .with_transition("q0", "a", "q1")
                 .with_accept("q1"))
    accepts_nothing = accepts_a.without_accept("q1")
    assert counterexample(accepts_a, accepts_nothing) == "a"
    assert not equivalent(accepts_a, accepts_nothing)


# ---------------------------------------------------------------------------
# The empty string is a legal counterexample
# ---------------------------------------------------------------------------


def test_disagreeing_initial_states_give_the_empty_string():
    accepting_start = a_star()
    rejecting_start = a_star().without_accept("q0")
    word = counterexample(accepting_start, rejecting_start)
    assert word is not None, "'' is a counterexample, not the absence of one"
    assert word == ""
    assert not equivalent(accepting_start, rejecting_start)


def test_the_empty_string_counterexample_is_falsy_but_not_none():
    """The trap this exists to catch: `if counterexample(a, b):` reads a real
    disagreement about '' as agreement. Only a None test is correct."""
    word = counterexample(a_star(), a_star().without_accept("q0"))
    assert not word
    assert word is not None
    assert (word is None) == equivalent(a_star(), a_star().without_accept("q0"))


# ---------------------------------------------------------------------------
# Partial delta: the crux
# ---------------------------------------------------------------------------


def test_a_word_only_one_machine_can_read_distinguishes_them():
    """Both accept nothing they can read, but one can read 'a' into an
    accepting state and the other simply halts."""
    reads_a = (DFA()
               .with_states(["q0", "q1"])
               .with_transition("q0", "a", "q1")
               .with_accept("q1"))
    halts = DFA().with_states(["r0"]).with_symbol("a")
    assert counterexample(reads_a, halts) == "a"
    assert counterexample(halts, reads_a) == "a"


def test_a_missing_transition_is_not_a_self_loop():
    """Nothing-happens is the tempting wrong model: it would make these two
    agree on 'a', because the accepting start state would survive reading it."""
    no_arrows = DFA().with_states(["q0"]).with_symbol("a").with_accept("q0")
    self_loop = a_star()
    assert accepts(no_arrows, "") and accepts(self_loop, "")
    assert counterexample(no_arrows, self_loop) == "a"


def test_the_sink_is_absorbing():
    """Both recognise exactly {'a'}, but only one of them gets there by dying.

    If a missing transition were read as "stay put", the left machine would be
    back at q0 after 'b' and would then accept 'ba' -- a counterexample that
    does not exist. The run really stopped, and nothing revives it.
    """
    dies_on_b = (DFA()
                 .with_states(["q0", "qa"])
                 .with_transition("q0", "a", "qa")
                 .with_symbol("b")
                 .with_accept("qa"))
    absorbs_b = (DFA()
                 .with_states(["r0", "ra", "rb"])
                 .with_transition("r0", "a", "ra")
                 .with_transition("r0", "b", "rb")
                 .with_transition("rb", "a", "rb")
                 .with_accept("ra"))
    assert not accepts(dies_on_b, "ba") and not accepts(absorbs_b, "ba")
    assert counterexample(dies_on_b, absorbs_b) is None
    assert equivalent(dies_on_b, absorbs_b)


def test_dying_early_is_distinguishing_only_where_the_verdicts_differ():
    """One machine reads 'ba' into an accepting state; the other cannot read
    'b' at all. The shortest disagreement is that word, not the prefix."""
    dies_on_b = (DFA()
                 .with_states(["q0", "q1", "q2"])
                 .with_transition("q0", "a", "q1")
                 .with_transition("q1", "a", "q2")
                 .with_symbol("b")
                 .with_accept("q2"))
    reads_b = (DFA()
               .with_states(["r0", "r1", "r2"])
               .with_transition("r0", "a", "r1")
               .with_transition("r1", "a", "r2")
               .with_transition("r0", "b", "r1")
               .with_accept("r2"))
    assert counterexample(dies_on_b, reads_b) == "ba"
    assert first_disagreement(dies_on_b, reads_b, 4) == "ba"


def test_completion_preserves_equivalence():
    """complete() is supposed to change the reason for a rejection and never
    the verdict, which is exactly a claim about equivalence."""
    partial = (DFA()
               .with_states(["q0", "q1"])
               .with_transition("q0", "a", "q0")
               .with_transition("q0", "b", "q1")
               .with_accept("q1"))
    completed, trap = complete(partial)
    assert trap is not None
    assert equivalent(partial, completed)
    assert counterexample(completed, partial) is None


def test_an_incomplete_machine_differs_from_one_with_an_accepting_sink():
    """Completing to a *non*-dead trap would change the language, and the
    counterexample is the first word that used to halt."""
    partial = (DFA()
               .with_states(["q0", "q1"])
               .with_transition("q0", "a", "q1")
               .with_symbol("b")
               .with_accept("q1"))
    forgiving = (partial
                 .with_state("t")
                 .with_transition("q0", "b", "t")
                 .with_transition("t", "a", "t")
                 .with_transition("t", "b", "t")
                 .with_accept("t"))
    assert counterexample(partial, forgiving) == "b"


# ---------------------------------------------------------------------------
# Alphabets that do not match
# ---------------------------------------------------------------------------


def test_a_symbol_only_one_machine_knows_can_distinguish():
    knows_b = (DFA()
               .with_states(["q0", "q1"])
               .with_transition("q0", "b", "q1")
               .with_accept("q1"))
    knows_only_a = DFA().with_states(["r0"]).with_symbol("a")
    assert "b" not in knows_only_a.alphabet
    assert counterexample(knows_b, knows_only_a) == "b"


def test_a_symbol_only_one_machine_knows_need_not_distinguish():
    """An unused symbol is rejected by both, one for want of an arrow and one
    for want of the letter. Same verdict, so it proves nothing."""
    over_a = (DFA()
              .with_states(["q0", "q1"])
              .with_transition("q0", "a", "q1")
              .with_accept("q1"))
    over_ab = over_a.with_symbol("b")
    assert over_a.alphabet == {"a"} and over_ab.alphabet == {"a", "b"}
    assert equivalent(over_a, over_ab)


def test_a_symbol_in_neither_alphabet_is_never_searched():
    """Both machines reject it out of hand, so it cannot be a counterexample --
    and the search over the union never invents it."""
    left, right = a_star(), a_star_in_three_states()
    assert not accepts(left, "z") and not accepts(right, "z")
    assert equivalent(left, right)


# ---------------------------------------------------------------------------
# No initial state
# ---------------------------------------------------------------------------


def test_two_machines_without_a_start_state_are_equivalent():
    left = a_star().with_initial(None)
    right = a_star_in_three_states().with_initial(None)
    assert equivalent(left, right)
    assert equivalent(DFA(), DFA())


def test_no_start_state_matches_the_empty_language():
    """Both accept nothing, one because it cannot start and one because it
    never arrives anywhere accepting."""
    startless = a_star().with_initial(None)
    assert counterexample(startless, empty_language()) is None
    assert equivalent(empty_language(), startless)


def test_no_start_state_differs_from_a_machine_that_accepts_something():
    startless = a_star_in_three_states().with_initial(None)
    assert counterexample(startless, a_star()) == "", "'' is already accepted"

    accepts_only_a = (DFA()
                      .with_states(["q0", "q1"])
                      .with_transition("q0", "a", "q1")
                      .with_accept("q1"))
    assert counterexample(startless, accepts_only_a) == "a"
    assert counterexample(accepts_only_a, startless) == "a"


def test_an_empty_automaton_differs_from_one_accepting_the_empty_string():
    assert counterexample(DFA(), a_star()) == ""
    assert not equivalent(DFA(), a_star())


# ---------------------------------------------------------------------------
# Shortest, and deterministically so
# ---------------------------------------------------------------------------


def test_the_counterexample_is_the_shortest_disagreement():
    """Multiples of 3 against multiples of 5: they agree on '', 'a' and 'aa',
    and part company at three."""
    left, right = multiples_of(3, "p"), multiples_of(5, "m")
    word = counterexample(left, right)
    assert word == "aaa"
    for shorter in words_up_to(["a"], 2):
        assert accepts(left, shorter) == accepts(right, shorter)
    assert accepts(left, word) != accepts(right, word)


def test_breadth_first_beats_a_deep_disagreement():
    """A depth-first walk would follow the 'a' self-loop for ever and never
    reach the one-symbol answer sitting next to the start state."""
    left = (DFA()
            .with_states(["q0", "q1"])
            .with_transition("q0", "a", "q0")
            .with_transition("q0", "b", "q1")
            .with_transition("q1", "a", "q1")
            .with_transition("q1", "b", "q1")
            .with_accept("q1"))
    right = left.without_accept("q1")
    assert counterexample(left, right) == "b"


def test_ties_within_a_length_are_broken_alphabetically():
    """Two disagreements of the same length: the answer is stable, not
    whichever the dictionary happened to yield first."""
    left = (DFA()
            .with_states(["q0", "q1"])
            .with_transition("q0", "a", "q1")
            .with_transition("q0", "b", "q1")
            .with_accept("q1"))
    right = left.without_accept("q1")
    assert accepts(left, "a") and accepts(left, "b")
    assert counterexample(left, right) == "a"
    for _ in range(5):
        assert counterexample(left, right) == "a", "deterministic across runs"


def test_a_long_shortest_counterexample_is_still_found():
    """The disagreement is |Q| deep, so nothing may give up early."""
    length = 9
    ids = [f"q{i}" for i in range(length + 1)]
    chain = DFA().with_states(ids)
    for index in range(length):
        chain = chain.with_transition(ids[index], "a", ids[index + 1])
    left = chain.with_accept(ids[length])
    right = chain
    word = counterexample(left, right)
    assert word == "a" * length
    assert accepts(left, word) != accepts(right, word)


# ---------------------------------------------------------------------------
# Randomised machines
# ---------------------------------------------------------------------------


def random_dfa(rng: random.Random, pool: str = "ab", max_states: int = 4) -> DFA:
    """Small, often partial, occasionally start-less."""
    alphabet = rng.sample(pool, rng.randrange(1, len(pool) + 1))
    ids = [f"q{i}" for i in range(rng.randrange(1, max_states + 1))]
    automaton = DFA().with_states(ids)
    for symbol in alphabet:
        automaton = automaton.with_symbol(symbol)
    for state in ids:
        for symbol in alphabet:
            if rng.random() < 0.7:
                automaton = automaton.with_transition(state, symbol, rng.choice(ids))
    for state in ids:
        if rng.random() < 0.4:
            automaton = automaton.with_accept(state)
    if rng.random() < 0.1:
        automaton = automaton.with_initial(None)
    return automaton


def renamed_copy(automaton: DFA) -> DFA:
    """The same machine with every state renamed: equivalent by construction."""
    fresh = {state: f"{state}#" for state in automaton.states}
    return DFA(
        states=frozenset(fresh.values()),
        alphabet=automaton.alphabet,
        transitions={
            (fresh[source], symbol): fresh[target]
            for (source, symbol), target in automaton.transitions.items()
        },
        initial=None if automaton.initial is None else fresh[automaton.initial],
        accept=frozenset(fresh[state] for state in automaton.accept),
    )


def split_one_state(automaton: DFA, rng: random.Random) -> DFA:
    """Clone a state and send some of its incoming edges to the copy.

    The copy has the same outgoing edges and the same acceptance, so it behaves
    identically -- a language-preserving change that is not a renaming, which
    is what makes it a useful source of equivalent-but-different machines.
    """
    original = rng.choice(sorted(automaton.states))
    clone = f"{original}~"
    transitions = dict(automaton.transitions)
    for symbol, target in automaton.outgoing(original).items():
        transitions[(clone, symbol)] = target
    for (source, symbol), target in automaton.transitions.items():
        if target == original and rng.random() < 0.5:
            transitions[(source, symbol)] = clone
    accept = set(automaton.accept)
    if original in accept:
        accept.add(clone)
    return DFA(
        states=automaton.states | {clone},
        alphabet=automaton.alphabet,
        transitions=transitions,
        initial=automaton.initial,
        accept=frozenset(accept),
    )


def perturbed(automaton: DFA, rng: random.Random) -> DFA:
    """One small edit, which usually -- not always -- changes the language."""
    choice = rng.randrange(3)
    if choice == 0:
        return automaton.with_accept_toggled(rng.choice(sorted(automaton.states)))
    if choice == 1 and automaton.transitions:
        source, symbol = rng.choice(sorted(automaton.transitions))
        return automaton.without_transition(source, symbol)
    if automaton.alphabet:
        state = rng.choice(sorted(automaton.states))
        symbol = rng.choice(sorted(automaton.alphabet))
        return automaton.with_transition(state, symbol,
                                         rng.choice(sorted(automaton.states)))
    return automaton


def random_pairs(rng: random.Random, count: int) -> List[tuple]:
    """A mix of equivalent, nearly-equivalent and unrelated pairs.

    Purely random pairs almost always differ, which would leave the ``None``
    branch untested; the language-preserving variants are what exercise it.
    """
    pairs: List[tuple] = []
    for index in range(count):
        left = random_dfa(rng)
        kind = index % 5
        if kind == 0:
            right = left
        elif kind == 1:
            right = renamed_copy(left)
        elif kind == 2:
            right = split_one_state(left, rng)
        elif kind == 3:
            right = perturbed(complete(left)[0], rng)
        else:
            right = random_dfa(rng)
        pairs.append((left, right))
    return pairs


def test_the_counterexample_agrees_with_an_exhaustive_search():
    """The whole contract at once, against brute force: None exactly when no
    word up to the bound differs, and otherwise *the* first differing word --
    shortest, and alphabetically least among the shortest."""
    rng = random.Random(20260811)
    bound = 7
    verdicts = {True: 0, False: 0}
    for left, right in random_pairs(rng, 400):
        word = counterexample(left, right)
        brute = first_disagreement(left, right, bound)
        verdicts[word is None] += 1
        if word is None:
            assert brute is None, f"missed {brute!r} on {left!r} vs {right!r}"
        else:
            assert accepts(left, word) != accepts(right, word), \
                f"{word!r} is not distinguishing for {left!r} vs {right!r}"
            if len(word) <= bound:
                assert word == brute, f"expected {brute!r}, got {word!r}"
            else:
                assert brute is None, "brute force found a shorter one"
    assert verdicts[True] > 50, "the equivalent case must actually be covered"
    assert verdicts[False] > 50, "so must the inequivalent one"


def test_equivalent_is_exactly_the_absence_of_a_counterexample():
    rng = random.Random(7)
    for left, right in random_pairs(rng, 200):
        assert equivalent(left, right) == (counterexample(left, right) is None)


def test_every_machine_is_equivalent_to_itself():
    rng = random.Random(99)
    for _ in range(150):
        automaton = random_dfa(rng, pool="ab01", max_states=6)
        assert counterexample(automaton, automaton) is None
        assert equivalent(automaton, automaton)


def test_the_search_is_symmetric():
    """Swapping the arguments swaps the sides of every pair and nothing else,
    so the same word must come back."""
    rng = random.Random(1234)
    for left, right in random_pairs(rng, 200):
        assert counterexample(left, right) == counterexample(right, left)


def test_language_preserving_edits_are_never_reported_as_differences():
    """Renaming and state-splitting cannot change a language; if either ever
    produces a counterexample, the search has a false positive."""
    rng = random.Random(31337)
    for _ in range(150):
        automaton = random_dfa(rng, pool="ab01", max_states=5)
        assert equivalent(automaton, renamed_copy(automaton))
        assert equivalent(automaton, split_one_state(automaton, rng))
        assert equivalent(automaton, complete(automaton)[0])


def test_equivalence_is_transitive_over_the_variants():
    rng = random.Random(2718)
    for _ in range(100):
        first = random_dfa(rng)
        second = renamed_copy(first)
        third = split_one_state(second, rng)
        assert equivalent(first, second) and equivalent(second, third)
        assert equivalent(first, third)
