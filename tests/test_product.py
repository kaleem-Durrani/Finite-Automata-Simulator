"""The product construction and the boolean operations over it.

One claim dominates this file: for every word ``w``,

    accepts(op(a, b), w) == op_on_booleans(accepts(a, w), accepts(b, w))

including the words that are *not* over both alphabets, which is where naive
product implementations go wrong. The rest tests the two things that make that
claim hold -- the union alphabet and the finished-side marker -- plus
complement's refusal to work on a partial machine.

Imports no pygame and touches no display.
"""

import random
from typing import FrozenSet

import pytest

from fsa import DFA, accepts, is_complete, ops, run
from fsa.errors import AutomatonError, IncompleteAutomatonError
from fsa.language import words_up_to
from fsa.product import (
    complement,
    difference,
    intersection,
    product,
    symmetric_difference,
    union,
)
from fsa.simulate import Verdict

# ---------------------------------------------------------------------------
# Fixtures by hand
# ---------------------------------------------------------------------------


def only_a() -> DFA:
    """Exactly the word "a", over the alphabet {a}."""
    return (DFA()
            .with_states(["s0", "s1"])
            .with_transition("s0", "a", "s1")
            .with_accept("s1"))


def only_b() -> DFA:
    """Exactly the word "b", over the alphabet {b}."""
    return (DFA()
            .with_states(["t0", "t1"])
            .with_transition("t0", "b", "t1")
            .with_accept("t1"))


def even_zeros() -> DFA:
    """An even number of 0s, over {0, 1}. Complete."""
    return (DFA()
            .with_states(["e", "o"])
            .with_transition("e", "0", "o")
            .with_transition("o", "0", "e")
            .with_transition("e", "1", "e")
            .with_transition("o", "1", "o")
            .with_accept("e"))


def ends_in_one() -> DFA:
    """Ends in 1, over {0, 1}. Complete."""
    return (DFA()
            .with_states(["n", "y"])
            .with_transition("n", "0", "n")
            .with_transition("n", "1", "y")
            .with_transition("y", "0", "n")
            .with_transition("y", "1", "y")
            .with_accept("y"))


def partial_a_star_b() -> DFA:
    """a*b, with q1's outgoing arrows deliberately missing."""
    return (DFA()
            .with_states(["q0", "q1"])
            .with_transition("q0", "a", "q0")
            .with_transition("q0", "b", "q1")
            .with_accept("q1"))


# ---------------------------------------------------------------------------
# The alphabets may differ -- the subtlety the whole module exists for
# ---------------------------------------------------------------------------


def test_the_product_alphabet_is_the_union():
    assert union(only_a(), only_b()).alphabet == frozenset("ab")
    assert intersection(only_a(), only_b()).alphabet == frozenset("ab")


def test_union_over_disjoint_alphabets_accepts_words_from_both():
    """The headline case. A symbol foreign to one side makes that side's delta
    undefined; if that undefinedness is allowed to kill the *pair*, this union
    accepts nothing at all."""
    both = union(only_a(), only_b())
    assert accepts(both, "a")
    assert accepts(both, "b")
    assert not accepts(both, "")
    assert not accepts(both, "ab")
    assert not accepts(both, "aa")


def test_a_foreign_symbol_is_a_rejection_for_that_side_not_a_crash():
    """The right operand has never heard of 'a'; it must simply lose."""
    only_left = difference(only_a(), only_b())
    assert accepts(only_left, "a")
    assert not accepts(only_left, "b")


def test_intersection_over_disjoint_alphabets_keeps_only_shared_words():
    left = only_a().with_accept("s0")     # {"", "a"}
    right = only_b().with_accept("t0")    # {"", "b"}
    shared = intersection(left, right)
    assert accepts(shared, "")
    assert not accepts(shared, "a")
    assert not accepts(shared, "b")


def test_the_product_is_complete_even_when_both_operands_are_partial():
    """The finished-side pair is a genuine state with genuine edges, so delta
    on the result is total. That is what lets complement() be applied to a
    product without completing it first."""
    combined = union(partial_a_star_b(), only_a())
    assert is_complete(combined)
    for state in combined.states:
        for symbol in combined.alphabet:
            assert combined.target(state, symbol) is not None


# ---------------------------------------------------------------------------
# Only reachable pairs
# ---------------------------------------------------------------------------


def test_only_reachable_pairs_are_built():
    """A machine against itself reaches only its own diagonal, plus the pair
    where both have failed together."""
    chain = (DFA()
             .with_states(["c0", "c1", "c2", "c3"])
             .with_transition("c0", "a", "c1")
             .with_transition("c1", "a", "c2")
             .with_transition("c2", "a", "c3")
             .with_accept("c3"))
    squared = intersection(chain, chain)
    assert len(squared.states) == 5, "4 diagonal pairs and (-,-), not 16"
    assert accepts(squared, "aaa")
    assert not accepts(squared, "aa")


def test_the_finished_pair_is_only_built_when_something_reaches_it():
    complete_pair = intersection(even_zeros(), ends_in_one())
    assert len(complete_pair.states) == 4
    assert all("-" not in state for state in complete_pair.states)


# ---------------------------------------------------------------------------
# Naming the pairs
# ---------------------------------------------------------------------------


def test_state_ids_spell_the_pair_they_stand_for():
    combined = intersection(even_zeros(), ends_in_one())
    assert combined.initial == "(e,n)"
    assert combined.states == {"(e,n)", "(o,n)", "(e,y)", "(o,y)"}


def test_a_failed_side_is_visible_in_the_id():
    combined = union(only_a(), only_b())
    assert "(-,-)" in combined.states
    assert "(s1,-)" in combined.states


def test_ambiguous_pair_names_are_disambiguated():
    """State ids are opaque strings, so a comma in one is not the engine's
    business -- but two different pairs must still not share an id."""
    left = (DFA()
            .with_states(["a,b", "a"])
            .with_transition("a,b", "0", "a"))
    right = (DFA()
             .with_states(["c", "b,c"])
             .with_transition("c", "0", "b,c"))
    combined = union(left, right)
    assert combined.initial == "(a,b,c)"
    assert "(a,b,c)#1" in combined.states, "the second pair spells the same"
    assert len(combined.states) == len(set(combined.states))
    assert combined.target("(a,b,c)", "0") == "(a,b,c)#1"


def test_a_state_literally_named_dash_does_not_collide_with_the_marker():
    """"-" is how a finished side is written, but it cannot be reserved: state
    ids are the front end's to choose. A real state called "-" is disambiguated
    like any other clash."""
    left = (DFA()
            .with_states(["l0"])
            .with_transition("l0", "a", "l0")
            .with_transition("l0", "b", "l0"))
    right = (DFA()
             .with_states(["r0", "-"])
             .with_transition("r0", "a", "-"))
    combined = intersection(left, right)
    assert combined.target("(l0,r0)", "a") == "(l0,-)", "the real state"
    assert combined.target("(l0,r0)", "b") == "(l0,-)#1", "the finished side"
    assert len(combined.states) == 3


def test_the_result_is_deterministic():
    """Structural equality on the output is only meaningful if set iteration
    order cannot leak into the state names."""
    first = symmetric_difference(even_zeros(), ends_in_one())
    second = symmetric_difference(even_zeros(), ends_in_one())
    assert first == second


# ---------------------------------------------------------------------------
# An operand with no start state
# ---------------------------------------------------------------------------


def test_a_missing_start_state_propagates_as_undefined_not_as_empty():
    """"No language yet" is not "the empty language". Treating it as empty
    would make `union(a, b)` quietly equal `L(b)` while the user was midway
    through building `a`."""
    for combined in (union(only_a().with_initial(None), only_b()),
                     union(only_a(), only_b().with_initial(None))):
        assert combined.initial is None
        assert combined.states == frozenset()
        assert combined.alphabet == frozenset("ab")
        assert run(combined, "b").verdict is Verdict.NO_INITIAL_STATE


# ---------------------------------------------------------------------------
# complement
# ---------------------------------------------------------------------------


def test_complement_flips_the_accepting_set_and_nothing_else():
    original = even_zeros()
    flipped = complement(original)
    assert flipped.states == original.states
    assert flipped.alphabet == original.alphabet
    assert dict(flipped.transitions) == dict(original.transitions)
    assert flipped.initial == original.initial
    assert flipped.accept == frozenset({"o"})


def test_complement_is_its_own_inverse():
    assert complement(complement(even_zeros())) == even_zeros()


def test_complement_flips_membership_for_every_word_over_the_alphabet():
    original = even_zeros()
    flipped = complement(original)
    for word in words_up_to("01", 6):
        assert accepts(flipped, word) != accepts(original, word)


def test_complement_is_relative_to_the_alphabet():
    """A word that is not over Sigma is in neither language: no DFA over Sigma
    accepts it, so complementing cannot pick it up."""
    flipped = complement(even_zeros())
    assert not accepts(even_zeros(), "01x")
    assert not accepts(flipped, "01x")
    assert run(flipped, "01x").verdict is Verdict.REJECT_SYMBOL_NOT_IN_ALPHABET


def test_complement_refuses_a_partial_automaton():
    with pytest.raises(IncompleteAutomatonError) as raised:
        complement(partial_a_star_b())
    message = str(raised.value)
    assert "fsa.ops.complete" in message, "the error must say what to do"
    assert "q1" in message, "and where the hole is"
    assert issubclass(IncompleteAutomatonError, AutomatonError)


def test_the_refusal_is_the_lesson_and_completing_lifts_it():
    partial = partial_a_star_b()
    completed, _ = ops.complete(partial)
    flipped = complement(completed)
    for word in words_up_to("ab", 5):
        assert accepts(flipped, word) != accepts(partial, word)


def test_completing_first_is_not_the_same_as_flipping_in_place():
    """Why the exception exists: the naive answer disagrees with the real one
    on exactly the words the partial machine could not read."""
    partial = partial_a_star_b()
    naive = DFA(states=partial.states, alphabet=partial.alphabet,
                transitions=partial.transitions, initial=partial.initial,
                accept=partial.states - partial.accept)
    completed, _ = ops.complete(partial)
    correct = complement(completed)
    assert not accepts(naive, "ba"), "the run dies before the flipped F matters"
    assert accepts(correct, "ba")


def test_complement_of_a_start_less_automaton_stays_undefined():
    automaton = even_zeros().with_initial(None)
    flipped = complement(automaton)
    assert flipped.initial is None
    assert flipped.accept == frozenset({"o"})
    assert run(flipped, "0").verdict is Verdict.NO_INITIAL_STATE


def test_complement_needs_no_completion_after_a_product():
    """Products are complete by construction, partial operands included."""
    combined = union(partial_a_star_b(), only_a())
    flipped = complement(combined)
    for word in words_up_to("ab", 5):
        assert accepts(flipped, word) != accepts(combined, word)


def test_a_complete_automaton_with_an_empty_alphabet_can_be_complemented():
    automaton = DFA().with_states(["q0"])
    assert complement(automaton).accept == frozenset({"q0"})


# ---------------------------------------------------------------------------
# The property: membership in the result agrees with the boolean combination
# ---------------------------------------------------------------------------


def random_dfa(rng: random.Random, index: int) -> DFA:
    """Small, often partial, over a random subset of "abc".

    State ids are prefixed per operand so that a pair's id is readable when a
    property test fails; the construction itself does not care whether the two
    operands share ids.
    """
    alphabet = rng.sample("abc", rng.randrange(1, 4))
    ids = [f"{index}s{i}" for i in range(rng.randrange(1, 5))]
    automaton = DFA().with_states(ids)
    for symbol in alphabet:
        automaton = automaton.with_symbol(symbol)
    for state in ids:
        for symbol in alphabet:
            if rng.random() < 0.6:
                automaton = automaton.with_transition(state, symbol, rng.choice(ids))
    for state in ids:
        if rng.random() < 0.4:
            automaton = automaton.with_accept(state)
    return automaton


OPERATIONS = [
    ("union", union, lambda left, right: left or right),
    ("intersection", intersection, lambda left, right: left and right),
    ("difference", difference, lambda left, right: left and not right),
    ("symmetric_difference", symmetric_difference,
     lambda left, right: left != right),
]


def test_every_boolean_operation_agrees_with_the_booleans():
    """The whole point, over random machines with deliberately mismatched
    alphabets, and over words that include a symbol neither one knows."""
    rng = random.Random(2026)
    for _ in range(60):
        left = random_dfa(rng, 1)
        right = random_dfa(rng, 2)
        words = list(words_up_to(sorted(set(left.alphabet | right.alphabet) | {"z"}), 3))
        for name, operation, on_booleans in OPERATIONS:
            combined = operation(left, right)
            assert is_complete(combined), f"{name} left delta partial"
            for word in words:
                expected = on_booleans(accepts(left, word), accepts(right, word))
                assert accepts(combined, word) == expected, (
                    f"{name} disagreed on {word!r}: {left!r} vs {right!r}")


def test_the_agreement_holds_for_longer_words_too():
    """Length 3 cannot separate machines that only differ deeper in."""
    rng = random.Random(7)
    for _ in range(25):
        left = random_dfa(rng, 1)
        right = random_dfa(rng, 2)
        words = list(words_up_to(left.alphabet | right.alphabet, 6))
        for name, operation, on_booleans in OPERATIONS:
            combined = operation(left, right)
            for word in words:
                expected = on_booleans(accepts(left, word), accepts(right, word))
                assert accepts(combined, word) == expected, f"{name} on {word!r}"


def test_an_arbitrary_accept_test_works_too():
    """product() is not limited to the four wrappers. NOR is the interesting
    case: it accepts the pair where *both* sides have failed, so the marker
    pair has to be a real, addressable state."""
    rng = random.Random(99)
    for _ in range(25):
        left = random_dfa(rng, 1)
        right = random_dfa(rng, 2)
        combined = product(left, right,
                           lambda in_left, in_right: not (in_left or in_right))
        # Only over the product alphabet: no DFA over Sigma accepts a word
        # containing a symbol outside it, however the accept test votes.
        for word in words_up_to(left.alphabet | right.alphabet, 4):
            expected = not (accepts(left, word) or accepts(right, word))
            assert accepts(combined, word) == expected, f"NOR on {word!r}"


def test_complement_agrees_with_negation_on_random_complete_machines():
    rng = random.Random(11)
    for _ in range(60):
        automaton, _ = ops.complete(random_dfa(rng, 1))
        flipped = complement(automaton)
        for word in words_up_to(automaton.alphabet, 5):
            assert accepts(flipped, word) != accepts(automaton, word)


def test_de_morgan_holds_on_the_constructions():
    """A cross-check that does not go through accepts(): complement of a union
    is the intersection of the complements, as languages."""
    rng = random.Random(5)
    for _ in range(20):
        left, _ = ops.complete(random_dfa(rng, 1))
        right, _ = ops.complete(random_dfa(rng, 2))
        alphabet = left.alphabet | right.alphabet
        # Completed over their own alphabets only, so widen before flipping.
        left_all, _ = ops.complete(_widen(left, alphabet))
        right_all, _ = ops.complete(_widen(right, alphabet))
        left_of = complement(union(left_all, right_all))
        right_of = intersection(complement(left_all), complement(right_all))
        for word in words_up_to(alphabet, 4):
            assert accepts(left_of, word) == accepts(right_of, word), word


def _widen(automaton: DFA, alphabet: FrozenSet[str]) -> DFA:
    """Add symbols to the alphabet without adding transitions on them."""
    for symbol in alphabet:
        automaton = automaton.with_symbol(symbol)
    return automaton
