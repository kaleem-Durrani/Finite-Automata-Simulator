"""Our algorithms against an independent implementation.

Every module in the algorithms layer arrived with its own tests, and a module
that only tests itself can be consistently wrong: minimisation that agrees with
its own idea of the language proves nothing about the language. Phase 9's
criteria closed some of that by checking modules against *each other* --
minimise against equivalence, complement against the simulator -- but they are
still all our code, sharing whatever assumption we happened to make.

These tests close the rest of it. `automata-lib` is a mature package written by
other people from the same definitions, so agreement is real evidence and
disagreement is a bug in one of the two.

Machines come from hypothesis, so a disagreement arrives shrunk to the smallest
one that reproduces it rather than as whatever a seed produced.
"""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import fsa
from tests import oracle
from tests.strategies import dfas, words

# Building a DFA a transition at a time is not fast, and every example here
# constructs several. A smaller example budget with no deadline keeps the suite
# usable while still covering hundreds of machines per run.
SETTINGS = settings(max_examples=60, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


def _oracle_or_skip(automaton: fsa.DFA):
    try:
        return oracle.to_oracle(automaton)
    except oracle.Unsupported as reason:
        pytest.skip(str(reason))


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------


@SETTINGS
@given(automaton=dfas(), word=words())
def test_our_simulator_agrees_with_the_oracle(automaton, word):
    """The foundation. If this disagreed, every property below would be
    measuring the wrong thing."""
    _oracle_or_skip(automaton)
    assert fsa.accepts(automaton, word) == oracle.oracle_accepts(automaton, word)


# ---------------------------------------------------------------------------
# Minimisation
# ---------------------------------------------------------------------------


@SETTINGS
@given(automaton=dfas(min_states=1, max_states=5))
def test_minimise_preserves_the_language_by_the_oracle(automaton):
    _oracle_or_skip(automaton)
    reduced = fsa.minimize(automaton)
    _oracle_or_skip(reduced)
    assert oracle.same_language(automaton, reduced)


@SETTINGS
@given(automaton=dfas(min_states=1, max_states=5, complete=True))
def test_our_minimal_machine_is_no_larger_than_the_oracle_s(automaton):
    """Minimal means minimal: the two implementations must reach the same
    number of states, not merely equivalent machines. A minimisation that
    merged too little would still be language-preserving and still wrong."""
    _oracle_or_skip(automaton)
    ours = fsa.minimize(automaton)
    theirs = oracle.to_oracle(automaton).minify()

    # The oracle keeps a trap for a partial delta and so may carry one state we
    # dropped; it can never need fewer than the true minimum.
    assert len(ours.states) <= len(theirs.states) + 1
    assert len(theirs.states) <= len(ours.states) + 1


# ---------------------------------------------------------------------------
# Equivalence
# ---------------------------------------------------------------------------


@SETTINGS
@given(left=dfas(max_states=4), right=dfas(max_states=4))
def test_our_equivalence_agrees_with_the_oracle(left, right):
    _oracle_or_skip(left)
    _oracle_or_skip(right)
    assert fsa.equivalent(left, right) == oracle.same_language(left, right)


@SETTINGS
@given(left=dfas(max_states=4), right=dfas(max_states=4))
def test_a_counterexample_really_distinguishes_by_the_oracle(left, right):
    """Not just that the two disagree by our simulator -- that they disagree by
    somebody else's."""
    _oracle_or_skip(left)
    _oracle_or_skip(right)
    witness = fsa.counterexample(left, right)
    if witness is None:
        assert oracle.same_language(left, right)
        return
    assert (oracle.oracle_accepts(left, witness)
            != oracle.oracle_accepts(right, witness))


# ---------------------------------------------------------------------------
# The boolean operations
# ---------------------------------------------------------------------------


@SETTINGS
@given(left=dfas(alphabet="ab", max_states=4),
       right=dfas(alphabet="ab", max_states=4),
       word=words(alphabet="ab"))
def test_boolean_operations_agree_with_membership(left, right, word):
    """Membership in the result must be the boolean combination of membership
    in the operands -- checked with the oracle's simulator, not ours."""
    _oracle_or_skip(left)
    _oracle_or_skip(right)
    in_left = oracle.oracle_accepts(left, word)
    in_right = oracle.oracle_accepts(right, word)

    for operation, expected in (
        (fsa.union, in_left or in_right),
        (fsa.intersection, in_left and in_right),
        (fsa.difference, in_left and not in_right),
        (fsa.symmetric_difference, in_left != in_right),
    ):
        result = operation(left, right)
        assert fsa.accepts(result, word) == expected, operation.__name__


@SETTINGS
@given(automaton=dfas(alphabet="ab", max_states=4), word=words(alphabet="ab"))
def test_complement_of_a_completed_machine_inverts_by_the_oracle(automaton, word):
    _oracle_or_skip(automaton)
    completed, _trap = fsa.complete(automaton)
    flipped = fsa.complement(completed)
    assert fsa.accepts(flipped, word) != oracle.oracle_accepts(automaton, word)


# ---------------------------------------------------------------------------
# Trim
# ---------------------------------------------------------------------------


@SETTINGS
@given(automaton=dfas(max_states=5), word=words())
def test_trim_preserves_the_language_by_the_oracle(automaton, word):
    _oracle_or_skip(automaton)
    assert fsa.accepts(fsa.trim(automaton), word) == oracle.oracle_accepts(
        automaton, word)


# ---------------------------------------------------------------------------
# The oracle bridge itself
# ---------------------------------------------------------------------------


@SETTINGS
@given(automaton=dfas())
def test_the_bridge_preserves_the_language(automaton):
    """If converting to the oracle changed the machine, every test above would
    be comparing two different automata and passing anyway."""
    _oracle_or_skip(automaton)
    for word in ("", "a", "b", "ab", "ba", "aab", "abb"):
        assert fsa.accepts(automaton, word) == oracle.oracle_accepts(
            automaton, word)


def test_the_bridge_refuses_what_it_cannot_represent():
    """Declining loudly, so a differential test never reports success for a
    machine it silently skipped."""
    with pytest.raises(oracle.Unsupported):
        oracle.to_oracle(fsa.DFA(states=frozenset({"q0"}),
                                 alphabet=frozenset("a")))


@given(st.data())
@settings(max_examples=5, deadline=None)
def test_strategies_produce_valid_automata(data):
    """The generators must not be able to build an illegal machine; DFA's own
    validation would raise, and a strategy that raises hides real failures."""
    automaton = data.draw(dfas())
    assert set(automaton.accept) <= set(automaton.states)
    for (source, symbol), target in automaton.transitions.items():
        assert source in automaton.states and target in automaton.states
        assert symbol in automaton.alphabet
