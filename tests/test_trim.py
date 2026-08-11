"""The trim operation.

Trimming deletes states, which is the kind of change that quietly alters a
language if any part of it is wrong -- so most of this file is one claim,
approached from several sides: the words accepted before are exactly the words
accepted after. The rest pins down the two behaviours that look like bugs and
are not: delta comes back *more* partial than it went in, and a machine that
accepts nothing trims away to nothing at all.

Imports no pygame and touches no display.
"""

import itertools
import random
from typing import FrozenSet, Iterator

import fsa
from fsa import DFA
from fsa.ops import complete, trim
from fsa.simulate import Verdict

# ---------------------------------------------------------------------------
# Fixtures by hand
# ---------------------------------------------------------------------------


def mixed_dfa() -> DFA:
    """One of each: a useful path, a dead branch, an unreachable island.

    q0 -a-> q1 -a-> q2*        the only accepting run
    q1 -b-> q3 -a-> q3         reachable, but nothing accepting downstream
    q4* -a-> q4                accepting, but no word ever arrives
    """
    return (DFA()
            .with_states(["q0", "q1", "q2", "q3", "q4"])
            .with_transition("q0", "a", "q1")
            .with_transition("q1", "a", "q2")
            .with_transition("q1", "b", "q3")
            .with_transition("q3", "a", "q3")
            .with_transition("q4", "a", "q4")
            .with_accept("q2")
            .with_accept("q4"))


def completed_a_star_b() -> DFA:
    """a*b, completed -- so delta is total and one state is a trap."""
    automaton = (DFA()
                 .with_states(["q0", "q1"])
                 .with_transition("q0", "a", "q0")
                 .with_transition("q0", "b", "q1")
                 .with_accept("q1"))
    return complete(automaton)[0]


# ---------------------------------------------------------------------------
# The basic contract
# ---------------------------------------------------------------------------


def test_trim_removes_the_unreachable_and_the_dead():
    after = trim(mixed_dfa())
    assert after.states == {"q0", "q1", "q2"}
    assert after.initial == "q0"
    assert after.accept == {"q2"}, "q4 accepted, but no word could get there"


def test_what_survives_is_exactly_what_an_accepting_run_visits():
    """The postcondition, stated as the analysis module would state it."""
    after = trim(mixed_dfa())
    assert fsa.dead_states(after) == frozenset()
    assert fsa.unreachable_states(after) == frozenset()
    # And concretely: the one accepting run still runs, unchanged.
    accepting_run = fsa.run(after, "aa")
    assert accepting_run.accepted
    assert set(accepting_run.path) == after.states


def test_an_already_trim_automaton_comes_back_as_the_same_object():
    """Nothing to remove means no copy is made -- and idempotence falls out."""
    once = trim(mixed_dfa())
    assert trim(once) is once


def test_an_empty_automaton_is_a_no_op():
    automaton = DFA()
    assert trim(automaton) is automaton


def test_trim_only_ever_removes():
    """No state, edge or accepting state is invented on the way through."""
    rng = random.Random(11)
    for _ in range(120):
        before = random_dfa(rng)
        after = trim(before)
        assert after.states <= before.states
        assert after.accept <= before.accept
        assert dict(after.transitions).items() <= dict(before.transitions).items()
        assert after.alphabet == before.alphabet
        assert after.initial in (before.initial, None)


# ---------------------------------------------------------------------------
# delta gets more partial, on purpose
# ---------------------------------------------------------------------------


def test_the_edge_into_a_removed_state_goes_with_it():
    before = mixed_dfa()
    assert before.target("q1", "b") == "q3"
    after = trim(before)
    assert after.target("q1", "b") is None, \
        "the arrow led only to q3, which no longer exists"


def test_trimming_a_complete_automaton_makes_it_incomplete():
    """This is the surprise worth asserting: completion and trimming pull in
    opposite directions, and trimming wins where they meet."""
    before = completed_a_star_b()
    assert fsa.is_complete(before)
    after = trim(before)
    assert not fsa.is_complete(after)
    assert ("q1", "a") in fsa.missing_transitions(after)


def test_a_run_that_used_to_die_in_a_trap_now_dies_for_want_of_an_arrow():
    """Same rejection, better reason -- the edge really did lead nowhere."""
    before = completed_a_star_b()
    after = trim(before)
    assert fsa.run(before, "ba").verdict is Verdict.REJECT_NON_ACCEPTING
    assert fsa.run(after, "ba").verdict is Verdict.REJECT_NO_TRANSITION
    assert not fsa.accepts(before, "ba") and not fsa.accepts(after, "ba")


def test_the_alphabet_is_left_alone():
    """Symbols are not states. Narrowing Sigma would turn "no transition" into
    "not in the alphabet", which is a different claim about the machine."""
    after = trim(completed_a_star_b())
    assert after.alphabet == {"a", "b"}
    rejected = fsa.run(after, "ba")
    assert rejected.verdict is Verdict.REJECT_NO_TRANSITION
    assert rejected.offending_symbol == "a"


# ---------------------------------------------------------------------------
# When nothing is useful, nothing survives
# ---------------------------------------------------------------------------


def test_an_automaton_with_no_accepting_states_trims_to_nothing():
    automaton = (DFA()
                 .with_states(["q0", "q1"])
                 .with_transition("q0", "a", "q1"))
    after = trim(automaton)
    assert after.states == frozenset()
    assert after.initial is None, "no state left to start in"
    assert after.alphabet == {"a"}, "the alphabet is still the alphabet"


def test_a_dead_start_state_takes_the_whole_automaton_with_it():
    """q0 can never reach q2, so no state is on an accepting run -- not even
    q2, which nothing can reach either."""
    automaton = (DFA()
                 .with_states(["q0", "q1", "q2"])
                 .with_transition("q0", "a", "q1")
                 .with_transition("q1", "a", "q0")
                 .with_accept("q2"))
    assert trim(automaton).states == frozenset()


def test_a_startless_automaton_trims_to_nothing():
    """With no initial state nothing is reachable, so nothing is useful. The
    result already meant "no language yet" and still does."""
    automaton = mixed_dfa().with_initial(None)
    after = trim(automaton)
    assert after.states == frozenset()
    assert after.initial is None


def test_losing_the_start_state_is_all_or_nothing():
    """The documented reason `initial=None` needs no special handling: a start
    state that is not useful makes every other state useless too."""
    rng = random.Random(12)
    for _ in range(200):
        after = trim(random_dfa(rng))
        if after.initial is None:
            assert after.states == frozenset(), \
                "states survived a start state that did not"


def test_an_emptied_automaton_still_rejects_everything():
    """`initial=None` reads as "undefined" rather than "empty", but no word can
    tell the two apart, which is why trimming is free to prefer it."""
    automaton = DFA().with_states(["q0"]).with_transition("q0", "a", "q0")
    after = trim(automaton)
    for word in words_up_to(frozenset("a"), 4):
        assert not fsa.accepts(automaton, word)
        assert not fsa.accepts(after, word)
    assert any(defect.kind == "no_initial_state" for defect in fsa.defects(after))


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def test_labels_follow_their_states():
    before = mixed_dfa().with_label("q2", "done").with_label("q4", "island")
    after = trim(before)
    assert after.label_of("q2") == "done"
    assert "q4" not in after.labels, "a name for a state nobody can visit"


# ---------------------------------------------------------------------------
# THE property: trimming never changes the language
# ---------------------------------------------------------------------------


def random_dfa(rng: random.Random) -> DFA:
    """Small and sparse, so useless states are common rather than rare.

    Sometimes start-less, sometimes with nothing accepting -- both of which
    trim away to nothing, and both of which are legal automata.
    """
    alphabet = rng.sample("ab01", rng.randrange(1, 4))
    ids = [f"q{i}" for i in range(rng.randrange(1, 7))]
    automaton = DFA().with_states(ids)
    for symbol in alphabet:
        automaton = automaton.with_symbol(symbol)
    for state in ids:
        for symbol in alphabet:
            if rng.random() < 0.45:
                automaton = automaton.with_transition(state, symbol, rng.choice(ids))
    for state in ids:
        if rng.random() < 0.3:
            automaton = automaton.with_accept(state)
    if rng.random() < 0.15:
        automaton = automaton.with_initial(None)
    return automaton


def words_up_to(alphabet: FrozenSet[str], length: int) -> Iterator[str]:
    """Every word over ``alphabet`` of length ``0..length``, shortest first."""
    letters = sorted(alphabet)
    for size in range(length + 1):
        for combination in itertools.product(letters, repeat=size):
            yield "".join(combination)


def test_trimming_never_changes_the_language():
    """Exhaustive over short words rather than sampled: the words that expose a
    wrongly-deleted state are the short ones, and there are few enough of them
    to just enumerate. 'z' is in no generated alphabet, so rejection for a
    foreign symbol has to survive trimming too."""
    rng = random.Random(2026)
    for _ in range(120):
        before = random_dfa(rng)
        after = trim(before)
        for word in words_up_to(before.alphabet | {"z"}, 4):
            assert fsa.accepts(before, word) == fsa.accepts(after, word), \
                f"{word!r} changed verdict on {before!r}"


def test_trimming_leaves_nothing_useless_behind():
    rng = random.Random(13)
    for _ in range(200):
        after = trim(random_dfa(rng))
        assert fsa.dead_states(after) == frozenset()
        assert fsa.unreachable_states(after) == frozenset()


def test_trim_is_idempotent():
    rng = random.Random(14)
    for _ in range(200):
        once = trim(random_dfa(rng))
        assert trim(once) == once
        assert trim(once) is once, "and cheaply: the second pass finds nothing"


def test_completing_before_trimming_makes_no_difference():
    """complete() only ever adds a dead trap and edges into it, and trim()
    removes exactly those -- so the two compose to the plain trim."""
    rng = random.Random(15)
    for _ in range(120):
        before = random_dfa(rng)
        completed, _ = complete(before)
        assert trim(completed) == trim(before)
