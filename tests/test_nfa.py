"""The NFA value type, epsilon closure, and the set-based simulator.

Three things carry this file.

The **epsilon cycle** gets its own section. ``q0 -eps-> q1 -eps-> q0`` is legal,
it is what Thompson's construction will produce by the dozen in Phase 13, and a
closure that walks it without a visited set never returns. It is the single case
naive implementations break on, so it is tested from every direction here: a
self-loop, a two-state cycle, a cycle with a tail, and a cycle nothing outside
it reaches.

**Partial delta is the default**, not an edge case. A missing key means the
empty set, so every query is asked of a state that has no move on the symbol as
well as one that does -- that is the case the whole engine is written around.

And the simulator is checked against a **second, independent implementation**
in this file: ours explores every branch at once as a set of states, the
reference backtracks down one path at a time. Two implementations of the same
definition agreeing over hypothesis-generated machines is much stronger evidence
than either one agreeing with a list of examples somebody wrote by hand.

Imports no pygame and touches no display.
"""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path
from typing import Set, Tuple

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import fsa
from fsa import DFA
from fsa.errors import (
    DuplicateStateError,
    IllegalSymbolError,
    NondeterministicError,
    UnknownStateError,
)
from fsa.nfa import EPSILON, NFA, NfaRun, accepts, from_dfa, run, to_dfa
from fsa.simulate import Verdict
from fsa.symbols import StateId
from tests.strategies import dfas, words

# Building a machine a transition at a time is not fast and every example here
# builds several, so the budget is small and the deadline off -- the same
# settings the differential tests use.
SETTINGS = settings(max_examples=50, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


# ---------------------------------------------------------------------------
# Fixtures by hand
# ---------------------------------------------------------------------------


def ends_in_ab() -> NFA:
    """The textbook NFA: words over {a,b} ending in "ab".

    q0 keeps its options open on every symbol and *also* guesses that the a it
    just read begins the final "ab". Determinising this is the standard first
    exercise, and no DFA writes it in three states, so a simulator that quietly
    followed only the first branch would fail on "aab".
    """
    return (NFA()
            .with_states(["q0", "q1", "q2"])
            .with_transition("q0", "a", "q0")
            .with_transition("q0", "b", "q0")
            .with_transition("q0", "a", "q1")
            .with_transition("q1", "b", "q2")
            .with_accept("q2"))


def epsilon_chain() -> NFA:
    """q0 -eps-> q1 -eps-> q2, accepting in q2.

    Accepts the empty string without reading anything, so it fails immediately
    if the closure is not taken *before* the first symbol.
    """
    return (NFA()
            .with_states(["q0", "q1", "q2"])
            .with_transition("q0", EPSILON, "q1")
            .with_transition("q1", EPSILON, "q2")
            .with_accept("q2"))


def epsilon_cycle() -> NFA:
    """q0 -eps-> q1 -eps-> q0, with a real edge hanging off q1.

    The machine a naive closure hangs on. It accepts "a" -- but only if the
    closure of {q0} reaches q1 and stops rather than going round again.
    """
    return (NFA()
            .with_states(["q0", "q1", "q2"])
            .with_transition("q0", EPSILON, "q1")
            .with_transition("q1", EPSILON, "q0")
            .with_transition("q1", "a", "q2")
            .with_accept("q2"))


def partial_nfa() -> NFA:
    """One state, one symbol in the alphabet, and no move on it at all."""
    return (NFA(states=frozenset({"q0"}), alphabet=frozenset("ab"))
            .with_initial("q0")
            .with_accept("q0"))


# ---------------------------------------------------------------------------
# An independent simulator to check ours against
# ---------------------------------------------------------------------------


def backtracking_accepts(automaton: NFA, word: str) -> bool:
    """Whether some single path through the machine accepts the word.

    The definition rather than the algorithm: one branch at a time, depth
    first, backtracking on failure. Our simulator instead advances a whole set
    of states in step, which is a different program with a different failure
    mode -- so where the two agree, the agreement is evidence.

    ``seen`` is what makes this terminate on an epsilon cycle, and it is sound
    for the same reason it is in any DFS: a ``(state, index)`` is only revisited
    after it has already been explored and failed, or while it is still on the
    stack -- and a path that returns to where it started has learnt nothing.
    """
    if automaton.initial is None:
        return False

    seen: Set[Tuple[StateId, int]] = set()

    def explore(state: StateId, index: int) -> bool:
        if (state, index) in seen:
            return False
        seen.add((state, index))

        if index == len(word):
            if state in automaton.accept:
                return True
        elif word[index] in automaton.alphabet:
            for target in sorted(automaton.targets(state, word[index])):
                if explore(target, index + 1):
                    return True

        return any(explore(target, index)
                   for target in sorted(automaton.targets(state, EPSILON)))

    return explore(automaton.initial, 0)


@st.composite
def nfas(draw: st.DrawFn, *, max_states: int = 4,
         alphabet: str = "ab", epsilons: bool = True) -> NFA:
    """A nondeterministic machine, partial and epsilon-ridden by default.

    Deliberately unlike :func:`tests.strategies.dfas`: several targets on one
    symbol is the common case here, not the exception, and roughly a third of
    the states get an epsilon move so that closures actually have work to do.
    """
    count = draw(st.integers(min_value=1, max_value=max_states))
    ids = [f"q{index}" for index in range(count)]

    automaton = NFA(states=frozenset(ids),
                    alphabet=frozenset(alphabet)).with_initial(ids[0])

    for state in ids:
        if draw(st.booleans()):
            automaton = automaton.with_accept(state)
        for symbol in sorted(alphabet):
            for target in draw(st.lists(st.sampled_from(ids), max_size=2)):
                automaton = automaton.with_transition(state, symbol, target)
        if epsilons and draw(st.integers(min_value=0, max_value=2)) == 0:
            automaton = automaton.with_transition(
                state, EPSILON, draw(st.sampled_from(ids)))
    return automaton


# ---------------------------------------------------------------------------
# The value type
# ---------------------------------------------------------------------------


def test_a_fresh_nfa_is_empty_and_defines_no_language():
    automaton = NFA()
    assert automaton.states == frozenset()
    assert automaton.initial is None
    assert run(automaton, "").verdict is Verdict.NO_INITIAL_STATE


def test_the_value_is_frozen():
    automaton = ends_in_ab()
    with pytest.raises(dataclasses.FrozenInstanceError):
        automaton.initial = "q1"


def test_the_mappings_handed_out_cannot_be_written_to():
    """A snapshot that shares a mutable dict is not a snapshot -- the lesson
    that made every field here immutable in the first place."""
    automaton = ends_in_ab()
    with pytest.raises(TypeError):
        automaton.transitions[("q0", "a")] = frozenset({"q2"})
    with pytest.raises(TypeError):
        automaton.labels["q0"] = "start"


def test_equality_is_structural_and_order_of_construction_does_not_matter():
    forwards = (NFA().with_states(["q0", "q1"])
                .with_transition("q0", "a", "q0")
                .with_transition("q0", "a", "q1"))
    backwards = (NFA().with_states(["q0", "q1"])
                 .with_transition("q0", "a", "q1")
                 .with_transition("q0", "a", "q0"))
    assert forwards == backwards
    assert hash(forwards) == hash(backwards)
    assert len({forwards, backwards}) == 1


def test_an_nfa_is_never_equal_to_a_dfa():
    """Different types, even for the same machine. The whole point of the split
    is that a caller can tell which kind it is holding."""
    automaton = DFA().with_state("q0").with_accept("q0")
    assert from_dfa(automaton) != automaton
    assert automaton != from_dfa(automaton)


def test_an_empty_target_set_is_spelled_by_leaving_the_key_out():
    """One fact, one spelling. Storing the empty set would make two machines
    that behave identically compare unequal, and every later value -- a saved
    file, an undo entry, a table key -- would inherit the split."""
    explicit = NFA(states=frozenset({"q0"}), alphabet=frozenset("a"),
                   transitions={("q0", "a"): frozenset()})
    absent = NFA(states=frozenset({"q0"}), alphabet=frozenset("a"))
    assert explicit == absent
    assert explicit.transitions == {}
    assert ("q0", "a") not in explicit.transitions


def test_any_iterable_of_states_is_accepted_as_a_target_set():
    """A caller with a list or a set should not have to convert first."""
    made = NFA(states=frozenset({"q0", "q1"}), alphabet=frozenset("a"),
               transitions={("q0", "a"): ["q1", "q1"]})
    assert made.targets("q0", "a") == frozenset({"q1"})


def test_a_bare_state_id_as_the_target_set_is_refused():
    """frozenset("q10") is {"q", "1", "0"}: a bare id would be shredded into one
    target per character. A state id is a string and a string is iterable, so
    nothing else in the constructor can catch this."""
    with pytest.raises(TypeError) as raised:
        NFA(states=frozenset({"q0", "q1"}), alphabet=frozenset("a"),
            transitions={("q0", "a"): "q1"})
    assert "bare state" in str(raised.value)


def test_the_repr_says_how_many_epsilon_moves_there_are():
    assert "eps=2" in repr(epsilon_chain())
    assert "eps=0" in repr(ends_in_ab())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_a_transition_must_name_states_the_machine_has():
    with pytest.raises(UnknownStateError):
        NFA(states=frozenset({"q0"}), alphabet=frozenset("a"),
            transitions={("nowhere", "a"): frozenset({"q0"})})
    with pytest.raises(UnknownStateError):
        NFA(states=frozenset({"q0"}), alphabet=frozenset("a"),
            transitions={("q0", "a"): frozenset({"q0", "nowhere"})})


def test_a_transition_must_be_on_a_symbol_in_the_alphabet():
    with pytest.raises(UnknownStateError):
        NFA(states=frozenset({"q0"}), alphabet=frozenset("a"),
            transitions={("q0", "b"): frozenset({"q0"})})


def test_an_epsilon_move_needs_nothing_from_the_alphabet():
    """The alphabet may be empty and the machine still moves. That is what
    keying epsilon as None buys: it is not a letter, so it cannot be missing
    from Sigma."""
    automaton = NFA(states=frozenset({"q0", "q1"}),
                    transitions={("q0", EPSILON): frozenset({"q1"})})
    assert automaton.alphabet == frozenset()
    assert automaton.epsilon_closure(["q0"]) == frozenset({"q0", "q1"})


def test_epsilon_can_never_be_added_to_the_alphabet():
    with pytest.raises(IllegalSymbolError):
        NFA().with_symbol(EPSILON)


def test_the_initial_and_accepting_states_must_exist():
    with pytest.raises(UnknownStateError):
        NFA(states=frozenset({"q0"}), initial="q1")
    with pytest.raises(UnknownStateError):
        NFA(states=frozenset({"q0"}), accept=frozenset({"q1"}))


# ---------------------------------------------------------------------------
# delta is partial: a missing key is the empty set
# ---------------------------------------------------------------------------


def test_a_missing_move_is_the_empty_set_and_not_an_error():
    automaton = partial_nfa()
    assert automaton.targets("q0", "a") == frozenset()
    assert automaton.targets("q0", EPSILON) == frozenset()


def test_an_unknown_state_answers_with_the_empty_set_too():
    """Same rule, so callers need no guard: nothing leads out of a state that
    is not there either."""
    assert ends_in_ab().targets("nowhere", "a") == frozenset()


def test_targets_returns_every_branch():
    assert ends_in_ab().targets("q0", "a") == frozenset({"q0", "q1"})


# ---------------------------------------------------------------------------
# Epsilon closure -- and the cycle, which is the case that breaks naive ones
# ---------------------------------------------------------------------------


def test_the_closure_contains_the_states_it_was_given():
    assert ends_in_ab().epsilon_closure(["q0", "q2"]) == frozenset({"q0", "q2"})


def test_the_closure_of_nothing_is_nothing():
    assert epsilon_chain().epsilon_closure([]) == frozenset()


def test_the_closure_follows_a_chain_all_the_way():
    """Transitivity: q0 reaches q2 through q1 without a second call."""
    assert epsilon_chain().epsilon_closure(["q0"]) == frozenset({"q0", "q1", "q2"})


def test_the_closure_terminates_on_a_two_state_epsilon_cycle():
    """q0 -eps-> q1 -eps-> q0. A closure without a visited set never returns
    from this, so a hang here is the bug, not a slow test."""
    assert epsilon_cycle().epsilon_closure(["q0"]) == frozenset({"q0", "q1"})
    assert epsilon_cycle().epsilon_closure(["q1"]) == frozenset({"q0", "q1"})


def test_the_closure_terminates_on_an_epsilon_self_loop():
    automaton = (NFA().with_state("q0").with_transition("q0", EPSILON, "q0"))
    assert automaton.epsilon_closure(["q0"]) == frozenset({"q0"})


def test_the_closure_terminates_on_a_cycle_with_a_tail():
    """The cycle is entered from outside and left again, so the walk has to
    both stop going round and carry on past it."""
    automaton = (NFA()
                 .with_states(["in", "a", "b", "out"])
                 .with_transition("in", EPSILON, "a")
                 .with_transition("a", EPSILON, "b")
                 .with_transition("b", EPSILON, "a")
                 .with_transition("b", EPSILON, "out"))
    assert automaton.epsilon_closure(["in"]) == frozenset({"in", "a", "b", "out"})


def test_a_run_crosses_an_epsilon_cycle():
    """Not just the closure in isolation: the simulator has to survive it too,
    once before the first symbol and once after every step."""
    assert accepts(epsilon_cycle(), "a")
    assert not accepts(epsilon_cycle(), "aa")


def test_the_empty_string_is_accepted_through_epsilon_moves_alone():
    """The closure must be taken before the first symbol is read, not after."""
    assert accepts(epsilon_chain(), "")
    assert run(epsilon_chain(), "").start == frozenset({"q0", "q1", "q2"})


@SETTINGS
@given(automaton=nfas())
def test_the_closure_is_idempotent_and_only_grows(automaton):
    """Two properties that together say it really is a closure: applying it
    again adds nothing, and it never loses a state it was given."""
    for state in sorted(automaton.states):
        once = automaton.epsilon_closure([state])
        assert state in once
        assert automaton.epsilon_closure(once) == once


# ---------------------------------------------------------------------------
# Determinism is a property of delta alone
# ---------------------------------------------------------------------------


def test_a_partial_machine_is_still_deterministic():
    """The distinction :class:`NondeterministicError` exists to keep: a state
    with no move has nothing to choose between."""
    assert partial_nfa().is_deterministic()


def test_two_targets_on_one_symbol_is_nondeterministic():
    assert not ends_in_ab().is_deterministic()


def test_an_epsilon_move_is_nondeterministic_on_its_own():
    """Even though nothing branches: an epsilon move is a choice about whether
    to move at all, and no DFA has one."""
    automaton = (NFA().with_states(["q0", "q1"])
                 .with_transition("q0", EPSILON, "q1"))
    assert not automaton.is_deterministic()


def test_an_empty_machine_is_deterministic():
    assert NFA().is_deterministic()


@SETTINGS
@given(automaton=dfas())
def test_every_dfa_read_as_an_nfa_is_deterministic(automaton):
    assert from_dfa(automaton).is_deterministic()


# ---------------------------------------------------------------------------
# Derived views
# ---------------------------------------------------------------------------


def test_grouped_transitions_puts_epsilon_on_the_edge_that_carries_it():
    """One edge can be labelled with both a symbol and epsilon, and the
    renderer has to draw both on it."""
    automaton = (NFA().with_states(["q0", "q1"])
                 .with_transition("q0", "a", "q1")
                 .with_transition("q0", EPSILON, "q1"))
    assert automaton.grouped_transitions() == {("q0", "q1"): frozenset({"a", EPSILON})}


def test_grouped_transitions_splits_a_branching_move_into_two_edges():
    grouped = ends_in_ab().grouped_transitions()
    assert grouped[("q0", "q0")] == frozenset({"a", "b"})
    assert grouped[("q0", "q1")] == frozenset({"a"})


def test_outgoing_lists_every_move_leaving_a_state():
    automaton = epsilon_cycle()
    assert dict(automaton.outgoing("q1")) == {
        EPSILON: frozenset({"q0"}),
        "a": frozenset({"q2"}),
    }
    assert dict(automaton.outgoing("q2")) == {}


def test_sorted_transitions_puts_epsilon_first_and_sorts_the_targets():
    """``sorted(transitions.items())`` cannot do this -- None does not compare
    with str -- which is the entire reason the method exists."""
    automaton = (NFA().with_states(["q0", "q1", "q2"])
                 .with_transition("q0", "b", "q2")
                 .with_transition("q0", "a", "q2")
                 .with_transition("q0", "a", "q1")
                 .with_transition("q0", EPSILON, "q2"))
    assert automaton.sorted_transitions() == (
        ("q0", EPSILON, ("q2",)),
        ("q0", "a", ("q1", "q2")),
        ("q0", "b", ("q2",)),
    )


def test_labels_are_cosmetic():
    automaton = ends_in_ab().with_label("q0", "start")
    assert automaton.label_of("q0") == "start"
    assert automaton.label_of("q1") == "q1"
    assert accepts(automaton, "ab") == accepts(ends_in_ab(), "ab")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def test_with_transition_adds_a_branch_where_the_dfa_would_replace_one():
    """The one behavioural difference between the two builders, and the reason
    the GUI could not draw an NFA before: its add_transition overwrote."""
    automaton = (NFA().with_states(["q0", "q1", "q2"])
                 .with_transition("q0", "a", "q1")
                 .with_transition("q0", "a", "q2"))
    assert automaton.targets("q0", "a") == frozenset({"q1", "q2"})


def test_with_transition_on_a_new_symbol_extends_the_alphabet():
    assert "z" in NFA().with_state("q0").with_transition("q0", "z", "q0").alphabet


def test_an_epsilon_move_does_not_extend_the_alphabet():
    automaton = NFA().with_state("q0").with_transition("q0", EPSILON, "q0")
    assert automaton.alphabet == frozenset()


def test_the_first_state_added_becomes_the_initial_one():
    assert NFA().with_state("q0").with_state("q1").initial == "q0"


def test_adding_a_state_twice_is_refused():
    with pytest.raises(DuplicateStateError):
        NFA().with_state("q0").with_state("q0")


def test_without_transition_can_remove_one_branch_or_the_whole_move():
    automaton = ends_in_ab()
    assert automaton.without_transition("q0", "a", "q1").targets(
        "q0", "a") == frozenset({"q0"})
    assert automaton.without_transition("q0", "a").targets("q0", "a") == frozenset()


def test_removing_the_last_branch_removes_the_key():
    """Otherwise delta would hold an empty set, which is the one spelling of
    "no move" this type does not allow."""
    automaton = ends_in_ab().without_transition("q1", "b", "q2")
    assert ("q1", "b") not in automaton.transitions


def test_removing_a_state_takes_the_moves_into_it_and_leaves_the_others():
    automaton = ends_in_ab().without_state("q1")
    assert automaton.states == frozenset({"q0", "q2"})
    assert automaton.targets("q0", "a") == frozenset({"q0"})
    assert ("q1", "b") not in automaton.transitions


def test_removing_the_initial_state_leaves_the_machine_without_one():
    """No replacement is elected: choosing one would change the language
    without saying so."""
    assert ends_in_ab().without_state("q0").initial is None


def test_without_symbol_leaves_epsilon_moves_alone():
    automaton = (NFA().with_states(["q0", "q1"])
                 .with_transition("q0", "a", "q1")
                 .with_transition("q0", EPSILON, "q1"))
    stripped = automaton.without_symbol("a")
    assert stripped.alphabet == frozenset()
    assert stripped.targets("q0", EPSILON) == frozenset({"q1"})


def test_without_symbol_of_epsilon_removes_every_epsilon_move():
    """It reads oddly and it is exactly right: epsilon was never in Sigma, so
    removing it can only mean the moves."""
    stripped = epsilon_cycle().without_symbol(EPSILON)
    assert stripped.is_deterministic()
    assert stripped.alphabet == frozenset("a")


def test_every_builder_leaves_the_original_untouched():
    """Undo is "keep the previous value", which only works if nothing mutates."""
    original = ends_in_ab()
    before = (original.states, original.transitions, original.accept, hash(original))
    original.with_state("q9").with_transition("q0", "b", "q1").with_accept("q0")
    original.without_state("q1").with_label("q0", "x").without_accept("q2")
    assert (original.states, original.transitions,
            original.accept, hash(original)) == before


def test_clearing_a_label_is_not_the_same_as_setting_it_to_the_id():
    """Equality can tell them apart even though the screen cannot, so writing
    the id back would spend an undo slot on an invisible change."""
    plain = ends_in_ab()
    assert plain.with_label("q0", "hello").with_label_removed("q0") == plain
    assert plain.with_label("q0", "q0") != plain


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def test_the_classic_nondeterministic_machine_is_simulated_correctly():
    """The discriminating case is "aab": the branch that guessed at the first a
    dies, and the run must still accept on the branch that guessed at the
    second. A simulator following one target would get this wrong."""
    automaton = ends_in_ab()
    for word in ("ab", "aab", "abab", "bbab"):
        assert accepts(automaton, word), word
    for word in ("", "a", "b", "ba", "aba", "abb"):
        assert not accepts(automaton, word), word


def test_a_run_records_the_configuration_at_every_position():
    """What the GUI will animate: one set per position, recoverable in full."""
    result = run(ends_in_ab(), "ab")
    assert result.configurations == (
        frozenset({"q0"}),
        frozenset({"q0", "q1"}),
        frozenset({"q0", "q2"}),
    )
    assert [step.symbol for step in result.steps] == ["a", "b"]
    assert [step.index for step in result.steps] == [0, 1]


def test_each_step_starts_where_the_last_one_ended():
    """The invariant that lets a UI index the word and the run by one counter."""
    result = run(ends_in_ab(), "abab")
    assert len(result.configurations) == len(result.steps) + 1
    for step, configuration in zip(result.steps, result.configurations):
        assert step.source == configuration
    assert result.final_states == result.configurations[-1]


def test_a_word_the_machine_runs_out_of_branches_on_stops_where_it_died():
    """Every alternative explored, every one dead. The reason is distinct from
    a plain rejection, and the position is the symbol nothing could read."""
    result = run(partial_nfa(), "ab")
    assert result.verdict is Verdict.REJECT_NO_TRANSITION
    assert result.stopped_at == 0
    assert result.offending_symbol == "a"
    assert result.consumed == "" and result.remaining == "ab"
    assert result.final_states == frozenset({"q0"}), (
        "the last live configuration is the useful one, not the empty set")


def test_a_symbol_outside_the_alphabet_is_rejected_exactly_as_a_dfa_does_it():
    result = run(ends_in_ab(), "aXb")
    assert result.verdict is Verdict.REJECT_SYMBOL_NOT_IN_ALPHABET
    assert result.stopped_at == 1
    assert result.offending_symbol == "X"
    assert "not in the alphabet" in result.explain()


def test_a_machine_with_no_start_state_says_so_rather_than_rejecting():
    result = run(NFA(states=frozenset({"q0"})), "")
    assert result.verdict is Verdict.NO_INITIAL_STATE
    assert result.start is None
    assert result.configurations == ()
    assert result.final_states == frozenset()


def test_accepting_needs_only_one_surviving_branch():
    """Two states in the final configuration, one of them accepting."""
    result = run(ends_in_ab(), "ab")
    assert result.final_states == frozenset({"q0", "q2"})
    assert result.accepted


def test_every_explanation_names_its_states_in_sorted_order():
    """A message built from a set reads differently in every process unless it
    is sorted -- the lesson that made sorting a house rule."""
    assert run(ends_in_ab(), "ab").explain() == (
        "'ab' was accepted in {q0, q2}")
    assert run(ends_in_ab(), "a").explain() == (
        "'a' was rejected: the whole string was read, but nothing in "
        "{q0, q1} is an accepting state")


def test_running_out_of_branches_is_not_reported_as_a_defect():
    """A DFA that cannot read a symbol is incomplete and the message says so.
    An NFA whose branches all die is doing exactly what it is supposed to, and
    telling a student to draw more arrows would be teaching the wrong thing --
    the same distinction the complete/trim cycle makes."""
    explanation = run(partial_nfa(), "a").explain()
    assert "every branch died" in explanation
    assert "incomplete" not in explanation


def test_the_verdict_reuses_the_dfa_simulator_s_enum():
    """Not a second vocabulary for the same five outcomes: a UI that can
    explain a DFA run must not need a second code path for this one."""
    assert run(ends_in_ab(), "ab").verdict is fsa.Verdict.ACCEPT
    assert isinstance(run(ends_in_ab(), "ab"), NfaRun)


@SETTINGS
@given(automaton=nfas(), word=words())
def test_our_set_simulator_agrees_with_a_backtracking_one(automaton, word):
    """The differential check. Ours advances a set of states in lockstep; the
    reference walks one path at a time and backtracks. Same definition, two
    programs, and hypothesis shrinks any disagreement to the smallest machine
    that shows it."""
    assert accepts(automaton, word) == backtracking_accepts(automaton, word)


@SETTINGS
@given(automaton=nfas(), word=words())
def test_a_run_is_internally_consistent(automaton, word):
    result = run(automaton, word)
    assert len(result.configurations) == len(result.steps) + 1
    assert result.consumed + result.remaining == word
    assert 0 <= result.stopped_at <= len(word)
    for configuration in result.configurations:
        assert configuration, "a live configuration is never empty"
        assert configuration <= automaton.states

    if result.accepted:
        assert result.stopped_at == len(word)
        assert result.final_states & automaton.accept
    elif result.verdict is Verdict.REJECT_NON_ACCEPTING:
        assert result.stopped_at == len(word)
        assert not result.final_states & automaton.accept


# ---------------------------------------------------------------------------
# Between the two machines
# ---------------------------------------------------------------------------


def test_a_dfa_read_as_an_nfa_keeps_everything_but_the_type():
    automaton = (DFA().with_states(["q0", "q1"])
                 .with_transition("q0", "a", "q1")
                 .with_accept("q1")
                 .with_label("q1", "done"))
    converted = from_dfa(automaton)
    assert converted.states == automaton.states
    assert converted.alphabet == automaton.alphabet
    assert converted.initial == automaton.initial
    assert converted.accept == automaton.accept
    assert dict(converted.labels) == dict(automaton.labels)
    assert converted.targets("q0", "a") == frozenset({"q1"})


def test_to_dfa_refuses_a_machine_with_a_choice_to_make():
    with pytest.raises(NondeterministicError) as raised:
        to_dfa(ends_in_ab())
    assert "q0" in str(raised.value) and "2 targets" in str(raised.value)


def test_to_dfa_refuses_a_machine_with_an_epsilon_move():
    with pytest.raises(NondeterministicError) as raised:
        to_dfa(epsilon_chain())
    assert "epsilon" in str(raised.value)


def test_to_dfa_accepts_a_partial_machine():
    """Partial is not nondeterministic, and refusing it would turn away most of
    the machines this function exists for."""
    converted = to_dfa(partial_nfa())
    assert converted.states == frozenset({"q0"})
    assert converted.target("q0", "a") is None


def test_a_partial_machine_rejects_for_the_same_reason_on_both_sides():
    """The rejection reason survives the conversion, not just the verdict."""
    assert (run(partial_nfa(), "a").verdict
            is fsa.run(to_dfa(partial_nfa()), "a").verdict
            is Verdict.REJECT_NO_TRANSITION)


@SETTINGS
@given(automaton=dfas())
def test_from_dfa_and_to_dfa_round_trip_any_dfa(automaton):
    assert to_dfa(from_dfa(automaton)) == automaton


@SETTINGS
@given(automaton=dfas(with_initial=False))
def test_the_round_trip_survives_a_machine_with_no_start_state(automaton):
    assert to_dfa(from_dfa(automaton)) == automaton


@SETTINGS
@given(automaton=dfas(), word=words())
def test_reading_a_dfa_as_an_nfa_changes_nothing_about_the_language(automaton, word):
    assert accepts(from_dfa(automaton), word) == fsa.accepts(automaton, word)


@SETTINGS
@given(automaton=dfas(), word=words())
def test_reading_a_dfa_as_an_nfa_changes_nothing_about_the_reason_either(
        automaton, word):
    """Stronger than language equality, and the claim from_dfa's docstring
    makes: the same verdict, at the same position, for the same reason."""
    ours = run(from_dfa(automaton), word)
    theirs = fsa.run(automaton, word)
    assert ours.verdict is theirs.verdict
    assert ours.stopped_at == theirs.stopped_at
    assert ours.offending_symbol == theirs.offending_symbol
    assert ours.final_states == (frozenset() if theirs.final_state is None
                                 else frozenset({theirs.final_state}))


# ---------------------------------------------------------------------------
# Determinism across processes
# ---------------------------------------------------------------------------


_ACROSS_PROCESSES = """
from fsa.nfa import EPSILON, NFA, run

names = [f"q{i}" for i in range(10)]
a = NFA(states=frozenset(names), alphabet=frozenset("ab")).with_initial("q0")
for index, name in enumerate(names):
    a = a.with_transition(name, "a", names[(index * 5 + 1) % 10])
    a = a.with_transition(name, "a", names[(index + 2) % 10])
    a = a.with_transition(name, "b", names[(index + 3) % 10])
    if index % 3 == 0:
        a = a.with_transition(name, EPSILON, names[(index + 7) % 10])
a = a.with_accept("q7")

print(a.sorted_transitions())
print(sorted(a.epsilon_closure(["q0"])))
print(run(a, "abba").explain())

# The edges in the mapping's own order, only the frozensets sorted: a set's
# repr is not stable across processes either, and the point here is whether
# grouped_transitions built its dict in a fixed order.
print([(edge, sorted(symbols, key=lambda s: (s is not None, s or "")))
       for edge, symbols in a.grouped_transitions().items()])
"""


def test_everything_that_leaves_the_module_is_the_same_in_every_process():
    """Python randomises string hashing per process, so a frozenset of state
    ids iterates differently every run. Calling twice in one process cannot
    catch that -- the order is fixed for the life of the interpreter -- so this
    runs the same machine under three hash seeds and compares the output."""
    src = Path(__file__).resolve().parent.parent / "src"
    outputs = []
    for seed in ("0", "1", "1000"):
        result = subprocess.run(
            [sys.executable, "-c", _ACROSS_PROCESSES],
            cwd=src, capture_output=True, text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1] == outputs[2]
    assert "q7" in outputs[0]
