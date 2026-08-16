"""The subset construction, checked from four directions.

The property that matters is one line -- the DFA accepts exactly what the NFA
accepts -- and it is checked here against hypothesis-generated machines, against
a hand-built set of the cases that break naive implementations, and against
`automata-lib`, which implements the same construction from the same definition
without sharing any of our assumptions.

The fourth direction is the one a language test cannot see: *which* DFA came
back. Subset names are the whole teaching value of this algorithm, so they are
tested as carefully as the language -- including across three processes with
different hash seeds, because a name derived from an unsorted set would differ
between runs and nobody would notice until two saved files failed to diff.

The cases with a history: an epsilon *cycle* (the closure that never returns), a
machine with no move at all on some symbol (the empty subset, which is where
completeness comes from), and a machine with no start state (where inventing one
would quietly change the claim being made).
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Set

import pytest
from automata.fa.dfa import DFA as OracleDFA
from automata.fa.nfa import NFA as OracleNFA
from hypothesis import HealthCheck, find, given, settings

import fsa
from fsa.nfa import EPSILON, NFA, from_dfa
from fsa.nfa import accepts as nfa_accepts
from fsa.nfa import run as nfa_run
from fsa.simulate import Verdict
from fsa.subset import EMPTY_SUBSET, TRAP_LABEL, determinize, subset_name
from tests import oracle
from tests.strategies import dfas, nfas, words

# Building a machine a transition at a time is not fast and every example here
# builds several, so the budget is small and the deadline off -- the settings
# the rest of the property suites use.
SETTINGS = settings(max_examples=50, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


# ---------------------------------------------------------------------------
# Machines by hand
# ---------------------------------------------------------------------------


def ends_in_ab() -> NFA:
    """The textbook NFA: words over {a,b} ending in "ab".

    q0 keeps every option open and *also* guesses that the a it just read begins
    the final "ab". This is the first determinisation exercise in most courses,
    and the answer is known: three subsets, {q0}, {q0,q1} and {q0,q2}.
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

    Accepts the empty string without reading anything, so it fails at once if
    the start subset is not the closure of the initial state.
    """
    return (NFA()
            .with_states(["q0", "q1", "q2"])
            .with_transition("q0", EPSILON, "q1")
            .with_transition("q1", EPSILON, "q2")
            .with_accept("q2"))


def epsilon_cycle() -> NFA:
    """q0 -eps-> q1 -eps-> q0, with a real edge hanging off q1.

    The machine a closure without a visited set spins on forever, and the one
    the construction would spin on twice over -- once per subset it builds.
    """
    return (NFA()
            .with_states(["q0", "q1", "q2"])
            .with_transition("q0", EPSILON, "q1")
            .with_transition("q1", EPSILON, "q0")
            .with_transition("q1", "a", "q2")
            .with_accept("q2"))


def dead_end() -> NFA:
    """One accepting state, an alphabet of two, and no move on either.

    Accepts the empty string and nothing else, so every non-empty word has to
    die somewhere -- which is what forces the empty subset into existence.
    """
    return (NFA(states=frozenset({"q0"}), alphabet=frozenset("ab"))
            .with_initial("q0")
            .with_accept("q0"))


def nth_from_the_end(n: int) -> NFA:
    """"the nth symbol from the end is an a", over {a,b}.

    The standard witness that the exponential is real: this needs n + 1 NFA
    states and the minimal DFA provably needs 2^n, because the machine has to
    remember the last n symbols and every one of those histories is a state a
    word can distinguish.
    """
    ids = [f"q{index}" for index in range(n + 1)]
    automaton = NFA(states=frozenset(ids), alphabet=frozenset("ab"))
    automaton = automaton.with_initial("q0")
    automaton = (automaton
                 .with_transition("q0", "a", "q0")
                 .with_transition("q0", "b", "q0")
                 .with_transition("q0", "a", "q1"))
    for index in range(1, n):
        for symbol in "ab":
            automaton = automaton.with_transition(ids[index], symbol,
                                                  ids[index + 1])
    return automaton.with_accept(ids[n])


# ---------------------------------------------------------------------------
# The oracle, extended to nondeterministic machines
# ---------------------------------------------------------------------------


def to_oracle_nfa(automaton: NFA) -> OracleNFA:
    """Our NFA in `automata-lib`'s representation.

    One difference beyond the nesting that :func:`tests.oracle.to_oracle`
    already bridges for DFAs: the oracle spells an epsilon move as the empty
    string, where we use ``None`` so that no character has to be reserved. Every
    state gets an entry, empty or not, so a machine with a dead end converts
    rather than looking malformed.

    Lives here rather than in :mod:`tests.oracle` because the subset
    construction is the only thing that needs it so far; it should move if a
    second caller appears.
    """
    if automaton.initial is None:
        raise oracle.Unsupported("the oracle requires an initial state")
    if not automaton.alphabet:
        raise oracle.Unsupported("the oracle requires a non-empty alphabet")

    nested: Dict[str, Dict[str, Set[str]]] = {
        state: {} for state in automaton.states}
    for source, symbol, targets in automaton.sorted_transitions():
        nested[source]["" if symbol is EPSILON else symbol] = set(targets)

    return OracleNFA(
        states=set(automaton.states),
        input_symbols=set(automaton.alphabet),
        transitions=nested,
        initial_state=automaton.initial,
        final_states=set(automaton.accept),
    )


def oracle_determinized(automaton: NFA) -> OracleDFA:
    """The oracle's own subset construction.

    ``minify=False`` on purpose: minimising is a different algorithm with a
    different answer, and folding it in would compare our determinisation
    against their determinisation *and* their minimisation at once. The
    languages must match either way -- the point is that a disagreement should
    name one algorithm.
    """
    return OracleDFA.from_nfa(to_oracle_nfa(automaton), minify=False)


def _oracle_or_skip(automaton: fsa.DFA) -> OracleDFA:
    try:
        return oracle.to_oracle(automaton)
    except oracle.Unsupported as reason:
        pytest.skip(str(reason))


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def test_a_subset_is_named_after_its_members_in_sorted_order():
    assert subset_name(frozenset({"q2", "q0", "q1"})) == "{q0,q1,q2}"


def test_the_name_does_not_depend_on_how_the_set_was_built():
    assert subset_name(frozenset(["q1", "q0"])) == subset_name(
        frozenset(["q0", "q1"]))


def test_ids_are_sorted_the_way_the_rest_of_the_engine_sorts_them():
    """Lexicographic, so q10 precedes q2. Stated as a test because it looks
    wrong at a glance, and because a numeric sort here would be a second
    ordering rule disagreeing with every list the UI shows."""
    assert subset_name(frozenset({"q2", "q10"})) == "{q10,q2}"


def test_a_singleton_keeps_its_braces():
    """{q0} and q0 are different things -- a set of NFA states and an NFA state
    -- and a table where they look alike is a table nobody can read."""
    assert subset_name(frozenset({"q0"})) == "{q0}"


def test_the_empty_subset_is_named_by_the_very_same_rule():
    assert subset_name(frozenset()) == "{}"
    assert EMPTY_SUBSET == "{}"


# ---------------------------------------------------------------------------
# The textbook machine
# ---------------------------------------------------------------------------


def test_the_textbook_machine_determinizes_to_the_textbook_answer():
    result = determinize(ends_in_ab())
    assert result.states == frozenset({"{q0}", "{q0,q1}", "{q0,q2}"})
    assert result.initial == "{q0}"
    assert result.accept == frozenset({"{q0,q2}"})


def test_the_textbook_machine_keeps_its_language():
    machine = ends_in_ab()
    result = determinize(machine)
    for word in ("", "a", "b", "ab", "ba", "aab", "abb", "abab", "bbab"):
        assert fsa.accepts(result, word) == nfa_accepts(machine, word), word
    assert fsa.accepts(result, "aab")
    assert not fsa.accepts(result, "aba")


def test_every_edge_of_the_textbook_answer_is_the_one_by_hand():
    """The states could be right and the arrows wrong, and the language tests
    above would still pass on the words they happen to try."""
    result = determinize(ends_in_ab())
    assert result.target("{q0}", "a") == "{q0,q1}"
    assert result.target("{q0}", "b") == "{q0}"
    assert result.target("{q0,q1}", "a") == "{q0,q1}"
    assert result.target("{q0,q1}", "b") == "{q0,q2}"
    assert result.target("{q0,q2}", "a") == "{q0,q1}"
    assert result.target("{q0,q2}", "b") == "{q0}"


# ---------------------------------------------------------------------------
# Complete, deterministic, reachable
# ---------------------------------------------------------------------------


@SETTINGS
@given(automaton=nfas())
def test_the_result_is_complete(automaton):
    """Phase exit criterion. delta is total, so no run of the result can ever
    stop for want of an arrow."""
    result = determinize(automaton)
    assert fsa.is_complete(result)
    assert fsa.missing_transitions(result) == ()


@SETTINGS
@given(automaton=nfas())
def test_the_result_is_deterministic_in_shape_and_not_only_in_type(automaton):
    """A DFA is deterministic by type, which proves nothing about the machine
    this function built. Read back as an NFA, it must have no epsilon move and
    no state with two targets -- the same question asked of the same shape."""
    assert from_dfa(determinize(automaton)).is_deterministic()


@SETTINGS
@given(automaton=nfas())
def test_every_state_of_the_result_is_reachable(automaton):
    """The point of walking outward instead of taking the powerset. A subset
    nothing can reach is never named, so this is exact rather than a bound."""
    result = determinize(automaton)
    assert fsa.reachable(result) == result.states


@SETTINGS
@given(automaton=nfas(max_states=4))
def test_the_result_never_exceeds_the_powerset(automaton):
    assert len(determinize(automaton).states) <= 2 ** len(automaton.states)


# ---------------------------------------------------------------------------
# The empty subset
# ---------------------------------------------------------------------------


def test_a_dead_end_produces_the_empty_subset_and_it_traps():
    result = determinize(dead_end())
    assert result.states == frozenset({"{q0}", EMPTY_SUBSET})
    assert result.target("{q0}", "a") == EMPTY_SUBSET
    assert result.target(EMPTY_SUBSET, "a") == EMPTY_SUBSET
    assert result.target(EMPTY_SUBSET, "b") == EMPTY_SUBSET
    assert EMPTY_SUBSET not in result.accept


def test_the_empty_subset_is_a_trap_by_the_engine_s_own_definition():
    """Derived from the edges by fsa.analysis, not flagged by us -- the rule
    that keeps the drawn machine and the simulated one from disagreeing."""
    assert fsa.is_trap(determinize(dead_end()), EMPTY_SUBSET)


def test_a_machine_with_a_move_everywhere_never_grows_a_trap():
    """The empty subset appears only when a branch really dies. Determinising a
    complete machine must not hand the user a state they have no use for."""
    result = determinize(from_dfa(fsa.complete(
        fsa.DFA()
        .with_states(["q0", "q1"])
        .with_transition("q0", "a", "q1")
        .with_accept("q1"))[0]))
    assert EMPTY_SUBSET not in result.states
    assert fsa.is_complete(result)


def test_the_verdict_survives_the_construction_even_though_the_reason_changes():
    """The NFA runs out of branches and stops; the DFA walks into the trap and
    reads the rest of the word sitting in it. Same answer, different story --
    and the story is the price of a total delta, not a defect."""
    machine = dead_end()
    result = determinize(machine)
    assert nfa_run(machine, "ab").verdict is Verdict.REJECT_NO_TRANSITION
    assert fsa.run(result, "ab").verdict is Verdict.REJECT_NON_ACCEPTING
    assert nfa_accepts(machine, "ab") == fsa.accepts(result, "ab") is False


# ---------------------------------------------------------------------------
# The language -- the phase exit criterion
# ---------------------------------------------------------------------------


@SETTINGS
@given(automaton=nfas(alphabet="ab"), word=words(alphabet="ab"))
def test_determinizing_preserves_the_language(automaton, word):
    assert nfa_accepts(automaton, word) == fsa.accepts(determinize(automaton),
                                                       word)


@SETTINGS
@given(automaton=nfas(), word=words(alphabet="abc01xyz"))
def test_it_preserves_the_language_outside_the_alphabet_too(automaton, word):
    """The alphabet travels with the machine, so a word using a symbol the
    machine never declared is rejected identically by both -- at the same
    position, for the same reason. Narrowing or widening Sigma here would show
    up as a changed *reason* long before it changed a verdict."""
    result = determinize(automaton)
    assert result.alphabet == automaton.alphabet
    assert nfa_accepts(automaton, word) == fsa.accepts(result, word)

    theirs = nfa_run(automaton, word)
    ours = fsa.run(result, word)
    if theirs.verdict is Verdict.REJECT_SYMBOL_NOT_IN_ALPHABET:
        assert ours.verdict is theirs.verdict
        assert ours.stopped_at == theirs.stopped_at
        assert ours.offending_symbol == theirs.offending_symbol


@SETTINGS
@given(automaton=dfas(max_states=5), word=words())
def test_a_dfa_survives_the_round_trip_through_nondeterminism(automaton, word):
    """DFA -> NFA -> DFA. The names change and the machine may gain a trap, but
    nothing about the language may."""
    assert fsa.accepts(determinize(from_dfa(automaton)), word) == fsa.accepts(
        automaton, word)


@SETTINGS
@given(automaton=dfas(max_states=5))
def test_a_machine_that_was_already_deterministic_gains_at_most_a_trap(automaton):
    """Determinising something deterministic must not multiply its states: each
    reachable state becomes the singleton subset holding it, plus the empty
    subset if delta had a gap."""
    result = determinize(from_dfa(automaton))
    assert len(result.states) <= len(fsa.reachable(automaton)) + 1


@SETTINGS
@given(automaton=nfas(alphabet="ab"))
def test_the_result_is_equivalent_by_the_engine_s_own_equivalence(automaton):
    """A second opinion from inside the codebase: `equivalent` walks the two
    machines in lockstep looking for a distinguishing word, which is a different
    program from the simulator the property above uses."""
    result = determinize(automaton)
    assert fsa.equivalent(result, determinize(from_dfa(result)))


# ---------------------------------------------------------------------------
# Epsilon moves
# ---------------------------------------------------------------------------


def test_the_start_state_is_the_closure_of_the_initial_state():
    result = determinize(epsilon_chain())
    assert result.initial == "{q0,q1,q2}"
    assert fsa.accepts(result, ""), "accepting by epsilon moves alone"


def test_an_epsilon_cycle_does_not_hang_the_construction():
    """The case naive implementations never return from. If this test hangs
    rather than fails, the closure lost its visited set."""
    machine = epsilon_cycle()
    result = determinize(machine)
    assert result.initial == "{q0,q1}"
    assert fsa.accepts(result, "a")
    for word in ("", "a", "aa", "b"):
        assert fsa.accepts(result, word) == nfa_accepts(machine, word), word


def test_epsilon_moves_are_gone_from_the_result_entirely():
    """Not merely followed -- consumed. Every symbol of the result's delta is a
    real symbol of the alphabet, which is half of what deterministic means."""
    result = determinize(epsilon_cycle())
    assert all(symbol in result.alphabet
               for _, symbol in result.transitions)


@SETTINGS
@given(automaton=nfas(epsilons=True, alphabet="ab"), word=words(alphabet="ab"))
def test_machines_full_of_epsilon_moves_still_determinize_correctly(automaton, word):
    assert nfa_accepts(automaton, word) == fsa.accepts(determinize(automaton),
                                                       word)


# ---------------------------------------------------------------------------
# Reachable subsets only, and the blow-up that is real
# ---------------------------------------------------------------------------


def test_only_the_subsets_some_word_reaches_are_built():
    """Three NFA states have eight subsets. This machine reaches three of them,
    and the powerset construction as written in the textbook would draw all
    eight -- five of which no word can ever enter."""
    assert len(determinize(ends_in_ab()).states) == 3


def test_the_exponential_is_paid_where_the_language_demands_it():
    """"the 3rd symbol from the end is an a" needs 2^3 states and gets them.
    Building only reachable subsets is not a trick that avoids the blow-up; it
    is a refusal to pay for the blow-up when it is not there."""
    machine = nth_from_the_end(3)
    result = determinize(machine)
    assert len(result.states) == 8
    assert fsa.accepts(result, "abb"), "3rd from the end is a"
    assert not fsa.accepts(result, "bab")
    for word in ("", "a", "ab", "aab", "baab", "bbbab"):
        assert fsa.accepts(result, word) == nfa_accepts(machine, word), word


# ---------------------------------------------------------------------------
# No initial state
# ---------------------------------------------------------------------------


def test_a_machine_with_no_start_state_determinizes_to_one_with_none():
    """Nothing is reachable when there is nowhere to start. Inventing a start
    state would turn "no language defined yet" into "the empty language", which
    this codebase reads as a different claim -- the rule trim and
    DFA.without_state already follow."""
    machine = NFA(states=frozenset({"q0"}), alphabet=frozenset("ab"))
    result = determinize(machine)
    assert result.states == frozenset()
    assert result.initial is None
    assert result.accept == frozenset()


def test_the_alphabet_survives_a_machine_with_no_start_state():
    result = determinize(NFA(states=frozenset({"q0"}), alphabet=frozenset("ab")))
    assert result.alphabet == frozenset("ab")


def test_both_machines_say_no_initial_state_rather_than_rejecting():
    machine = NFA(states=frozenset({"q0"}), alphabet=frozenset("ab"))
    result = determinize(machine)
    for word in ("", "a", "ab"):
        assert nfa_run(machine, word).verdict is Verdict.NO_INITIAL_STATE
        assert fsa.run(result, word).verdict is Verdict.NO_INITIAL_STATE


def test_an_empty_result_is_complete_vacuously():
    """The promise of completeness has to survive the degenerate case, or the
    docstring is lying in exactly the situation nobody checks."""
    result = determinize(NFA(states=frozenset({"q0"}), alphabet=frozenset("ab")))
    assert fsa.is_complete(result)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def test_a_subset_wears_the_display_names_of_its_members():
    machine = (ends_in_ab()
               .with_label("q0", "start")
               .with_label("q2", "seen ab"))
    result = determinize(machine)
    assert result.label_of("{q0}") == "{start}"
    assert result.label_of("{q0,q2}") == "{start,seen ab}"


def test_a_member_with_no_label_shows_as_its_own_id_inside_the_subset():
    result = determinize(ends_in_ab().with_label("q0", "start"))
    assert result.label_of("{q0,q1}") == "{start,q1}"


def test_an_unlabelled_machine_produces_no_labels_at_all():
    """Writing the id back would read identically and count as an edit --
    dirtying the file for a change nobody can see. Same rule as
    DFA.with_label_removed."""
    result = determinize(ends_in_ab())
    assert dict(result.labels) == {}
    assert result.label_of("{q0}") == "{q0}"


def test_the_trap_says_on_screen_what_it_is():
    """`{}` in a circle reads as a state whose name failed to render. The label
    is the one place that can say "every branch died here"."""
    assert determinize(dead_end()).label_of(EMPTY_SUBSET) == TRAP_LABEL


def test_labels_cannot_change_the_language():
    machine = ends_in_ab()
    labelled = machine.with_label("q0", "start").with_label("q1", "guessing")
    for word in ("", "a", "ab", "aab", "abb"):
        assert fsa.accepts(determinize(machine), word) == fsa.accepts(
            determinize(labelled), word)


# ---------------------------------------------------------------------------
# The same answer every time
# ---------------------------------------------------------------------------


def test_determinizing_twice_gives_the_same_value():
    assert determinize(ends_in_ab()) == determinize(ends_in_ab())
    assert hash(determinize(ends_in_ab())) == hash(determinize(ends_in_ab()))


def test_the_input_is_never_touched():
    machine = ends_in_ab()
    before = (machine.states, machine.sorted_transitions(), machine.accept)
    determinize(machine)
    assert (machine.states, machine.sorted_transitions(), machine.accept) == before


def test_determinizing_a_determinized_machine_only_nests_the_names():
    """Idempotent in language, not in names: the subsets of an already
    deterministic machine are singletons, so `{q0,q1}` becomes `{{q0,q1}}`. The
    nesting is the naming rule being honest about what it was handed."""
    once = determinize(ends_in_ab())
    twice = determinize(from_dfa(once))
    assert twice.states == frozenset({"{{q0}}", "{{q0,q1}}", "{{q0,q2}}"})
    assert fsa.equivalent(once, twice)


_ACROSS_PROCESSES = """
from fsa.nfa import EPSILON, NFA
from fsa.subset import determinize

names = [f"q{i}" for i in range(8)]
a = NFA(states=frozenset(names), alphabet=frozenset("ab")).with_initial("q0")
for index, name in enumerate(names):
    a = a.with_transition(name, "a", names[(index * 3 + 1) % 8])
    a = a.with_transition(name, "a", names[(index + 2) % 8])
    a = a.with_transition(name, "b", names[(index + 5) % 8])
    if index % 3 == 0:
        a = a.with_transition(name, EPSILON, names[(index + 4) % 8])
a = a.with_accept("q5").with_label("q3", "third")

d = determinize(a)
print(sorted(d.states))
print(d.initial)
print(sorted(d.accept))
print(sorted(d.transitions.items()))
print(sorted(d.labels.items()))
"""


def test_the_names_are_the_same_in_every_process():
    """Python randomises string hashing per process, so a frozenset of state ids
    iterates differently on every run and a name built without sorting would
    come out differently too. Calling twice in one process cannot catch that --
    the order is fixed for the life of the interpreter -- so this runs the same
    machine under three hash seeds.

    The collections are printed sorted on purpose: what is under test is the
    *names*, which are built out of sets, not the order this script prints them
    in. An unsorted name is a different string and would survive the sort.
    """
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
    assert "{q0," in outputs[0], "the machine really did build multi-state subsets"


# ---------------------------------------------------------------------------
# Against an independent implementation
# ---------------------------------------------------------------------------


@SETTINGS
@given(automaton=nfas())
def test_our_determinize_agrees_with_the_oracle(automaton):
    """Phase exit criterion. Languages, never state names: their construction
    numbers its states and ours names them after subsets, and both are right."""
    ours = _oracle_or_skip(determinize(automaton))
    assert ours == oracle_determinized(automaton)


@SETTINGS
@given(automaton=nfas(alphabet="ab"), word=words(alphabet="ab"))
def test_membership_after_determinizing_agrees_with_the_oracle(automaton, word):
    """Language equality compares two whole machines; this asks about one word
    at a time, which is what a failure has to be reported in anyway."""
    assert fsa.accepts(determinize(automaton), word) == oracle_determinized(
        automaton).accepts_input(word)


@SETTINGS
@given(automaton=nfas(alphabet="ab"), word=words(alphabet="ab"))
def test_the_nfa_bridge_preserves_the_language(automaton, word):
    """If converting to the oracle changed the machine, the two tests above
    would be comparing different automata and passing anyway. Epsilon is the
    part most likely to be mistranslated -- ours is None, theirs is ""."""
    assert nfa_accepts(automaton, word) == to_oracle_nfa(automaton).accepts_input(word)


def test_the_bridge_refuses_a_machine_it_cannot_represent():
    """Loudly, so a differential test never reports success for something it
    silently skipped."""
    with pytest.raises(oracle.Unsupported):
        to_oracle_nfa(NFA(states=frozenset({"q0"}), alphabet=frozenset("a")))


# ---------------------------------------------------------------------------
# The generator itself
# ---------------------------------------------------------------------------


@SETTINGS
@given(automaton=nfas())
def test_the_strategy_builds_only_legal_machines(automaton):
    """A strategy that raised, or that quietly built something illegal, would
    hide every failure the properties above exist to find."""
    assert automaton.accept <= automaton.states
    for (source, symbol), targets in automaton.transitions.items():
        assert source in automaton.states
        assert targets <= automaton.states
        assert symbol is EPSILON or symbol in automaton.alphabet


def test_the_strategy_can_produce_the_cases_that_matter():
    """`find` rather than a property, because these are existence claims: a
    generator that never branched or never emitted an epsilon move would leave
    the whole suite above green and vacuous. `find` raises if it cannot."""
    find(nfas(), lambda machine: any(
        symbol is EPSILON for _, symbol in machine.transitions))
    find(nfas(), lambda machine: any(
        len(targets) > 1 for targets in machine.transitions.values()))
    find(nfas(), lambda machine: not machine.is_deterministic())
