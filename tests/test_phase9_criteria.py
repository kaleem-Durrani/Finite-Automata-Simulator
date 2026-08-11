"""Phase 9's exit criteria, checked across module boundaries.

Each algorithm arrived with its own tests, and a module that only tests itself
can be consistently wrong: minimisation that agrees with its own idea of the
language proves nothing. Everything here deliberately checks one module against
a *different* one -- minimisation against equivalence, complement against the
simulator -- so agreement means two independent implementations concur rather
than one implementation repeating itself.

The random automata are built here rather than imported from another test
module, for the same reason.
"""

import math
import random
from typing import List

import pytest

import fsa
from fsa import analysis, language, ops
from fsa.layout import Layout

ALPHABET = ("a", "b")


def random_dfa(rng: random.Random, states: int = 5,
               symbols=ALPHABET, density: float = 0.8) -> fsa.DFA:
    """A random, possibly partial, possibly unreachable automaton.

    `density` below 1 leaves delta undefined in places on purpose: a partial
    transition function is the case these algorithms get wrong, so it has to be
    the common case here.
    """
    ids = [f"q{i}" for i in range(states)]
    automaton = fsa.DFA(states=frozenset(ids), alphabet=frozenset(symbols))
    automaton = automaton.with_initial(ids[0])
    for state in ids:
        if rng.random() < 0.4:
            automaton = automaton.with_accept(state)
        for symbol in symbols:
            if rng.random() < density:
                automaton = automaton.with_transition(
                    state, symbol, rng.choice(ids))
    return automaton


def words(symbols=ALPHABET, max_length: int = 6) -> List[str]:
    """Every word up to `max_length`, shortest first."""
    out = [""]
    frontier = [""]
    for _ in range(max_length):
        frontier = [w + s for w in frontier for s in symbols]
        out.extend(frontier)
    return out


WORDS = words()


# ---------------------------------------------------------------------------
# 1. minimisation
# ---------------------------------------------------------------------------


def test_minimise_preserves_the_language_over_300_automata():
    """Criterion 1, checked with `equivalent` rather than with minimize's own
    notion of sameness -- two modules agreeing is evidence, one agreeing with
    itself is not."""
    rng = random.Random(20260812)
    for _ in range(300):
        automaton = random_dfa(rng, states=rng.randrange(1, 7))
        if automaton.initial is None:
            continue
        reduced = fsa.minimize(automaton)
        assert fsa.equivalent(automaton, reduced), automaton
        assert len(reduced.states) <= len(automaton.states)


def test_minimising_twice_changes_nothing_further():
    rng = random.Random(4242)
    for _ in range(120):
        automaton = random_dfa(rng, states=rng.randrange(2, 7))
        once = fsa.minimize(automaton)
        twice = fsa.minimize(once)
        assert len(twice.states) == len(once.states)
        assert fsa.equivalent(once, twice)


def test_a_minimal_machine_has_no_two_equivalent_states():
    """The property that makes it minimal, read off the marking table -- the
    artifact the GUI shows -- rather than off the state count."""
    rng = random.Random(99)
    for _ in range(60):
        automaton = random_dfa(rng, states=rng.randrange(2, 6))
        reduced = fsa.minimize(automaton)
        if len(reduced.states) < 2:
            continue
        table = fsa.marking_table(reduced)
        # A pair absent from `marks` is the algorithm concluding the two states
        # are equivalent. In a minimal machine there must be no such pair --
        # except any trap the table had to invent to complete a partial delta,
        # which is not a state of the machine we minimised.
        invented = {table.invented_trap} if table.invented_trap else set()
        unmarked = [pair for pair in table.pairs
                    if not table.is_distinguishable(*pair)
                    and not invented.intersection(pair)]
        assert not unmarked, f"{unmarked} are equivalent in a minimal machine"


# ---------------------------------------------------------------------------
# 2. complement
# ---------------------------------------------------------------------------


def test_complement_of_a_completed_machine_inverts_every_verdict():
    """Criterion 2. Completion first is the whole lesson: complement is *wrong*
    on a partial machine, because a word that simply halts is not accepted by
    either the machine or its complement."""
    rng = random.Random(7)
    for _ in range(60):
        automaton = random_dfa(rng, states=rng.randrange(1, 5))
        completed, _trap = ops.complete(automaton)
        flipped = fsa.complement(completed)
        for word in WORDS[:40]:
            assert fsa.accepts(flipped, word) != fsa.accepts(automaton, word), \
                (automaton, word)


def test_complement_refuses_a_partial_machine():
    """And says so in terms that name the fix."""
    automaton = fsa.DFA(states=frozenset({"q0"}),
                        alphabet=frozenset(ALPHABET)).with_initial("q0")
    assert not analysis.is_complete(automaton)
    with pytest.raises(fsa.IncompleteAutomatonError) as caught:
        fsa.complement(automaton)
    assert "complete" in str(caught.value).lower()


# ---------------------------------------------------------------------------
# 3. counterexamples
# ---------------------------------------------------------------------------


def test_a_counterexample_exists_exactly_when_the_languages_differ():
    """Criterion 3, both directions, and the witness must actually witness."""
    rng = random.Random(1234)
    for _ in range(200):
        left = random_dfa(rng, states=rng.randrange(1, 5))
        right = random_dfa(rng, states=rng.randrange(1, 5))
        witness = fsa.counterexample(left, right)

        assert (witness is None) == fsa.equivalent(left, right)
        if witness is not None:
            assert fsa.accepts(left, witness) != fsa.accepts(right, witness)


def test_the_counterexample_is_the_shortest_one():
    """BFS order is the point: a shortest distinguishing word is the useful
    feedback, and any longer one would also be 'correct'."""
    rng = random.Random(555)
    for _ in range(80):
        left = random_dfa(rng, states=rng.randrange(1, 5))
        right = random_dfa(rng, states=rng.randrange(1, 5))
        witness = fsa.counterexample(left, right)
        if witness is None:
            continue
        for word in WORDS:
            if len(word) >= len(witness):
                break
            assert fsa.accepts(left, word) == fsa.accepts(right, word), \
                f"{word} is shorter than {witness!r} and also distinguishes"


def test_a_machine_is_equivalent_to_itself():
    rng = random.Random(3)
    for _ in range(50):
        automaton = random_dfa(rng)
        assert fsa.equivalent(automaton, automaton)
        assert fsa.counterexample(automaton, automaton) is None


# ---------------------------------------------------------------------------
# 4. the demo's language
# ---------------------------------------------------------------------------


def test_sample_language_matches_the_verified_demo_table():
    """Criterion 4. The README's table is generated and test-locked elsewhere;
    this pins the same six words against the sampler."""
    import main as main_module

    automaton = main_module.demo_document().automaton
    accepted = language.sample_language(automaton, limit=6, max_length=3)
    assert accepted == ["b", "ab", "bb", "aab", "abb", "bbb"]
    for word in accepted:
        assert fsa.accepts(automaton, word)


# ---------------------------------------------------------------------------
# 6. generated automata get usable coordinates
# ---------------------------------------------------------------------------


#: The GUI draws states at this radius. Written numerically rather than
#: imported, because src/fsa must not depend on the renderer.
STATE_RADIUS = 30.0


def test_auto_layout_never_overlaps_two_states():
    """Criterion 6's geometry half. Every algorithm here emits states nobody
    placed, so a layout that stacks them makes the result unreadable."""
    rng = random.Random(2026)
    for _ in range(40):
        automaton = random_dfa(rng, states=12, density=0.6)
        layout = Layout.auto(automaton)
        assert set(layout.positions) == set(automaton.states)

        placed = sorted(layout.positions.items())
        for i, (_a, first) in enumerate(placed):
            for _b, second in placed[i + 1:]:
                assert math.dist(first, second) >= 2 * STATE_RADIUS


def test_auto_layout_places_the_output_of_a_real_algorithm():
    """The case it exists for: minimise produces states the user never placed."""
    rng = random.Random(11)
    automaton = random_dfa(rng, states=8)
    reduced = fsa.minimize(automaton)
    layout = Layout.auto(reduced)

    assert set(layout.positions) == set(reduced.states)
    assert len({tuple(p) for p in layout.positions.values()}) == len(reduced.states)
