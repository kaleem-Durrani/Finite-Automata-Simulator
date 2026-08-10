"""The completion operation.

Completion exists to make delta total *without changing the language*; nearly
every test here is some form of that claim. Imports no pygame and touches no
display -- if anything here needs one, the boundary has been broken.
"""

import random

import fsa
from fsa import DFA, Document, complete, ops

# ---------------------------------------------------------------------------
# Fixtures by hand
# ---------------------------------------------------------------------------


def incomplete_dfa() -> DFA:
    """a*b, with q1's arrows deliberately missing."""
    return (DFA()
            .with_states(["q0", "q1"])
            .with_transition("q0", "a", "q0")
            .with_transition("q0", "b", "q1")
            .with_accept("q1"))


# ---------------------------------------------------------------------------
# The basic contract
# ---------------------------------------------------------------------------


def test_completion_makes_delta_total():
    after, trap = complete(incomplete_dfa())
    assert fsa.is_complete(after)
    assert trap in after.states
    for symbol in after.alphabet:
        assert after.target(trap, symbol) == trap, "the trap loops on everything"
    assert trap in fsa.dead_states(after), "derived from the edges, not flagged"


def test_a_complete_automaton_comes_back_as_the_same_object():
    """Idempotence, and cheaply: no copy is made when there is nothing to do."""
    after, _ = complete(incomplete_dfa())
    again, trap = complete(after)
    assert again is after
    assert trap is None


def test_an_empty_alphabet_is_a_no_op():
    automaton = DFA().with_states(["q0", "q1"])
    after, trap = complete(automaton)
    assert after is automaton and trap is None


def test_an_empty_automaton_is_a_no_op():
    automaton = DFA()
    after, trap = complete(automaton)
    assert after is automaton and trap is None


def test_completion_does_not_invent_a_start_state():
    """with_state promotes its state to initial when there is none. Promoting
    the trap would turn "language not defined yet" into "the empty language"
    without telling anyone -- the same silent-promotion defect the old model
    had when deleting the initial state."""
    automaton = incomplete_dfa().with_initial(None)
    after, trap = complete(automaton)
    assert trap is not None
    assert after.initial is None


# ---------------------------------------------------------------------------
# Naming the trap
# ---------------------------------------------------------------------------


def test_an_explicit_trap_id_is_honoured():
    after, trap = complete(incomplete_dfa(), trap_id="sink")
    assert trap == "sink"
    assert "sink" in after.states
    assert fsa.is_complete(after)


def test_a_fresh_name_is_chosen_when_trap_is_already_a_state():
    automaton = incomplete_dfa().with_state("trap")
    after, trap = complete(automaton)
    assert trap == "trap1"
    assert fsa.is_complete(after)


def test_a_fresh_name_is_chosen_after_an_explicit_trap_took_the_default():
    first, trap = complete(incomplete_dfa(), trap_id="trap")
    assert trap == "trap"
    widened = first.with_state("q9")
    second, fresh = complete(widened)
    assert fresh == "trap1"
    assert fsa.is_complete(second)


def test_an_explicit_trap_id_may_name_an_existing_state():
    automaton = incomplete_dfa().with_state("trap")
    after, trap = complete(automaton, trap_id="trap")
    assert trap == "trap"
    assert "trap1" not in after.states, "reused, not duplicated"
    assert fsa.is_complete(after)
    for symbol in after.alphabet:
        assert after.target("trap", symbol) == "trap"


# ---------------------------------------------------------------------------
# THE property: completion never changes the language
# ---------------------------------------------------------------------------


def random_dfa(rng: random.Random) -> DFA:
    """Small, sometimes partial, sometimes complete, sometimes start-less."""
    alphabet = rng.sample("ab01", rng.randrange(1, 4))
    ids = [f"q{i}" for i in range(rng.randrange(1, 6))]
    automaton = DFA().with_states(ids)
    for symbol in alphabet:
        automaton = automaton.with_symbol(symbol)
    for state in ids:
        for symbol in alphabet:
            if rng.random() < 0.55:
                automaton = automaton.with_transition(state, symbol, rng.choice(ids))
    for state in ids:
        if rng.random() < 0.4:
            automaton = automaton.with_accept(state)
    if rng.random() < 0.15:
        automaton = automaton.with_initial(None)
    return automaton


def random_word(rng: random.Random) -> str:
    """Over "ab01" plus "xz", which no generated alphabet contains -- rejection
    for a foreign symbol must survive completion too."""
    return "".join(rng.choice("ab01xz") for _ in range(rng.randrange(0, 9)))


def test_completion_never_changes_the_language():
    """A trap may change the *reason* a word is rejected, never the verdict."""
    rng = random.Random(2026)
    for _ in range(150):
        before = random_dfa(rng)
        after, _ = complete(before)
        assert fsa.is_complete(after)
        for _ in range(40):
            word = random_word(rng)
            assert fsa.accepts(before, word) == fsa.accepts(after, word), \
                f"{word!r} changed verdict on {before!r}"


# ---------------------------------------------------------------------------
# Document.complete keeps the picture in step
# ---------------------------------------------------------------------------


def test_document_complete_places_the_trap():
    document = Document.of(incomplete_dfa())
    fixed, trap = document.complete()
    assert trap is not None
    assert fsa.is_complete(fixed.automaton)
    assert trap in fixed.layout.positions, "an invented state must be visible"
    positions = list(fixed.layout.positions.values())
    assert len(set(positions)) == len(positions), "and not under another state"
    assert fixed.next_id == document.next_id


def test_document_complete_is_a_no_op_when_already_complete():
    document = Document.of(complete(incomplete_dfa())[0])
    fixed, trap = document.complete()
    assert fixed is document
    assert trap is None


def test_the_operation_is_exported_once():
    assert fsa.complete is ops.complete
