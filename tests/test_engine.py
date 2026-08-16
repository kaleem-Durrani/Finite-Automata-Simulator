"""Tests for the fsa engine.

Deliberately imports no pygame and touches no display. If anything here needs
one, the boundary has been broken.
"""

import random
from typing import Dict, List, Optional, Tuple

import pytest

import fsa
from fsa import DFA, Verdict

# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["a", "Z", "0", "!", "#", "α"])
def test_legal_symbols(value):
    assert fsa.is_legal_symbol(value)


@pytest.mark.parametrize("value", ["", "ab", " ", "\t", "\n", None, 1, "a b"])
def test_illegal_symbols(value):
    assert not fsa.is_legal_symbol(value)


def test_alphabet_rejects_illegal_symbols():
    with pytest.raises(fsa.IllegalSymbolError):
        fsa.normalize_alphabet(["a", "bb"])


def test_reserved_letters_are_ordinary_symbols():
    """q, w, r, n and p were unusable in the old UI because keys owned them.

    The engine has no such notion; a symbol is a character.
    """
    automaton = (DFA()
                 .with_states(["s0", "s1"])
                 .with_transition("s0", "q", "s1")
                 .with_transition("s1", "w", "s0")
                 .with_accept("s1"))
    assert fsa.accepts(automaton, "q")
    assert fsa.accepts(automaton, "qwq")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@pytest.fixture
def a_star_b_plus() -> DFA:
    """The demo automaton: a*b+."""
    return (DFA()
            .with_states(["q0", "q1", "q2"])
            .with_transition("q0", "a", "q0")
            .with_transition("q0", "b", "q1")
            .with_transition("q1", "a", "q2")
            .with_transition("q1", "b", "q1")
            .with_transition("q2", "a", "q2")
            .with_transition("q2", "b", "q2")
            .with_accept("q1"))


def test_empty_automaton_is_valid():
    automaton = DFA()
    assert automaton.states == frozenset()
    assert automaton.initial is None
    assert fsa.run(automaton, "").verdict is Verdict.NO_INITIAL_STATE


def test_first_state_becomes_initial():
    automaton = DFA().with_state("a").with_state("b")
    assert automaton.initial == "a"


def test_operations_do_not_mutate():
    original = DFA().with_state("q0")
    derived = original.with_state("q1")
    assert original.states == {"q0"}
    assert derived.states == {"q0", "q1"}


def test_equality_is_structural():
    one = DFA().with_states(["q0", "q1"]).with_transition("q0", "a", "q1")
    two = DFA().with_states(["q0", "q1"]).with_transition("q0", "a", "q1")
    assert one == two
    assert hash(one) == hash(two)
    assert one != two.with_accept("q1")


def test_automata_are_hashable_and_usable_as_values():
    """Undo is 'keep the previous value', so values must behave like values."""
    automaton = DFA().with_state("q0")
    history = {automaton, automaton.with_state("q1"), automaton}
    assert len(history) == 2


def test_deleting_a_state_removes_its_transitions(a_star_b_plus):
    smaller = a_star_b_plus.without_state("q1")
    assert "q1" not in smaller.states
    assert all("q1" not in (source, target)
               for (source, _s), target in smaller.transitions.items())
    assert "q1" not in smaller.accept


def test_deleting_the_initial_state_leaves_none(a_star_b_plus):
    """The old model silently promoted an arbitrary state, changing the language."""
    orphaned = a_star_b_plus.without_state("q0")
    assert orphaned.initial is None
    assert fsa.run(orphaned, "b").verdict is Verdict.NO_INITIAL_STATE


def test_transitions_are_flat_so_a_sink_needs_no_entry():
    """The nested representation raised KeyError on states with no edges."""
    automaton = DFA().with_states(["q0", "q1"]).with_transition("q0", "a", "q1")
    assert automaton.outgoing("q1") == {}
    assert automaton.target("q1", "a") is None
    assert automaton.without_state("q1").states == {"q0"}


def test_redefining_a_transition_replaces_it(a_star_b_plus):
    changed = a_star_b_plus.with_transition("q0", "b", "q0")
    assert changed.target("q0", "b") == "q0"
    assert len(changed.transitions) == len(a_star_b_plus.transitions)


def test_unknown_states_are_rejected():
    automaton = DFA().with_state("q0")
    with pytest.raises(fsa.UnknownStateError):
        automaton.with_transition("q0", "a", "nope")
    with pytest.raises(fsa.UnknownStateError):
        automaton.with_accept("nope")
    with pytest.raises(fsa.UnknownStateError):
        automaton.without_state("nope")
    with pytest.raises(fsa.UnknownStateError):
        automaton.with_initial("nope")


def test_duplicate_state_is_rejected():
    with pytest.raises(fsa.DuplicateStateError):
        DFA().with_state("q0").with_state("q0")


def test_constructing_with_a_dangling_transition_is_rejected():
    with pytest.raises(fsa.UnknownStateError):
        DFA(states=frozenset({"q0"}), alphabet=frozenset({"a"}),
            transitions={("q0", "a"): "ghost"})


def test_constructing_with_a_transition_from_an_unknown_state_is_rejected():
    with pytest.raises(fsa.UnknownStateError):
        DFA(states=frozenset({"q0"}), alphabet=frozenset({"a"}),
            transitions={("ghost", "a"): "q0"})


def test_constructing_with_an_off_alphabet_transition_is_rejected():
    with pytest.raises(fsa.UnknownStateError, match="not in the alphabet"):
        DFA(states=frozenset({"q0"}), alphabet=frozenset({"a"}),
            transitions={("q0", "b"): "q0"})


def test_constructing_with_an_unknown_accepting_state_is_rejected():
    with pytest.raises(fsa.UnknownStateError):
        DFA(states=frozenset({"q0"}), accept=frozenset({"ghost"}))


def test_constructing_with_an_unknown_initial_state_is_rejected():
    with pytest.raises(fsa.UnknownStateError):
        DFA(states=frozenset({"q0"}), initial="ghost")


def test_labels_for_unknown_states_are_dropped():
    automaton = DFA(states=frozenset({"q0"}), labels={"q0": "start", "gone": "x"})
    assert dict(automaton.labels) == {"q0": "start"}


def test_comparison_with_other_types():
    assert DFA() != "not an automaton"
    assert (DFA() == 42) is False


def test_repr_is_informative(a_star_b_plus):
    text = repr(a_star_b_plus)
    assert "|Q|=3" in text
    assert "initial='q0'" in text


def test_toggling_acceptance(a_star_b_plus):
    toggled = a_star_b_plus.with_accept_toggled("q1")
    assert "q1" not in toggled.accept
    assert "q1" in toggled.with_accept_toggled("q1").accept


def test_clearing_the_initial_state(a_star_b_plus):
    assert a_star_b_plus.with_initial(None).initial is None


def test_label_for_an_unknown_state_is_rejected(a_star_b_plus):
    with pytest.raises(fsa.UnknownStateError):
        a_star_b_plus.with_label("ghost", "x")


def test_removing_a_transition_that_is_not_there_is_harmless(a_star_b_plus):
    assert a_star_b_plus.without_transition("q0", "zzz") == a_star_b_plus


def test_grouped_transitions_is_derived(a_star_b_plus):
    grouped = a_star_b_plus.grouped_transitions()
    assert grouped[("q2", "q2")] == frozenset({"a", "b"})
    assert grouped[("q0", "q1")] == frozenset({"b"})
    # And it tracks the transition function, because it is computed from it.
    changed = a_star_b_plus.without_transition("q2", "a")
    assert changed.grouped_transitions()[("q2", "q2")] == frozenset({"b"})


def test_labels_are_cosmetic(a_star_b_plus):
    labelled = a_star_b_plus.with_label("q1", "accepting")
    assert labelled.label_of("q1") == "accepting"
    assert labelled.label_of("q0") == "q0"
    for word in ["", "b", "ab", "ba"]:
        assert fsa.accepts(labelled, word) == fsa.accepts(a_star_b_plus, word)


def test_removing_a_symbol_removes_its_transitions(a_star_b_plus):
    without_a = a_star_b_plus.without_symbol("a")
    assert "a" not in without_a.alphabet
    assert all(symbol != "a" for _state, symbol in without_a.transitions)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def test_the_four_verdicts_are_distinguishable():
    """The whole product thesis: a rejection says which kind it is."""
    automaton = (DFA()
                 .with_states(["q0", "q1"])
                 .with_transition("q0", "a", "q1")
                 .with_symbol("b")
                 .with_accept("q1"))

    assert fsa.run(automaton, "a").verdict is Verdict.ACCEPT
    assert fsa.run(automaton, "").verdict is Verdict.REJECT_NON_ACCEPTING
    assert fsa.run(automaton, "b").verdict is Verdict.REJECT_NO_TRANSITION
    assert fsa.run(automaton, "z").verdict is Verdict.REJECT_SYMBOL_NOT_IN_ALPHABET
    assert fsa.run(automaton.with_initial(None), "a").verdict is Verdict.NO_INITIAL_STATE


def test_explanations_name_the_cause():
    automaton = (DFA()
                 .with_states(["q0", "q1"])
                 .with_transition("q0", "a", "q1")
                 .with_symbol("b")
                 .with_accept("q1"))

    assert "accepted" in fsa.run(automaton, "a").explain()
    assert "not an accepting state" in fsa.run(automaton, "").explain()
    assert "incomplete" in fsa.run(automaton, "b").explain()
    assert "not in the alphabet" in fsa.run(automaton, "z").explain()
    assert "no initial state" in fsa.run(automaton.with_initial(None), "a").explain()


def test_empty_word_is_simulable(a_star_b_plus):
    """The GUI blocks this; the engine must not."""
    result = fsa.run(a_star_b_plus, "")
    assert result.path == ("q0",)
    assert result.steps == ()
    assert "empty string" in result.explain()

    accepting_start = a_star_b_plus.with_accept("q0")
    assert fsa.accepts(accepting_start, "")


def test_run_records_where_it_stopped():
    automaton = (DFA()
                 .with_states(["q0", "q1"])
                 .with_transition("q0", "a", "q1")
                 .with_symbol("b")
                 .with_accept("q1"))

    result = fsa.run(automaton, "ab")
    assert result.stopped_at == 1
    assert result.offending_symbol == "b"
    assert result.consumed == "a"
    assert result.remaining == "b"
    assert result.path == ("q0", "q1")


def test_symbol_outside_alphabet_reports_its_position(a_star_b_plus):
    result = fsa.run(a_star_b_plus, "abz")
    assert result.stopped_at == 2
    assert result.offending_symbol == "z"
    assert result.path == ("q0", "q0", "q1")


def test_steps_are_contiguous(a_star_b_plus):
    result = fsa.run(a_star_b_plus, "aab")
    assert [step.index for step in result.steps] == [0, 1, 2]
    for earlier, later in zip(result.steps, result.steps[1:]):
        assert earlier.target == later.source


def test_a_trap_state_no_longer_overrides_the_transition_function():
    """The defect this engine exists to remove.

    q1 looks like a trap and would have been flagged as one by hand, but delta
    leads from it to an accepting state. The language is what delta says.
    """
    automaton = (DFA()
                 .with_states(["q0", "q1", "q2"])
                 .with_transition("q0", "a", "q1")
                 .with_transition("q1", "a", "q2")
                 .with_transition("q2", "a", "q2")
                 .with_accept("q2"))

    assert fsa.accepts(automaton, "aa")
    assert fsa.run(automaton, "aa").path == ("q0", "q1", "q2")
    assert fsa.dead_states(automaton) == frozenset()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def test_completeness(a_star_b_plus):
    assert fsa.is_complete(a_star_b_plus)
    assert fsa.missing_transitions(a_star_b_plus) == ()

    partial = a_star_b_plus.without_transition("q0", "a")
    assert not fsa.is_complete(partial)
    assert fsa.missing_transitions(partial) == (("q0", "a"),)


def test_dead_states_are_derived_not_declared(a_star_b_plus):
    """q2 is a genuine trap: nothing accepting is reachable from it."""
    assert fsa.dead_states(a_star_b_plus) == frozenset({"q2"})
    assert fsa.is_trap(a_star_b_plus, "q2")
    assert not fsa.is_trap(a_star_b_plus, "q0")


def test_dead_states_follow_the_edges(a_star_b_plus):
    """Give the trap a way out and it stops being a trap. No flag to update."""
    escaped = a_star_b_plus.with_transition("q2", "b", "q1")
    assert fsa.dead_states(escaped) == frozenset()


def test_reachability(a_star_b_plus):
    assert fsa.reachable(a_star_b_plus) == {"q0", "q1", "q2"}
    assert fsa.unreachable_states(a_star_b_plus) == frozenset()

    with_island = a_star_b_plus.with_state("island")
    assert fsa.unreachable_states(with_island) == frozenset({"island"})


def test_reachability_of_an_automaton_with_no_start():
    automaton = DFA().with_states(["q0", "q1"]).with_initial(None)
    assert fsa.reachable(automaton) == frozenset()
    assert fsa.unreachable_states(automaton) == {"q0", "q1"}


def test_defects_reports_an_incomplete_automaton():
    automaton = (DFA()
                 .with_states(["q0", "q1"])
                 .with_transition("q0", "a", "q1")
                 .with_symbol("b")
                 .with_accept("q1"))

    kinds = {defect.kind for defect in fsa.defects(automaton)}
    assert "incomplete" in kinds

    incomplete = next(d for d in fsa.defects(automaton) if d.kind == "incomplete")
    assert ("q0", "b") in incomplete.pairs
    assert ("q1", "a") in incomplete.pairs


def test_incomplete_defect_summarises_long_lists():
    """Six states over two symbols with no edges: 12 missing pairs."""
    automaton = (DFA(alphabet=frozenset({"a", "b"}))
                 .with_states([f"s{i}" for i in range(6)])
                 .with_accept("s0"))
    incomplete = next(d for d in fsa.defects(automaton) if d.kind == "incomplete")
    assert len(incomplete.pairs) == 12
    assert "and 8 more" in incomplete.message


def test_defects_reports_blocking_problems_first():
    automaton = DFA().with_states(["q0"]).with_initial(None)
    found = fsa.defects(automaton)
    assert found[0].kind == "no_initial_state"
    assert found[0].is_blocking
    assert any(d.kind == "no_accepting_states" for d in found)


def test_a_clean_automaton_has_no_defects():
    automaton = (DFA()
                 .with_states(["e", "o"])
                 .with_transition("e", "0", "o")
                 .with_transition("e", "1", "e")
                 .with_transition("o", "0", "e")
                 .with_transition("o", "1", "o")
                 .with_accept("e"))
    assert fsa.defects(automaton) == ()


def test_defect_messages_are_human_readable(a_star_b_plus):
    for defect in fsa.defects(a_star_b_plus):
        assert defect.message
        assert defect.message[0].isupper()
        assert defect.message.endswith(".")


# ---------------------------------------------------------------------------
# Property tests against an independent reference
# ---------------------------------------------------------------------------


def reference_delta_hat(
    transitions: Dict[Tuple[str, str], str],
    accept: set,
    alphabet: set,
    start: Optional[str],
    word: str,
) -> Tuple[bool, List[str]]:
    """A deliberately naive re-implementation of delta-hat.

    Written from the definition, in this file, with no reference to the engine.
    Six lines of logic whose only job is to disagree if the engine is wrong.
    """
    if start is None:
        return False, []
    state = start
    path = [state]
    for symbol in word:
        if symbol not in alphabet or (state, symbol) not in transitions:
            return False, path
        state = transitions[(state, symbol)]
        path.append(state)
    return state in accept, path


def random_automaton(rng: random.Random) -> DFA:
    """A small random automaton, often partial, sometimes with no start."""
    n = rng.randrange(1, 6)
    states = [f"s{i}" for i in range(n)]
    alphabet = rng.sample(["a", "b", "c"], rng.randrange(1, 4))

    automaton = DFA(alphabet=frozenset(alphabet)).with_states(states)
    for state in states:
        for symbol in alphabet:
            if rng.random() < 0.75:  # deliberately partial 25% of the time
                automaton = automaton.with_transition(state, symbol,
                                                      rng.choice(states))
    for state in states:
        if rng.random() < 0.35:
            automaton = automaton.with_accept(state)
    if rng.random() < 0.1:
        automaton = automaton.with_initial(None)
    return automaton


def test_simulation_agrees_with_the_reference_on_random_input():
    """200 random automata x 200 random words."""
    rng = random.Random(20260808)
    alphabet_pool = "abcz"  # 'z' is often outside the alphabet, on purpose

    for _ in range(200):
        automaton = random_automaton(rng)
        transitions = dict(automaton.transitions)
        accept = set(automaton.accept)
        alphabet = set(automaton.alphabet)

        for _ in range(200):
            word = "".join(rng.choice(alphabet_pool)
                           for _ in range(rng.randrange(0, 7)))
            result = fsa.run(automaton, word)
            expected_accept, expected_path = reference_delta_hat(
                transitions, accept, alphabet, automaton.initial, word)

            assert result.accepted == expected_accept, (
                f"{automaton!r} on {word!r}: {result.explain()}")
            assert list(result.path) == expected_path, f"{automaton!r} on {word!r}"


def test_path_invariant_holds_for_random_runs():
    rng = random.Random(99)
    for _ in range(300):
        automaton = random_automaton(rng)
        word = "".join(rng.choice("abc") for _ in range(rng.randrange(0, 8)))
        result = fsa.run(automaton, word)

        if result.start is None:
            assert result.path == ()
        else:
            assert len(result.path) == len(result.steps) + 1
        assert result.consumed + result.remaining == word
        assert result.explain()


def test_a_complete_automaton_always_consumes_the_whole_word():
    """If delta is total and the word is over the alphabet, nothing stops early."""
    rng = random.Random(7)
    for _ in range(100):
        states = [f"s{i}" for i in range(rng.randrange(1, 5))]
        alphabet = ["a", "b"]
        automaton = DFA(alphabet=frozenset(alphabet)).with_states(states)
        for state in states:
            for symbol in alphabet:
                automaton = automaton.with_transition(state, symbol,
                                                      rng.choice(states))
        assert fsa.is_complete(automaton)

        word = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 10)))
        result = fsa.run(automaton, word)
        assert result.stopped_at == len(word)
        assert result.verdict in (Verdict.ACCEPT, Verdict.REJECT_NON_ACCEPTING)


def test_dead_states_never_accept_anything():
    """Definitional: no word read from a dead state can be accepted."""
    rng = random.Random(4242)
    for _ in range(150):
        automaton = random_automaton(rng)
        dead = fsa.dead_states(automaton)
        if not dead:
            continue
        start = sorted(dead)[0]
        rerooted = automaton.with_initial(start)
        for _ in range(25):
            word = "".join(rng.choice("abc") for _ in range(rng.randrange(0, 6)))
            assert not fsa.accepts(rerooted, word)


# ---------------------------------------------------------------------------
# The boundary
# ---------------------------------------------------------------------------


def test_engine_imports_without_pygame():
    """The engine must be usable with no display and no third-party packages."""
    import subprocess
    import sys
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src"
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys\n"
         "for blocked in ('pygame', 'networkx', 'numpy'):\n"
         "    sys.modules[blocked] = None\n"
         "import fsa\n"
         "a = fsa.DFA().with_state('q0').with_accept('q0')\n"
         "assert fsa.accepts(a, '')\n"
         "print(len(fsa.__all__))\n"],
        cwd=src, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) > 15


def test_engine_uses_only_the_standard_library():
    """No third-party import may appear anywhere in the engine.

    Parses the import statements rather than searching the text. A substring
    search matches the engine's own docstrings, which explain this very rule --
    the first version of this check reported the documentation as a violation.
    """
    import ast
    import pathlib

    # `greenery` converts regex to FSM and back, which is exactly what
    # fsa.regex does, so it is the one package most tempting to reach for and
    # the one that would hollow the module out. It is a test oracle only.
    banned = {"pygame", "networkx", "numpy", "automata", "frozendict",
              "greenery", "hypothesis", "lark", "pydot", "graphviz"}
    engine = pathlib.Path(fsa.__file__).parent

    offenders = []
    for module in engine.rglob("*.py"):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.split(".")[0] in banned:
                    offenders.append(f"{module.name}:{node.lineno} imports {name}")

    assert offenders == [], offenders


def test_the_import_guard_would_catch_a_real_violation():
    """The guard above only means something if it can fail.

    Feeds it a module that imports pygame in each of the two syntaxes, and one
    that merely mentions the word in a docstring, and checks it tells them
    apart.
    """
    import ast

    def imports_banned(source: str) -> bool:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.split(".")[0] == "pygame" for name in names):
                return True
        return False

    assert imports_banned("import pygame")
    assert imports_banned("import pygame.draw as draw")
    assert imports_banned("from pygame import Rect")
    assert imports_banned("def f():\n    import pygame\n")
    assert not imports_banned('"""This module never imports pygame."""')
    assert not imports_banned("# pygame is deliberately not used here")
