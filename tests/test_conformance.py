"""Runs the DFA conformance specification in tests/conformance/cases.json.

Every expected verdict and path was computed by hand from the transition
function, so this suite says what a simulator *should* do rather than recording
what one happens to do. It was written against the old model, survived the
engine being built, and survived the old model being deleted -- unchanged,
because it is stated in terms of (automaton, word, verdict, path) and mentions
no class at all.

Four cases carry a ``known_wrong`` note. Those are the ones the previous model
failed, all for the same reason: a hand-set "dead end" flag overrode the
transition function, so the tool computed a different language than the diagram
showed. They ran as strict xfail until the engine landed. They are kept as
history, and asserted to pass, so a regression to that behaviour is caught.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

import fsa

CASES_PATH = Path(__file__).parent / "conformance" / "cases.json"
SPEC: Dict[str, Any] = json.loads(CASES_PATH.read_text(encoding="utf-8"))

ACCEPTING_VERDICT = "accept"
KNOWN_VERDICTS = {
    "accept",
    "reject_non_accepting",
    "reject_no_transition",
    "reject_symbol_not_in_alphabet",
    "no_initial_state",
}


# ---------------------------------------------------------------------------
# The same specification, run against the engine
# ---------------------------------------------------------------------------


def build_engine(spec: Dict[str, Any]) -> fsa.DFA:
    """Construct an fsa.DFA from a spec entry.

    Note what is absent: there is no dead-end flag to set. Trap states are
    derived from the transition function, so the spec's ``dead_end`` list is
    deliberately ignored -- and the cases the legacy model fails are exactly
    the ones where honouring that flag gave the wrong answer.
    """
    automaton = fsa.DFA(alphabet=frozenset(spec["alphabet"]))

    state_ids: List[str] = []
    for source, _symbol, target in spec["transitions"]:
        for state_id in (source, target):
            if state_id not in state_ids:
                state_ids.append(state_id)
    for state_id in list(spec["accept"]) + list(spec["dead_end"]):
        if state_id not in state_ids:
            state_ids.append(state_id)
    if spec["initial"] and spec["initial"] not in state_ids:
        state_ids.insert(0, spec["initial"])

    automaton = automaton.with_states(state_ids)
    for source, symbol, target in spec["transitions"]:
        automaton = automaton.with_transition(source, symbol, target)
    for state_id in spec["accept"]:
        automaton = automaton.with_accept(state_id)
    return automaton.with_initial(spec["initial"])


@pytest.mark.parametrize(
    "case", [pytest.param(case, id=case["id"]) for case in SPEC["cases"]])
def test_engine_conformance(case: Dict[str, Any]):
    """The engine must satisfy every case, including the four the old model fails.

    No xfail markers here. This is the whole point of the rewrite.
    """
    automaton = build_engine(SPEC["automata"][case["automaton"]])
    result = fsa.run(automaton, case["word"])

    assert result.verdict.value == case["verdict"], (
        f"{case['id']}: {result.explain()}")
    assert list(result.path) == case["path"], f"{case['id']}: wrong run"
    assert result.accepted is (case["verdict"] == ACCEPTING_VERDICT)


@pytest.mark.parametrize(
    "case", [pytest.param(case, id=case["id"]) for case in SPEC["cases"]])
def test_engine_path_invariant(case: Dict[str, Any]):
    """len(path) == len(steps) + 1 whenever there is a start state."""
    automaton = build_engine(SPEC["automata"][case["automaton"]])
    result = fsa.run(automaton, case["word"])

    if result.start is None:
        assert result.path == ()
    else:
        assert len(result.path) == len(result.steps) + 1

    assert 0 <= result.stopped_at <= len(case["word"])
    assert result.consumed + result.remaining == case["word"]


@pytest.mark.parametrize(
    "case", [pytest.param(case, id=case["id"]) for case in SPEC["cases"]])
def test_every_case_explains_itself(case: Dict[str, Any]):
    """Every outcome produces a non-empty, specific sentence."""
    automaton = build_engine(SPEC["automata"][case["automaton"]])
    result = fsa.run(automaton, case["word"])

    explanation = result.explain()
    assert explanation and explanation[0].isprintable()
    assert "rejected" in explanation or "accepted" in explanation or "could not" in explanation

    if result.offending_symbol is not None:
        assert f"'{result.offending_symbol}'" in explanation
        assert str(result.stopped_at) in explanation


def test_the_four_legacy_failures_pass_on_the_engine():
    """Name the burn-down explicitly.

    These four cases are marked xfail against the legacy model. Each one is a
    place where a user-set flag overruled the transition function. The engine
    has no flag, so it computes the language the diagram actually describes.
    """
    failing = [case for case in SPEC["cases"] if case.get("known_wrong")]
    assert len(failing) == 4

    for case in failing:
        automaton = build_engine(SPEC["automata"][case["automaton"]])
        result = fsa.run(automaton, case["word"])
        assert result.verdict.value == case["verdict"], case["id"]
        assert list(result.path) == case["path"], case["id"]
