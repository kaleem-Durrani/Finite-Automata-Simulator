"""Runs the DFA conformance specification in tests/conformance/cases.json.

The cases are written from the theory -- each expected verdict and path was
computed by hand from the transition function -- so this suite says what the
simulator *should* do, independent of what it currently does. That is the point:
it survives the engine rewrite unchanged, because it is stated in terms of
(automaton, word, verdict, path) and not in terms of any class.

Cases the current implementation gets wrong carry a "known_wrong" note and run
as strict xfail. Fixing the implementation turns them green; CI then fails until
the marker is removed, so the burn-down cannot silently drift.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

import fsa
from core.dfa import DFA

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


def build(spec: Dict[str, Any]) -> DFA:
    """Construct a DFA from a spec entry.

    Goes through from_dict rather than add_state/add_transition so that the
    automaton under test is built the same way a loaded file would be.
    """
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

    def state_type(state_id: str) -> str:
        if state_id in spec["accept"]:
            return "accept"
        if state_id in spec["dead_end"]:
            return "dead_end"
        return "normal"

    transitions: Dict[str, Dict[str, str]] = {state_id: {} for state_id in state_ids}
    for source, symbol, target in spec["transitions"]:
        transitions[source][symbol] = target

    dfa = DFA()
    dfa.from_dict(
        {
            "states": {
                state_id: {"position": [0, 0], "state_type": state_type(state_id)}
                for state_id in state_ids
            },
            "transitions": transitions,
            "alphabet": list(spec["alphabet"]),
            "initial_state": spec["initial"],
            "accept_states": list(spec["accept"]),
            "dead_end_states": list(spec["dead_end"]),
            "next_state_id": len(state_ids),
        }
    )
    return dfa


def _cases() -> List[Tuple[str, Dict[str, Any]]]:
    return [(case["id"], case) for case in SPEC["cases"]]


def _parametrize():
    params = []
    for case_id, case in _cases():
        marks = []
        if case.get("known_wrong"):
            marks.append(pytest.mark.xfail(strict=True, reason=case["known_wrong"]))
        params.append(pytest.param(case, id=case_id, marks=marks))
    return params


def test_spec_is_well_formed():
    """Guard the spec itself: unknown verdicts or automata are typos."""
    seen_ids = set()
    for case in SPEC["cases"]:
        assert case["id"] not in seen_ids, f"duplicate case id {case['id']}"
        seen_ids.add(case["id"])
        assert case["verdict"] in KNOWN_VERDICTS, f"{case['id']}: unknown verdict"
        assert case["automaton"] in SPEC["automata"], f"{case['id']}: unknown automaton"


def test_spec_paths_are_self_consistent():
    """A run that consumes the whole word visits len(word) + 1 states."""
    for case in SPEC["cases"]:
        if case["verdict"] in ("accept", "reject_non_accepting"):
            assert len(case["path"]) == len(case["word"]) + 1, (
                f"{case['id']}: a completed run must visit one more state than "
                f"it reads symbols"
            )
        elif case["verdict"] == "no_initial_state":
            assert case["path"] == []


@pytest.mark.parametrize("case", _parametrize())
def test_conformance(case: Dict[str, Any]):
    dfa = build(SPEC["automata"][case["automaton"]])
    accepted, path = dfa.process_string(case["word"])

    expected_accepted = case["verdict"] == ACCEPTING_VERDICT
    assert accepted is expected_accepted, (
        f"{case['id']}: expected {case['verdict']}, "
        f"got {'accept' if accepted else 'reject'}"
    )
    assert path == case["path"], f"{case['id']}: wrong run"


def test_known_wrong_cases_are_all_dead_end_related():
    """
    Every currently-failing case must have a recorded reason.

    All of them trace to one defect in the legacy model: StateType.DEAD_END
    short-circuits acceptance, so a user-set flag overrides the transition
    function. The engine below has no such flag and passes all of them.
    """
    failing = [c for c in SPEC["cases"] if c.get("known_wrong")]
    assert failing, "expected the spec to record the known DEAD_END divergence"
    for case in failing:
        assert "DEAD_END" in case["known_wrong"] or "flag" in case["known_wrong"]


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
