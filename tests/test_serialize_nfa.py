"""Version 3 of the document format: a file that holds a nondeterministic machine.

Two halves, and the first one matters more.

**Nothing about version 2 may move.** A file that opens today has to open
tomorrow, and a DFA has to serialise to the same bytes it serialised to before
this format grew a second version -- there is an example file in the repository,
a README table generated from it, and a byte-for-byte test in
``test_document.py`` all resting on that. So the compatibility section here
pins the actual bytes as a literal rather than only re-round-tripping them: a
round trip agrees with itself even when both halves have changed together.

**The second half is what version 3 adds**, which is exactly two things: a move
may have several targets, and a move may read nothing. Both are the case an
implementation gets wrong by *dropping*, so the epsilon move is tested against
an empty alphabet -- an epsilon key must never be checked against the alphabet
it was never in -- and the several targets are tested by writing the same
``(source, symbol)`` twice in a hand-written file and demanding both survive.

Imports no pygame and touches no display.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fsa import DFA, Document, Layout, serialize
from fsa.errors import NondeterministicError
from fsa.nfa import EPSILON, NFA, from_dfa
from fsa.serialize import DocumentFormatError
from tests.strategies import dfas, nfas

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

#: Building a machine one transition at a time is not fast, and every example
#: here builds a whole file around one; the same budget the other engine
#: property tests use.
SETTINGS = settings(max_examples=50, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


# ---------------------------------------------------------------------------
# Fixtures by hand
# ---------------------------------------------------------------------------


def branching() -> NFA:
    """Two targets on one symbol, plus an epsilon move and a label.

    Everything version 3 has to carry that version 2 does not, in one machine
    small enough to read.
    """
    return (NFA()
            .with_states(["q0", "q1", "q2"])
            .with_transition("q0", "a", "q1")
            .with_transition("q0", "a", "q2")
            .with_transition("q1", EPSILON, "q2")
            .with_accept("q2")
            .with_label("q0", "start"))


def drawn(automaton: NFA) -> Layout:
    """Somewhere to put every state, so a round trip has coordinates to carry."""
    return Layout({state: (60.0 * index, 40.0)
                   for index, state in enumerate(sorted(automaton.states))})


@st.composite
def nfa_files(draw: st.DrawFn) -> Tuple[NFA, Layout, int]:
    """The three values a version 3 file holds, ready to be written.

    The machine comes from the house strategy; the three variants named here
    are the shapes a *format* loses and a simulator does not care about, so
    they are drawn from explicitly rather than left to chance. Each of them is
    something absent -- no start state, no alphabet, no coordinates -- and an
    absent thing reads back as an absent thing, which is exactly why nobody
    notices it has gone until a real file does it.

    Arc offsets are drawn from a handful of values rather than from
    ``st.floats``: :class:`~fsa.layout.Layout` drops a zero offset but rounds
    the rest, so an offset under half a thousandth of a pixel survives
    construction and vanishes on reload. That is a property of ``Layout``, not
    of this format, and it belongs in that module's tests rather than being
    rediscovered here.
    """
    automaton = draw(st.one_of(
        nfas(),
        # No start state -- "no language defined yet", which a file must not
        # quietly replace with a guess.
        nfas(min_states=0, with_initial=False),
        # An empty alphabet, so the only moves in the machine are epsilon ones
        # and a reader that filtered them by the alphabet would lose all of it.
        nfas(alphabet=()),
    ))
    ids = sorted(automaton.states)

    for state in ids:
        if draw(st.booleans()):
            automaton = automaton.with_label(state, draw(st.text(max_size=4)))

    coordinate = st.floats(min_value=-900, max_value=900,
                           allow_nan=False, allow_infinity=False)
    positions = {}
    for state in ids:
        # Not every state: a partial layout is what a hand-written file has,
        # and it has to survive the trip as the partial layout it is.
        if draw(st.booleans()):
            positions[state] = (draw(coordinate), draw(coordinate))

    arcs = {}
    for source in ids:
        for target in ids:
            offset = draw(st.sampled_from([0.0, 0.0, 0.0, 28.5, -40.0, 12.25]))
            if offset:
                arcs[(source, target)] = offset

    # At least one past the highest id in the machine, which is the floor the
    # loader raises a smaller number to; see the test below that pins it.
    next_id = len(ids) + draw(st.integers(min_value=0, max_value=3))
    return automaton, Layout(positions, arcs), next_id


# ---------------------------------------------------------------------------
# Compatibility: version 2 must not have moved
# ---------------------------------------------------------------------------


#: What ``dumps`` produced for this document before version 3 existed, pasted
#: in whole. A round trip only proves the writer and the reader agree with each
#: other; this proves the writer still agrees with what is already on disk.
GOLDEN_V2 = """{
  "version": 2,
  "automaton": {
    "states": [
      "q0",
      "q1"
    ],
    "alphabet": [
      "a",
      "b"
    ],
    "initial": "q0",
    "accept": [
      "q1"
    ],
    "transitions": [
      [
        "q0",
        "a",
        "q1"
      ],
      [
        "q1",
        "b",
        "q1"
      ]
    ],
    "labels": {
      "q0": "start"
    }
  },
  "layout": {
    "positions": {
      "q0": [
        10.0,
        20.0
      ],
      "q1": [
        90.0,
        20.0
      ]
    },
    "arcs": [
      [
        "q0",
        "q1",
        28.5
      ]
    ]
  },
  "next_id": 2
}
"""


def golden_document() -> Document:
    """The document :data:`GOLDEN_V2` is the serialisation of."""
    document = Document()
    document, _ = document.add_state((10.0, 20.0), "q0")
    document, _ = document.add_state((90.0, 20.0), "q1")
    document = document.add_transition("q0", "a", "q1", arc=28.5)
    document = document.add_transition("q1", "b", "q1")
    document = document.toggle_accept("q1")
    return Document(document.automaton.with_label("q0", "start"),
                    document.layout, document.next_id)


def test_a_dfa_still_serialises_to_exactly_the_bytes_it_used_to():
    assert serialize.dumps(golden_document()) == GOLDEN_V2


def test_the_version_a_dfa_is_written_as_is_still_2():
    assert serialize.VERSION == 2
    assert json.loads(serialize.dumps(Document()))["version"] == 2


def test_the_example_file_round_trips_byte_for_byte():
    """The checked-in example is in the pre-versioning format, so this is also
    the one exercise :func:`serialize.read_legacy` still gets."""
    document = serialize.load(str(EXAMPLES / "simple_binary.json"))
    once = serialize.dumps(document)
    assert serialize.dumps(serialize.loads(once)) == once
    assert json.loads(once)["version"] == 2


def test_the_example_file_also_opens_as_an_nfa():
    """Every DFA is an NFA, so the version 3 door opens every older file too."""
    automaton, layout, next_id = serialize.load_nfa(
        str(EXAMPLES / "simple_binary.json"))
    assert automaton == serialize.load(str(EXAMPLES / "simple_binary.json")).automaton
    assert set(layout.positions) == {"q0", "q1", "q2"}
    assert next_id == 3


def test_an_unknown_version_is_still_refused():
    with pytest.raises(DocumentFormatError, match="version"):
        serialize.loads('{"version": 99}')
    with pytest.raises(DocumentFormatError, match="version"):
        serialize.loads_nfa('{"version": 99}')


def test_the_refusal_names_both_versions_it_can_read():
    with pytest.raises(DocumentFormatError, match="2 and 3"):
        serialize.loads('{"version": 4}')


# ---------------------------------------------------------------------------
# The shape of a version 3 file
# ---------------------------------------------------------------------------


def test_an_nfa_is_written_as_version_3():
    body = json.loads(serialize.dumps_nfa(branching(), drawn(branching()), 3))
    assert body["version"] == serialize.NFA_VERSION == 3


def test_a_deterministic_nfa_is_still_written_as_version_3():
    """The version says which type wrote the file, not what the machine looks
    like. Otherwise adding one epsilon move would silently change the format a
    file is stored in, and removing it would change it back."""
    deterministic = (NFA().with_states(["q0", "q1"])
                     .with_transition("q0", "a", "q1"))
    assert deterministic.is_deterministic()
    text = serialize.dumps_nfa(deterministic, drawn(deterministic), 2)
    assert json.loads(text)["version"] == 3


def test_epsilon_is_written_as_json_null():
    """Not "e", "" or the character epsilon -- every one of those is a legal
    alphabet symbol here, so it would give one entry two meanings."""
    text = serialize.dumps_nfa(branching(), drawn(branching()), 3)
    rows = json.loads(text)["automaton"]["transitions"]
    assert ["q1", None, "q2"] in rows
    assert "ε" not in text


def test_two_targets_on_one_symbol_are_two_entries():
    rows = json.loads(
        serialize.dumps_nfa(branching(), drawn(branching()), 3)
    )["automaton"]["transitions"]
    assert ["q0", "a", "q1"] in rows
    assert ["q0", "a", "q2"] in rows


def test_the_file_is_sorted_with_epsilon_first():
    """Sorted so the file diffs like source; epsilon first because that is the
    order :meth:`NFA.sorted_transitions` and the UI both use."""
    automaton = (NFA().with_states(["q0", "q1"])
                 .with_transition("q0", "b", "q1")
                 .with_transition("q0", "a", "q1")
                 .with_transition("q0", EPSILON, "q1"))
    body = json.loads(serialize.dumps_nfa(automaton, drawn(automaton), 2))["automaton"]
    assert body["transitions"] == [["q0", None, "q1"],
                                   ["q0", "a", "q1"],
                                   ["q0", "b", "q1"]]
    assert body["states"] == sorted(body["states"])
    assert body["alphabet"] == sorted(body["alphabet"])


def test_the_envelope_has_the_same_keys_as_version_2():
    """One format with two versions, not two formats."""
    document = golden_document()
    two = json.loads(serialize.dumps(document))
    three = json.loads(serialize.dumps_nfa(
        document.automaton, document.layout, document.next_id))
    assert list(two) == list(three)
    assert list(two["automaton"]) == list(three["automaton"])
    assert list(two["layout"]) == list(three["layout"])


def test_the_layout_survives_the_trip():
    automaton = branching()
    layout = Layout({"q0": (10.0, 20.5), "q1": (90.0, 20.0), "q2": (170.0, 60.0)},
                    {("q0", "q1"): 28.5})
    _, restored, _ = serialize.loads_nfa(serialize.dumps_nfa(automaton, layout, 3))
    assert restored == layout


# ---------------------------------------------------------------------------
# Round-tripping
# ---------------------------------------------------------------------------


@given(nfa_files())
@SETTINGS
def test_every_nfa_file_round_trips(written: Tuple[NFA, Layout, int]):
    automaton, layout, next_id = written
    assert serialize.loads_nfa(serialize.dumps_nfa(automaton, layout, next_id)) == written


@given(nfa_files())
@SETTINGS
def test_writing_an_nfa_file_is_byte_stable(written: Tuple[NFA, Layout, int]):
    """Read what was written and write it again: the same bytes, or a saved
    file gets a spurious diff every time it is opened."""
    once = serialize.dumps_nfa(*written)
    reread = serialize.loads_nfa(once)
    assert serialize.dumps_nfa(*reread) == once


def test_a_machine_with_no_initial_state_round_trips():
    """``None`` means "no language defined yet", and a file that lost it would
    silently pick a start state for the user."""
    automaton = branching().with_initial(None)
    restored, _, _ = serialize.loads_nfa(
        serialize.dumps_nfa(automaton, drawn(automaton), 3))
    assert restored.initial is None
    assert restored == automaton


def test_a_machine_with_an_empty_alphabet_round_trips():
    automaton = (NFA().with_states(["q0", "q1"])
                 .with_transition("q0", EPSILON, "q1")
                 .with_accept("q1"))
    assert automaton.alphabet == frozenset()
    restored, _, _ = serialize.loads_nfa(
        serialize.dumps_nfa(automaton, drawn(automaton), 2))
    assert restored == automaton
    assert restored.targets("q0", EPSILON) == frozenset({"q1"})


def test_an_empty_machine_round_trips():
    assert serialize.loads_nfa(serialize.dumps_nfa(NFA(), Layout(), 0)) == (
        NFA(), Layout(), 0)


def test_an_epsilon_cycle_round_trips():
    """The shape Thompson's construction produces by the dozen, and the one a
    reader that walks the moves as it reads them would hang on."""
    automaton = (NFA().with_states(["q0", "q1"])
                 .with_transition("q0", EPSILON, "q1")
                 .with_transition("q1", EPSILON, "q0"))
    restored, _, _ = serialize.loads_nfa(
        serialize.dumps_nfa(automaton, drawn(automaton), 2))
    assert restored == automaton
    assert restored.epsilon_closure(["q0"]) == frozenset({"q0", "q1"})


def test_labels_survive():
    restored, _, _ = serialize.loads_nfa(
        serialize.dumps_nfa(branching(), drawn(branching()), 3))
    assert restored.label_of("q0") == "start"
    assert restored.label_of("q1") == "q1"


# ---------------------------------------------------------------------------
# Reading a file somebody typed
# ---------------------------------------------------------------------------


def hand_written(**automaton: object) -> str:
    """A version 3 file with only the keys the caller cared about."""
    body: Dict[str, object] = {"states": [], "alphabet": [], "transitions": [],
                               "initial": None, "accept": []}
    body.update(automaton)
    return json.dumps({"version": 3, "automaton": body})


def test_a_repeated_pair_in_a_hand_written_file_is_nondeterminism():
    """The point of the shape. Reading with assignment rather than union would
    keep the last entry and quietly delete the branch."""
    automaton, _, _ = serialize.loads_nfa(hand_written(
        states=["q0", "q1", "q2"], alphabet=["a"],
        transitions=[["q0", "a", "q1"], ["q0", "a", "q2"]]))
    assert automaton.targets("q0", "a") == frozenset({"q1", "q2"})


def test_an_epsilon_entry_is_not_checked_against_the_alphabet():
    """The mistake this format invites: an epsilon key is not a symbol, so
    filtering it by the alphabet drops every epsilon move in the file."""
    automaton, _, _ = serialize.loads_nfa(hand_written(
        states=["q0", "q1"], alphabet=[],
        transitions=[["q0", None, "q1"]]))
    assert automaton.targets("q0", EPSILON) == frozenset({"q1"})
    assert automaton.alphabet == frozenset()


def test_an_entry_on_a_symbol_outside_the_alphabet_is_dropped():
    """As in version 2: junk, not a corrupt file."""
    automaton, _, _ = serialize.loads_nfa(hand_written(
        states=["q0", "q1"], alphabet=["a"],
        transitions=[["q0", "a", "q1"], ["q0", "z", "q1"]]))
    assert automaton.sorted_transitions() == (("q0", "a", ("q1",)),)


def test_entries_naming_unknown_states_are_dropped():
    automaton, _, _ = serialize.loads_nfa(hand_written(
        states=["q0"], alphabet=["a"],
        transitions=[["q0", "a", "ghost"], ["ghost", "a", "q0"]]))
    assert automaton.transitions == {}


def test_an_unknown_initial_state_becomes_none():
    automaton, _, _ = serialize.loads_nfa(hand_written(
        states=["q0"], initial="ghost"))
    assert automaton.initial is None


def test_a_malformed_entry_is_refused():
    with pytest.raises(DocumentFormatError, match="transition"):
        serialize.loads_nfa(hand_written(states=["q0"], transitions=[["q0", "a"]]))


def test_a_missing_automaton_is_refused():
    with pytest.raises(DocumentFormatError, match="automaton"):
        serialize.loads_nfa('{"version": 3}')


def test_malformed_json_is_refused():
    with pytest.raises(DocumentFormatError, match="JSON"):
        serialize.loads_nfa("{not json")


def test_a_hand_written_file_need_not_carry_a_next_id():
    """And a wrong one is raised to something safe, rather than handing out an
    id a state in the file already has."""
    _, _, fresh = serialize.loads_nfa(hand_written(states=["q0", "q7"]))
    assert fresh == 8
    _, _, declared = serialize.loads_nfa(json.dumps({
        "version": 3, "next_id": 0,
        "automaton": {"states": ["q0", "q7"]}}))
    assert declared == 8


def test_a_declared_next_id_above_the_floor_is_kept():
    _, _, fresh = serialize.loads_nfa(json.dumps({
        "version": 3, "next_id": 40,
        "automaton": {"states": ["q0"]}}))
    assert fresh == 40


def test_a_position_for_a_state_that_is_not_there_is_dropped():
    _, layout, _ = serialize.loads_nfa(json.dumps({
        "version": 3,
        "automaton": {"states": ["q0"]},
        "layout": {"positions": {"q0": [1, 2], "ghost": [3, 4]},
                   "arcs": [["q0", "ghost", 10]]}}))
    assert layout.positions == {"q0": (1.0, 2.0)}
    assert layout.arc_offsets == {}


def test_a_file_with_no_coordinates_is_left_without_them():
    """Unlike the document reader, which lays out what it opens. This one hands
    back the file's three values as written, so the round trip is exact;
    placing a state belongs to whatever holds the machine and the drawing
    together, which is Phase 12b's job."""
    _, layout, _ = serialize.loads_nfa(hand_written(states=["q0", "q1"]))
    assert layout == Layout()


# ---------------------------------------------------------------------------
# The two doors: what each version opens as
# ---------------------------------------------------------------------------


def test_a_deterministic_version_3_file_opens_as_a_document():
    automaton = (NFA().with_states(["q0", "q1"])
                 .with_transition("q0", "a", "q1")
                 .with_accept("q1"))
    document = serialize.loads(
        serialize.dumps_nfa(automaton, drawn(automaton), 2))
    assert document.as_dfa().target("q0", "a") == "q1"
    assert document.automaton.accept == frozenset({"q1"})


def test_a_nondeterministic_version_3_file_opens_as_a_document_too():
    """It used to be refused, because a document could only hold a DFA. Now the
    machine in the file is the machine in the document, which is the only
    reading that does not either lose it or quietly show a different one."""
    automaton = branching()
    document = serialize.loads(serialize.dumps_nfa(automaton, drawn(automaton), 3))
    assert document.automaton == automaton
    assert not document.is_deterministic


def test_the_refusal_moves_to_as_dfa_and_still_names_the_state():
    """"This machine is nondeterministic" is not actionable; "q0 has 2 targets
    on 'a'" is. The file opens; asking it for a DFA is what fails."""
    automaton = (NFA().with_states(["q0", "q1", "q2"])
                 .with_transition("q0", "a", "q1")
                 .with_transition("q0", "a", "q2"))
    document = serialize.loads(serialize.dumps_nfa(automaton, drawn(automaton), 3))
    with pytest.raises(NondeterministicError, match=r"q0 has 2 targets on 'a'"):
        document.as_dfa()


def test_an_epsilon_move_alone_is_enough_to_refuse_a_dfa_view():
    automaton = (NFA().with_states(["q0", "q1"])
                 .with_transition("q0", EPSILON, "q1"))
    document = serialize.loads(serialize.dumps_nfa(automaton, drawn(automaton), 2))
    with pytest.raises(NondeterministicError, match="epsilon"):
        document.as_dfa()


def test_a_version_2_file_opens_as_an_nfa():
    document = golden_document()
    automaton, layout, next_id = serialize.loads_nfa(serialize.dumps(document))
    assert automaton == document.automaton
    assert layout == document.layout
    assert next_id == document.next_id


@given(dfas())
@SETTINGS
def test_a_dfa_written_as_version_3_comes_back_the_same_dfa(automaton: DFA):
    """Both doors, on every deterministic machine hypothesis can build: out as
    version 3, back in as a document, and equal to what went out. Partial delta
    included -- a state with no move on a symbol is deterministic, and a reader
    that treated a missing entry as a defect would refuse most real files."""
    text = serialize.dumps_nfa(from_dfa(automaton), Layout.auto(automaton),
                               len(automaton.states))
    assert serialize.loads(text).as_dfa() == automaton
    assert serialize.loads_nfa(text)[0] == from_dfa(automaton)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def test_a_machine_survives_a_file(tmp_path: Path):
    path = str(tmp_path / "machine.json")
    serialize.save_nfa(branching(), drawn(branching()), 3, path)
    assert serialize.load_nfa(path) == (branching(), drawn(branching()), 3)


def test_saving_reports_a_failure_rather_than_raising(tmp_path: Path):
    ok, error = serialize.save_nfa_or_error(NFA(), Layout(), 0,
                                            str(tmp_path / "fine.json"))
    assert (ok, error) == (True, "")

    # A directory is never a writable file, on any platform.
    ok, error = serialize.save_nfa_or_error(NFA(), Layout(), 0, str(tmp_path))
    assert not ok and error


def test_loading_reports_a_failure_rather_than_raising(tmp_path: Path):
    written: Optional[Tuple[NFA, Layout, int]]
    written, error = serialize.load_nfa_or_error(str(tmp_path / "nowhere.json"))
    assert written is None and error

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    written, error = serialize.load_nfa_or_error(str(bad))
    assert written is None and "JSON" in error

    good = tmp_path / "good.json"
    serialize.save_nfa(branching(), drawn(branching()), 3, str(good))
    written, error = serialize.load_nfa_or_error(str(good))
    assert error == "" and written is not None and written[0] == branching()


# ---------------------------------------------------------------------------
# Determinism across processes
# ---------------------------------------------------------------------------


_ACROSS_PROCESSES = """
import sys
sys.path.insert(0, ".")
from fsa.layout import Layout
from fsa.nfa import EPSILON, NFA
from fsa.serialize import dumps_nfa

a = NFA().with_states(["q%d" % i for i in range(8)])
for i in range(7):
    a = a.with_transition("q%d" % i, "a", "q%d" % (i + 1))
    a = a.with_transition("q%d" % i, "a", "q0")
    a = a.with_transition("q%d" % i, EPSILON, "q%d" % (7 - i))
sys.stdout.write(dumps_nfa(a, Layout.grid(sorted(a.states)), 8))
"""


def test_a_saved_file_is_the_same_bytes_in_every_process():
    """Python randomises string hashing per process, so a frozenset of state
    ids iterates differently every run. A file whose lines shuffled between
    saves could not be diffed, which is most of what this format is for --
    and calling twice in one process cannot catch it, because the order is
    fixed for the life of the interpreter."""
    src = Path(__file__).resolve().parent.parent / "src"
    outputs: List[str] = []
    for seed in ("0", "1", "1000"):
        result = subprocess.run(
            [sys.executable, "-c", _ACROSS_PROCESSES],
            cwd=src, capture_output=True, text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1] == outputs[2]
    assert '"version": 3' in outputs[0]
