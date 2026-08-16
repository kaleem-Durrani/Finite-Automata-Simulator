"""A document holds an NFA, always.

Phase 12a put nondeterminism in the engine; this is the phase that let it reach
a file, a canvas and a command line. The design was chosen by changing the type
and reading what broke -- see :mod:`fsa.document` -- and the tests here pin the
three consequences that choice has to earn:

**Reads keep working; only the DFA-only operations move.** ``NFA`` mirrors
``DFA``'s surface, so a document's states, alphabet, accepting set and edges are
read exactly as before. What changed is that ``target`` is gone and every
algorithm defined on a transition *function* now goes through
:meth:`~fsa.document.Document.as_dfa`. There is therefore one place where "this
machine has a choice" can be discovered, and these tests check it says so
usefully -- naming the state and the symbol -- from the file reader, the CLI and
the editor alike.

**Nondeterminism is a fact, not a defect.** It is never in
:func:`fsa.analysis.defects`, no panel offers to fix it, and the phrase the
interface uses is "determinize it first" rather than anything about the machine
being wrong. This codebase has already shipped a Fix button next to a legal
design choice once (docs/LESSONS.md, the complete/trim cycle); the tests below
are what stops it happening again.

**Version 2 does not move.** A deterministic document writes exactly the bytes
it wrote before -- example files, a byte-for-byte round trip and the generated
README table all rest on that -- so a file only becomes a version 3 file when
it holds something version 2 cannot say. Adding a branch and removing it again
must leave the file it started as, byte for byte.
"""

import json

import pytest
from hypothesis import HealthCheck, given, settings

import fsa
from editor import EditorModel
from fsa import Document, Layout, serialize
from fsa.cli import NO, OK, USAGE, main
from fsa.errors import NondeterministicError
from fsa.nfa import EPSILON, NFA, from_dfa
from tests.strategies import dfas, nfa_documents

SETTINGS = settings(max_examples=50, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


# ---------------------------------------------------------------------------
# Fixtures by hand
# ---------------------------------------------------------------------------


def branching() -> Document:
    """Two ``a``-edges out of q0: the smallest thing version 2 cannot hold."""
    document = Document().add_symbol("a")
    document, q0 = document.add_state((0.0, 0.0))
    document, q1 = document.add_state((200.0, 0.0))
    document, q2 = document.add_state((200.0, 200.0))
    return (document.add_transition(q0, "a", q1)
                    .add_transition(q0, "a", q2)
                    .toggle_accept(q2)
                    .set_initial(q0))


def deterministic() -> Document:
    """The same three states with one target per pair."""
    document = Document().add_symbol("a")
    document, q0 = document.add_state((0.0, 0.0))
    document, q1 = document.add_state((200.0, 0.0))
    document, q2 = document.add_state((200.0, 200.0))
    return (document.add_transition(q0, "a", q1)
                    .add_transition(q1, "a", q2)
                    .toggle_accept(q2)
                    .set_initial(q0))


def written(tmp_path, document: Document, name: str = "machine.json") -> str:
    path = tmp_path / name
    serialize.save(document, str(path))
    return str(path)


def run_cli(*argv):
    """Run the CLI, returning (code, stdout, stderr)."""
    import io
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# The type
# ---------------------------------------------------------------------------


def test_a_document_holds_an_nfa():
    assert isinstance(Document().automaton, NFA)
    assert isinstance(branching().automaton, NFA)


def test_a_dfa_handed_over_is_converted_rather_than_refused():
    """Callers holding the output of an algorithm should not have to lift it."""
    machine = (fsa.DFA().with_states(["q0", "q1"])
               .with_transition("q0", "a", "q1")
               .with_accept("q1"))
    for document in (Document(machine), Document.of(machine),
                     Document().with_automaton(machine)):
        assert document.automaton == from_dfa(machine)
        assert document.as_dfa() == machine


def test_an_nfa_handed_over_is_stored_as_it_is():
    machine = (NFA().with_states(["q0", "q1"])
               .with_transition("q0", EPSILON, "q1"))
    assert Document(machine).automaton == machine
    assert Document.of(machine).automaton == machine


def test_of_still_counts_past_the_highest_numbered_state():
    document = Document.of(NFA().with_states(["q0", "q7"]))
    assert document.next_id == 8
    assert document.fresh_id() == "q8"


def test_the_empty_document_is_deterministic():
    """Vacuously, and it matters: a brand-new file must still be a version 2
    file, and the whole of the old behaviour hangs off this being true."""
    assert Document().is_deterministic
    assert Document().as_dfa() == fsa.DFA()


# ---------------------------------------------------------------------------
# The deterministic view
# ---------------------------------------------------------------------------


def test_as_dfa_names_the_state_and_symbol_that_refused():
    """"This machine is nondeterministic" is not actionable. "q0 has 2 targets
    on 'a'" is, because it says where to look."""
    with pytest.raises(NondeterministicError, match=r"q0 has 2 targets on 'a'"):
        branching().as_dfa()


def test_an_epsilon_move_alone_is_enough_to_refuse():
    document, _ = Document().add_state((0.0, 0.0), "q0")
    document, _ = document.add_state((100.0, 0.0), "q1")
    document = document.add_transition("q0", EPSILON, "q1")

    assert not document.is_deterministic
    with pytest.raises(NondeterministicError, match="epsilon"):
        document.as_dfa()


def test_a_partial_delta_is_still_deterministic():
    """A state with no move on a symbol has nothing to choose between. Reading
    partiality as nondeterminism would refuse most real files."""
    document = Document().add_symbol("a").add_symbol("b")
    document, q0 = document.add_state((0.0, 0.0))
    document = document.add_transition(q0, "a", q0)

    assert document.is_deterministic
    assert document.as_dfa().target(q0, "b") is None


@given(dfas())
@SETTINGS
def test_every_dfa_survives_the_round_trip_through_a_document(automaton: fsa.DFA):
    document = Document(automaton)
    assert document.is_deterministic
    assert document.as_dfa() == automaton


# ---------------------------------------------------------------------------
# Editing: the canvas accepts two edges on one symbol
# ---------------------------------------------------------------------------


def test_a_second_edge_on_one_symbol_is_kept():
    """It used to replace the first, silently. That single line is what made
    nondeterminism undrawable."""
    assert branching().automaton.targets("q0", "a") == {"q1", "q2"}


def test_removing_one_branch_leaves_the_others():
    document = branching().remove_transition("q0", "a", "q1")
    assert document.automaton.targets("q0", "a") == {"q2"}


def test_removing_the_whole_move_takes_every_branch():
    document = branching().remove_transition("q0", "a")
    assert document.automaton.targets("q0", "a") == frozenset()


def test_removing_a_branch_that_is_not_there_changes_nothing():
    before = branching()
    assert before.remove_transition("q0", "a", "q0").automaton == before.automaton


def test_a_bow_survives_while_any_arrow_between_the_two_states_remains():
    """The bow belongs to the edge, not to the symbol: it goes when the last
    arrow between the pair does, and not before."""
    document = branching().add_symbol("b")
    document = document.add_transition("q0", "b", "q1", arc=30.0)
    assert document.layout.arc_of("q0", "q1") == 30.0

    document = document.remove_transition("q0", "a", "q1")
    assert document.layout.arc_of("q0", "q1") == 30.0, "'b' still draws it"

    document = document.remove_transition("q0", "b", "q1")
    assert document.layout.arc_of("q0", "q1") == 0.0


def test_an_epsilon_edge_is_an_ordinary_edit():
    document = deterministic().add_transition("q0", EPSILON, "q2")
    assert ("q0", "q2") in document.automaton.grouped_transitions()
    assert EPSILON not in document.automaton.alphabet, "epsilon is not a letter"

    document = document.remove_transition("q0", EPSILON, "q2")
    assert document.is_deterministic


# ---------------------------------------------------------------------------
# Serialisation: the version follows the fact, and version 2 does not move
# ---------------------------------------------------------------------------


def test_a_deterministic_document_is_still_written_as_version_2():
    assert json.loads(serialize.dumps(deterministic()))["version"] == 2
    assert json.loads(serialize.dumps(Document()))["version"] == 2


def test_a_nondeterministic_document_is_written_as_version_3():
    assert json.loads(serialize.dumps(branching()))["version"] == 3


def test_a_branch_added_and_removed_leaves_the_file_it_started_as():
    """Byte for byte. The version is chosen by the fact, so an edit that
    changes the fact changes the format -- and undoing that edit has to change
    it back, or every round trip through the editor rewrites the file."""
    before = serialize.dumps(deterministic())
    branched = deterministic().add_transition("q0", "a", "q2")
    assert json.loads(serialize.dumps(branched))["version"] == 3

    after = serialize.dumps(branched.remove_transition("q0", "a", "q2"))
    assert after == before


def test_a_nondeterministic_document_round_trips():
    document = branching().add_transition("q1", EPSILON, "q2")
    restored = serialize.loads(serialize.dumps(document))
    assert restored == document
    assert restored.automaton.targets("q0", "a") == {"q1", "q2"}
    assert restored.automaton.targets("q1", EPSILON) == {"q2"}


@given(nfa_documents())
@SETTINGS
def test_every_document_round_trips(document: Document):
    once = serialize.dumps(document)
    assert serialize.loads(once) == document
    assert serialize.dumps(serialize.loads(once)) == once


@given(nfa_documents())
@SETTINGS
def test_the_version_written_is_the_determinism_of_the_machine(document: Document):
    written_version = json.loads(serialize.dumps(document))["version"]
    assert written_version == (2 if document.is_deterministic else 3)


def test_a_saved_nondeterministic_machine_survives_a_file(tmp_path):
    path = written(tmp_path, branching())
    assert serialize.load(path) == branching()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_auto_layout_places_a_nondeterministic_machine():
    """Placement reads ``grouped_transitions``, which both machines have, so a
    branch or an epsilon move is just another edge to walk."""
    automaton = branching().add_transition("q1", EPSILON, "q2").automaton
    layout = Layout.auto(automaton)
    assert set(layout.positions) == set(automaton.states)
    assert len(set(layout.positions.values())) == len(automaton.states)


# ---------------------------------------------------------------------------
# The editor never crashes because someone drew a second edge
# ---------------------------------------------------------------------------


def test_the_editor_survives_a_second_edge():
    editor = EditorModel(deterministic())
    assert editor.add_transition("q0", "a", "q2")

    assert not editor.is_deterministic
    assert editor.automaton.targets("q0", "a") == {"q1", "q2"}
    # The two readings the frame loop makes every frame. Neither may raise.
    assert editor.analysis() == (frozenset(), frozenset(), True)
    assert editor.defects() == ()


def test_nondeterminism_is_never_reported_as_a_defect():
    """The lesson of the complete/trim cycle: a legal design choice must not be
    labelled a fault. Nothing in the list may even mention it."""
    editor = EditorModel(branching())
    assert all("nondetermin" not in defect.message.lower()
               for defect in editor.defects())
    assert all(defect.kind != "nondeterministic" for defect in editor.defects())


def test_the_analysis_comes_back_when_the_branch_goes():
    """Degrading to "no answer" while the machine branches, not staying there.
    The cache is keyed on the automaton value, so this also proves it clears."""
    editor = EditorModel(deterministic())
    editor.add_transition("q0", "a", "q2")
    assert editor.analysis()[0] == frozenset()

    editor.remove_transition("q0", "a", "q2")
    assert editor.is_deterministic
    assert editor.defects(), "an incomplete delta is still reported"


def test_the_editor_removes_one_branch_at_a_time():
    editor = EditorModel(branching())
    editor.remove_transition("q0", "a", "q2")
    assert editor.automaton.targets("q0", "a") == {"q1"}

    assert editor.undo() is not None
    assert editor.automaton.targets("q0", "a") == {"q1", "q2"}


# ---------------------------------------------------------------------------
# The CLI: every DFA-only verb refuses, and says what to run instead
# ---------------------------------------------------------------------------


#: The verbs that need a transition function, and the arguments each takes.
#: A phase exit criterion: every one of them has to refuse a nondeterministic
#: document with an error naming ``fsa determinize``, and none may pretend to
#: have an answer.
DFA_ONLY = [
    ("test", lambda path: ("test", path, "a")),
    ("run", lambda path: ("run", path, "a")),
    ("check", lambda path: ("check", path)),
    ("sample", lambda path: ("sample", path)),
    ("minimize", lambda path: ("minimize", path)),
    ("complete", lambda path: ("complete", path)),
    ("equiv", lambda path: ("equiv", path, path)),
    ("export", lambda path: ("export", path)),
]


@pytest.mark.parametrize("verb, argv", DFA_ONLY, ids=[v for v, _ in DFA_ONLY])
def test_a_dfa_only_verb_refuses_and_names_determinize(verb, argv, tmp_path):
    path = written(tmp_path, branching())
    code, out, err = run_cli(*argv(path))

    assert code == USAGE, f"{verb} must not answer a question it cannot answer"
    assert "determinize" in err, f"{verb} must say what to run instead"
    assert "q0" in err and "'a'" in err, f"{verb} must say what made it refuse"
    assert out == "", f"{verb} printed an answer as well as refusing"


@pytest.mark.parametrize("_verb, argv", DFA_ONLY, ids=[v for v, _ in DFA_ONLY])
def test_a_dfa_only_verb_is_untouched_by_a_deterministic_document(_verb, argv,
                                                                  tmp_path):
    """The other half of the criterion: the guard must not refuse what it used
    to accept."""
    path = written(tmp_path, deterministic())
    code, _out, err = run_cli(*argv(path))
    assert code in (OK, NO), err
    assert "determinize" not in err


def test_determinize_accepts_what_the_others_refuse(tmp_path):
    path = written(tmp_path, branching())
    code, out, err = run_cli("determinize", path, "-o", str(tmp_path / "out.json"))
    assert code == OK, err

    assert "3 states ->" in out
    rebuilt = serialize.load(str(tmp_path / "out.json"))
    assert rebuilt.is_deterministic
    for word in ("", "a", "aa"):
        assert (fsa.accepts(rebuilt.as_dfa(), word)
                == fsa.nfa.accepts(branching().automaton, word))


def test_show_prints_the_choice_rather_than_refusing(tmp_path):
    """``show`` is the one reading verb that stays on the NFA: when the answer
    is "more targets than you expected", showing them is the whole point."""
    path = written(tmp_path, branching().add_transition("q1", EPSILON, "q2"))
    code, out, _err = run_cli("show", path)

    assert code == OK
    assert "{q1,q2}" in out, "the branch is drawn as the subset it will become"
    assert "ε" in out, "the epsilon column appears only when it is needed"
    assert "a choice" in out


def test_show_on_a_deterministic_document_says_nothing_extra(tmp_path):
    path = written(tmp_path, deterministic())
    _code, out, _err = run_cli("show", path)
    assert "ε" not in out and "a choice" not in out
    assert out.rstrip().endswith("-> initial, * accepting, - undefined")
