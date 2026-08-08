"""The command line interface.

Every test asserts the exit code, because that is the part a shell script or an
autograder actually reads:

    0   yes -- accepted, or no defects
    1   no  -- rejected, or defects found
    2   could not run

Nothing here touches a display. If any of it needs pygame, the boundary broke.
"""

import io
import json

import pytest

import fsa
from fsa import Document, serialize
from fsa.cli import NO, OK, USAGE, main


def run(*argv):
    """Run the CLI, returning (code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def machine(tmp_path):
    """0*1+ over {0,1}, with a trap. Written to a file, path returned."""
    document = Document().add_symbol("0").add_symbol("1")
    document, q0 = document.add_state((0.0, 0.0))
    document, q1 = document.add_state((200.0, 0.0))
    document, q2 = document.add_state((100.0, 160.0))
    document = (document.add_transition(q0, "0", q0)
                        .add_transition(q0, "1", q1)
                        .add_transition(q1, "0", q2)
                        .add_transition(q1, "1", q1)
                        .add_transition(q2, "0", q2)
                        .add_transition(q2, "1", q2))
    document = document.toggle_accept(q1).set_initial(q0)

    path = tmp_path / "machine.json"
    serialize.save(document, str(path))
    return str(path)


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------


def test_an_accepted_word_exits_zero(machine):
    code, out, _ = run("test", machine, "011")
    assert code == OK
    assert "accepted" in out


def test_a_rejected_word_exits_one(machine):
    code, out, _ = run("test", machine, "101")
    assert code == NO
    assert "rejected" in out


def test_a_symbol_outside_the_alphabet_says_which_and_where(machine):
    code, out, _ = run("test", machine, "1x2")
    assert code == NO
    assert "'x'" in out
    assert "position 1" in out
    assert "not in the alphabet" in out


def test_quiet_prints_the_verdict_alone(machine):
    code, out, _ = run("test", machine, "101", "-q")
    assert code == NO
    assert out.strip() == "reject_non_accepting"


def test_the_empty_word_can_be_tested(machine):
    code, out, _ = run("test", machine, "")
    assert code == NO
    assert "empty string" in out


def test_a_missing_file_exits_two(tmp_path):
    code, _, err = run("test", str(tmp_path / "nope.json"), "1")
    assert code == USAGE
    assert "nope.json" in err


def test_a_malformed_file_exits_two(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    code, _, err = run("test", str(path), "1")
    assert code == USAGE
    assert "JSON" in err


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_shows_every_step(machine):
    code, out, _ = run("run", machine, "011")
    assert code == OK
    assert out.count("q0") >= 2
    assert "accepting" in out
    # One row per symbol read, plus a header, a rule and the final state.
    rows = [line for line in out.splitlines() if line.strip()]
    assert len(rows) >= 3 + len("011")


def test_run_on_a_rejected_word_exits_one(machine):
    code, out, _ = run("run", machine, "101")
    assert code == NO
    assert "not accepting" in out


def test_run_without_an_initial_state_says_so(tmp_path):
    document, _ = Document().add_state((0.0, 0.0))
    document = document.set_initial(None)
    path = tmp_path / "no_start.json"
    serialize.save(document, str(path))

    code, out, _ = run("run", str(path), "a")
    assert code == NO
    assert "no initial state" in out


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def test_check_reports_defects_and_exits_one(machine):
    code, out, _ = run("check", machine)
    assert code == NO
    assert "dead_states" in out


def test_check_on_a_clean_automaton_exits_zero(tmp_path):
    document = Document().add_symbol("a")
    document, q0 = document.add_state((0.0, 0.0))
    document = document.add_transition(q0, "a", q0).toggle_accept(q0)
    path = tmp_path / "clean.json"
    serialize.save(document, str(path))

    code, out, _ = run("check", str(path))
    assert code == OK
    assert "no defects" in out


def test_check_summarises_the_automaton(machine):
    _, out, _ = run("check", machine)
    assert "3 states" in out
    assert "alphabet {0, 1}" in out


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------


def test_sample_lists_accepted_words_shortest_first(machine):
    code, out, _ = run("sample", machine, "-n", "5")
    assert code == OK
    words = out.split()
    assert words == ["1", "01", "11", "001", "011"]
    assert [len(w) for w in words] == sorted(len(w) for w in words)


def test_sample_is_deterministic(machine):
    assert run("sample", machine, "-n", "8")[1] == run("sample", machine, "-n", "8")[1]


def test_sample_on_an_empty_language_exits_one(tmp_path):
    document = Document().add_symbol("a")
    document, q0 = document.add_state((0.0, 0.0))
    path = tmp_path / "empty.json"
    serialize.save(document.add_transition(q0, "a", q0), str(path))

    code, _, err = run("sample", str(path), "--max-length", "4")
    assert code == NO
    assert "no accepted words" in err


def test_sample_can_show_rejected_words_too(machine):
    code, out, _ = run("sample", machine, "-n", "3", "--rejected")
    assert code == OK
    assert "accepted" in out and "rejected" in out


def test_the_empty_word_prints_as_epsilon(tmp_path):
    document = Document().add_symbol("a")
    document, q0 = document.add_state((0.0, 0.0))
    path = tmp_path / "eps.json"
    serialize.save(document.toggle_accept(q0), str(path))

    code, out, _ = run("sample", str(path))
    assert code == OK
    assert out.strip() == "ε"


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["dot", "tikz", "svg"])
def test_export_writes_to_stdout(machine, fmt):
    code, out, _ = run("export", machine, "-f", fmt)
    assert code == OK
    assert len(out) > 100


@pytest.mark.parametrize("fmt", ["dot", "tikz", "svg"])
def test_export_writes_to_a_file(machine, tmp_path, fmt):
    target = tmp_path / f"out.{fmt}"
    code, out, _ = run("export", machine, "-f", fmt, "-o", str(target))
    assert code == OK
    assert target.exists() and target.stat().st_size > 100
    assert "wrote" in out


def test_export_to_an_unwritable_path_exits_two(machine, tmp_path):
    code, _, err = run("export", machine, "-o", str(tmp_path))
    assert code == USAGE
    assert err


def test_exported_svg_is_byte_stable(machine):
    assert run("export", machine, "-f", "svg")[1] == run("export", machine, "-f", "svg")[1]


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


def test_show_prints_a_transition_table(machine):
    code, out, _ = run("show", machine)
    assert code == OK
    assert "->" in out and "*" in out
    for state in ("q0", "q1", "q2"):
        assert state in out


def test_show_marks_undefined_transitions(tmp_path):
    document = Document().add_symbol("a").add_symbol("b")
    document, q0 = document.add_state((0.0, 0.0))
    path = tmp_path / "partial.json"
    serialize.save(document.add_transition(q0, "a", q0), str(path))

    _, out, _ = run("show", str(path))
    assert "-" in out
    assert "undefined" in out


def test_show_on_an_empty_automaton(tmp_path):
    path = tmp_path / "void.json"
    serialize.save(Document(), str(path))
    code, out, _ = run("show", str(path))
    assert code == OK
    assert "empty" in out


# ---------------------------------------------------------------------------
# new
# ---------------------------------------------------------------------------


def test_new_creates_a_loadable_document(tmp_path):
    target = tmp_path / "fresh.json"
    code, out, _ = run("new", str(target), "-a", "0", "1")
    assert code == OK
    assert "wrote" in out

    document = serialize.load(str(target))
    assert sorted(document.automaton.alphabet) == ["0", "1"]
    assert document.automaton.states == frozenset()


def test_new_refuses_to_clobber(tmp_path):
    target = tmp_path / "fresh.json"
    run("new", str(target))
    code, _, err = run("new", str(target))
    assert code == USAGE
    assert "exists" in err

    assert run("new", str(target), "--force")[0] == OK


def test_new_refuses_an_illegal_symbol(tmp_path):
    code, _, err = run("new", str(tmp_path / "x.json"), "-a", "ab")
    assert code == USAGE
    assert "symbol" in err


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def test_no_command_prints_help():
    code, out, _ = run()
    assert code == USAGE
    assert "usage" in out.lower()


def test_help_lists_every_command():
    code, out, _ = run()
    for command in ("test", "run", "check", "sample", "export", "show", "new"):
        assert command in out


def test_the_gui_command_explains_itself_without_pygame(monkeypatch):
    """The engine must not depend on the app, so this is a runtime lookup."""
    import fsa.cli as cli
    monkeypatch.setattr(cli.os.path, "isfile", lambda _path: False)
    code, _, err = run("gui")
    assert code == USAGE
    assert "not installed" in err


def test_the_cli_never_imports_pygame():
    """It has to work on a machine with no graphics stack at all."""
    import ast
    import pathlib

    source = pathlib.Path(fsa.cli.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        assert "pygame" not in [n.split(".")[0] for n in names]


def test_a_legacy_file_can_be_driven_from_the_cli(tmp_path):
    """Old files still open, so a saved assignment from before still works."""
    legacy = {
        "states": {"q0": {"position": [0, 0], "state_type": "normal"},
                   "q1": {"position": [90, 0], "state_type": "accept"}},
        "transitions": {"q0": {"a": "q1"}},
        "alphabet": ["a"],
        "initial_state": "q0",
        "accept_states": ["q1"],
        "dead_end_states": [],
        "next_state_id": 2,
    }
    path = tmp_path / "old.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    assert run("test", str(path), "a")[0] == OK
    assert run("export", str(path), "-f", "svg")[0] == OK
