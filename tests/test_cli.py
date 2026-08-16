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
from fsa import Document, regex, serialize
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


def test_epsilon_falls_back_where_the_terminal_cannot_encode_it(tmp_path):
    """A Windows console still defaults to a code page with no Greek in it, so
    printing the character raised UnicodeEncodeError partway through the output
    -- ``fsa sample`` aborted with a traceback on any machine that accepts the
    empty word. Asked of the stream, so a pipe or a UTF-8 terminal is
    unaffected."""
    document = Document().add_symbol("a")
    document, q0 = document.add_state((0.0, 0.0))
    document = document.toggle_accept(q0)

    # `sample` writes the empty word as epsilon; `show` heads the epsilon
    # column with it. The second machine has to be a separate file: an epsilon
    # move is what `show` is being asked to print, and it is also what stops a
    # DFA-only verb like `sample` running at all.
    plain = tmp_path / "empty-word.json"
    serialize.save(document, str(plain))
    moving = tmp_path / "epsilon-move.json"
    serialize.save(document.add_transition(q0, None, q0), str(moving))

    for argv in (["sample", str(plain)], ["show", str(moving)]):
        raw = io.BytesIO()
        out = io.TextIOWrapper(raw, encoding="cp1252", newline="")
        assert main(argv, out=out, err=io.StringIO()) == OK
        out.flush()

        text = raw.getvalue().decode("cp1252")
        assert "eps" in text and "ε" not in text
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
# from-regex / to-regex
#
# The two halves of Kleene's theorem, so the tests are mostly about the seam
# between them. Almost nothing here asserts the *spelling* of an expression:
# one language has infinitely many, and a test that pinned the string would
# freeze state elimination around whatever it happened to emit first. What is
# asserted is the language, decided on the machines, and the exit codes.
# ---------------------------------------------------------------------------


def a_pattern_machine(pattern, tmp_path, name="built.json"):
    """``fsa from-regex pattern -o <file>``; returns the path, as a string."""
    target = tmp_path / name
    assert run("from-regex", pattern, "-o", str(target))[0] == OK
    return str(target)


def test_from_regex_writes_the_machine_a_pattern_denotes(tmp_path):
    target = tmp_path / "built.json"
    code, out, _ = run("from-regex", "a*b+", "-o", str(target))
    assert code == OK
    assert "wrote" in out

    document = serialize.load(str(target))
    assert sorted(document.automaton.alphabet) == ["a", "b"]
    # Thompson's machine as it comes: the epsilon moves are the thing worth
    # looking at, and the summary line names the verb that removes them.
    assert not document.is_deterministic
    assert "determinize" in out


def test_the_machine_a_pattern_denotes_recognises_that_pattern(tmp_path):
    """End to end and through the filesystem, which is how anyone chains these:
    build the machine, determinize it, run words against it."""
    built = a_pattern_machine("a*b+", tmp_path)
    deterministic = tmp_path / "dfa.json"
    assert run("determinize", built, "-o", str(deterministic))[0] == OK

    for word in ("b", "ab", "aabbb", "bb"):
        assert run("test", str(deterministic), word)[0] == OK, word
    for word in ("", "a", "ba", "aba"):
        assert run("test", str(deterministic), word)[0] == NO, word


def test_from_regex_prints_the_document_when_no_path_is_given():
    code, out, _ = run("from-regex", "ab")
    assert code == OK
    # The document follows the summary, the same way `determinize` prints one.
    document = serialize.loads(out[out.index("{\n"):])
    assert sorted(document.automaton.alphabet) == ["a", "b"]


def test_a_malformed_pattern_exits_two_and_says_where(tmp_path):
    """The parser names the character and its index. Swallowing that for a
    tidier message would send a student back to stare at the whole line."""
    target = tmp_path / "never-written.json"
    code, out, err = run("from-regex", "a(b|c", "-o", str(target))
    assert code == USAGE
    assert "position 1" in err      # the bracket that opened, not the end
    assert "^" in err               # ...and a caret under it
    assert not target.exists()
    assert out == ""


def test_from_regex_gives_every_state_somewhere_to_be(tmp_path):
    """A machine an algorithm built has no coordinates until Layout.auto gives
    it some, and states with none all land on the origin in one pile, where
    hit-testing finds the topmost and the rest are lost."""
    document = serialize.load(a_pattern_machine("(a|b)*abb", tmp_path))
    positions = document.layout.positions

    assert set(positions) == set(document.automaton.states)
    assert len(set(positions.values())) == len(positions)


def test_to_regex_describes_the_machines_language(machine):
    code, out, _ = run("to-regex", machine)
    assert code == OK
    rebuilt = fsa.determinize(regex.to_nfa(out.strip()))
    assert fsa.equivalent(rebuilt, serialize.load(machine).as_dfa())


def test_to_regex_accepts_a_nondeterministic_document(tmp_path):
    """It is defined on both machines, and the one `from-regex` writes is
    nondeterministic -- a verb that refused it would be refusing its own
    output, and sending the user through a determinisation that cannot change
    the answer."""
    document = Document().add_symbol("a")
    document, q0 = document.add_state((0.0, 0.0))
    document, q1 = document.add_state((100.0, 0.0))
    document = (document.add_transition(q0, "a", q0)
                        .add_transition(q0, "a", q1)
                        .add_transition(q0, None, q1)
                        .toggle_accept(q1).set_initial(q0))
    path = tmp_path / "nfa.json"
    serialize.save(document, str(path))

    code, out, _ = run("to-regex", str(path))
    assert code == OK
    rebuilt = fsa.determinize(regex.to_nfa(out.strip()))
    assert fsa.equivalent(rebuilt, fsa.determinize(document.automaton))


def test_to_regex_on_a_machine_that_accepts_nothing(tmp_path):
    document = Document().add_symbol("a")
    document, q0 = document.add_state((0.0, 0.0))
    path = tmp_path / "nothing.json"
    serialize.save(document.set_initial(q0), str(path))

    code, out, _ = run("to-regex", str(path))
    assert code == OK
    assert out.strip() == regex.EMPTY_LANGUAGE


def test_to_regex_on_a_machine_that_accepts_only_the_empty_word(tmp_path):
    document = Document().add_symbol("a")
    document, q0 = document.add_state((0.0, 0.0))
    path = tmp_path / "epsilon.json"
    serialize.save(document.set_initial(q0).toggle_accept(q0), str(path))

    code, out, _ = run("to-regex", str(path))
    assert code == OK
    assert out.strip() == regex.EMPTY_WORD


def test_an_expression_the_terminal_cannot_encode_falls_back_and_says_so(tmp_path):
    """ε and ∅ are the two answers a Windows code page cannot print, and here
    the character is the whole answer rather than a heading on a table.

    Neither stand-in parses back, so the substitution is announced -- on
    stderr, which is also cp1252 here, so the note has to name the codepoint
    rather than show it or it would raise the very error it is explaining.
    """
    document = Document().add_symbol("a")
    document, q0 = document.add_state((0.0, 0.0))
    empty_word = tmp_path / "empty-word.json"
    serialize.save(document.set_initial(q0).toggle_accept(q0), str(empty_word))
    no_words = tmp_path / "no-words.json"
    serialize.save(document.set_initial(q0), str(no_words))

    for path, stood_in_for, codepoint in ((empty_word, "eps", "U+03B5"),
                                          (no_words, "{}", "U+2205")):
        streams = [io.BytesIO(), io.BytesIO()]
        out, err = [io.TextIOWrapper(raw, encoding="cp1252", newline="")
                    for raw in streams]
        assert main(["to-regex", str(path)], out=out, err=err) == OK
        out.flush()
        err.flush()

        printed, explained = [raw.getvalue().decode("cp1252")
                              for raw in streams]
        assert printed.strip() == stood_in_for
        assert codepoint in explained


def test_to_regex_without_an_initial_state_exits_two(tmp_path):
    """"No language yet" is not "the empty language", and ∅ would claim it was.
    The engine cannot keep that distinction -- an expression denotes a language
    and "undefined" is not one -- so the CLI keeps it instead."""
    document, _ = Document().add_state((0.0, 0.0))
    path = tmp_path / "no_start.json"
    serialize.save(document.set_initial(None), str(path))

    code, out, err = run("to-regex", str(path))
    assert code == USAGE
    assert "no initial state" in err
    assert out == ""


def test_to_regex_minimize_is_exactly_minimising_the_machine_first(tmp_path):
    """What the flag claims, as an equation between two ways of asking. Not
    "the answer gets shorter" -- it usually does and sometimes does not, since
    every loop in a small machine costs a star that a big tree of a machine
    never pays -- but "the answer is the one the minimised machine gives"."""
    built = a_pattern_machine("a*b+", tmp_path)
    deterministic, smallest = tmp_path / "dfa.json", tmp_path / "min.json"
    assert run("determinize", built, "-o", str(deterministic))[0] == OK
    assert run("minimize", str(deterministic), "-o", str(smallest))[0] == OK

    assert run("to-regex", built, "--minimize") == run("to-regex", str(smallest))


def test_to_regex_minimize_keeps_the_language(machine):
    code, out, _ = run("to-regex", machine, "--minimize")
    assert code == OK
    rebuilt = fsa.determinize(regex.to_nfa(out.strip()))
    assert fsa.equivalent(rebuilt, serialize.load(machine).as_dfa())


def test_a_pattern_survives_the_round_trip_through_files(tmp_path):
    """Pattern -> machine -> pattern -> machine, entirely through the CLI, and
    the two machines are compared by the CLI too."""
    first = a_pattern_machine("(a|b)*abb", tmp_path, "first.json")

    code, pattern, _ = run("to-regex", first)
    assert code == OK
    second = a_pattern_machine(pattern.strip(), tmp_path, "second.json")

    left, right = tmp_path / "left.json", tmp_path / "right.json"
    assert run("determinize", first, "-o", str(left))[0] == OK
    assert run("determinize", second, "-o", str(right))[0] == OK
    assert run("equiv", str(left), str(right))[0] == OK


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def test_no_command_prints_help():
    code, out, _ = run()
    assert code == USAGE
    assert "usage" in out.lower()


def test_help_lists_every_command():
    code, out, _ = run()
    for command in ("test", "run", "check", "sample", "export", "show", "new",
                    "from-regex", "to-regex"):
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
