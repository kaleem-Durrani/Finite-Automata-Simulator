"""The command line interface.

Every test asserts the exit code, because that is the part a shell script or an
autograder actually reads:

    0   yes -- accepted, or no defects
    1   no  -- rejected, or defects found
    2   could not run

Nothing here touches a display. If any of it needs pygame, the boundary broke.
"""

import csv
import importlib.util
import io
import json

import pytest

import fsa
from fsa import Document, exercise, regex, serialize
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
# check --against, and mark
#
# Grading from the shell. Two things are being tested and they are not the same
# thing:
#
# * **The exit code**, so a marking script can branch on the status alone. Every
#   test here asserts it, because a grader that has to parse prose is a grader
#   that breaks the first time the prose improves.
# * **The sentence**, which is the whole reason this exists. The study behind
#   the phase found a weaker cohort beating a stronger one purely from getting
#   counterexample strings back, so "wrong" is not an acceptable output and the
#   tests say what the output has to contain.
#
# The empty word gets its own tests in both shapes -- as a sentence, where it is
# spelled out in words, and as a table cell, where it is ε with the usual ASCII
# fallback -- because it is the commonest counterexample there is and it is the
# one that renders as nothing at all if nobody thinks about it.
# ---------------------------------------------------------------------------


#: An even number of a's over {a, b}: the reference every submission below is
#: marked against. Three spellings of the same language, so "correct" is tested
#: as a property of the language and not of a machine anyone drew.
EVEN_AS = "b*(ab*ab*)*"
CORRECT_ANSWERS = (EVEN_AS, "(b|ab*a)*", "(ab*a|b)*", "b*(ab*ab*)*b*")

#: Wrong answers, each with the shortest word that proves it wrong. The words
#: are asserted, not just their existence: the counterexample being the
#: *shortest* one is what makes it small enough to trace by hand.
WRONG_ANSWERS = (
    ("(a|b)*", "a"),            # accepts everything
    ("b*", "aa"),               # no a's at all
    ("(aa)*", "b"),             # forgot the b's
    ("a*b*", "a"),              # counted nothing
    ("(ab*ab*)*", "b"),         # a leading b falls off
    ("b*(ab*a)*b*", "aabaa"),   # right until the fifth symbol
)


def write_exercise(directory, name, pattern, alphabet="ab", *, title=""):
    """An exercise file, written the way a marker's directory holds one."""
    path = directory / f"{name}{exercise.EXTENSION}"
    path.write_text(json.dumps({
        "version": exercise.VERSION,
        "kind": exercise.KIND,
        "title": title or name,
        "prompt": "Build a DFA that accepts exactly the right words.",
        "alphabet": sorted(set(alphabet)),
        "reference": {"regex": pattern},
        "examples": {"accept": [], "reject": []},
    }), encoding="utf-8")
    return str(path)


def write_submission(path, pattern, *, determinize=True):
    """A submission, as the machine a pattern denotes.

    Determinized by default, because that is what the exercise asked for. With
    ``determinize=False`` it is Thompson's machine written out whole -- epsilon
    moves and all, exactly what ``fsa from-regex`` produces -- which is a
    legitimate thing for a student to hand in and a different code path.
    """
    machine = regex.to_nfa(pattern)
    if determinize:
        machine = fsa.determinize(machine)
    document = Document.of(machine, fsa.Layout.auto(machine))
    serialize.save(document, str(path))
    return str(path)


@pytest.fixture
def task(tmp_path):
    """``exercises/even_as.fsx``, alone in its directory."""
    directory = tmp_path / "exercises"
    directory.mkdir()
    return write_exercise(directory, "even_as", EVEN_AS,
                          title="An even number of a's")


@pytest.fixture
def submissions(tmp_path):
    """An empty ``submissions/`` for a test to fill."""
    directory = tmp_path / "submissions"
    directory.mkdir()
    return directory


@pytest.mark.parametrize("pattern", CORRECT_ANSWERS)
def test_a_correct_submission_exits_zero(tmp_path, task, pattern):
    """Correct is a property of the language, so four machines that share no
    states pass the same exercise."""
    attempt = write_submission(tmp_path / "attempt.json", pattern)
    code, out, _ = run("check", attempt, "--against", task)
    assert code == OK
    assert "correct" in out
    assert "An even number of a's" in out       # which exercise, not just yes


@pytest.mark.parametrize("pattern,word", WRONG_ANSWERS)
def test_a_wrong_submission_exits_one_and_names_the_word(tmp_path, task,
                                                         pattern, word):
    """The counterexample is the feature. It is named, it is the shortest one,
    and the sentence says which side accepts it -- "wrong" would be useless and
    "your machine accepts 'aabaa'" points at an arrow."""
    attempt = write_submission(tmp_path / "attempt.json", pattern)
    code, out, _ = run("check", attempt, "--against", task)
    assert code == NO
    assert repr(word) in out
    assert ("accepts" in out) and ("rejects" in out)


def test_the_counterexample_really_does_distinguish_the_two(tmp_path, task):
    """Fed back through the CLI: the word the grader named is a word the
    submission and the reference genuinely answer differently. Nothing here
    trusts the sentence -- it is re-run against both machines."""
    attempt = write_submission(tmp_path / "attempt.json", "b*")
    reference = write_submission(tmp_path / "reference.json", EVEN_AS)

    code, out, _ = run("check", attempt, "--against", task)
    assert code == NO
    word = out.strip().splitlines()[-1].split("'")[1]

    assert run("test", attempt, word)[0] != run("test", reference, word)[0]


def test_a_submission_may_be_nondeterministic(tmp_path, task):
    """An NFA is a machine recognising a language, which is all the exercise
    asked for -- and it is exactly what ``fsa from-regex`` writes. Refusing it
    would fail an answer that is right."""
    attempt = write_submission(tmp_path / "nfa.json", EVEN_AS, determinize=False)
    assert not serialize.load(attempt).is_deterministic

    code, out, _ = run("check", attempt, "--against", task)
    assert code == OK
    assert "correct" in out


def test_a_submission_over_the_wrong_alphabet_is_told_so(tmp_path, task):
    """A machine drawn over {0,1} handed in for a task over {a,b} disagrees on
    a one-symbol word, and "your machine accepts '0'" on its own reads as
    nonsense until you notice which alphabet it is in."""
    attempt = write_submission(tmp_path / "attempt.json", "(0|1)*")
    code, out, _ = run("check", attempt, "--against", task)
    assert code == NO
    assert "not in this exercise's alphabet" in out
    assert "{a, b}" in out


def test_a_submission_with_no_initial_state_is_wrong_not_unrunnable(tmp_path,
                                                                    task):
    """Exit 1, not 2. The file opened and the question was asked; the answer is
    that this is not the reference. And the remedy is named, because "no initial
    state" is a sentence a student can read twice without knowing what to
    click."""
    document, _ = Document().add_symbol("a").add_state((0.0, 0.0))
    path = tmp_path / "no_start.json"
    serialize.save(document.set_initial(None), str(path))

    code, out, _ = run("check", str(path), "--against", task)
    assert code == NO
    assert "no initial state" in out
    assert "mark one state as the start" in out


def test_the_empty_word_is_spelled_out_in_the_sentence(tmp_path):
    """The commonest counterexample there is: two machines that disagree about
    their start states disagree about "" before reading anything. ``your machine
    rejects ''`` reads like a bug in the marker, so it comes out in words."""
    everything = write_exercise(tmp_path, "everything", "(a|b)*")
    attempt = write_submission(tmp_path / "attempt.json", "(a|b)(a|b)*")

    code, out, _ = run("check", attempt, "--against", everything)
    assert code == NO
    assert "the empty word" in out
    assert "''" not in out


def test_the_sentence_survives_a_terminal_that_cannot_encode_greek(tmp_path):
    """The empty word is the commonest counterexample and ε is the character a
    Windows code page cannot print, so this pairing is the one that would take
    the grader down. It does not, because the sentence never uses ε."""
    everything = write_exercise(tmp_path, "everything", "(a|b)*")
    attempt = write_submission(tmp_path / "attempt.json", "(a|b)(a|b)*")

    raw = io.BytesIO()
    out = io.TextIOWrapper(raw, encoding="cp1252", newline="")
    assert main(["check", attempt, "--against", everything],
                out=out, err=io.StringIO()) == NO
    out.flush()
    assert "the empty word" in raw.getvalue().decode("cp1252")


def test_check_without_against_still_asks_the_structural_question(tmp_path,
                                                                   machine):
    """Two questions under one verb, and they are genuinely different: this
    machine has a trap state nobody can escape -- a defect -- while recognising
    exactly the language its exercise asks for. Neither answer is the other's,
    which is why the flag has to choose."""
    theirs = write_exercise(tmp_path, "zeros_then_ones", "0*1+", alphabet="01")

    assert run("check", machine)[0] == NO                        # defects
    assert run("check", machine, "--against", theirs)[0] == OK   # correct


def test_grading_against_a_missing_exercise_exits_two(tmp_path):
    attempt = write_submission(tmp_path / "attempt.json", EVEN_AS)
    code, _, err = run("check", attempt, "--against", str(tmp_path / "no.fsx"))
    assert code == USAGE
    assert "no.fsx" in err


def test_grading_against_a_document_says_it_is_the_wrong_kind(machine):
    """Both formats are JSON with a `version`, so the confusing failure is the
    likely one. It is named rather than blamed on a version number."""
    code, _, err = run("check", machine, "--against", machine)
    assert code == USAGE
    assert "automaton document rather than an exercise" in err


def test_grading_a_missing_submission_exits_two(tmp_path, task):
    code, _, err = run("check", str(tmp_path / "gone.json"), "--against", task)
    assert code == USAGE
    assert "gone.json" in err


# --- mark ------------------------------------------------------------------


def read_csv(path):
    """The results file, parsed by the csv module rather than by eye."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_mark_writes_a_row_for_every_submission(task, submissions, tmp_path):
    """Exit criterion: twenty submissions, a CSV a marker can open. Read back
    with the csv module, so quoting and line endings are the module's problem
    and are actually exercised."""
    patterns = ([CORRECT_ANSWERS[index % len(CORRECT_ANSWERS)]
                 for index in range(12)]
                + [WRONG_ANSWERS[index % len(WRONG_ANSWERS)][0]
                   for index in range(8)])
    for index, pattern in enumerate(patterns):
        write_submission(submissions / f"s{index:02d}.json", pattern)

    results = tmp_path / "results.csv"
    code, out, _ = run("mark", str(task), str(submissions), "-o", str(results))
    assert code == NO                       # not everyone was right

    rows = read_csv(str(results))
    assert len(rows) == 20
    assert sum(1 for row in rows if row["correct"] == "yes") == 12
    assert sum(1 for row in rows if row["correct"] == "no") == 8
    assert all(row["exercise"] == "even_as" for row in rows)
    assert "20 submissions: 12 correct, 8 wrong" in out


def test_the_csv_carries_the_counterexample_and_the_sentence(task, submissions,
                                                             tmp_path):
    """The two columns a marker actually reads. The message is the one from
    `fsa.exercise`, verbatim -- the CLI does not paraphrase it."""
    write_submission(submissions / "wrong.json", "b*")
    results = tmp_path / "results.csv"
    assert run("mark", str(task), str(submissions), "-o", str(results))[0] == NO

    row, = read_csv(str(results))
    assert row["submission"] == "wrong.json"
    assert row["counterexample"] == "aa"
    assert row["attempt"] == "rejects"
    assert row["message"] == "your machine rejects 'aa', the answer accepts it"


def test_the_empty_word_is_told_apart_from_no_word_at_all(submissions,
                                                          tmp_path):
    """Three rows leave the counterexample cell blank and they mean three
    different things. `attempt` is what separates them: it is filled in exactly
    when a distinguishing word exists, which is the invariant Result promises,
    so (no, blank, rejects) is "differs on the empty word" while (no, blank,
    blank) is "there was no language to compare"."""
    everything = write_exercise(tmp_path, "everything", "(a|b)*")
    write_submission(submissions / "nonempty.json", "(a|b)(a|b)*")
    write_submission(submissions / "right.json", "(a|b)*")
    document, _ = Document().add_symbol("a").add_state((0.0, 0.0))
    serialize.save(document.set_initial(None),
                   str(submissions / "no_start.json"))

    results = tmp_path / "results.csv"
    assert run("mark", everything, str(submissions), "-o", str(results))[0] == NO

    rows = {row["submission"]: row for row in read_csv(str(results))}
    assert rows["nonempty.json"]["correct"] == "no"
    assert rows["nonempty.json"]["counterexample"] == ""
    assert rows["nonempty.json"]["attempt"] == "rejects"

    assert rows["right.json"]["correct"] == "yes"
    assert rows["right.json"]["counterexample"] == ""
    assert rows["right.json"]["attempt"] == ""

    assert rows["no_start.json"]["correct"] == "no"
    assert rows["no_start.json"]["attempt"] == ""
    assert "no initial state" in rows["no_start.json"]["message"]


def test_the_table_writes_the_empty_word_as_epsilon(submissions, tmp_path):
    """A sentence says "the empty word"; a column four characters wide says ε,
    the spelling `sample` and `show` already use -- with the same fallback for
    the terminal that cannot encode it, which is the one this runs on."""
    everything = write_exercise(tmp_path, "everything", "(a|b)*")
    write_submission(submissions / "nonempty.json", "(a|b)(a|b)*")

    code, out, _ = run("mark", everything, str(submissions))
    assert code == NO
    assert "ε" in out

    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", newline="")
    assert main(["mark", everything, str(submissions)],
                out=stream, err=io.StringIO()) == NO
    stream.flush()
    text = raw.getvalue().decode("cp1252")
    assert "eps" in text and "ε" not in text


def test_a_malformed_submission_is_a_row_not_a_crash(task, submissions,
                                                     tmp_path):
    """The one that matters on real submissions: a file that will not open
    costs its own row, not the other nineteen students' marks."""
    write_submission(submissions / "alice.json", EVEN_AS)
    (submissions / "bob.json").write_text("{not json", encoding="utf-8")
    write_submission(submissions / "carol.json", "b*")
    (submissions / "dave.json").write_text("[]", encoding="utf-8")

    results = tmp_path / "results.csv"
    code, out, err = run("mark", str(task), str(submissions), "-o", str(results))
    assert code == NO

    rows = {row["submission"]: row for row in read_csv(str(results))}
    assert set(rows) == {"alice.json", "bob.json", "carol.json", "dave.json"}
    assert rows["alice.json"]["correct"] == "yes"
    assert rows["carol.json"]["correct"] == "no"

    # Neither right nor wrong: a third answer, so a marker can sort the files
    # to go and look at into a pile of their own.
    assert rows["bob.json"]["correct"] == "error"
    assert rows["dave.json"]["correct"] == "error"
    assert "JSON" in rows["bob.json"]["message"]
    assert "bob.json" in err                    # and said out loud, once
    assert "1 correct, 1 wrong, 2 could not be marked" in out


def test_mark_exits_zero_when_every_submission_is_correct(task, submissions):
    """So `fsa mark exercises/ submissions/ && echo all passed` says something
    true, and a CI check on a worked-solutions directory is one line."""
    for index, pattern in enumerate(CORRECT_ANSWERS):
        write_submission(submissions / f"s{index}.json", pattern)

    code, out, _ = run("mark", str(task), str(submissions))
    assert code == OK
    assert "4 submissions: 4 correct, 0 wrong" in out
    assert "could not be marked" not in out     # only named when it happened


def test_mark_pairs_a_submission_with_the_exercise_its_path_names(tmp_path,
                                                                  submissions):
    """With more than one exercise, the pairing is by name -- either a
    directory or part of the filename, since both are layouts a marker will
    already have."""
    exercises = tmp_path / "exercises"
    exercises.mkdir()
    write_exercise(exercises, "even_as", EVEN_AS)
    write_exercise(exercises, "ends_with_b", "(a|b)*b")

    nested = submissions / "ends_with_b"
    nested.mkdir()
    write_submission(nested / "alice.json", "(a|b)*b")
    write_submission(submissions / "bob.even_as.json", EVEN_AS)
    # Correct for `even_as`, but it is not the question this one was asked.
    write_submission(nested / "carol.json", EVEN_AS)

    results = tmp_path / "results.csv"
    assert run("mark", str(exercises), str(submissions),
               "-o", str(results))[0] == NO

    rows = {row["submission"]: row for row in read_csv(str(results))}
    assert rows["ends_with_b/alice.json"]["exercise"] == "ends_with_b"
    assert rows["ends_with_b/alice.json"]["correct"] == "yes"
    assert rows["bob.even_as.json"]["exercise"] == "even_as"
    assert rows["bob.even_as.json"]["correct"] == "yes"
    assert rows["ends_with_b/carol.json"]["correct"] == "no"


def test_an_unpaired_submission_is_not_guessed_at(tmp_path, submissions):
    """Marking it against an arbitrary exercise would hand a student a
    counterexample to a question they were not answering: confident feedback
    about the wrong thing, which is worse than none."""
    exercises = tmp_path / "exercises"
    exercises.mkdir()
    write_exercise(exercises, "even_as", EVEN_AS)
    write_exercise(exercises, "ends_with_b", "(a|b)*b")
    write_submission(submissions / "anonymous.json", EVEN_AS)

    results = tmp_path / "results.csv"
    code, _, err = run("mark", str(exercises), str(submissions),
                       "-o", str(results))
    assert code == NO

    row, = read_csv(str(results))
    assert row["correct"] == "error"
    assert row["exercise"] == ""
    assert "no exercise" in row["message"]
    assert "anonymous.json" in err


def test_a_single_exercise_marks_everything_however_it_is_named(tmp_path,
                                                                submissions):
    """One task, twenty students, filenames nobody agreed in advance: with
    nothing to choose between, there is nothing to pair."""
    lone = write_exercise(tmp_path, "even_as", EVEN_AS)
    write_submission(submissions / "22i-1234.json", EVEN_AS)

    code, out, _ = run("mark", lone, str(submissions))
    assert code == OK
    assert "22i-1234.json" in out


def test_mark_prints_plain_columns_when_rich_is_absent(task, submissions):
    """rich is optional and not installed here, so this is the path that runs.
    The table is the same shape `run` and `show` print: headings, an ASCII rule,
    two spaces between columns."""
    if importlib.util.find_spec("rich") is not None:
        pytest.skip("rich is installed, so this is not the path taken")
    write_submission(submissions / "alice.json", EVEN_AS)
    write_submission(submissions / "bob.json", "b*")

    code, out, _ = run("mark", str(task), str(submissions))
    assert code == NO

    heading, rule, first, second = out.strip().splitlines()[:4]
    assert heading.split() == ["submission", "exercise", "correct",
                               "counterexample"]
    assert set(rule) == {"-", " "}
    assert first.split() == ["alice.json", "even_as", "yes"]
    assert second.split() == ["bob.json", "even_as", "no", "aa"]


def test_the_rich_table_declines_rather_than_half_drawing():
    """The import is the capability check and it happens before anything is
    printed, so a missing library costs a plainer table and never half of one.
    rich is not a dependency of this project and is not installed here, which
    is why the plain columns above are what actually runs."""
    import fsa.cli as cli

    out = io.StringIO()
    if importlib.util.find_spec("rich") is None:
        assert cli._rich_summary([], out) is False
        assert out.getvalue() == ""
    else:
        assert cli._rich_summary([], out) is True


def test_mark_without_an_output_path_writes_no_file(task, submissions,
                                                    tmp_path):
    write_submission(submissions / "alice.json", EVEN_AS)
    code, out, _ = run("mark", str(task), str(submissions))
    assert code == OK
    assert "alice.json" in out
    assert not list(tmp_path.glob("*.csv"))


def test_mark_keeps_the_table_when_the_csv_cannot_be_written(task, submissions,
                                                             tmp_path):
    """Exit 2, because the run could not deliver -- but the results are still
    on stdout, which is the marker's only remaining copy of the work."""
    write_submission(submissions / "alice.json", EVEN_AS)
    code, out, err = run("mark", str(task), str(submissions), "-o",
                         str(tmp_path))       # a directory: not writable as one
    assert code == USAGE
    assert "alice.json" in out
    assert err


def test_marking_the_same_directory_twice_gives_the_same_bytes(task,
                                                               submissions,
                                                               tmp_path):
    """So a re-run can be diffed against the last one, which is how a marker
    finds out what changed after a round of resubmissions."""
    for index, pattern in enumerate(CORRECT_ANSWERS + ("b*", "(a|b)*")):
        write_submission(submissions / f"s{index}.json", pattern)

    first, second = tmp_path / "one.csv", tmp_path / "two.csv"
    assert run("mark", str(task), str(submissions), "-o", str(first))[0] == NO
    assert run("mark", str(task), str(submissions), "-o", str(second))[0] == NO
    assert first.read_bytes() == second.read_bytes()


def test_mark_on_an_empty_directory_exits_two(task, submissions):
    code, _, err = run("mark", str(task), str(submissions))
    assert code == USAGE
    assert "no .json submission here" in err


def test_a_path_that_is_not_there_says_so_rather_than_that_it_is_empty(
        task, tmp_path):
    """Two failures fixed in two different places: a typo on the command line,
    and a directory that really is empty."""
    code, _, err = run("mark", str(task), str(tmp_path / "typo"))
    assert code == USAGE
    assert "no such file or directory" in err


def test_mark_without_a_readable_exercise_exits_two(tmp_path, submissions):
    """A directory of exercises that are all broken is a run that cannot
    start -- as against a broken submission, which is a row."""
    exercises = tmp_path / "exercises"
    exercises.mkdir()
    (exercises / "broken.fsx").write_text("{not json", encoding="utf-8")
    write_submission(submissions / "alice.json", EVEN_AS)

    code, _, err = run("mark", str(exercises), str(submissions))
    assert code == USAGE
    assert "broken.fsx" in err
    assert "nothing to mark against" in err


def test_a_broken_exercise_does_not_stop_the_others(tmp_path, submissions):
    """The same courtesy a broken submission gets, for the same reason."""
    exercises = tmp_path / "exercises"
    exercises.mkdir()
    write_exercise(exercises, "even_as", EVEN_AS)
    (exercises / "broken.fsx").write_text("{}", encoding="utf-8")
    write_submission(submissions / "alice.even_as.json", EVEN_AS)

    code, out, err = run("mark", str(exercises), str(submissions))
    assert code == OK
    assert "broken.fsx" in err
    assert "1 correct" in out


def test_marking_the_checked_in_examples(submissions):
    """The exercises that ship with the project, marked by the tool that ships
    with them. Nothing here is written by the test but the submission."""
    write_submission(submissions / "even_as.json", EVEN_AS)
    write_submission(submissions / "ends_with_01.json", "(0|1)*01")

    code, out, _ = run("mark", "examples", str(submissions))
    assert code == OK, out
    assert "2 correct, 0 wrong" in out


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
                    "from-regex", "to-regex", "mark"):
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
