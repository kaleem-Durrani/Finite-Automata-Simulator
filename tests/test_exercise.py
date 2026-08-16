"""Exercises, self-grading, and the sentence a student actually reads.

Three claims carry this module.

* **The verdict is a language comparison and nothing else.** A submission is
  correct exactly when no word tells it and the reference apart -- whether it
  was drawn as a DFA or an NFA, and whatever alphabet it happens to be over.
* **A wrong submission always comes back with the shortest word that proves
  it**, and that word really is one the two machines answer differently. Those
  two are the properties at the bottom of the file, checked over machines
  hypothesis builds rather than over the handful anybody would think to write.
* **The message is the feature.** So the wording is asserted literally, string
  by string, and not merely tested for being non-empty. Rewording it should
  cost a deliberate edit here, because "your machine accepts 'bb', the answer
  rejects it" is the whole reason this phase exists and "incorrect" is not.

The empty word gets its own tests for the same reason it gets its own spelling:
it is the commonest counterexample there is, and ``your machine accepts ''``
reads like a bug in the marker.

Imports no pygame and touches no display.
"""

import itertools
import json
from typing import FrozenSet, Iterator, Optional, Sequence

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import fsa
from fsa import exercise, regex, serialize
from fsa.equivalence import equivalent
from fsa.exercise import (
    CORRECT_MESSAGE,
    NO_INITIAL_MESSAGE,
    Exercise,
    ExerciseFormatError,
    check,
)
from fsa.nfa import NFA
from fsa.nfa import accepts as nfa_accepts
from fsa.simulate import accepts
from fsa.subset import determinize
from tests.strategies import dfas, nfas

SETTINGS = settings(max_examples=60, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])

EXAMPLES = ["even_as.fsx", "ends_with_01.fsx"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def envelope(pattern: str, alphabet: str = "ab", *,
             accept: Sequence[str] = (), reject: Sequence[str] = (),
             prompt: str = "Accept the right words.",
             title: str = "") -> str:
    """An exercise file, as text. Every fixture below goes through the real
    reader, so the format is exercised by every test rather than only by the
    ones about the format."""
    return json.dumps({
        "version": exercise.VERSION,
        "kind": exercise.KIND,
        "title": title,
        "prompt": prompt,
        "alphabet": sorted(set(alphabet)),
        "reference": {"regex": pattern},
        "examples": {"accept": list(accept), "reject": list(reject)},
    })


def task(pattern: str, alphabet: str = "ab", *,
         accept: Sequence[str] = (), reject: Sequence[str] = (),
         prompt: str = "Accept the right words.",
         title: str = "") -> Exercise:
    """An exercise whose answer is ``pattern``."""
    return exercise.loads(envelope(pattern, alphabet, accept=accept,
                                   reject=reject, prompt=prompt, title=title))


def machine(pattern: str) -> fsa.DFA:
    """A submission: the deterministic machine a pattern denotes, over exactly
    the symbols the pattern mentions -- so passing a pattern over the wrong
    letters is how the wrong-alphabet cases below are written."""
    return determinize(regex.to_nfa(pattern))


def headless() -> fsa.DFA:
    """A submission with no start state. Built directly rather than with
    ``with_state``, which would choose one."""
    return fsa.DFA(states=frozenset({"q0"}), alphabet=frozenset("ab"))


def example_path(name: str) -> str:
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "examples", name)


def shorter_words(alphabet: FrozenSet[str], length: int) -> Iterator[str]:
    """Every word over ``alphabet`` shorter than ``length``, shortest first.

    The union of the two alphabets is enough to prove minimality: a symbol
    neither machine knows is rejected by both whatever surrounds it, so it can
    never be part of a distinguishing word (see :mod:`fsa.equivalence`).
    """
    for size in range(length):
        for letters in itertools.product(sorted(alphabet), repeat=size):
            yield "".join(letters)


@st.composite
def exercises(draw: st.DrawFn) -> Exercise:
    """An exercise around a generated reference machine.

    No visible examples: :class:`Exercise` checks those against the reference,
    and a generated pair would either be a tautology or a rejected draw. They
    get their own tests instead.
    """
    reference = draw(dfas(with_initial=True))
    return Exercise(prompt="Accept the right words.",
                    alphabet=reference.alphabet,
                    reference=reference,
                    accept_examples=(),
                    reject_examples=())


# ---------------------------------------------------------------------------
# The format
# ---------------------------------------------------------------------------


def test_an_exercise_carries_the_prompt_the_alphabet_and_the_examples():
    task_ = task("a*", "ab", accept=["", "a", "aa"], reject=["b", "ab"],
                 prompt="Accept the words made only of a's.", title="Only a's")
    assert task_.prompt == "Accept the words made only of a's."
    assert task_.title == "Only a's"
    assert task_.alphabet == frozenset("ab")
    assert task_.accept_examples == ("", "a", "aa")
    assert task_.reject_examples == ("b", "ab")


def test_the_reference_is_widened_to_the_declared_alphabet():
    """``a*`` never mentions b, but the task is set over {a, b}.

    The reference has to know b exists, or a front end showing it would show a
    different alphabet from the one the prompt talks about. The extra symbol
    gets no arrow, which is exactly "every word with a b in it is rejected".
    """
    reference = task("a*", "ab").reference
    assert reference.alphabet == frozenset("ab")
    assert accepts(reference, "aa")
    assert not accepts(reference, "ab")


def test_the_examples_are_checked_against_the_reference():
    with pytest.raises(ExerciseFormatError) as caught:
        task("a*", "ab", accept=["ab"])
    assert "'ab'" in str(caught.value)
    assert "rejects it" in str(caught.value)

    with pytest.raises(ExerciseFormatError) as caught:
        task("a*", "ab", reject=["aa"])
    assert "'aa'" in str(caught.value)
    assert "accepts it" in str(caught.value)


def test_a_bad_example_naming_the_empty_word_says_so_in_words():
    """The failure message obeys the same rule the feedback does."""
    with pytest.raises(ExerciseFormatError) as caught:
        task("aa*", "ab", accept=[""])
    assert "the empty word" in str(caught.value)
    assert "''" not in str(caught.value)


def test_an_exercise_without_a_prompt_is_refused():
    with pytest.raises(ExerciseFormatError, match="no prompt"):
        task("a*", "ab", prompt="   ")


def test_a_reference_using_an_undeclared_symbol_is_refused():
    with pytest.raises(ExerciseFormatError) as caught:
        task("a*c", "ab")
    assert "'c'" in str(caught.value)
    assert "{a, b}" in str(caught.value)


def test_a_reference_with_no_start_state_is_refused():
    """Directly, not through a file: the rule belongs to the value."""
    with pytest.raises(ExerciseFormatError, match="no initial state"):
        Exercise(prompt="p", alphabet=frozenset("ab"),
                 reference=fsa.DFA(states=frozenset({"q0"}),
                                   alphabet=frozenset("ab")),
                 accept_examples=(), reject_examples=())


def test_the_title_is_optional_and_the_prompt_is_the_fallback_name():
    assert task("a*", title="Only a's").name() == "Only a's"
    assert task("a*", prompt="Accept a's.\nBe careful.").name() == "Accept a's."


def test_a_file_round_trips_exactly():
    """Not merely to an equivalent exercise: to the same value.

    The reference is normalised on the way in, so the machine a pattern loads
    to depends on the language and not on how it was spelled.
    """
    original = task("b*(ab*ab*)*", "ab", accept=["", "aa"], reject=["a"],
                    prompt="Even a's.", title="Even")
    assert exercise.loads(exercise.dumps(original)) == original


def test_writing_a_file_twice_produces_the_same_bytes():
    """The property `fsa.serialize` is built around, kept here.

    Before the reference was put in a normal form this alternated between two
    spellings of the same answer forever, so a rewritten exercise never stopped
    showing up in a diff.
    """
    once = exercise.dumps(task("(0|1)*01", "01"))
    twice = exercise.dumps(exercise.loads(once))
    assert once == twice


def test_two_spellings_of_one_answer_load_to_the_same_value():
    assert task("a*", "ab").reference == task("(a|a)*", "ab").reference
    assert task("a*", "ab").reference == task("aa*|", "ab").reference


def test_the_envelope_is_versioned_and_names_its_kind():
    data = json.loads(exercise.dumps(task("a*", "ab")))
    assert data["version"] == exercise.VERSION
    assert data["kind"] == exercise.KIND
    assert data["alphabet"] == ["a", "b"]


def test_the_written_answer_is_a_pattern_and_never_a_diagram():
    """The obfuscation claim, such as it is.

    A marker who opens the file to fix a typo in the prompt should not have the
    answer laid out as a table of arrows. The answer is still *in* the file --
    one `fsa from-regex` away -- which is why the module docstring calls this
    obfuscation rather than security.
    """
    text = exercise.dumps(task("b*(ab*ab*)*", "ab"))
    assert '"regex"' in text
    assert "transitions" not in text
    assert "states" not in text


def test_examples_keep_the_order_they_were_written_in():
    """Not sorted: they are written shortest-first and in teaching order."""
    written = ["", "b", "aa", "abba"]
    data = json.loads(exercise.dumps(
        task("b*(ab*ab*)*", "ab", accept=written)))
    assert data["examples"]["accept"] == written


def test_a_reference_may_be_a_whole_automaton_instead():
    """The other half of the format, read through the document reader itself.

    An exercise authored in the editor has a machine and no pattern, so both
    forms have to open -- and both have to land on the same value, or the
    format would have two normal forms and the round trip would depend on which
    one a file happened to use.
    """
    drawn = machine("b*(ab*ab*)*").with_symbol("b")
    document = fsa.Document.of(drawn, fsa.Layout.auto(drawn))
    data = json.loads(envelope("b*(ab*ab*)*", "ab"))
    data["reference"] = {"automaton": serialize.to_dict(document)}

    from_diagram = exercise.loads(json.dumps(data))
    assert from_diagram.reference == task("b*(ab*ab*)*", "ab").reference


def test_a_nondeterministic_reference_automaton_is_read_too():
    """Thompson's machine, epsilon moves and all, embedded as it comes."""
    thompson = regex.to_nfa("(a|b)*a")
    data = json.loads(envelope("(a|b)*a", "ab"))
    data["reference"] = {
        "automaton": serialize.to_nfa_dict(thompson, fsa.Layout(), 0)}

    loaded = exercise.loads(json.dumps(data))
    assert equivalent(loaded.reference, machine("(a|b)*a"))


def test_a_reference_automaton_with_no_start_state_says_so():
    empty = NFA(states=frozenset({"q0"}), alphabet=frozenset("ab"))
    data = json.loads(envelope("a*", "ab"))
    data["reference"] = {"automaton": serialize.to_nfa_dict(empty, fsa.Layout(), 0)}

    with pytest.raises(ExerciseFormatError, match="no initial state"):
        exercise.loads(json.dumps(data))


@pytest.mark.parametrize("text, fragment", [
    ("{not json", "not valid JSON"),
    ('{"version": 1, "kind": "document"}', "not an exercise"),
    ('{"version": 99, "kind": "exercise"}', "unsupported version"),
    ('{"version": 1, "kind": "exercise"}', "missing 'reference'"),
    ('{"version": 1, "kind": "exercise", "reference": {}}', "neither"),
    ('{"version": 1, "kind": "exercise", "reference": {"automaton": 7}}',
     "not an object"),
    ('{"version": 1, "kind": "exercise", "alphabet": ["ab"],'
     ' "reference": {"regex": "a"}}', "alphabet is not usable"),
    ('[1, 2, 3]', "not an object"),
])
def test_a_malformed_file_is_refused_with_a_reason(text, fragment):
    with pytest.raises(ExerciseFormatError, match=fragment):
        exercise.loads(text)


def test_a_document_handed_to_the_exercise_reader_is_told_what_it_is():
    """The specific confusion two JSON formats with a `version` key invite.

    Complaining about the version number would send someone to fix the wrong
    line of the wrong file.
    """
    document = serialize.dumps(fsa.Document.of(machine("a*"),
                                               fsa.Layout.auto(machine("a*"))))
    with pytest.raises(ExerciseFormatError, match="automaton document"):
        exercise.loads(document)


def test_a_bad_reference_pattern_keeps_its_position_and_caret():
    """`RegexSyntaxError` travels out unwrapped, deliberately.

    Its position is what puts a mark under the character to fix; an
    ExerciseFormatError carrying only the text would throw that away. Both are
    AutomatonErrors, so a caller catching the base class still catches this.
    """
    with pytest.raises(regex.RegexSyntaxError) as caught:
        task("(a|b", "ab")
    assert caught.value.position == 0
    assert "^" in caught.value.caret()
    assert isinstance(caught.value, fsa.AutomatonError)


def test_a_file_round_trips_through_the_disk(tmp_path):
    original = task("(0|1)*01", "01", reject=[""], title="Ends with 01")
    path = str(tmp_path / ("t" + exercise.EXTENSION))
    exercise.save(original, path)
    assert exercise.load(path) == original


def test_a_failure_to_read_comes_back_as_a_reason_not_an_exception(tmp_path):
    missing, reason = exercise.load_or_error(str(tmp_path / "absent.fsx"))
    assert missing is None
    assert reason

    bad = tmp_path / "bad.fsx"
    bad.write_text("{oops", encoding="utf-8")
    broken, reason = exercise.load_or_error(str(bad))
    assert broken is None
    assert "JSON" in reason


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def test_the_reference_answer_is_correct():
    task_ = task("b*(ab*ab*)*", "ab")
    result = check(task_.reference, task_)
    assert result.correct
    assert result.counterexample is None
    assert result.attempt_accepts is None
    assert result.message == CORRECT_MESSAGE


def test_a_differently_drawn_machine_for_the_same_language_is_correct():
    """Correctness is about the language, not about the drawing."""
    task_ = task("b*(ab*ab*)*", "ab")
    assert check(machine("(b|ab*a)*"), task_).correct


def test_a_wrong_machine_is_wrong():
    result = check(machine("(a|b)*"), task("a*", "ab"))
    assert not result.correct
    assert result.counterexample == "b"
    assert result.attempt_accepts is True


def test_a_nondeterministic_submission_is_graded_not_refused():
    """A student may legitimately hand in an NFA. Thompson's machine for the
    right language is one, epsilon moves and all."""
    task_ = task("(a|b)*a", "ab")
    submission = regex.to_nfa("(a|b)*a")
    assert not submission.is_deterministic()
    assert check(submission, task_).correct


def test_a_nondeterministic_wrong_submission_still_gets_a_word():
    result = check(regex.to_nfa("(a|b)*b"), task("(a|b)*a", "ab"))
    assert not result.correct
    assert result.counterexample == "a"
    assert result.attempt_accepts is False


def test_a_submission_with_no_start_state_is_wrong_and_says_why():
    result = check(headless(), task("a*", "ab"))
    assert not result.correct
    assert result.counterexample is None
    assert result.attempt_accepts is None
    assert result.message == NO_INITIAL_MESSAGE


def test_no_start_state_is_wrong_even_when_the_answer_accepts_nothing():
    """The one place this could quietly mark a blank page correct.

    The engine reads `initial=None` as "no language defined yet" rather than
    "the empty language" -- the distinction `fsa determinize` and `fsa to-regex`
    refuse on -- so there is nothing here to be equivalent to.
    """
    nothing = task(regex.EMPTY_LANGUAGE, "ab", reject=["", "a", "b"])
    assert not accepts(nothing.reference, "")
    assert not check(headless(), nothing).correct


def test_a_machine_with_no_states_at_all_is_the_same_case():
    assert check(fsa.DFA(), task("a*", "ab")).message == NO_INITIAL_MESSAGE


def test_a_submission_over_a_different_alphabet_is_compared_anyway():
    """Not refused: it is a machine recognising a language, which is what was
    asked for. It is simply the wrong language."""
    result = check(machine("0*"), task("a*", "ab"))
    assert not result.correct
    assert result.counterexample == "0"
    assert result.attempt_accepts is True


def test_a_submission_missing_a_symbol_can_still_be_right():
    """`a*` over {a} and `a*` over {a, b} recognise the same language: neither
    accepts a word containing b, one for want of an arrow and the other for
    want of the letter."""
    assert check(machine("a*"), task("a*", "ab")).correct


# ---------------------------------------------------------------------------
# The message, which is the feature
# ---------------------------------------------------------------------------


def test_the_message_names_the_word_and_the_side_that_accepts_it():
    result = check(machine("(a|b)*"), task("a*", "ab"))
    assert result.message == "your machine accepts 'b', the answer rejects it"


def test_the_message_turns_round_when_the_submission_is_the_one_rejecting():
    result = check(machine("a*"), task("(a|b)*", "ab"))
    assert result.message == "your machine rejects 'b', the answer accepts it"


def test_the_empty_word_is_named_in_words_not_in_empty_quotes():
    """`your machine accepts ''` reads like a bug in the marker."""
    result = check(machine("a*"), task("aa*", "a"))
    assert result.counterexample == ""
    assert result.message == (
        "your machine accepts the empty word, the answer rejects it")
    assert "''" not in result.message


def test_the_empty_word_turns_round_too():
    result = check(machine("aa*"), task("a*", "a"))
    assert result.counterexample == ""
    assert result.message == (
        "your machine rejects the empty word, the answer accepts it")


def test_a_word_outside_the_exercises_alphabet_is_pointed_out():
    """The invisible mistake: a machine drawn over {0, 1} handed in against a
    task set over {a, b}. Without the clause, "your machine accepts '0'" reads
    as nonsense until you notice which alphabet it is in."""
    result = check(machine("0*"), task("a*", "ab"))
    assert result.message == (
        "your machine accepts '0', the answer rejects it "
        "-- '0' is not in this exercise's alphabet {a, b}")


def test_several_foreign_symbols_are_listed_and_the_verb_agrees():
    result = check(machine("a*|01"), task("a*", "a"))
    assert result.counterexample == "01"
    assert result.message == (
        "your machine accepts '01', the answer rejects it "
        "-- '0' and '1' are not in this exercise's alphabet {a}")


def test_the_correct_message_says_why_rather_than_only_yes():
    assert CORRECT_MESSAGE == (
        "correct: no word tells your machine and the answer apart")


def test_the_no_start_state_message_names_the_remedy():
    assert NO_INITIAL_MESSAGE == (
        "your machine has no initial state, so there is no language to compare "
        "-- mark one state as the start")


def test_no_message_ever_contains_a_character_a_windows_console_would_refuse():
    """cp1252 has no Greek in it, and printing one aborted `fsa sample` with a
    traceback on every machine that accepted the empty word. Nothing here has a
    stream to ask, so nothing here may produce the character at all."""
    task_ = task("aa*", "a")
    messages = [check(machine("a*"), task_).message,
                check(machine("a*"), task_).message,
                CORRECT_MESSAGE, NO_INITIAL_MESSAGE]
    for message in messages:
        message.encode("cp1252")


# ---------------------------------------------------------------------------
# The checked-in exercises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", EXAMPLES)
def test_the_shipped_exercises_load(name):
    task_ = exercise.load(example_path(name))
    assert task_.prompt.strip()
    assert task_.title.strip()
    assert task_.alphabet


@pytest.mark.parametrize("name", EXAMPLES)
def test_the_shipped_exercises_agree_with_their_own_examples(name):
    """`Exercise` enforces this on the way in, so a failure here means the
    enforcement stopped working rather than that the file drifted."""
    task_ = exercise.load(example_path(name))
    assert task_.accept_examples and task_.reject_examples
    for word in task_.accept_examples:
        assert accepts(task_.reference, word), word
    for word in task_.reject_examples:
        assert not accepts(task_.reference, word), word


@pytest.mark.parametrize("name", EXAMPLES)
def test_the_shipped_exercises_hide_the_answer(name):
    with open(example_path(name), "r", encoding="utf-8") as handle:
        text = handle.read()
    assert '"regex"' in text
    assert "transitions" not in text


def test_the_even_as_exercise_grades_a_right_and_a_wrong_answer():
    task_ = exercise.load(example_path("even_as.fsx"))
    assert check(machine("(b|ab*a)*"), task_).correct

    # The classic near miss: counting a's but forgetting that zero is even.
    result = check(machine("b*ab*a(b|ab*a)*"), task_)
    assert not result.correct
    assert result.counterexample == ""
    assert result.message == (
        "your machine rejects the empty word, the answer accepts it")


def test_the_ends_with_01_exercise_grades_a_right_and_a_wrong_answer():
    task_ = exercise.load(example_path("ends_with_01.fsx"))
    assert check(machine("(1|0)*01"), task_).correct

    # Reading the pair the wrong way round.
    result = check(machine("(0|1)*10"), task_)
    assert not result.correct
    assert result.message == "your machine rejects '01', the answer accepts it"


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def verdicts(attempt: fsa.AnyAutomaton, word: str) -> bool:
    """Whether ``attempt`` accepts ``word``, whichever kind of machine it is."""
    if isinstance(attempt, NFA):
        return nfa_accepts(attempt, word)
    return accepts(attempt, word)


def assert_shortest(attempt: fsa.AnyAutomaton, task_: Exercise,
                    word: Optional[str]) -> None:
    """No word shorter than ``word`` distinguishes the two machines."""
    if word is None or len(word) > 5:
        return  # enumerating below length 6 is cheap; above it is not
    union = task_.alphabet | attempt.alphabet
    for shorter in shorter_words(union, len(word)):
        assert verdicts(attempt, shorter) == accepts(task_.reference, shorter), (
            f"{shorter!r} distinguishes them and is shorter than {word!r}")


@given(exercises())
@SETTINGS
def test_the_reference_is_always_correct(task_):
    """The property the whole thing rests on: an answer equal to the answer is
    correct, on every machine hypothesis can build."""
    result = check(task_.reference, task_)
    assert result.correct
    assert result.counterexample is None
    assert result.attempt_accepts is None


@given(exercises())
@SETTINGS
def test_an_equivalent_answer_is_always_correct(task_):
    """Completion adds a trap state and changes no word's verdict, so the
    result is a different machine that must still mark correct."""
    completed, _trap = fsa.complete(task_.reference)
    assert check(completed, task_).correct


@given(exercises(), dfas())
@SETTINGS
def test_a_wrong_answer_gets_a_word_both_sides_genuinely_disagree_about(
        task_, attempt):
    result = check(attempt, task_)
    if result.correct:
        assert result.counterexample is None
        assert equivalent(attempt, task_.reference)
        return

    word = result.counterexample
    assert word is not None
    assert accepts(attempt, word) != accepts(task_.reference, word)
    assert result.attempt_accepts == accepts(attempt, word)


@given(exercises(), dfas())
@SETTINGS
def test_the_word_handed_back_is_the_shortest_one(task_, attempt):
    assert_shortest(attempt, task_, check(attempt, task_).counterexample)


@given(exercises(), nfas())
@SETTINGS
def test_a_nondeterministic_submission_is_graded_by_its_own_language(
        task_, attempt):
    """The determinisation is invisible from outside: the verdict and the word
    are the ones the student's machine gives, not the subset machine's."""
    result = check(attempt, task_)
    word = result.counterexample
    if word is None:
        assert result.correct
        return
    assert nfa_accepts(attempt, word) != accepts(task_.reference, word)
    assert result.attempt_accepts == nfa_accepts(attempt, word)
    assert_shortest(attempt, task_, word)


@given(exercises(), dfas())
@SETTINGS
def test_the_result_fields_never_contradict_each_other(task_, attempt):
    result = check(attempt, task_)
    assert (result.counterexample is None) == (result.attempt_accepts is None)
    if result.correct:
        assert result.counterexample is None
    assert result.message


@given(exercises())
@SETTINGS
def test_every_exercise_survives_a_round_trip(task_):
    """Not equality -- a hand-built reference is not in normal form -- but the
    language, the prompt and the alphabet all come back."""
    reloaded = exercise.loads(exercise.dumps(task_))
    assert reloaded.prompt == task_.prompt
    assert reloaded.alphabet == task_.alphabet
    assert equivalent(reloaded.reference, task_.reference)
    assert exercise.dumps(reloaded) == exercise.dumps(
        exercise.loads(exercise.dumps(reloaded)))
