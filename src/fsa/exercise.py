"""Exercises, and the sentence that turns a "no" into something to act on.

Everything hard about self-grading is already written. :func:`fsa.equivalence.
counterexample` returns the *shortest* word two machines disagree about, and the
CLI already exits 0 for yes and 1 for no. What was missing is the task itself:
something that says what the student was asked to build, so that "wrong" can be
"wrong on this input".

That last part is the whole feature, not a detail of it. The study this is built
on -- reported in ``IMPROVEMENT_PLAN.md`` -- found a weaker cohort beating a
stronger one on DFA, RegEx and PDA construction for one reason: they were handed
counterexample strings when they were wrong. So :attr:`Result.message` is held
to a standard the boolean is not. It names the word, and it says which side
accepts it, because "your machine accepts 'bb', the answer rejects it" points at
an arrow and "incorrect" points at nothing.

**The empty word is spelled in words.** It is the commonest counterexample there
is -- two machines that disagree about their start states disagree about ``""``
before they have read anything -- and ``your machine accepts ''`` reads like a
bug in the marker rather than a fact about the submission. So it comes out as
"the empty word". Nothing in this module ever prints ``ε`` either: that is one
character a Windows console has repeatedly failed to encode (see
``fsa.cli._epsilon_for``), and this module has no stream to ask.

The format
==========

``.fsx``, JSON, a versioned envelope in the style of :mod:`fsa.serialize`::

    {
      "version": 1,
      "kind": "exercise",
      "title": "Even number of a's",
      "prompt": "Accept exactly the words with an even number of a's.",
      "alphabet": ["a", "b"],
      "reference": {"regex": "b*(ab*ab*)*"},
      "examples": {"accept": ["", "aa"], "reject": ["a", "ba"]}
    }

**The reference may be a regular expression or a whole automaton.** Both are
read; a pattern is what :func:`dumps` writes, and what the checked-in examples
use, because a marker who opens an exercise to fix a typo in the prompt should
not have the answer laid out as a readable transition table.

That is obfuscation and not security, and saying otherwise would be a lie: the
answer is in the file either way, and ``fsa from-regex`` turns the pattern back
into a diagram in one command. What it buys is that the answer is not read *by
accident*, which is the only threat a file a student is never given actually
faces.

**An exercise validates itself.** Every visible example is run against the
reference when the value is built, so an accept example the reference rejects is
a load-time error naming the word rather than a student staring at an example
that contradicts the task. The reference must have a start state, and may not
mention a symbol the exercise did not declare -- an exercise whose language
depends on a letter the student was never told about is unanswerable, not
merely hard.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from fsa import regex, serialize
from fsa.automaton import DFA
from fsa.equivalence import counterexample
from fsa.errors import AutomatonError
from fsa.layout import AnyAutomaton
from fsa.minimize import minimize
from fsa.nfa import NFA, to_dfa
from fsa.simulate import accepts
from fsa.subset import determinize
from fsa.symbols import StateId, Symbol, normalize_alphabet

#: The version an exercise is written as.
VERSION = 1

#: Every version this build understands.
READABLE_VERSIONS = (VERSION,)

#: What the envelope calls itself. A document (:mod:`fsa.serialize`) and an
#: exercise are both JSON with a ``version`` key, so without this the failure
#: mode of handing one to the other is a puzzling complaint about a version
#: number rather than "that is the wrong kind of file".
KIND = "exercise"

#: The conventional extension, for a file picker or a glob.
EXTENSION = ".fsx"


class ExerciseFormatError(AutomatonError):
    """The bytes are not an exercise this version understands.

    Also raised for a file that parses but does not describe a usable task --
    a reference with no start state, an example the reference contradicts. The
    distinction between "malformed" and "unanswerable" is not one the person
    who has to fix the file cares about, and both are fixed in the same editor.
    """


# ----------------------------------------------------------------------
# Values
# ----------------------------------------------------------------------

def _spell(word: str) -> str:
    """A word as it should appear in a sentence shown to a student.

    The empty word gets words rather than an empty pair of quotes. See the
    module docstring: it is the commonest counterexample and the one whose
    obvious spelling reads like a defect in the marker.
    """
    return "the empty word" if not word else repr(word)


def _listed(items: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c`` -- an English list, not a repr."""
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _shown(alphabet: FrozenSet[Symbol]) -> str:
    """An alphabet as the subject writes one: ``{a, b}``, sorted."""
    return "{" + ", ".join(sorted(alphabet)) + "}"


@dataclass(frozen=True, slots=True)
class Exercise:
    """A task, its reference answer, and the examples a student may see.

    Args:
        prompt: What the student is asked to build, in prose. Required and
            non-empty: an exercise with no prompt grades an answer to a
            question nobody asked.
        alphabet: The alphabet the task is set over. The reference is widened
            to exactly this, so a pattern that never mentions ``b`` still
            describes a machine that knows ``b`` exists and rejects it.
        reference: The answer, as a deterministic machine. Must have a start
            state -- the engine reads ``initial=None`` as "no language defined
            yet" rather than "the empty language", and a task whose answer has
            no language cannot be answered.
        accept_examples: Words the student is shown as accepted.
        reject_examples: Words the student is shown as rejected.
        title: A short name, for a list or a window caption. Optional; the
            prompt is the part that carries the meaning.

    Raises:
        ExerciseFormatError: If the reference has no start state, mentions a
            symbol outside ``alphabet``, or contradicts one of the examples.
    """

    prompt: str
    alphabet: FrozenSet[Symbol]
    reference: DFA
    accept_examples: Tuple[str, ...]
    reject_examples: Tuple[str, ...]
    title: str = ""

    def __post_init__(self) -> None:
        """Normalise every field, then check the exercise is answerable."""
        alphabet = normalize_alphabet(self.alphabet)
        reference = self.reference

        if not self.prompt.strip():
            raise ExerciseFormatError("the exercise has no prompt")

        if reference.initial is None:
            raise ExerciseFormatError(
                "the reference has no initial state, so it recognises no "
                "language for an answer to be compared against")

        undeclared = reference.alphabet - alphabet
        if undeclared:
            stray = _listed([repr(symbol) for symbol in sorted(undeclared)])
            raise ExerciseFormatError(
                f"the reference uses {stray}, which the exercise's alphabet "
                f"{_shown(alphabet)} does not declare")

        # Widened rather than merely checked. The exercise declares the
        # alphabet, so the reference should know every letter of it: the extra
        # symbols get no transitions, which is exactly "the answer rejects any
        # word containing one", and it means a front end showing the reference
        # shows the same alphabet the prompt talks about.
        for symbol in sorted(alphabet - reference.alphabet):
            reference = reference.with_symbol(symbol)

        object.__setattr__(self, "alphabet", alphabet)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "accept_examples", tuple(self.accept_examples))
        object.__setattr__(self, "reject_examples", tuple(self.reject_examples))

        # The examples are the part a student is shown, so an example the
        # reference disagrees with is worse than no example at all: it teaches
        # the wrong language, and it does it with the tool's authority. Checked
        # here rather than in `loads` so that it holds for every Exercise
        # however it was built.
        for word in self.accept_examples:
            if not accepts(reference, word):
                raise ExerciseFormatError(
                    f"{_spell(word)} is listed as accepted, but the reference "
                    f"rejects it")
        for word in self.reject_examples:
            if accepts(reference, word):
                raise ExerciseFormatError(
                    f"{_spell(word)} is listed as rejected, but the reference "
                    f"accepts it")

    def name(self) -> str:
        """Something to call this exercise in a list: the title, or the prompt.

        A title is optional, and a menu entry is not. The prompt's first line
        is the fallback because an exercise is written prompt-first -- the
        title is the thing an author forgets.
        """
        if self.title.strip():
            return self.title.strip()
        return self.prompt.strip().splitlines()[0]


@dataclass(frozen=True, slots=True)
class Result:
    """The verdict on one submission, and the sentence that explains it.

    Three of the four fields exist so that a front end can do something with
    the word rather than only print it -- the GUI puts it in the test field and
    runs it, so the student watches their own machine take the wrong path.

    The invariants, which the tests pin:

    * ``attempt_accepts`` is ``None`` exactly when ``counterexample`` is.
    * When there is a counterexample, the reference's verdict on it is
      ``not attempt_accepts``. That is what "counterexample" means, and it is
      why one bool is enough to write the message from.
    * ``correct`` implies ``counterexample is None``. The converse does *not*
      hold: a submission with no start state is wrong and has no word to prove
      it, because it has no language for a word to be compared against.
    """

    correct: bool
    counterexample: Optional[str]
    attempt_accepts: Optional[bool]
    message: str


# ----------------------------------------------------------------------
# Checking
# ----------------------------------------------------------------------

#: What a submission with no start state is told. Not a counterexample case:
#: there is no language to find a word in. The remedy is named, because "no
#: initial state" is a sentence a student can read twice without knowing what
#: to click.
NO_INITIAL_MESSAGE = (
    "your machine has no initial state, so there is no language to compare "
    "-- mark one state as the start")

#: What a correct submission is told. Says *why* it is correct in the terms the
#: rest of the feedback uses, so the two sentences are read as one idea: the
#: check is always about whether a word exists.
CORRECT_MESSAGE = "correct: no word tells your machine and the answer apart"


def _as_dfa(attempt: AnyAutomaton) -> DFA:
    """The submission as a deterministic machine.

    A student may legitimately submit an NFA -- it is a machine over the same
    alphabet recognising a language, which is all the exercise asked for -- so
    this determinises rather than refusing. Two doors, and which one is taken
    matters:

    * A machine that is already deterministic goes through
      :func:`fsa.nfa.to_dfa`, which is exact and keeps the student's own state
      names and their partial delta.
    * Anything else goes through the subset construction, which invents new
      states. That is the price of an answer at all, and it is paid only by the
      submissions that actually branch.
    """
    if isinstance(attempt, DFA):
        return attempt
    if attempt.is_deterministic():
        return to_dfa(attempt)
    return determinize(attempt)


def _wrong_message(word: str, attempt_accepts: bool,
                   alphabet: FrozenSet[Symbol]) -> str:
    """The sentence for a wrong submission: the word, and who accepts it.

    One bool decides both verbs, because a counterexample is by definition a
    word the two sides answer differently -- so the reference's verdict is the
    negation and does not have to be passed in and kept in step.

    A word using a symbol the exercise never declared gets a clause of its own.
    It is a common and completely invisible mistake -- a machine drawn over
    ``{0,1}`` submitted against a task set over ``{a,b}`` disagrees on the
    one-letter word ``0``, and "your machine accepts '0'" on its own reads as
    nonsense until you notice which alphabet it is in.
    """
    if attempt_accepts:
        mine, theirs = "accepts", "rejects"
    else:
        mine, theirs = "rejects", "accepts"
    sentence = f"your machine {mine} {_spell(word)}, the answer {theirs} it"

    foreign = sorted(set(word) - alphabet)
    if foreign:
        verb = "is" if len(foreign) == 1 else "are"
        sentence += (f" -- {_listed([repr(s) for s in foreign])} {verb} not in "
                     f"this exercise's alphabet {_shown(alphabet)}")
    return sentence


def check(attempt: AnyAutomaton, exercise: Exercise) -> Result:
    """Grade one submission against an exercise.

    The comparison is :func:`fsa.equivalence.counterexample` and nothing else,
    so what "correct" means here is exactly what it means everywhere else in
    the engine: no word gets a different verdict from the two machines. That
    includes words over symbols only one of them knows -- the alphabets need
    not match, and a submission drawn over the wrong one is wrong for a reason
    the message spells out rather than being refused.

    Args:
        attempt: The student's machine, deterministic or not. Never mutated.
        exercise: The task, whose ``reference`` is the answer.

    Returns:
        A :class:`Result`. When it is not correct there is a shortest
        distinguishing word, except in the one structural case below.

    An attempt with **no initial state** is wrong and gets no counterexample.
    The engine reads ``initial=None`` as "no language defined yet" rather than
    "the empty language" -- the same distinction ``fsa determinize`` and ``fsa
    to-regex`` refuse on -- so there is no language for a word to distinguish,
    and marking such a submission correct against a reference that happens to
    accept nothing would be an accident, not a judgement.
    """
    if attempt.initial is None:
        return Result(correct=False, counterexample=None, attempt_accepts=None,
                      message=NO_INITIAL_MESSAGE)

    machine = _as_dfa(attempt)
    word = counterexample(machine, exercise.reference)
    if word is None:
        return Result(correct=True, counterexample=None, attempt_accepts=None,
                      message=CORRECT_MESSAGE)

    # Asked of the determinised machine, which is the one the comparison was
    # made against; asking the original NFA would be a second implementation of
    # acceptance and a second thing to keep true.
    mine = accepts(machine, word)
    return Result(correct=False, counterexample=word, attempt_accepts=mine,
                  message=_wrong_message(word, mine, exercise.alphabet))


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------

def to_dict(exercise: Exercise) -> Dict[str, Any]:
    """A plain, sorted, JSON-ready snapshot.

    The reference goes out as a **pattern**, always, derived from the machine
    with :func:`fsa.regex.from_automaton`. Two consequences, and both are
    deliberate:

    * A file this writes is never a readable diagram of the answer, whichever
      form it was read from. That is the point of the format (see the module
      docstring), and a writer that preserved the input form would lose it for
      every exercise authored in the editor.
    * The round trip preserves the *language* of the reference rather than the
      machine. ``loads(dumps(e))`` has the same prompt, title, alphabet and
      examples, and a reference equivalent to ``e``'s -- equal to it, in fact,
      whenever ``e`` itself came from :func:`loads`, because both sides are then
      the normal form :func:`_reference` produces. That is what makes rewriting
      a file idempotent: the bytes settle after one pass rather than alternating
      between two spellings of one answer, so an exercise can be diffed like
      source, exactly as a document can.
    """
    return {
        "version": VERSION,
        "kind": KIND,
        "title": exercise.title,
        "prompt": exercise.prompt,
        "alphabet": sorted(exercise.alphabet),
        "reference": {"regex": regex.from_automaton(exercise.reference)},
        # Author order, not sorted. Examples are written shortest-first and
        # roughly in the order they teach; sorting them would shuffle a
        # deliberate sequence into alphabetical noise.
        "examples": {
            "accept": list(exercise.accept_examples),
            "reject": list(exercise.reject_examples),
        },
    }


def dumps(exercise: Exercise) -> str:
    """Serialise an exercise to text."""
    return json.dumps(to_dict(exercise), indent=2) + "\n"


def save(exercise: Exercise, path: str) -> None:
    """Write an exercise to a file."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(dumps(exercise))


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------

def _text(data: Dict[str, Any], key: str) -> str:
    """A string field, defaulting to empty rather than raising on a missing
    one. What a missing prompt costs is decided in one place --
    :meth:`Exercise.__post_init__` -- so that a hand-written file gets the same
    message however the key went astray."""
    value = data.get(key)
    return value if isinstance(value, str) else ""


def _words(body: Any, key: str) -> Tuple[str, ...]:
    """A list of example words. Anything that is not a string is dropped."""
    if not isinstance(body, dict):
        return ()
    return tuple(str(word) for word in body.get(key, []) if isinstance(word, str))


def _widened(automaton: DFA, alphabet: FrozenSet[Symbol]) -> DFA:
    """``automaton`` over the exercise's whole alphabet.

    The added symbols get no transitions, which is the correct reading: a
    pattern that never mentions ``b`` denotes a language containing no word
    with a ``b`` in it, and a machine with no arrow on ``b`` rejects exactly
    those. Done after minimisation rather than before, so the reference does not
    carry a trap state nobody asked for.
    """
    for symbol in sorted(alphabet - automaton.alphabet):
        automaton = automaton.with_symbol(symbol)
    return automaton


def _canonical(automaton: DFA) -> DFA:
    """The same machine with its states renamed ``q0, q1, ...``.

    Numbered in breadth-first order from the start state, ties broken by symbol
    order, so the names depend on the machine's shape and on nothing else. That
    is what makes :func:`_reference` a *normal form* rather than merely a small
    machine: a minimal DFA is unique up to the names of its states, so once the
    names are derived too, two spellings of one answer load to values that
    compare equal -- and a file rewritten twice does not change, which is the
    property :mod:`fsa.serialize` is built around and the reason an exercise can
    be diffed like source.

    Labels are dropped rather than carried. They came from the subset
    construction and name sets of states in a machine that no longer exists
    (``{q0,q3}``, ``trap``); keeping them would make the normal form depend on
    which of two equal machines happened to be determinised.
    """
    if automaton.initial is None:
        return automaton

    symbols = sorted(automaton.alphabet)
    order: List[StateId] = [automaton.initial]
    seen = {automaton.initial}
    index = 0
    while index < len(order):
        state = order[index]
        index += 1
        for symbol in symbols:
            target = automaton.target(state, symbol)
            if target is not None and target not in seen:
                seen.add(target)
                order.append(target)
    # Minimisation drops unreachable states, so this appends nothing in
    # practice. It is here because the function promises a renaming of *this*
    # machine, and one that silently lost a state would be a different promise.
    order.extend(sorted(automaton.states - seen))

    names = {state: f"q{number}" for number, state in enumerate(order)}
    return DFA(
        states=frozenset(names.values()),
        alphabet=automaton.alphabet,
        transitions={(names[source], symbol): names[target]
                     for (source, symbol), target in automaton.transitions.items()},
        initial=names[automaton.initial],
        accept=frozenset(names[state] for state in automaton.accept),
    )


def _reference(machine: NFA, alphabet: FrozenSet[Symbol]) -> DFA:
    """A reference answer in normal form, whichever form the file wrote it in.

    Determinise, minimise, widen to the declared alphabet, rename. One pipeline
    for both readers rather than one each, so a task stored as a pattern and the
    same task stored as a diagram load to the same value -- which is what lets
    :func:`dumps` write either of them out as the same bytes.

    The determinisation is not skipped for a machine that is already
    deterministic. It costs little on the machines an exercise holds, and taking
    the short cut would make the normal form depend on how the file was written:
    :func:`fsa.minimize.minimize` gives back a partial machine when it is handed
    one and a complete machine when it is handed one, and only the complete
    minimal DFA is unique.
    """
    return _canonical(_widened(minimize(determinize(machine)), alphabet))


def _reference_from_regex(pattern: str, alphabet: FrozenSet[Symbol]) -> DFA:
    """The machine a reference pattern denotes.

    Raises:
        ExerciseFormatError: If the pattern uses a symbol the exercise did not
            declare. Checked here, on the parsed pattern, rather than left to
            :class:`Exercise`, because naming the *pattern's* stray symbol is
            what tells the author where to look.
        fsa.regex.RegexSyntaxError: If the pattern does not parse. Deliberately
            not wrapped: its position and its caret are the whole reason that
            error type exists, and an ``ExerciseFormatError`` carrying only the
            message would throw away the part that gets the typo fixed. It is
            an :class:`~fsa.errors.AutomatonError` too, so a caller catching the
            base class already has both.
    """
    node = regex.parse(pattern)
    undeclared = node.alphabet - alphabet
    if undeclared:
        stray = _listed([repr(symbol) for symbol in sorted(undeclared)])
        raise ExerciseFormatError(
            f"the reference pattern uses {stray}, which the exercise's "
            f"alphabet {_shown(alphabet)} does not declare")
    return _reference(regex.thompson(node), alphabet)


def _reference_from_automaton(body: Any, alphabet: FrozenSet[Symbol]) -> DFA:
    """The machine an embedded document envelope holds.

    Read with :func:`fsa.serialize.from_nfa_dict`, so an exercise can embed any
    version of the document format this build opens and a nondeterministic
    reference is as welcome as a deterministic one -- the reader is reused
    rather than reimplemented, which is what keeps the two formats from drifting
    apart. The layout travels with it and is dropped: an exercise's reference is
    a language, and where its author happened to drag the states says nothing
    about the task.
    """
    if not isinstance(body, dict):
        raise ExerciseFormatError("the reference automaton is not an object")
    try:
        machine, _layout, _next_id = serialize.from_nfa_dict(body)
    except (serialize.DocumentFormatError, AutomatonError, ValueError,
            TypeError, KeyError) as exc:
        raise ExerciseFormatError(
            f"the reference automaton is malformed: {exc}") from exc
    if machine.initial is None:
        # determinize would answer a machine with no states here, and the
        # complaint Exercise makes about that is the right one -- but it would
        # arrive after minimisation had quietly discarded the states the file
        # does contain, so say it while the file is still in view.
        raise ExerciseFormatError(
            "the reference automaton has no initial state, so it recognises no "
            "language for an answer to be compared against")
    return _reference(machine, alphabet)


def from_dict(data: Dict[str, Any]) -> Exercise:
    """Rebuild an exercise from a parsed envelope.

    Reading is the tolerant direction, as it is in :mod:`fsa.serialize`. A
    missing ``kind`` is accepted -- a hand-written file will not have thought of
    it -- but a *wrong* one is refused, because the whole reason the key exists
    is to catch a document handed to the exercise reader.
    """
    if not isinstance(data, dict):
        raise ExerciseFormatError("not an object")

    kind = data.get("kind")
    if kind is not None and kind != KIND:
        raise ExerciseFormatError(f"not an exercise: kind is {kind!r}")
    if "automaton" in data and "prompt" not in data:
        # The specific confusion this format is most likely to meet: `fsa`'s
        # own document files are JSON with a `version` too, and complaining
        # about the version number would send someone to fix the wrong line.
        raise ExerciseFormatError(
            "this looks like an automaton document rather than an exercise")

    version = data.get("version")
    if version not in READABLE_VERSIONS:
        raise ExerciseFormatError(
            f"unsupported version {version!r}; this build reads "
            f"{' and '.join(str(known) for known in READABLE_VERSIONS)}")

    try:
        alphabet = normalize_alphabet(data.get("alphabet", []))
    except AutomatonError as exc:
        raise ExerciseFormatError(f"the alphabet is not usable: {exc}") from exc

    body = data.get("reference")
    if not isinstance(body, dict):
        raise ExerciseFormatError("missing 'reference'")

    # The pattern wins if a hand-edited file somehow carries both. It is the
    # form this module writes and the form the checked-in exercises use, so it
    # is the one an author is looking at when they change the answer.
    pattern = body.get("regex")
    if isinstance(pattern, str):
        reference = _reference_from_regex(pattern, alphabet)
    elif "automaton" in body:
        reference = _reference_from_automaton(body.get("automaton"), alphabet)
    else:
        raise ExerciseFormatError(
            "the reference holds neither a 'regex' nor an 'automaton'")

    examples = data.get("examples")
    return Exercise(
        prompt=_text(data, "prompt"),
        alphabet=alphabet,
        reference=reference,
        accept_examples=_words(examples, "accept"),
        reject_examples=_words(examples, "reject"),
        title=_text(data, "title"),
    )


def loads(text: str) -> Exercise:
    """Parse an exercise from text.

    Raises:
        ExerciseFormatError: For anything structural -- bad JSON, a version
            this build does not read, a reference that contradicts an example.
        fsa.regex.RegexSyntaxError: If a reference pattern does not parse. See
            :func:`_reference_from_regex` for why that one is not wrapped.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExerciseFormatError(f"not valid JSON (line {exc.lineno})") from exc
    return from_dict(data)


def load(path: str) -> Exercise:
    """Read an exercise from a file."""
    with open(path, "r", encoding="utf-8") as handle:
        return loads(handle.read())


def load_or_error(path: str) -> Tuple[Optional[Exercise], str]:
    """Read a file, returning either the exercise or a reason.

    The shape :func:`fsa.serialize.load_or_error` uses, for the same reason:
    the failure has to reach a window or a table, and nothing in :mod:`fsa`
    prints.
    """
    try:
        return load(path), ""
    except OSError as exc:
        return None, exc.strerror or str(exc)
    except (AutomatonError, ValueError, TypeError, KeyError) as exc:
        return None, str(exc)


__all__: List[str] = [
    "VERSION", "READABLE_VERSIONS", "KIND", "EXTENSION",
    "ExerciseFormatError", "Exercise", "Result",
    "check", "NO_INITIAL_MESSAGE", "CORRECT_MESSAGE",
    "to_dict", "dumps", "save",
    "from_dict", "loads", "load", "load_or_error",
]
