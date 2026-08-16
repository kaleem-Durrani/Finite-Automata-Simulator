"""Command line interface.

Exit codes are the interesting part, because they are what makes this usable
from a shell script or an autograder without parsing anything:

    0   the answer is yes -- accepted, no defects found, every submission right
    1   the answer is no  -- rejected, defects found, a submission wrong
    2   the question was wrong -- bad usage, missing file, unreadable document

So ``fsa test machine.json 0110 && echo yes`` works, and a marking loop can run
``fsa check submission.json --against task.fsx`` and branch on the status.

Everything a command prints goes to stdout; everything about a *failure to run*
goes to stderr. Nothing here imports pygame, so the CLI works on a machine with
no display and no graphics stack.
"""

import argparse
import csv
import os
import sys
from typing import FrozenSet, List, NamedTuple, Optional, Sequence, TextIO, Tuple

import fsa
from fsa import analysis, exercise, language, regex, serialize
from fsa.export import FORMATS, render
from fsa.nfa import EPSILON
from fsa.symbols import StateId, Symbol

OK = 0
NO = 1
USAGE = 2

PROGRAM = "fsa"

#: How the empty word and an epsilon move are written for a person. A file
#: spells the move as JSON ``null`` and the engine spells it ``None``; neither
#: is a thing to print in a table.
EPSILON_SHOWN = "ε"

#: The spelling for a terminal that cannot encode the character. Not a
#: decoration -- see :func:`_epsilon_for`.
EPSILON_ASCII = "eps"

#: The same fallback for ``∅``, the empty language, which only
#: :func:`fsa.regex.from_automaton` ever answers with. ``{}`` is how a set with
#: nothing in it is written everywhere else in the subject, so it is one fewer
#: spelling to learn -- but it is a *description*, not a pattern this program
#: reads back. :func:`_pattern_for` is where that is said out loud.
EMPTY_LANGUAGE_ASCII = "{}"

#: The two characters :mod:`fsa.regex` reserves, each with the ASCII stand-in
#: above and the name to call it by when the stand-in has to be explained. The
#: character cannot appear in the explanation -- being unprintable here is the
#: whole reason there is one -- so the codepoint is named instead.
_SENTINELS: Tuple[Tuple[str, str, str], ...] = (
    (regex.EMPTY_WORD, EPSILON_ASCII, "U+03B5, the empty word"),
    (regex.EMPTY_LANGUAGE, EMPTY_LANGUAGE_ASCII, "U+2205, the empty language"),
)

#: The three answers the ``correct`` column of a marking run can hold. A
#: submission that would not open is neither right nor wrong: calling it "no"
#: would report a mistake about a language nobody made, and a marker who sorts
#: the column wants the unreadable files in a pile of their own to go and look
#: at rather than mixed in with the students who got it wrong.
MARK_CORRECT = "yes"
MARK_WRONG = "no"
MARK_ERROR = "error"

#: The extension a submission is expected to have -- the same documents every
#: other verb here reads.
SUBMISSION_EXTENSION = ".json"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _encodable(text: str, stream: TextIO) -> bool:
    """Whether ``stream`` can actually write ``text``.

    The question is asked of the stream rather than of the platform, so a
    redirected file, a pipe or a UTF-8 terminal is never handed a fallback it
    did not need, and only the one that genuinely cannot print a character
    gets one. A stream that declares no encoding -- a ``StringIO``, or
    anything a test drives -- is taken at its word and trusted with anything.
    """
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return True
    try:
        text.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def _epsilon_for(stream: TextIO) -> str:
    """``ε``, or an ASCII spelling where ``stream`` cannot encode it.

    A Windows console still defaults to a code page with no Greek in it, so
    printing the character raises ``UnicodeEncodeError`` partway through the
    output -- which is how ``fsa sample`` came to abort with a traceback on any
    machine that accepts the empty word.
    """
    return EPSILON_SHOWN if _encodable(EPSILON_SHOWN, stream) else EPSILON_ASCII


def _pattern_for(pattern: str, stream: TextIO) -> Tuple[str, Optional[str]]:
    """``pattern`` as ``stream`` can print it, and a note if that changed it.

    :func:`fsa.regex.from_automaton` answers ``ε`` for the language ``{""}``
    and ``∅`` for ``{}``, so this verb walks into exactly the trap
    :func:`_epsilon_for` was written for -- except that here the character is
    the *whole answer* rather than a heading on a table, and the substitute is
    not a pattern this program reads back: the grammar has no ASCII spelling of
    either language, and inventing one would be a second syntax to teach for
    the sake of one terminal. So the substitution is announced rather than made
    quietly. The announcement comes back as a second value for the caller to
    put on stderr, where whatever is reading the expression cannot mistake it
    for part of one.

    Only these two characters are substituted. A symbol the terminal cannot
    encode came out of the user's own file and is theirs to look at -- the same
    stance ``show`` and ``sample`` take. These two the tool invented.
    """
    if _encodable(pattern, stream):
        return pattern, None

    shown = pattern
    named: List[str] = []
    for character, ascii_form, description in _SENTINELS:
        if character in pattern and not _encodable(character, stream):
            shown = shown.replace(character, ascii_form)
            named.append(f"'{ascii_form}' stands for {description}")
    if not named:
        # Something else in the pattern is unprintable, which means a symbol of
        # the machine's own alphabet. Not this function's to rewrite.
        return pattern, None
    return shown, "this terminal cannot encode the expression; " + ", ".join(named)


def _load(path: str, err: TextIO) -> Optional[fsa.Document]:
    document, error = serialize.load_or_error(path)
    if document is None:
        print(f"{PROGRAM}: {path}: {error}", file=err)
    return document


def _load_dfa(path: str, err: TextIO) -> Optional[Tuple[fsa.Document, fsa.DFA]]:
    """A document and its deterministic view, or ``None`` with the reason
    already printed.

    Most verbs here run a DFA algorithm -- minimisation, completion,
    equivalence, the exporters, the simulator -- and none of them is defined on
    a machine that has a choice to make. This is the single door they all go
    through, so "this file is nondeterministic" is reported once, in one
    wording, naming the state and symbol responsible *and* the command that
    fixes it. A caller who only wanted the document uses :func:`_load`.
    """
    document = _load(path, err)
    if document is None:
        return None
    try:
        return document, document.as_dfa()
    except fsa.NondeterministicError as exc:
        print(f"{PROGRAM}: {path}: {exc}", file=err)
        print(f"{PROGRAM}: '{PROGRAM} determinize {path}' builds an equivalent "
              f"deterministic machine", file=err)
        return None


def _describe(automaton: fsa.NFA) -> str:
    return (f"{len(automaton.states)} states, "
            f"alphabet {{{', '.join(sorted(automaton.alphabet)) or 'empty'}}}, "
            f"{len(automaton.accept)} accepting")


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------

def cmd_test(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Run one word and report the verdict."""
    loaded = _load_dfa(args.file, err)
    if loaded is None:
        return USAGE

    _document, automaton = loaded
    result = fsa.run(automaton, args.word)
    if args.quiet:
        print(result.verdict.value, file=out)
    else:
        print(result.explain(), file=out)
    return OK if result.accepted else NO


def cmd_run(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Run one word and show every configuration along the way."""
    loaded = _load_dfa(args.file, err)
    if loaded is None:
        return USAGE
    _document, automaton = loaded

    result = fsa.run(automaton, args.word)
    if result.start is None:
        print(result.explain(), file=out)
        return NO

    # Wide enough for the widest state name *and* the column heading.
    width = max(max((len(s) for s in automaton.states), default=2), 5)
    print(f"{'':>4}  {'state':<{width}}  read  next", file=out)
    print(f"{'':>4}  {'-' * width}  ----  ----", file=out)
    for step in result.steps:
        print(f"{step.index:>4}  {step.source:<{width}}  {step.symbol:^4}  "
              f"{step.target}", file=out)

    final = result.final_state
    marker = "accepting" if final in automaton.accept else "not accepting"
    print(f"{'':>4}  {final:<{width}}  {'':^4}  ({marker})", file=out)
    print("", file=out)
    print(result.explain(), file=out)
    return OK if result.accepted else NO


def cmd_check(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Report structural problems with an automaton, or grade it.

    Two questions under one verb, and ``--against`` chooses which: without it,
    "is this machine well formed?"; with it, "is this machine the answer?".
    They are the same verb because they are the same act -- running the tool
    over someone's work to find out what is wrong with it -- and because the
    exit code means the same thing either way, so the marking loop that already
    says ``fsa check submission.json || retry`` keeps working when the marker
    starts pointing it at an exercise.

    The grading half lives in the Grading section below: it shares everything
    it knows with :func:`cmd_mark`, and nothing with the defect report.
    """
    if args.against is not None:
        return _graded(args, out, err)

    loaded = _load_dfa(args.file, err)
    if loaded is None:
        return USAGE

    document, automaton = loaded
    defects = analysis.defects(automaton)
    print(_describe(document.automaton), file=out)

    if not defects:
        print("no defects", file=out)
        return OK

    for defect in defects:
        print(f"  [{defect.kind}] {defect.message}", file=out)
    return NO


def cmd_sample(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """List words the automaton accepts, shortest first."""
    loaded = _load_dfa(args.file, err)
    if loaded is None:
        return USAGE

    _document, automaton = loaded
    accepted = language.sample_language(automaton, args.limit, args.max_length)
    # The empty word has to be written as something, and "" is a blank line.
    empty = _epsilon_for(out)

    if args.rejected:
        rejected = language.sample_rejected(automaton, args.limit, args.max_length)
        width = max((len(w) for w in accepted + rejected), default=1)
        for word in accepted:
            print(f"  {word or empty:<{max(width, 1)}}  accepted", file=out)
        for word in rejected:
            print(f"  {word or empty:<{max(width, 1)}}  rejected", file=out)
        return OK if accepted else NO

    for word in accepted:
        print(word or empty, file=out)

    if not accepted:
        print(f"{PROGRAM}: no accepted words up to length {args.max_length}",
              file=err)
        return NO
    return OK


def _write_document(document: "fsa.Document", path: Optional[str],
                    out: TextIO, err: TextIO) -> int:
    """Save a rebuilt document, or print it when no path was given."""
    if path is None:
        print(serialize.dumps(document), end="", file=out)
        return OK
    ok, error = serialize.save_or_error(document, path)
    if not ok:
        print(f"{PROGRAM}: {path}: {error}", file=err)
        return USAGE
    print(f"wrote {path}", file=out)
    return OK


def cmd_determinize(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Turn a nondeterministic machine into an equivalent deterministic one.

    Goes through :func:`_load` rather than :func:`_load_dfa`, because a
    nondeterministic document is exactly what it is for -- as it is for
    ``to-regex``, the other verb that has no need of a deterministic reading.
    """
    document = _load(args.file, err)
    if document is None:
        return USAGE

    automaton = document.automaton
    if automaton.initial is None:
        print(f"{PROGRAM}: no initial state, so there is no language to "
              f"preserve", file=err)
        return USAGE

    result = fsa.determinize(automaton)
    print(f"{len(automaton.states)} states -> {len(result.states)}", file=out)
    if document.is_deterministic:
        print("(it was already deterministic; the subset construction still "
              "completes delta)", file=out)

    # Subset states are new, so the drawing is new: nothing here was placed by
    # anyone.
    rebuilt = fsa.Document(result, fsa.Layout.auto(result), document.next_id)
    return _write_document(rebuilt, args.output, out, err)


def cmd_minimize(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Merge the states no word can tell apart."""
    loaded = _load_dfa(args.file, err)
    if loaded is None:
        return USAGE

    document, automaton = loaded
    if automaton.initial is None:
        print(f"{PROGRAM}: no initial state, so there is no language to "
              f"preserve", file=err)
        return USAGE

    reduced = fsa.minimize(automaton)
    print(f"{len(automaton.states)} states -> {len(reduced.states)}", file=out)
    # New states, so new coordinates: nothing here was placed by a user.
    rebuilt = fsa.Document(reduced, fsa.Layout.auto(reduced), document.next_id)
    return _write_document(rebuilt, args.output, out, err)


def cmd_complete(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Make delta total by routing undefined pairs to a trap."""
    loaded = _load_dfa(args.file, err)
    if loaded is None:
        return USAGE

    document, automaton = loaded
    missing = len(analysis.missing_transitions(automaton))
    completed, trap = document.complete()
    if trap is None:
        print("already complete", file=out)
        return OK

    print(f"added {trap} and routed {missing} missing "
          f"transition{'s' if missing != 1 else ''} to it", file=out)
    return _write_document(completed, args.output, out, err)


def cmd_equiv(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Say whether two automata recognise the same language.

    Exit 0 when they agree and 1 when they do not, so a marking script can
    branch on the status without parsing anything -- and when they disagree,
    print the shortest word that proves it. That word is the whole point: it
    turns "wrong" into "wrong on this input".
    """
    loaded_left = _load_dfa(args.left, err)
    loaded_right = _load_dfa(args.right, err)
    if loaded_left is None or loaded_right is None:
        return USAGE
    left, right = loaded_left[1], loaded_right[1]

    witness = fsa.counterexample(left, right)
    if witness is None:
        print("equivalent", file=out)
        return OK

    print(f"differ on {witness or 'the empty string'}", file=out)
    for name, automaton in ((args.left, left), (args.right, right)):
        verdict = "accepts" if fsa.accepts(automaton, witness) else "rejects"
        print(f"  {name} {verdict} it", file=out)
    return NO


# ----------------------------------------------------------------------
# Grading
#
# ``check --against`` and ``mark`` are one feature at two scales: a student
# asking about their own machine, and a marker asking about a class of them.
# Both are :func:`fsa.exercise.check`, whose answer carries the shortest word
# that tells the submission and the reference apart. That word is the feature.
# Being told a machine is wrong teaches nothing; being told it accepts 'bb'
# when the answer rejects it points at an arrow.
#
# So the CLI's job here is small and it is mostly about not losing things:
# print the sentence :mod:`fsa.exercise` already wrote, keep the exit code
# meaning what it means everywhere else, and make sure one unreadable file in a
# directory of twenty costs one row rather than nineteen marks.
# ----------------------------------------------------------------------

def _load_exercise(path: str, err: TextIO) -> Optional[exercise.Exercise]:
    """An exercise, or ``None`` with the reason already printed.

    The :func:`_load` of the other format. ``load_or_error`` catches the regex
    parser's error too -- it is an :class:`~fsa.errors.AutomatonError` -- so a
    typo in a reference pattern arrives here as a sentence rather than a
    traceback out of the middle of a marking run.
    """
    task, error = exercise.load_or_error(path)
    if task is None:
        print(f"{PROGRAM}: {path}: {error}", file=err)
    return task


def _graded(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """``fsa check submission.json --against task.fsx``.

    Reads the submission with :func:`_load` rather than :func:`_load_dfa`,
    because a student may legitimately hand in an NFA: the exercise asked for a
    machine recognising a language, and :func:`fsa.exercise.check` determinises
    whatever it is given. Refusing it here would fail an answer that is right.

    A submission with no initial state exits 1, not 2. The file opened and the
    question was asked; the answer is that it is not the reference. That is a
    wrong answer with a remedy printed, not a broken invocation.
    """
    task = _load_exercise(args.against, err)
    if task is None:
        return USAGE
    document = _load(args.file, err)
    if document is None:
        return USAGE

    result = exercise.check(document.automaton, task)
    # Which exercise was used, then the verdict. The name first because a
    # marking loop over several tasks prints many of these and "correct" with
    # nothing above it says nothing about what was correct.
    print(task.name(), file=out)
    # Printed exactly as `fsa.exercise` wrote it. That module spells the empty
    # word "the empty word" -- the commonest counterexample there is, and the
    # one whose obvious spelling ('') reads like a bug in the marker -- so
    # there is no epsilon to fall back from here. See `_cell_word` for the
    # table, where the same word is a cell rather than a sentence.
    print(result.message, file=out)
    return OK if result.correct else NO


class Row(NamedTuple):
    """One marked submission: one line of the CSV, one line of the table.

    The field names are the CSV header, so the columns cannot drift out of step
    with what is written under them.

    ``counterexample`` holds the word verbatim, which means the empty word --
    again, the commonest one there is -- lands in the same empty cell as "there
    was no counterexample". ``attempt`` is what tells them apart: it is
    ``accepts`` or ``rejects`` when there is a word and empty when there is
    not, which is exactly the invariant :class:`fsa.exercise.Result` promises
    about ``attempt_accepts``. It earns its place twice over, because it is
    also the column a marker scans to see whether a cohort is over-accepting or
    under-accepting.
    """

    submission: str
    exercise: str
    correct: str
    counterexample: str
    attempt: str
    message: str


def _files(root: str, extension: str) -> List[str]:
    """Every file at or under ``root`` with this extension, sorted.

    A single file is a legal root, so ``fsa mark task.fsx submissions/`` works
    without a directory holding one exercise. The walk is recursive because
    ``submissions/<exercise>/<student>.json`` is a layout a marker will already
    have -- and it is the layout :func:`_matching` reads to decide which task a
    submission answers. Sorted, so two runs over one directory produce byte-
    identical CSVs and a marker can diff a re-run.
    """
    if os.path.isfile(root):
        return [root]
    found: List[str] = []
    for directory, _subdirectories, names in os.walk(root):
        found.extend(os.path.join(directory, name) for name in names
                     if name.lower().endswith(extension))
    return sorted(found)


def _grading_files(root: str, extension: str, noun: str,
                   err: TextIO) -> Optional[List[str]]:
    """What :func:`_files` found, or ``None`` with the reason printed.

    Two failures, told apart because they are fixed differently: a path that is
    not there at all is a typo on the command line, and a path that is there
    with nothing of this kind under it is a marker pointing at the wrong
    directory -- or at a set of submissions saved under some other extension.
    """
    if not os.path.exists(root):
        print(f"{PROGRAM}: {root}: no such file or directory", file=err)
        return None
    found = _files(root, extension)
    if not found:
        print(f"{PROGRAM}: {root}: no {extension} {noun} here", file=err)
        return None
    return found


def _stem(path: str) -> str:
    """``exercises/even_as.fsx`` -> ``even_as``, the name an exercise goes by."""
    return os.path.splitext(os.path.basename(path))[0]


def _shown(path: str, root: str) -> str:
    """How a submission is named in the results: its path below ``root``.

    Relative, so a marker's home directory does not end up in a file they send
    on, and with forward slashes on every platform, so the CSV a Windows marker
    produces and the one CI produces are the same bytes.
    """
    return os.path.relpath(path, root or ".").replace(os.sep, "/")


def _matching(shown: str, tasks: Sequence[Tuple[str, exercise.Exercise]],
              ) -> Optional[Tuple[str, exercise.Exercise]]:
    """The exercise a submission answers, by name, or ``None``.

    One exercise is the common case -- one task, twenty students -- and then
    there is nothing to decide. With several, the pairing is by name: the
    exercise's stem has to appear somewhere in the submission's path, so
    ``submissions/even_as/alice.json`` and ``submissions/alice.even_as.json``
    both find ``exercises/even_as.fsx``. The longest match wins, so an exercise
    called ``even`` cannot steal a submission from ``even_as``.

    A submission that matches nothing gets a row saying so rather than a mark.
    Guessing would hand a student a counterexample to a question they were not
    answering, which is worse than no feedback: it is confident feedback about
    the wrong thing.
    """
    if len(tasks) == 1:
        return tasks[0]

    haystack = shown.lower()
    best: Optional[Tuple[str, exercise.Exercise]] = None
    for stem, task in tasks:
        if stem.lower() in haystack and (best is None or len(stem) > len(best[0])):
            best = (stem, task)
    return best


def _mark_one(path: str, shown: str,
              tasks: Sequence[Tuple[str, exercise.Exercise]],
              err: TextIO) -> Row:
    """Mark one submission. Returns a row for every outcome, including failure.

    This is the function that makes ``mark`` usable on real submissions: a file
    that will not open is one row saying so, not an exception that abandons the
    other nineteen students. Everything that can go wrong with someone else's
    file -- missing, truncated, saved from the wrong program, an automaton the
    engine refuses -- comes back as a row and a line on stderr.
    """
    pairing = _matching(shown, tasks)
    if pairing is None:
        reason = ("no exercise's name appears in this submission's path, so "
                  "there is nothing to mark it against")
        print(f"{PROGRAM}: {shown}: {reason}", file=err)
        return Row(shown, "", MARK_ERROR, "", "", reason)
    stem, task = pairing

    document, error = serialize.load_or_error(path)
    if document is None:
        print(f"{PROGRAM}: {shown}: {error}", file=err)
        return Row(shown, stem, MARK_ERROR, "", "", error)

    try:
        result = exercise.check(document.automaton, task)
    except fsa.AutomatonError as exc:
        # The engine raises this for anything it refuses, and the whole point
        # of the verb is that one hostile file costs one row.
        print(f"{PROGRAM}: {shown}: {exc}", file=err)
        return Row(shown, stem, MARK_ERROR, "", "", str(exc))

    return Row(
        submission=shown,
        exercise=stem,
        correct=MARK_CORRECT if result.correct else MARK_WRONG,
        # Verbatim, including "" for the empty word; `attempt` is what says
        # which of the two blanks this is. See :class:`Row`.
        counterexample=result.counterexample or "",
        attempt=("" if result.attempt_accepts is None else
                 "accepts" if result.attempt_accepts else "rejects"),
        message=result.message,
    )


def _cell_word(row: Row, empty: str) -> str:
    """The counterexample as a table cell rather than as a sentence.

    ``ε`` here, where :mod:`fsa.exercise` writes "the empty word": in a column
    four characters wide a sentence does not fit, and this is the same spelling
    ``sample`` and ``show`` already use for the same thing -- with the same
    ASCII fallback, since the terminal that cannot encode it is the one this
    project runs on.

    Keyed off ``attempt``, not off the word, because a blank word means either
    "the empty word" or "no word at all" and only one of those is a cell to
    fill in.
    """
    if not row.attempt:
        return ""
    return row.counterexample or empty


def _totals(rows: Sequence[Row]) -> str:
    """The line under the table: how many, and how they went.

    The third bucket is named only when it is not empty. A marker whose files
    all opened should not have to read the word "unmarkable" to find that out.
    """
    correct = sum(1 for row in rows if row.correct == MARK_CORRECT)
    wrong = sum(1 for row in rows if row.correct == MARK_WRONG)
    failed = sum(1 for row in rows if row.correct == MARK_ERROR)

    line = (f"{len(rows)} submission{'' if len(rows) == 1 else 's'}: "
            f"{correct} correct, {wrong} wrong")
    if failed:
        line += f", {failed} could not be marked"
    return line


#: The columns of the printed summary. Not every column of the CSV: the message
#: is a sentence and would wrap a terminal into uselessness, and the marker who
#: wants it opens the file this verb just wrote.
SUMMARY_COLUMNS = ("submission", "exercise", "correct", "counterexample")


def _summary_cells(rows: Sequence[Row], out: TextIO) -> List[Tuple[str, ...]]:
    """The table's body, as text, however it is about to be drawn."""
    empty = _epsilon_for(out)
    return [(row.submission, row.exercise, row.correct, _cell_word(row, empty))
            for row in rows]


def _rich_summary(rows: Sequence[Row], out: TextIO) -> bool:
    """Draw the summary with ``rich``, or answer ``False`` having drawn nothing.

    Optional, and the import is the capability check. ``rich`` is not a
    dependency of this project and is not installed on the machine it is
    developed on, so :func:`_plain_summary` is the path that actually runs and
    the path the tests exercise; this one is a nicety for a terminal that has
    it. Nothing below the import may fail, or a missing library would cost half
    a table instead of a plainer one.
    """
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        return False

    table = Table(box=None, pad_edge=False)
    for heading in SUMMARY_COLUMNS:
        table.add_column(heading, overflow="fold")
    for cells in _summary_cells(rows, out):
        table.add_row(*cells)
    Console(file=out).print(table)
    return True


def _plain_summary(rows: Sequence[Row], out: TextIO) -> None:
    """The same table in columns, for a terminal without ``rich``.

    Two spaces between columns and an ASCII rule under the headings, the same
    shape ``run`` and ``show`` print, so the tool looks like one tool. Widths
    come from the content, so a directory of short names does not get a table
    padded out to the width of a path nobody has.
    """
    cells = _summary_cells(rows, out)
    widths = [max(len(heading),
                  max((len(row[column]) for row in cells), default=0))
              for column, heading in enumerate(SUMMARY_COLUMNS)]

    def line(values: Sequence[str]) -> str:
        # Right-stripped: a trailing run of spaces after the last column is
        # invisible until someone diffs the output or pastes it somewhere.
        return "  ".join(value.ljust(width)
                         for value, width in zip(values, widths)).rstrip()

    print(line(SUMMARY_COLUMNS), file=out)
    print(line(["-" * width for width in widths]), file=out)
    for row in cells:
        print(line(row), file=out)


def _write_csv(rows: Sequence[Row], path: str, out: TextIO, err: TextIO) -> bool:
    """Write the marking rows where a marker can open them.

    ``newline=""`` is required by the :mod:`csv` module, which writes its own
    line endings: without it every row on Windows ends ``\\r\\r\\n``. UTF-8
    because a symbol is any character a student's alphabet contains, and the
    counterexample column is made of those.
    """
    try:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(Row._fields)
            writer.writerows(rows)
    except OSError as exc:
        print(f"{PROGRAM}: {path}: {exc.strerror or exc}", file=err)
        return False
    print(f"wrote {path}", file=out)
    return True


def cmd_mark(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Grade a directory of submissions against one or more exercises.

    Exit 0 when every submission is correct, 1 when any is wrong or would not
    open, and 2 when the run could not start at all -- no exercise, no
    submission, or a CSV that could not be written. So ``fsa mark exercises/
    submissions/ && echo all passed`` says something true, and an unreadable
    file is a "no" rather than a "could not run": the run happened, and the
    answer for that student is not yet correct.

    The table goes to stdout before the CSV is written, so a marker whose
    output path was wrong still has the results in front of them.
    """
    exercise_paths = _grading_files(args.exercises, exercise.EXTENSION,
                                    "exercise", err)
    if exercise_paths is None:
        return USAGE

    # A broken exercise is reported and skipped rather than fatal, for the
    # reason a broken submission is: the other tasks are still markable.
    tasks: List[Tuple[str, exercise.Exercise]] = []
    for path in exercise_paths:
        task = _load_exercise(path, err)
        if task is not None:
            tasks.append((_stem(path), task))
    if not tasks:
        print(f"{PROGRAM}: no exercise loaded, so there is nothing to mark "
              f"against", file=err)
        return USAGE

    submissions = _grading_files(args.submissions, SUBMISSION_EXTENSION,
                                 "submission", err)
    if submissions is None:
        return USAGE

    root = (args.submissions if os.path.isdir(args.submissions)
            else os.path.dirname(args.submissions))
    rows = [_mark_one(path, _shown(path, root), tasks, err)
            for path in submissions]

    if not _rich_summary(rows, out):
        _plain_summary(rows, out)
    print("", file=out)
    print(_totals(rows), file=out)

    if args.output is not None and not _write_csv(rows, args.output, out, err):
        return USAGE
    return OK if all(row.correct == MARK_CORRECT for row in rows) else NO


def cmd_from_regex(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Build the machine a regular expression denotes.

    Half of Kleene's theorem with a file on the end of it. What comes out is a
    document like any other, so the next thing done to it can be ``fsa show``,
    ``fsa determinize`` or the editor.

    Thompson's machine is written out as it comes, epsilon moves and all,
    rather than determinized on the way past. The shape of that machine mirrors
    the shape of the expression -- which is the thing worth looking at, and the
    reason the construction is taught -- and ``fsa determinize`` already exists
    for the reader who wants it small. Doing it here would also hide which step
    cost what, on the one verb where the exponential step is easiest to meet.
    """
    try:
        automaton = regex.to_nfa(args.pattern)
    except regex.RegexSyntaxError as exc:
        # The parser's message names the character and its index; the caret
        # puts a mark under that index. Both go to stderr, so a caller who
        # redirected stdout to a file still finds out why it is empty.
        print(f"{PROGRAM}: {exc}", file=err)
        print(exc.caret(), file=err)
        return USAGE

    # Every state here was invented by the construction seconds ago, so there
    # is no hand-placed coordinate to preserve and nothing to lay out around.
    document = fsa.Document.of(automaton, fsa.Layout.auto(automaton))
    print(_describe(automaton), file=out)
    if not document.is_deterministic:
        # The same courtesy _load_dfa pays: name the command that unblocks the
        # verb the user is about to reach for. Every pattern with an operator
        # in it lands here, so being told once, now, beats being refused later.
        print(f"nondeterministic, as Thompson's construction leaves it; "
              f"'{PROGRAM} determinize' builds the machine the other verbs "
              f"can run", file=out)
    return _write_document(document, args.output, out, err)


def cmd_to_regex(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Print a regular expression denoting the machine's language.

    The other half of Kleene's theorem, and like ``determinize`` it reads its
    file through :func:`_load` rather than :func:`_load_dfa`: state elimination
    never asks whether a move has a choice to make, so it is defined on a
    nondeterministic document as it stands -- which is exactly what the machine
    ``fsa from-regex`` just wrote is. Refusing that one would send someone
    through a determinisation that cannot change the answer.

    Stdout is the expression and nothing else, so ``pattern=$(fsa to-regex
    m.json)`` works. Anything to say *about* the expression is said on stderr.
    """
    document = _load(args.file, err)
    if document is None:
        return USAGE

    automaton = document.automaton
    if automaton.initial is None:
        # from_automaton answers ∅ here, and repeating that would be a lie the
        # engine is careful not to tell: a machine with no initial state has no
        # language *yet*, which is a different thing from the language with no
        # words in it. The engine cannot keep the distinction -- an expression
        # denotes a language and "undefined" is not one -- so the CLI keeps it.
        print(f"{PROGRAM}: no initial state, so there is no language to "
              f"describe", file=err)
        return USAGE

    # The expression tracks the machine it was given, so a smaller machine is
    # the road to a shorter expression -- usually. Measured, not assumed, and
    # it does not always hold: on `(a|b)*abb` the minimal four-state DFA
    # eliminates to 31 characters where Thompson's fourteen-state tree gives
    # 13, because every loop costs a star and a tree has none. That is one
    # reason this is a flag: it offers a different description of the same
    # language, and which of the two reads better is not a question the verb
    # can answer on the user's behalf. The other is that the subset
    # construction is exponential in the worst case, and a verb that usually
    # answers at once and occasionally takes a minute is one nobody puts in a
    # loop.
    machine: fsa.AnyAutomaton = (
        fsa.minimize(fsa.determinize(automaton)) if args.minimize else automaton)

    pattern, note = _pattern_for(regex.from_automaton(machine), out)
    if note is not None:
        print(f"{PROGRAM}: {note}", file=err)
    print(pattern, file=out)
    return OK


def cmd_export(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Write the automaton as a diagram."""
    loaded = _load_dfa(args.file, err)
    if loaded is None:
        return USAGE

    document, automaton = loaded
    try:
        text = render(automaton, document.layout, args.format)
    except ValueError as exc:
        print(f"{PROGRAM}: {exc}", file=err)
        return USAGE

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(text)
        except OSError as exc:
            print(f"{PROGRAM}: {args.output}: {exc.strerror or exc}", file=err)
            return USAGE
        print(f"wrote {args.output}", file=out)
    else:
        out.write(text)
    return OK


def cmd_new(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Create an empty document."""
    document = fsa.Document()
    for symbol in args.alphabet:
        try:
            document = document.add_symbol(symbol)
        except fsa.IllegalSymbolError as exc:
            print(f"{PROGRAM}: {exc}", file=err)
            return USAGE

    if os.path.exists(args.file) and not args.force:
        print(f"{PROGRAM}: {args.file} exists; pass --force to overwrite",
              file=err)
        return USAGE

    ok, error = serialize.save_or_error(document, args.file)
    if not ok:
        print(f"{PROGRAM}: {args.file}: {error}", file=err)
        return USAGE
    print(f"wrote {args.file}", file=out)
    return OK


def _cell(targets: FrozenSet[StateId]) -> str:
    """One cell of the table :func:`cmd_show` prints.

    A set in braces when a move branches, which is the same spelling
    :func:`fsa.subset.subset_name` gives the DFA state that branch becomes --
    so the table and the determinized machine read as one story. A singleton
    keeps no braces here, because in this table it is a cell rather than a
    state name and there is nothing to tell it apart from.
    """
    if not targets:
        return "-"
    if len(targets) == 1:
        return next(iter(targets))
    return "{" + ",".join(sorted(targets)) + "}"


def cmd_show(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Print the transition table.

    Reads the document's machine as it is, nondeterminism and all, rather than
    going through :func:`_load_dfa`: showing someone what they have is exactly
    what they need when the answer is "more targets than you expected".
    """
    document = _load(args.file, err)
    if document is None:
        return USAGE

    automaton = document.automaton
    states = sorted(automaton.states)
    if not states:
        print("(empty automaton)", file=out)
        return OK

    # An epsilon column only when the machine has an epsilon move to put in it.
    # A column of dashes on every deterministic file would be a permanent
    # reminder of a feature that file does not use.
    columns: List[Optional[Symbol]] = [s for s in sorted(automaton.alphabet)]
    if any(symbol is EPSILON for _source, symbol, _t in automaton.sorted_transitions()):
        columns.insert(0, EPSILON)

    cells = {(state, symbol): _cell(automaton.targets(state, symbol))
             for state in states for symbol in columns}
    width = max(max(len(s) for s in states) + 2,
                max((len(text) + 2 for text in cells.values()), default=0), 6)

    epsilon = _epsilon_for(out)
    header = "".join(
        f"{epsilon if symbol is EPSILON else symbol:^{width}}"
        for symbol in columns)
    print(f"{'':<{width}}{header}", file=out)

    for state in states:
        marks = ("->" if state == automaton.initial else "  ")
        marks += ("*" if state in automaton.accept else " ")
        row = "".join(f"{cells[(state, symbol)]:^{width}}" for symbol in columns)
        print(f"{marks}{state:<{width - 3}}{row}", file=out)

    # The legend grows only for what the table actually shows. Nondeterminism
    # is spelled out beside the other facts, not warned about: a second target
    # on one symbol is a legal thing to draw, and this table is where someone
    # comes to see it.
    legend = ["-> initial", "* accepting", "- undefined"]
    if any(text.startswith("{") for text in cells.values()):
        legend.append("{p,q} a choice")
    if EPSILON in columns:
        legend.append(f"{epsilon} reads nothing")

    print("", file=out)
    print(", ".join(legend), file=out)
    return OK


def cmd_gui(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Launch the graphical editor, if it is available.

    Looked up at run time rather than imported. The engine must not depend on
    the application -- that is the whole point of the boundary, and a static
    import here would reverse it. The editor also is not part of the installed
    package, so on a plain ``pip install`` there is nothing to find, and saying
    so is more useful than a traceback.
    """
    del args, out
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    entry = os.path.join(root, "main.py")
    if not os.path.isfile(entry):
        print(f"{PROGRAM}: the editor is not installed with this package; "
              f"run 'python main.py' from a checkout", file=err)
        return USAGE

    spec = importlib.util.spec_from_file_location("_fsa_gui", entry)
    if spec is None or spec.loader is None:
        print(f"{PROGRAM}: could not load {entry}", file=err)
        return USAGE

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        print(f"{PROGRAM}: the editor needs pygame: {exc}", file=err)
        return USAGE

    module.AutomatonSimulator().run()
    return OK


# ----------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Build, run and export finite automata.",
        epilog="Exit codes: 0 yes, 1 no, 2 could not run.",
    )
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {serialize.VERSION}.0")
    subs = parser.add_subparsers(dest="command", metavar="command")

    test = subs.add_parser("test", help="run a word and report the verdict")
    test.add_argument("file")
    test.add_argument("word")
    test.add_argument("-q", "--quiet", action="store_true",
                      help="print the verdict alone, for scripting")
    test.set_defaults(handler=cmd_test)

    run_cmd = subs.add_parser("run", help="run a word and show every step")
    run_cmd.add_argument("file")
    run_cmd.add_argument("word")
    run_cmd.set_defaults(handler=cmd_run)

    check = subs.add_parser("check", help="report structural problems, or grade")
    check.add_argument("file")
    check.add_argument("--against", metavar="EXERCISE",
                       help=f"grade against an exercise ({exercise.EXTENSION}) "
                            f"instead: exit 0 when the language is right, 1 "
                            f"when it is not, and say which word proves it")
    check.set_defaults(handler=cmd_check)

    sample = subs.add_parser("sample", help="list accepted words")
    sample.add_argument("file")
    sample.add_argument("-n", "--limit", type=int, default=10)
    sample.add_argument("--max-length", type=int,
                        default=language.DEFAULT_MAX_LENGTH)
    sample.add_argument("--rejected", action="store_true",
                        help="also list rejected words, labelled")
    sample.set_defaults(handler=cmd_sample)

    determinize = subs.add_parser(
        "determinize", help="subset construction: NFA to an equivalent DFA")
    determinize.add_argument("file")
    determinize.add_argument("-o", "--output",
                             help="write here instead of printing the document")
    determinize.set_defaults(handler=cmd_determinize)

    minimize = subs.add_parser("minimize", help="merge indistinguishable states")
    minimize.add_argument("file")
    minimize.add_argument("-o", "--output",
                          help="write here instead of printing the document")
    minimize.set_defaults(handler=cmd_minimize)

    complete = subs.add_parser("complete", help="route undefined pairs to a trap")
    complete.add_argument("file")
    complete.add_argument("-o", "--output",
                          help="write here instead of printing the document")
    complete.set_defaults(handler=cmd_complete)

    equiv = subs.add_parser("equiv", help="compare two automata")
    equiv.add_argument("left")
    equiv.add_argument("right")
    equiv.set_defaults(handler=cmd_equiv)

    mark = subs.add_parser(
        "mark", help="grade a directory of submissions against exercises")
    mark.add_argument("exercises",
                      help=f"an exercise ({exercise.EXTENSION}) or a directory "
                           f"of them")
    mark.add_argument("submissions",
                      help=f"a submission ({SUBMISSION_EXTENSION}) or a "
                           f"directory of them, searched recursively")
    mark.add_argument("-o", "--output", metavar="RESULTS.CSV",
                      help="write the rows here as CSV as well as summarising "
                           "them on stdout")
    mark.set_defaults(handler=cmd_mark)

    from_regex = subs.add_parser(
        "from-regex", help="build the machine a regular expression denotes")
    from_regex.add_argument("pattern")
    from_regex.add_argument("-o", "--output",
                            help="write here instead of printing the document")
    from_regex.set_defaults(handler=cmd_from_regex)

    to_regex = subs.add_parser(
        "to-regex", help="print a regular expression for the language")
    to_regex.add_argument("file")
    to_regex.add_argument("--minimize", action="store_true",
                          help="describe the minimised machine instead: the "
                               "same language, usually shorter")
    to_regex.set_defaults(handler=cmd_to_regex)

    export = subs.add_parser("export", help="write a diagram")
    export.add_argument("file")
    export.add_argument("-f", "--format", choices=sorted(FORMATS), default="svg")
    export.add_argument("-o", "--output", help="write here instead of stdout")
    export.set_defaults(handler=cmd_export)

    show = subs.add_parser("show", help="print the transition table")
    show.add_argument("file")
    show.set_defaults(handler=cmd_show)

    new = subs.add_parser("new", help="create an empty automaton")
    new.add_argument("file")
    new.add_argument("-a", "--alphabet", nargs="*", default=["a", "b"])
    new.add_argument("--force", action="store_true")
    new.set_defaults(handler=cmd_new)

    gui = subs.add_parser("gui", help="launch the graphical editor")
    gui.set_defaults(handler=cmd_gui)

    return parser


def main(argv: Optional[Sequence[str]] = None,
         out: Optional[TextIO] = None,
         err: Optional[TextIO] = None) -> int:
    """Run the CLI. Returns the exit code rather than calling sys.exit.

    The streams are parameters so tests can drive it without capturing global
    state, and so it can be embedded.
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    out = out or sys.stdout
    err = err or sys.stderr

    if not getattr(args, "handler", None):
        parser.print_help(out)
        return USAGE

    try:
        return int(args.handler(args, out, err))
    except BrokenPipeError:           # `fsa export ... | head`
        return OK
    except KeyboardInterrupt:
        print("", file=err)
        return USAGE


def run_main() -> None:
    """Console-script entry point."""
    sys.exit(main())


__all__: List[str] = ["main", "run_main", "build_parser", "OK", "NO", "USAGE"]


if __name__ == "__main__":          # python -m fsa.cli
    run_main()
