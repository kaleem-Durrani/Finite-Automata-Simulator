"""Command line interface.

Exit codes are the interesting part, because they are what makes this usable
from a shell script or an autograder without parsing anything:

    0   the answer is yes -- accepted, or no defects found
    1   the answer is no  -- rejected, or defects found
    2   the question was wrong -- bad usage, missing file, unreadable document

So ``fsa test machine.json 0110 && echo yes`` works, and a marking loop can run
``fsa check submission.json`` and branch on the status.

Everything a command prints goes to stdout; everything about a *failure to run*
goes to stderr. Nothing here imports pygame, so the CLI works on a machine with
no display and no graphics stack.
"""

import argparse
import os
import sys
from typing import FrozenSet, List, Optional, Sequence, TextIO, Tuple

import fsa
from fsa import analysis, language, regex, serialize
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
    """Report structural problems with an automaton."""
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

    check = subs.add_parser("check", help="report structural problems")
    check.add_argument("file")
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
