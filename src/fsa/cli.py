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
from fsa import analysis, language, serialize
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


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _epsilon_for(stream: TextIO) -> str:
    """``ε``, or an ASCII spelling where ``stream`` cannot encode it.

    A Windows console still defaults to a code page with no Greek in it, so
    printing the character raises ``UnicodeEncodeError`` partway through the
    output -- which is how ``fsa sample`` came to abort with a traceback on any
    machine that accepts the empty word. The question is asked of the stream
    rather than of the platform, so a redirected file, a pipe or a UTF-8
    terminal still gets the real character, and only the one that genuinely
    cannot print it gets the fallback.
    """
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return EPSILON_SHOWN
    try:
        EPSILON_SHOWN.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return EPSILON_ASCII
    return EPSILON_SHOWN


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

    The one verb that goes through :func:`_load` rather than :func:`_load_dfa`,
    because a nondeterministic document is exactly what it is for.
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
