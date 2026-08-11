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
from typing import List, Optional, Sequence, TextIO

import fsa
from fsa import analysis, language, serialize
from fsa.export import FORMATS, render

OK = 0
NO = 1
USAGE = 2

PROGRAM = "fsa"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _load(path: str, err: TextIO) -> Optional[fsa.Document]:
    document, error = serialize.load_or_error(path)
    if document is None:
        print(f"{PROGRAM}: {path}: {error}", file=err)
    return document


def _describe(automaton: fsa.DFA) -> str:
    return (f"{len(automaton.states)} states, "
            f"alphabet {{{', '.join(sorted(automaton.alphabet)) or 'empty'}}}, "
            f"{len(automaton.accept)} accepting")


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------

def cmd_test(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Run one word and report the verdict."""
    document = _load(args.file, err)
    if document is None:
        return USAGE

    result = fsa.run(document.automaton, args.word)
    if args.quiet:
        print(result.verdict.value, file=out)
    else:
        print(result.explain(), file=out)
    return OK if result.accepted else NO


def cmd_run(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Run one word and show every configuration along the way."""
    document = _load(args.file, err)
    if document is None:
        return USAGE

    result = fsa.run(document.automaton, args.word)
    if result.start is None:
        print(result.explain(), file=out)
        return NO

    # Wide enough for the widest state name *and* the column heading.
    width = max(max((len(s) for s in document.automaton.states), default=2), 5)
    print(f"{'':>4}  {'state':<{width}}  read  next", file=out)
    print(f"{'':>4}  {'-' * width}  ----  ----", file=out)
    for step in result.steps:
        print(f"{step.index:>4}  {step.source:<{width}}  {step.symbol:^4}  "
              f"{step.target}", file=out)

    final = result.final_state
    marker = "accepting" if final in document.automaton.accept else "not accepting"
    print(f"{'':>4}  {final:<{width}}  {'':^4}  ({marker})", file=out)
    print("", file=out)
    print(result.explain(), file=out)
    return OK if result.accepted else NO


def cmd_check(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Report structural problems with an automaton."""
    document = _load(args.file, err)
    if document is None:
        return USAGE

    automaton = document.automaton
    defects = analysis.defects(automaton)
    print(_describe(automaton), file=out)

    if not defects:
        print("no defects", file=out)
        return OK

    for defect in defects:
        print(f"  [{defect.kind}] {defect.message}", file=out)
    return NO


def cmd_sample(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """List words the automaton accepts, shortest first."""
    document = _load(args.file, err)
    if document is None:
        return USAGE

    automaton = document.automaton
    accepted = language.sample_language(automaton, args.limit, args.max_length)

    if args.rejected:
        rejected = language.sample_rejected(automaton, args.limit, args.max_length)
        width = max((len(w) for w in accepted + rejected), default=1)
        for word in accepted:
            print(f"  {word or 'ε':<{max(width, 1)}}  accepted", file=out)
        for word in rejected:
            print(f"  {word or 'ε':<{max(width, 1)}}  rejected", file=out)
        return OK if accepted else NO

    for word in accepted:
        print(word or "ε", file=out)

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


def cmd_minimize(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Merge the states no word can tell apart."""
    document = _load(args.file, err)
    if document is None:
        return USAGE

    automaton = document.automaton
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
    document = _load(args.file, err)
    if document is None:
        return USAGE

    missing = len(analysis.missing_transitions(document.automaton))
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
    left = _load(args.left, err)
    right = _load(args.right, err)
    if left is None or right is None:
        return USAGE

    witness = fsa.counterexample(left.automaton, right.automaton)
    if witness is None:
        print("equivalent", file=out)
        return OK

    print(f"differ on {witness or 'the empty string'}", file=out)
    for name, document in ((args.left, left), (args.right, right)):
        verdict = "accepts" if fsa.accepts(document.automaton, witness) else "rejects"
        print(f"  {name} {verdict} it", file=out)
    return NO


def cmd_export(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Write the automaton as a diagram."""
    document = _load(args.file, err)
    if document is None:
        return USAGE

    try:
        text = render(document.automaton, document.layout, args.format)
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


def cmd_show(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Print the transition table."""
    document = _load(args.file, err)
    if document is None:
        return USAGE

    automaton = document.automaton
    states = sorted(automaton.states)
    alphabet = sorted(automaton.alphabet)
    if not states:
        print("(empty automaton)", file=out)
        return OK

    width = max(max(len(s) for s in states) + 2, 6)
    header = "".join(f"{symbol:^{width}}" for symbol in alphabet)
    print(f"{'':<{width}}{header}", file=out)

    for state in states:
        marks = ("->" if state == automaton.initial else "  ")
        marks += ("*" if state in automaton.accept else " ")
        row = "".join(
            f"{automaton.target(state, symbol) or '-':^{width}}"
            for symbol in alphabet)
        print(f"{marks}{state:<{width - 3}}{row}", file=out)

    print("", file=out)
    print("-> initial, * accepting, - undefined", file=out)
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
