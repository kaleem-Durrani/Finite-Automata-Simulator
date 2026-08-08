"""The README must not lie.

Its examples were hand-written once, and the verdict was inverted on all three
of them -- for months, in the file that tells people what the project does. So
the table is generated from the tool now, and this regenerates it and compares.

If this fails, run it with ``--update`` semantics by copying the expected block
it prints into the README. The point is that drift is loud.
"""

import re
from pathlib import Path

import pytest

from fsa import language

README = Path(__file__).resolve().parent.parent / "README.md"
BLOCK = re.compile(r"<!-- generated: fsa sample -->\n(.*?)<!-- /generated -->",
                   re.DOTALL)


def demo():
    import main
    return main.demo_document()


def expected_table(rows: int = 6) -> str:
    """Build the table the README should contain, from the tool itself."""
    automaton = demo().automaton
    accepted = language.sample_language(automaton, rows)
    rejected = language.sample_rejected(automaton, rows)

    def cell(word: str) -> str:
        return f"`{word}`" if word else "*(empty)*"

    lines = ["| Accepted | Rejected |", "|---|---|"]
    for index in range(rows):
        left = cell(accepted[index]) if index < len(accepted) else ""
        right = cell(rejected[index]) if index < len(rejected) else ""
        lines.append(f"| {left} | {right} |")
    return "\n".join(lines) + "\n"


def test_the_readme_has_a_generated_block():
    match = BLOCK.search(README.read_text(encoding="utf-8"))
    assert match, "the generated example block is missing"


def test_the_readme_examples_match_the_tool():
    """Generated, not written. This is what stops them being wrong."""
    match = BLOCK.search(README.read_text(encoding="utf-8"))
    assert match is not None
    assert match.group(1) == expected_table(), (
        "README examples are stale. Replace the generated block with:\n\n"
        + expected_table())


def test_the_demo_language_is_what_the_readme_claims():
    """The prose says a*b+. Check that against the machine, not against itself."""
    automaton = demo().automaton
    import fsa

    for word in ("b", "ab", "aab", "bbb", "abb"):
        assert fsa.accepts(automaton, word), f"{word} should be accepted"
    for word in ("", "a", "aa", "ba", "aba"):
        assert not fsa.accepts(automaton, word), f"{word} should be rejected"


@pytest.mark.parametrize("command", [
    "fsa test", "fsa run", "fsa check", "fsa show",
    "fsa sample", "fsa export", "fsa new", "fsa gui",
])
def test_every_documented_command_exists(command):
    """The README used to advertise a shortcut that did not exist."""
    from fsa.cli import build_parser

    name = command.split()[1]
    parser = build_parser()
    subparsers = [action for action in parser._actions
                  if hasattr(action, "choices") and action.choices]
    assert subparsers, "no subcommands registered"
    assert name in subparsers[0].choices, f"{command} is documented but missing"


def test_documented_export_formats_are_real():
    from fsa.export import FORMATS

    text = README.read_text(encoding="utf-8")
    for fmt in FORMATS:
        assert fmt in text.lower(), f"{fmt} is supported but undocumented"


def test_the_readme_does_not_promise_things_that_do_not_exist():
    """A short list of claims that were wrong before, kept as a guard."""
    text = README.read_text(encoding="utf-8").lower()
    assert "nfa design" not in text
    assert "1-9, a-z" not in text
