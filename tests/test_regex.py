"""Regular expressions, both directions, and the claims each direction makes.

Three kinds of test, in the order they earn their keep.

The **syntax** tests are exact: precedence, associativity, what the empty
pattern means, and where an error says it is. These are the ones that pin a
decision, so they compare trees and positions and messages -- the parser is a
teaching artifact, and a student reading ``ab|c`` off the screen has to be shown
the same grouping the tool used.

The **language** tests never compare pattern strings. Two expressions denoting
one language may be written a hundred ways, and asserting that state
elimination produced the string somebody expected is how implementations of it
end up frozen around whatever they happened to emit first. Every claim about
elimination here is either "the language is the same", decided exactly by
:func:`fsa.equivalence.equivalent` on the machines, or "the answer is the same
on every run", which is about reproducibility rather than spelling.

The **differential** tests hand the same question to ``greenery``, an
independent regex/FSM package. It is never imported by ``src`` -- CI fails the
build if it is -- and it is used here in both directions: ours-versus-theirs on
membership pins Thompson's construction, and ours-versus-theirs on language
equality pins state elimination.

Imports no pygame and touches no display.
"""

import os
import random
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import fsa
from fsa import regex
from fsa.language import words_up_to
from fsa.nfa import EPSILON, NFA
from fsa.nfa import accepts as nfa_accepts
from fsa.regex import (
    EMPTY_LANGUAGE,
    EMPTY_WORD,
    Alt,
    Concat,
    EmptyLanguage,
    EmptyWord,
    Literal,
    Node,
    Plus,
    Question,
    RegexSyntaxError,
    Star,
)
from fsa.subset import determinize
from tests.strategies import nfas

SETTINGS = settings(max_examples=60, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def machine(pattern: str) -> fsa.DFA:
    """The deterministic machine for a pattern, for exact language questions."""
    return determinize(regex.to_nfa(pattern))


def same_language(left: str, right: str) -> bool:
    """Whether two patterns denote the same language, decided exactly.

    Through the machines, not through the strings: ``equivalent`` walks the
    product of the two and either finds a distinguishing word or proves there is
    none, so this is a decision and not a sample.
    """
    return fsa.equivalent(machine(left), machine(right))


def accepted(pattern: str, length: int = 4) -> List[str]:
    """Every word up to ``length`` the pattern's machine accepts, shortest first.

    Over the pattern's own alphabet widened with `a` and `b`, so that a word the
    expression *cannot* read still gets asked -- a machine whose alphabet had
    quietly shrunk would otherwise be indistinguishable from a correct one.
    """
    built = regex.to_nfa(pattern)
    alphabet = built.alphabet | frozenset("ab")
    return [word for word in words_up_to(alphabet, length)
            if nfa_accepts(built, word)]


# ---------------------------------------------------------------------------
# Precedence and associativity, which are decisions and so are tested exactly
# ---------------------------------------------------------------------------


def test_concatenation_binds_tighter_than_alternation():
    """`ab|c` is `(ab)|c` and not `a(b|c)`. The classic first misreading."""
    assert regex.parse("ab|c") == Alt((
        Concat((Literal("a"), Literal("b"))), Literal("c")))
    assert regex.parse("a|bc") == Alt((
        Literal("a"), Concat((Literal("b"), Literal("c")))))
    assert accepted("ab|c") == ["c", "ab"]
    assert accepted("a|bc") == ["a", "bc"]


def test_a_postfix_operator_binds_tighter_than_concatenation():
    assert regex.parse("ab*") == Concat((Literal("a"), Star(Literal("b"))))
    assert regex.parse("(ab)*") == Star(Concat((Literal("a"), Literal("b"))))
    assert accepted("ab*") == ["a", "ab", "abb", "abbb"]
    assert accepted("(ab)*") == ["", "ab", "abab"]


def test_a_postfix_operator_may_be_applied_twice():
    """`a**` is legal here, and is `Star(Star(a))` -- the parser records what
    was typed. It denotes the same language as `a*`, which is a fact about
    languages and is checked as one."""
    assert regex.parse("a**") == Star(Star(Literal("a")))
    assert same_language("a**", "a*")
    assert regex.simplify(regex.parse("a**")) == Star(Literal("a"))


def test_alternation_is_flat_rather_than_leaning():
    """`a|b|c` has three branches. Both binary operators are associative, so a
    tree that leaned left or right would carry a shape nobody wrote and every
    simplification rule would have to undo it."""
    assert regex.parse("a|b|c") == Alt((
        Literal("a"), Literal("b"), Literal("c")))
    assert regex.parse("abc") == Concat((
        Literal("a"), Literal("b"), Literal("c")))
    assert regex.parse("(a|b)|c") == Alt((
        Alt((Literal("a"), Literal("b"))), Literal("c"))), "brackets are kept"
    assert same_language("(a|b)|c", "a|b|c")


def test_brackets_change_what_an_operator_applies_to():
    assert not same_language("ab*", "(ab)*")
    assert not same_language("a|bc", "(a|b)c")
    assert same_language("(a)", "a")


# ---------------------------------------------------------------------------
# The two empty things
# ---------------------------------------------------------------------------


def test_the_empty_pattern_is_the_empty_word():
    """Concatenating no factors is the empty word, the way an empty sum is
    zero. The alternative reading -- that an empty pattern is the empty
    language -- would make `a|` mean `a`, and then `()` would have to mean
    something a third time."""
    assert regex.parse("") == EmptyWord()
    assert accepted("") == [""]


def test_the_three_spellings_of_the_empty_word_are_one_value():
    assert regex.parse("") == regex.parse("()") == regex.parse(EMPTY_WORD)
    assert EmptyWord().pattern() == EMPTY_WORD


def test_an_empty_branch_of_an_alternation_is_the_empty_word():
    assert regex.parse("a|") == Alt((Literal("a"), EmptyWord()))
    assert regex.parse("|a") == Alt((EmptyWord(), Literal("a")))
    assert accepted("a|") == ["", "a"]
    assert same_language("a|", "a?")


def test_the_empty_language_accepts_nothing_at_all():
    assert regex.parse(EMPTY_LANGUAGE) == EmptyLanguage()
    assert accepted(EMPTY_LANGUAGE) == []
    assert accepted("a" + EMPTY_LANGUAGE + "b") == [], "one dead factor is fatal"
    assert same_language("a|" + EMPTY_LANGUAGE, "a")
    assert same_language(EMPTY_LANGUAGE + "*", EMPTY_WORD)


def test_the_two_empties_are_different_values():
    """They look alike in prose -- "empty" -- and denote different languages:
    one word versus no words."""
    assert EmptyWord() != EmptyLanguage()
    assert not same_language(EMPTY_WORD, EMPTY_LANGUAGE)
    assert EmptyWord().nullable and not EmptyLanguage().nullable


# ---------------------------------------------------------------------------
# Escaping: every reserved character can still be a symbol
# ---------------------------------------------------------------------------


def test_a_reserved_character_can_be_a_symbol():
    assert regex.parse("\\*") == Literal("*")
    assert regex.alphabet_of("\\*") == frozenset("*")
    # Neither star is an operator here: the pattern denotes one word, `**`.
    assert accepted("\\*\\*") == ["**"]


def test_the_sentinels_themselves_can_be_escaped_into_symbols():
    """The price of reserving a character, paid back. Without this, an alphabet
    containing epsilon would simply be unwritable."""
    for sentinel in (EMPTY_WORD, EMPTY_LANGUAGE):
        pattern = "\\" + sentinel
        assert regex.parse(pattern) == Literal(sentinel)
        assert regex.alphabet_of(pattern) == frozenset({sentinel})
        assert nfa_accepts(regex.to_nfa(pattern), sentinel)
        assert not nfa_accepts(regex.to_nfa(pattern), "")


def test_an_escape_before_an_ordinary_character_is_that_character():
    """Not C's escapes: `\\n` is the letter n, because a symbol is one printable
    character and no alphabet here contains a newline to name."""
    assert regex.parse("\\a") == Literal("a")
    assert regex.parse("\\n") == Literal("n")
    assert regex.parse("\\\\") == Literal("\\")


def test_every_reserved_character_survives_being_printed_and_read_back():
    for character in sorted(regex.RESERVED):
        node = Literal(character)
        assert regex.parse(node.pattern()) == node, character


# ---------------------------------------------------------------------------
# Errors say where
# ---------------------------------------------------------------------------


def failure(pattern: str) -> RegexSyntaxError:
    with pytest.raises(RegexSyntaxError) as caught:
        regex.parse(pattern)
    return caught.value


def test_an_unclosed_bracket_names_the_bracket_that_opened():
    """`(a|b` deserves to be told the bracket never closed, and which one --
    not that something, somewhere, was invalid."""
    error = failure("(a|b")
    assert error.position == 0
    assert "never closed" in str(error)
    assert error.caret() == "(a|b\n^"


def test_the_bracket_named_is_the_inner_one_when_they_nest():
    error = failure("(a(b)")
    assert error.position == 0
    assert failure("((a)").position == 0
    assert failure("(a(b").position == 2, "the innermost unclosed one first"


def test_a_bracket_that_closes_nothing_names_itself():
    error = failure("a)b")
    assert error.position == 1
    assert "never opened" in str(error)
    assert error.caret() == "a)b\n ^"


def test_a_postfix_operator_with_nothing_to_repeat():
    for pattern, position in (("*a", 0), ("?", 0), ("a|*", 2), ("(*)", 1)):
        error = failure(pattern)
        assert error.position == position, pattern
        assert "nothing to repeat" in str(error), pattern


def test_an_escape_with_nothing_after_it():
    error = failure("ab\\")
    assert error.position == 2
    assert "nothing after it" in str(error)


def test_a_character_that_could_never_be_a_symbol():
    """A space is the case that turns up. Dropping it silently would choose
    between two readings of `a b` on the user's behalf."""
    error = failure("a b")
    assert error.position == 1
    assert "printable" in str(error)
    assert failure("a\\ b").position == 2, "escaping does not make it legal"


def test_the_error_is_an_engine_error_and_carries_the_pattern():
    assert issubclass(RegexSyntaxError, fsa.AutomatonError)
    error = failure("(")
    assert error.pattern == "("
    with pytest.raises(fsa.AutomatonError):
        regex.parse("(")
    with pytest.raises(RegexSyntaxError):
        regex.to_nfa("a|)")
    with pytest.raises(RegexSyntaxError):
        regex.alphabet_of("(((")


def test_the_guard_against_reading_past_the_end_of_the_pattern_fires():
    """`_concatenation` stops at the end of the input, so `_atom` never sees it
    through the public entry point -- and a guard that has never been shown to
    fail is not a guard (docs/LESSONS.md). Reaching for the production directly
    is what proves the message exists and does not crash instead."""
    with pytest.raises(RegexSyntaxError) as caught:
        regex._Parser("")._atom()
    assert caught.value.position == 0
    assert "ends at position" in str(caught.value)


# ---------------------------------------------------------------------------
# Writing an expression back out
# ---------------------------------------------------------------------------


CANONICAL = [
    "a", "ab", "abc", "a|b", "a|b|c", "ab|c", "a|bc", "a*", "a**", "a+", "a?",
    "a*b", "(ab)*", "(a|b)*", "(a|b)c", "a(b|c)", "(ab|c)*", "a?b+", "a*+",
    EMPTY_WORD, EMPTY_LANGUAGE, EMPTY_WORD + "*", "\\*|\\(", "a|" + EMPTY_WORD,
]


def test_a_canonical_pattern_survives_a_parse_and_a_print():
    for pattern in CANONICAL:
        assert regex.parse(pattern).pattern() == pattern


def test_stacked_postfix_operators_need_no_brackets():
    """`(a*)+` comes back as `a*+`, which parses to the very same tree: the
    repetition production takes a run of operators, so brackets around one of
    them say nothing. The same rule is what lets `a**` be written at all."""
    assert regex.parse("(a*)+") == regex.parse("a*+") == Plus(Star(Literal("a")))
    assert regex.parse("(a*)+").pattern() == "a*+"


def test_brackets_that_change_nothing_are_dropped():
    """The parser keeps what was typed and the printer writes what it means, so
    a bracket that groups an associative operator with itself survives the parse
    and not the print. That asymmetry is the right way round: `(a|b)|c` and
    `a|b|c` are the same expression, and only one of them is worth showing."""
    assert regex.parse("((a))").pattern() == "a"
    assert regex.parse("(a)|(b)").pattern() == "a|b"
    assert regex.parse("(a*)b").pattern() == "a*b"
    assert regex.parse("(a|b)|c").pattern() == "a|b|c"
    assert regex.parse("a(bc)").pattern() == "abc"
    assert regex.parse("(a|b)|c") != regex.parse("a|b|c"), "the parse is faithful"
    assert same_language("(a|b)|c", "a|b|c")


def test_a_hand_built_identity_prints_as_the_language_it_denotes():
    """Zero factors is the empty word and zero branches is the empty language.
    The parser never builds either, but a caller assembling a tree can, and
    printing nothing at all would be an invisible expression."""
    assert Concat(()).pattern() == EMPTY_WORD
    assert Alt(()).pattern() == EMPTY_LANGUAGE
    assert regex.parse(Concat(()).pattern()) == EmptyWord()
    assert regex.parse(Alt(()).pattern()) == EmptyLanguage()


# ---------------------------------------------------------------------------
# The nodes as values
# ---------------------------------------------------------------------------


def test_a_node_is_a_value():
    first = regex.parse("(a|b)*c")
    second = regex.parse("(a|b)*c")
    assert first == second and hash(first) == hash(second)
    assert {first, second} == {first}
    assert first != regex.parse("(a|b)*d")


NULLABLE = {
    "": True, "a": False, "a*": True, "a+": False, "(a*)+": True,
    "a?": True, "a|": True, "ab": False, "a*b*": True, "a*b": False,
    EMPTY_LANGUAGE: False, EMPTY_LANGUAGE + "*": True,
}


def test_nullable_says_whether_the_empty_word_is_in_the_language():
    """Checked against the machine as well as against the expected answer, so
    the property simplification relies on is the same one the simulator sees."""
    for pattern, expected in NULLABLE.items():
        node = regex.parse(pattern)
        assert node.nullable == expected, pattern
        assert node.nullable == nfa_accepts(regex.to_nfa(pattern), ""), pattern


def test_the_alphabet_is_what_the_pattern_mentions():
    assert regex.alphabet_of("(a|b)*abb") == frozenset("ab")
    assert regex.alphabet_of("") == frozenset()
    assert regex.alphabet_of(EMPTY_LANGUAGE + "a") == frozenset("a"), (
        "mentioned, even though nothing can ever read it")
    assert regex.parse("a\\|b").alphabet == frozenset("ab|")


def test_children_are_in_written_order():
    node = regex.parse("a|b|c")
    assert [child.pattern() for child in node.children] == ["a", "b", "c"]
    assert regex.parse("a").children == ()


# ---------------------------------------------------------------------------
# Thompson's construction
# ---------------------------------------------------------------------------


def test_the_machine_reads_exactly_the_symbols_the_pattern_mentions():
    assert regex.to_nfa("(a|b)*abb").alphabet == frozenset("ab")
    assert regex.to_nfa("").alphabet == frozenset()
    assert regex.to_nfa("a" + EMPTY_LANGUAGE).alphabet == frozenset("a")


def test_the_machine_has_one_way_in_and_one_way_out():
    """The invariant the whole construction rests on: with one accepting state
    and no transition out of it, gluing two fragments together is one epsilon
    move and no operator has to look inside another."""
    for pattern in CANONICAL:
        built = regex.to_nfa(pattern)
        assert len(built.accept) == 1, pattern
        only = next(iter(built.accept))
        assert built.outgoing(only) == {}, pattern
        assert built.initial == "q0", pattern


def test_a_fragment_is_two_states_per_node():
    assert len(regex.to_nfa("a").states) == 2
    assert len(regex.to_nfa("ab").states) == 4, "concatenation adds no states"
    assert len(regex.to_nfa("a|b").states) == 6
    assert len(regex.to_nfa("a*").states) == 4


def test_membership_on_a_pattern_anyone_can_check_by_hand():
    built = regex.to_nfa("(a|b)*abb")
    for word in ("abb", "aabb", "babb", "abbabb", "bbabb"):
        assert nfa_accepts(built, word), word
    for word in ("", "a", "ab", "abba", "bb", "abab"):
        assert not nfa_accepts(built, word), word


def test_the_epsilon_cycles_the_construction_makes_do_not_hang():
    """`(a*)*` puts an epsilon cycle around an epsilon cycle, which is the case
    a closure that does not check membership before enqueueing spins on."""
    for pattern in ("(a*)*", "(a?)*", "((a|)*)*", "(" + EMPTY_WORD + "*)*"):
        assert nfa_accepts(regex.to_nfa(pattern), ""), pattern
    assert same_language("(a*)*", "a*")
    assert same_language("((a|)*)*", "a*")


def test_thompson_can_be_given_a_tree_the_caller_already_has():
    node = regex.parse("a(b|c)*")
    assert regex.thompson(node) == regex.to_nfa("a(b|c)*")


def test_thompson_refuses_something_that_is_not_a_tree():
    with pytest.raises(TypeError):
        regex.thompson("a*")


# ---------------------------------------------------------------------------
# Simplification
# ---------------------------------------------------------------------------


SIMPLIFICATIONS = [
    ("a**", "a*"),
    ("(a+)*", "a*"),
    ("(a?)*", "a*"),
    ("(a*)+", "a*"),
    ("(a?)+", "a*"),
    ("(a+)?", "a*"),
    ("a|a", "a"),
    ("a|" + EMPTY_LANGUAGE, "a"),
    ("a|" + EMPTY_WORD, "a?"),
    (EMPTY_WORD + "|a*", "a*"),
    (EMPTY_WORD + "a" + EMPTY_WORD, "a"),
    (EMPTY_LANGUAGE + "*", EMPTY_WORD),
    ("a" + EMPTY_LANGUAGE + "b", EMPTY_LANGUAGE),
    ("aa*", "a+"),
    ("a*a", "a+"),
    ("a*a*", "a*"),
    ("a*a+", "a+"),
    ("ab(ab)*", "(ab)+"),
]


def test_the_rules_that_fire_produce_something_smaller():
    """Named cases, checked as strings *because they are about the spelling*:
    simplification's whole job is what the expression looks like. Its
    correctness -- that the language is untouched -- is the property below."""
    for before, after in SIMPLIFICATIONS:
        assert regex.simplify(regex.parse(before)).pattern() == after, before


def test_the_rules_that_would_be_unsound_do_not_fire():
    """`R+R+` accepts two or more and `R+R` the same, so neither collapses to
    `R+`. This is the mistake the star identities invite."""
    assert not same_language("a+a+", "a+")
    assert not same_language("a+a", "a+")
    assert regex.simplify(regex.parse("a+a+")) == regex.parse("a+a+")


def test_simplifying_a_leaf_changes_nothing():
    for pattern in ("a", "", EMPTY_LANGUAGE, "ab", "a|b"):
        node = regex.parse(pattern)
        assert regex.simplify(node) == node, pattern


def test_a_rule_deep_in_the_tree_enables_one_above_it():
    assert regex.simplify(regex.parse("(a**)b|" + EMPTY_LANGUAGE)).pattern() \
        == "a*b"


# ---------------------------------------------------------------------------
# State elimination
# ---------------------------------------------------------------------------


def even_number_of_as() -> fsa.DFA:
    """Two states, a cycle between them: the smallest machine with a loop."""
    return (fsa.DFA()
            .with_states(["q0", "q1"])
            .with_transition("q0", "a", "q1")
            .with_transition("q1", "a", "q0")
            .with_accept("q0"))


def ends_in_abb() -> fsa.DFA:
    """The textbook four-state machine for `(a|b)*abb`."""
    rows = {
        "q0": {"a": "q1", "b": "q0"},
        "q1": {"a": "q1", "b": "q2"},
        "q2": {"a": "q1", "b": "q3"},
        "q3": {"a": "q1", "b": "q0"},
    }
    automaton = fsa.DFA().with_states(sorted(rows))
    for source, row in rows.items():
        for symbol, target in row.items():
            automaton = automaton.with_transition(source, symbol, target)
    return automaton.with_initial("q0").with_accept("q3")


def branching_nfa() -> NFA:
    """Nondeterministic and epsilon-ridden: `a*` reached by an epsilon move,
    with two targets on one symbol."""
    return (NFA()
            .with_states(["s0", "s1", "s2"])
            .with_transition("s0", EPSILON, "s1")
            .with_transition("s1", "a", "s1")
            .with_transition("s1", "a", "s2")
            .with_transition("s2", "b", "s2")
            .with_accept("s2"))


MACHINES = [even_number_of_as(), ends_in_abb(), branching_nfa(),
            fsa.minimize(machine("(a|b)*abb")), machine("a*b+"), machine("a?"),
            machine(EMPTY_LANGUAGE), machine("")]


def assert_describes(automaton: fsa.AnyAutomaton) -> str:
    """The derived expression denotes exactly the machine's language."""
    derived = regex.from_automaton(automaton)
    reference = (automaton if isinstance(automaton, fsa.DFA)
                 else determinize(automaton))
    assert fsa.equivalent(reference, machine(derived)), (
        f"{derived!r} is not the language of {automaton!r}")
    return derived


def test_a_machine_and_the_expression_derived_from_it_agree():
    for automaton in MACHINES:
        assert_describes(automaton)


def test_either_kind_of_machine_can_be_described():
    """A DFA is read as an NFA first rather than the elimination being written
    twice, so this is one algorithm answering both questions."""
    deterministic = even_number_of_as()
    as_nondeterministic = fsa.nfa.from_dfa(deterministic)
    assert (regex.from_automaton(deterministic)
            == regex.from_automaton(as_nondeterministic))


def test_an_epsilon_move_becomes_the_empty_word_on_its_edge():
    """s0 reaches s1 by an epsilon move and s1 has two targets on `a`, so this
    machine is one no DFA-only elimination could take."""
    assert_describes(branching_nfa())
    assert same_language(regex.from_automaton(branching_nfa()), "a+b*")


def test_a_machine_that_accepts_nothing_is_the_empty_language():
    assert regex.from_automaton(fsa.DFA()) == EMPTY_LANGUAGE
    assert regex.from_automaton(ends_in_abb().without_accept("q3")) \
        == EMPTY_LANGUAGE
    unreachable = ends_in_abb().with_state("q9").with_accept("q9")
    assert same_language(regex.from_automaton(unreachable), "(a|b)*abb")


def test_a_machine_with_no_start_state_is_the_empty_language():
    """The one place the engine's distinction between "no language defined yet"
    and "the empty language" cannot be kept: a regular expression denotes a
    language, and "undefined" is not one."""
    assert regex.from_automaton(ends_in_abb().with_initial(None)) \
        == EMPTY_LANGUAGE


def test_a_machine_that_accepts_only_the_empty_word():
    lonely = fsa.DFA().with_state("q0").with_accept("q0").with_symbol("a")
    assert regex.from_automaton(lonely) == EMPTY_WORD


def test_a_dead_state_costs_time_and_not_output():
    """Ripping a state that cannot reach an accepting state only ever builds
    labels the empty language swallows, so completing a machine -- which adds a
    trap and nothing else -- cannot make the answer longer."""
    partial = ends_in_abb().without_transition("q0", "b")
    trapped, trap = fsa.complete(partial)
    assert trap is not None and trap in trapped.states
    assert_describes(trapped)
    assert same_language(regex.from_automaton(trapped),
                         regex.from_automaton(partial))
    assert len(regex.from_automaton(trapped)) <= len(
        regex.from_automaton(partial))


def test_the_two_ends_the_gnfa_adds_cannot_collide_with_real_states():
    """A state id is any string, so `<start>` is one a user can type. The two
    ends elimination adds take primes until they are nobody else's, or the
    machine's own start state would be ripped out from under the algorithm."""
    awkward = (fsa.DFA()
               .with_states(["<start>", "<accept>"])
               .with_transition("<start>", "a", "<accept>")
               .with_transition("<accept>", "b", "<start>")
               .with_initial("<start>")
               .with_accept("<accept>"))
    assert same_language(assert_describes(awkward), "a(ba)*")


def test_the_answer_is_the_same_every_time():
    assert regex.from_automaton(ends_in_abb()) == regex.from_automaton(
        ends_in_abb())


def test_the_answer_does_not_depend_on_hash_randomisation():
    """Python randomises string hashing per process, so an order derived from a
    set differs between runs of the same program. This project has been bitten
    by that before (docs/LESSONS.md); an expression written into a saved file or
    a diff has to be the same expression tomorrow."""
    script = (
        "from fsa import regex\n"
        "from fsa.subset import determinize\n"
        "machine = determinize(regex.to_nfa('(a|b)*abb|ab*a'))\n"
        "print(regex.from_automaton(machine))\n"
    )
    source = Path(fsa.__file__).resolve().parent.parent
    answers = set()
    for seed in ("0", "1", "9999"):
        environment = dict(os.environ,
                           PYTHONHASHSEED=seed, PYTHONIOENCODING="utf-8")
        result = subprocess.run([sys.executable, "-c", script], cwd=source,
                                env=environment, capture_output=True,
                                text=True, encoding="utf-8")
        assert result.returncode == 0, result.stderr
        answers.add(result.stdout.strip())
    assert len(answers) == 1, answers


def test_the_output_is_simplified_and_not_merely_correct():
    """A bound on the *size*, not a claim about the spelling. Every
    implementation of state elimination is correct about the language and the
    sloppy ones return a page; these three machines have short answers and an
    implementation that stopped simplifying would blow through the bound long
    before it produced a wrong language."""
    for automaton, bound in ((even_number_of_as(), 6),
                             (fsa.minimize(machine("a*b+")), 6),
                             (fsa.minimize(machine("(ab)*")), 8)):
        derived = assert_describes(automaton)
        assert len(derived) <= bound, derived


def test_a_derived_expression_can_be_parsed_again():
    """The output is input: the loop the round-trip property closes."""
    for automaton in MACHINES:
        derived = regex.from_automaton(automaton)
        assert regex.parse(derived).pattern() == derived


# ---------------------------------------------------------------------------
# Generated expressions
# ---------------------------------------------------------------------------


ATOMS = [Literal("a"), Literal("b"), EmptyWord(), EmptyLanguage()]


@st.composite
def nodes(draw: st.DrawFn, depth: int = 2) -> Node:
    """A tree shaped exactly like one the parser can produce.

    Built down the grammar rather than by picking node types at random, so an
    alternation never holds an alternation directly and a concatenation never
    holds a concatenation. Those flattened shapes are the only ones
    :func:`parse` returns, and generating the nested ones would make the
    printer/parser property below fail for a difference nobody can see in the
    text.
    """

    def atom(level: int) -> Node:
        if level <= 0 or draw(st.integers(0, 2)) == 0:
            return draw(st.sampled_from(ATOMS))
        return alternation(level - 1)

    def repetition(level: int) -> Node:
        node = atom(level)
        for _ in range(draw(st.integers(0, 2))):
            node = draw(st.sampled_from([Star, Plus, Question]))(node)
        return node

    def concatenation(level: int) -> Node:
        parts: List[Node] = []
        for _ in range(draw(st.integers(1, 3))):
            node = repetition(level)
            # Brackets around a concatenation *inside* a concatenation say
            # nothing, so the printer drops them and the parser gives back one
            # flat node. Splicing here keeps the generator to the trees that
            # survive the trip; `a(bc)` is a legal pattern and its tree is not
            # one of them.
            parts.extend(node.parts if isinstance(node, Concat) else [node])
        return parts[0] if len(parts) == 1 else Concat(tuple(parts))

    def alternation(level: int) -> Node:
        branches: List[Node] = []
        for _ in range(draw(st.integers(1, 3))):
            node = concatenation(level)
            branches.extend(
                node.branches if isinstance(node, Alt) else [node])
        return branches[0] if len(branches) == 1 else Alt(tuple(branches))

    return alternation(depth)


@SETTINGS
@given(node=nodes())
def test_the_printer_and_the_parser_are_inverse(node):
    """Not merely that the language survives -- that the *tree* does. This is
    what makes an expression the tool printed safe to hand back to it."""
    assert regex.parse(node.pattern()) == node


@SETTINGS
@given(node=nodes())
def test_simplifying_never_changes_the_language(node):
    simplified = regex.simplify(node)
    assert fsa.equivalent(determinize(regex.thompson(node)),
                          determinize(regex.thompson(simplified)))


@SETTINGS
@given(node=nodes())
def test_simplifying_is_idempotent(node):
    once = regex.simplify(node)
    assert regex.simplify(once) == once


@SETTINGS
@given(node=nodes())
def test_a_pattern_round_trips_through_a_machine_and_back(node):
    """Phase 13's first exit criterion, and the whole of Kleene's theorem in
    one line: expression to machine to expression denotes the same language."""
    pattern = node.pattern()
    built = machine(pattern)
    derived = regex.from_automaton(built)
    assert fsa.equivalent(built, machine(derived)), f"{pattern} -> {derived}"


@SETTINGS
@given(node=nodes(), word=st.text(alphabet="ab", max_size=6))
def test_accepts_agrees_between_a_machine_and_its_expression(node, word):
    pattern = node.pattern()
    built = regex.to_nfa(pattern)
    derived = regex.from_automaton(built)
    assert nfa_accepts(built, word) == nfa_accepts(regex.to_nfa(derived), word)


@SETTINGS
@given(automaton=nfas(max_states=3, alphabet="ab"),
       word=st.text(alphabet="ab", max_size=6))
def test_a_generated_machine_and_its_expression_agree_on_a_word(automaton, word):
    """The other exit criterion, from the machine end: whatever nondeterminism
    and epsilon moves hypothesis puts in, the derived expression accepts what
    the machine accepts."""
    derived = regex.from_automaton(automaton)
    assert nfa_accepts(automaton, word) == nfa_accepts(regex.to_nfa(derived),
                                                      word)


# ---------------------------------------------------------------------------
# Differential: greenery answers the same questions independently
# ---------------------------------------------------------------------------
#
# Imported defensively rather than with pytest.importorskip, which raises at
# collection time and would skip this whole file -- including the parser tests,
# which need no oracle at all -- on a machine where greenery is not installed.

try:
    import greenery
except ImportError:  # pragma: no cover
    greenery = None

needs_greenery = pytest.mark.skipif(
    greenery is None,
    reason="greenery is the regex oracle: pip install --user greenery")


def in_greenery(node: Node) -> str:
    """Our tree written in greenery's syntax, bracketed at every level.

    Deliberately dumb about precedence: bracketing everything means no
    assumption about *greenery's* grammar is being smuggled into the
    comparison, and it is also what makes our legal `a**` expressible at all --
    greenery rejects two postfix operators in a row, where this module allows
    them.

    Their spellings for the two empty languages are `()` and `[]`; ours are the
    textbook characters. That difference is the whole bridge.
    """
    if isinstance(node, EmptyWord):
        return "()"
    if isinstance(node, EmptyLanguage):
        return "[]"
    if isinstance(node, Literal):
        return node.symbol if node.symbol.isalnum() else "\\" + node.symbol
    if isinstance(node, Concat):
        if not node.parts:
            return "()"
        return "".join(f"({in_greenery(part)})" for part in node.parts)
    if isinstance(node, Alt):
        if not node.branches:
            return "[]"
        return "|".join(f"({in_greenery(branch)})" for branch in node.branches)
    if isinstance(node, Star):
        return f"({in_greenery(node.child)})*"
    if isinstance(node, Plus):
        return f"({in_greenery(node.child)})+"
    if isinstance(node, Question):
        return f"({in_greenery(node.child)})?"
    raise TypeError(node)


def theirs(pattern: str):
    """``pattern``, parsed by us and handed to greenery."""
    return greenery.parse(in_greenery(regex.parse(pattern)))


def random_node(rng: random.Random, depth: int = 3) -> Node:
    """A tree, from a seed rather than from hypothesis.

    Both have a place: hypothesis shrinks a failure to the smallest expression
    that reproduces it, and a seeded corpus is a fixed five hundred that the
    exit criteria can be counted against and that every run checks in full.
    """
    if depth <= 0 or rng.random() < 0.3:
        # The literals are drawn twice so that the two empty languages are the
        # unusual leaf rather than a third of every tree.
        return rng.choice(ATOMS + [Literal("a"), Literal("b")])
    roll = rng.random()
    if roll < 0.2:
        return Star(random_node(rng, depth - 1))
    if roll < 0.3:
        return Plus(random_node(rng, depth - 1))
    if roll < 0.4:
        return Question(random_node(rng, depth - 1))
    if roll < 0.7:
        return Concat(tuple(random_node(rng, depth - 1)
                            for _ in range(rng.randrange(2, 4))))
    return Alt(tuple(random_node(rng, depth - 1)
                     for _ in range(rng.randrange(2, 4))))


def corpus(count: int) -> List[str]:
    rng = random.Random(20130813)
    return [random_node(rng).pattern() for _ in range(count)]


CORPUS = corpus(520)
WORDS = list(words_up_to("ab", 4))


def test_the_corpus_is_varied_enough_to_mean_something():
    """A differential test over five hundred copies of `a` proves nothing."""
    assert len(CORPUS) >= 500
    assert len(set(CORPUS)) > 300
    assert sum(1 for pattern in CORPUS if "*" in pattern) > 50
    assert sum(1 for pattern in CORPUS if "|" in pattern) > 50
    assert sum(1 for pattern in CORPUS if len(pattern) > 8) > 50


@needs_greenery
def test_thompson_agrees_with_greenery_on_membership():
    """Our machine and their engine, over the same expressions and words."""
    for pattern in CORPUS:
        ours = regex.to_nfa(pattern)
        reference = theirs(pattern).to_fsm()
        for word in WORDS:
            assert nfa_accepts(ours, word) == reference.accepts(word), (
                f"{pattern!r} disagrees on {word!r}")


@needs_greenery
def test_state_elimination_agrees_with_greenery_on_the_language():
    """Languages, never strings. Their `equivalent` decides it, so agreement
    here is evidence from an implementation that shares no code with ours."""
    for pattern in CORPUS:
        derived = regex.from_automaton(machine(pattern))
        assert theirs(pattern).equivalent(theirs(derived)), (
            f"{pattern!r} was described as {derived!r}")


@needs_greenery
@settings(max_examples=25, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(node=nodes())
def test_greenery_agrees_about_generated_expressions_too(node):
    """The same two questions, with shrinking behind them.

    Fewer examples than the properties above run: deciding equivalence of two
    regular expressions is the expensive question here, the corpus already asks
    it five hundred times, and what this adds is a *small* failing expression
    when there is one.
    """
    pattern = node.pattern()
    ours = regex.to_nfa(pattern)
    reference = theirs(pattern).to_fsm()
    for word in ("", "a", "b", "ab", "ba", "aab", "abb"):
        assert nfa_accepts(ours, word) == reference.accepts(word), word
    derived = regex.from_automaton(machine(pattern))
    assert theirs(pattern).equivalent(theirs(derived)), derived


@needs_greenery
def test_the_bridge_to_greenery_is_not_a_no_op():
    """If the translation quietly produced the same pattern for everything, the
    two tests above would pass while checking nothing."""
    assert theirs("a").equivalent(greenery.parse("a"))
    assert not theirs("a").equivalent(greenery.parse("b"))
    assert theirs(EMPTY_LANGUAGE).equivalent(greenery.parse("[]"))
    assert theirs(EMPTY_WORD).equivalent(greenery.parse("()"))
    assert theirs("a**").equivalent(greenery.parse("a*"))
