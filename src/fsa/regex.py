"""Regular expressions, in both directions.

Kleene's theorem is the spine of the subject: a language is regular exactly when
some finite automaton recognises it, and exactly when some regular expression
denotes it. This module makes the second half of that sentence executable.
:func:`to_nfa` turns an expression into a machine by Thompson's construction and
:func:`from_automaton` turns a machine back into an expression by state
elimination, so the equivalence stops being a claim in a lecture and becomes two
buttons whose results can be compared.

Six decisions carry the module.

**Two characters are reserved for the two empty things.** ``ε`` (U+03B5) denotes
the language ``{""}`` and ``∅`` (U+2205) the language ``{}``. They are the
symbols the textbooks use, and this program is read alongside one; ``&``,
``@`` or ``!`` would have to be learned twice. Reserving a character is not free
-- a symbol is any single printable non-whitespace character, so ``ε`` could
have been in somebody's alphabet -- and the escape is what buys it back:
``\\ε`` is the *letter* epsilon, exactly as ``\\*`` is the asterisk. That is the
same trade :data:`fsa.nfa.EPSILON` refuses to make by being ``None``; here there
is nothing but text to say it in, so the escape carries it instead.

**The empty word is also spelled by writing nothing.** The empty pattern,
``()``, and an empty branch of an alternation (``a|``) all parse to ε, because
concatenating no factors *is* the empty word -- it is the identity of
concatenation, the way an empty sum is zero. That makes ``ε`` a character
nobody has to type, which matters on a keyboard that does not offer it: it
exists because :func:`from_automaton` has to be able to *print* the answer, and
a regular expression printed as nothing at all is unreadable. ``∅`` has no such
alias, because there is no way to build "no words at all" out of the other
operators; a machine that accepts nothing has to be describable, so the
character is the only spelling.

**The tree :func:`parse` returns is faithful, not simplified.** ``a**`` comes
back as ``Star(Star(...))`` and ``a|a`` keeps both branches. A parser that
quietly folded them would be lying about what was typed, and the tree is meant
to be shown to the person who typed it. :func:`simplify` is a separate function,
applied where an expression is *produced* rather than where one is read --
:func:`from_automaton` runs every intermediate label through it, which is what
keeps state elimination's output from doubling in size at every rip.

**Thompson's construction, one epsilon-machine per operator.** Each fragment has
exactly one start and one accepting state and is glued to its neighbours with
epsilon moves alone, so no fragment ever has to look inside another. The result
has about twice as many states as the pattern has operators and leaves --
worse, and deliberately, than a construction that merges states as it goes,
because the whole point is that the shape of the machine mirrors the shape of
the expression. :func:`fsa.subset.determinize` is where the machine gets
sensible, and it is a separate step for the same reason minimisation is.

**State elimination rips in a stable order.** Which state is removed first
changes only the size of the answer, never its language -- but "only the size"
is still visible, and an order derived from a set would produce a different
expression on different runs of the same program (see docs/LESSONS.md). The
state with the fewest ways through it goes first, ties broken by state id, so
the output is both small and reproducible.

**Nothing here parses on behalf of anybody else.** The grammar is four
productions and the recursive-descent parser is a teaching artifact in its own
right -- one method per production, meant to be read beside the grammar. A
parser generator would be the right answer to a larger syntax and the wrong
answer to this one.

The grammar, which the parser follows function for function::

    alternation   := concatenation ('|' concatenation)*
    concatenation := repetition*
    repetition    := atom ('*' | '+' | '?')*
    atom          := '(' alternation ')' | symbol | '\\' any | 'ε' | '∅'

Precedence falls out of the nesting: the postfix operators bind tightest, then
concatenation, then alternation, so ``ab|c`` is ``(ab)|c`` and ``a|bc`` is
``a|(bc)``. Both binary operators are associative, so the parser flattens them
into n-ary nodes rather than choosing a side to lean; ``a|b|c`` has three
branches and no shape to argue about.
"""

from dataclasses import dataclass
from typing import Dict, Final, FrozenSet, Iterable, List, Optional, Set, Tuple

from fsa.automaton import DFA
from fsa.errors import AutomatonError
from fsa.layout import AnyAutomaton
from fsa.nfa import EPSILON, NFA, from_dfa
from fsa.symbols import StateId, Symbol, is_legal_symbol

#: The empty word, ``{""}``: ε, GREEK SMALL LETTER EPSILON. Written as an escape
#: because ε, ϵ and ɛ are three different codepoints that look alike in most
#: fonts, and the parser compares characters -- so which one this is has to be
#: readable in the source rather than in a hex editor.
EMPTY_WORD: Final[str] = "\u03b5"

#: The empty language, ``{}``: ∅, EMPTY SET. Same reasoning -- it is not the
#: digit zero, not ⌀ and not the Scandinavian letter.
EMPTY_LANGUAGE: Final[str] = "\u2205"

#: What turns the next character into a literal symbol.
ESCAPE: Final[str] = "\\"

#: Every character the grammar has taken for itself. Each one can still be a
#: symbol, spelled with :data:`ESCAPE` in front of it, and :meth:`Node.pattern`
#: puts it back when it writes a literal out.
RESERVED: Final[FrozenSet[str]] = frozenset(
    "()|*+?" + ESCAPE + EMPTY_WORD + EMPTY_LANGUAGE)

# How tightly each kind of node binds. Only rendering reads these: a child binds
# at least as tightly as its parent needs, or it gets brackets.
_ALTERNATION: Final[int] = 0
_CONCATENATION: Final[int] = 1
_REPETITION: Final[int] = 2
_ATOM: Final[int] = 3


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RegexSyntaxError(AutomatonError):
    """A pattern could not be parsed, with the position that stopped it.

    "Invalid syntax" is a message that sends a student back to stare at the
    whole line. Every message raised here names the character and its index
    instead -- ``(a|b`` is told that the bracket opened at position 0 is never
    closed -- and the pattern and position travel on the exception so a front
    end can put a caret under it rather than re-deriving where to point.
    """

    def __init__(self, message: str, pattern: str, position: int) -> None:
        super().__init__(message)
        self.pattern = pattern
        """The pattern that failed to parse, unmodified."""

        self.position = position
        """Index into :attr:`pattern` of the character to blame.

        For an unclosed group that is the ``(`` that opened it, not the end of
        the text: the bracket is what has to be fixed, and pointing at the end
        of a long pattern says nothing at all."""

    def caret(self) -> str:
        """The pattern with a ``^`` under the offending character, two lines.

        For a terminal or a tooltip. Positions are counted in characters, which
        is what both the parser and a text cursor use.
        """
        return f"{self.pattern}\n{' ' * self.position}^"


# ---------------------------------------------------------------------------
# The syntax tree
# ---------------------------------------------------------------------------


class Node:
    """One node of a regular expression's syntax tree.

    A value like everything else in the engine: frozen, comparable, hashable, so
    two parses of one pattern are equal and a node can be a dictionary key
    during state elimination. Subclasses are the operators, one each, and the
    two empty languages are nodes rather than special cases -- ``∅`` is a
    perfectly ordinary regular expression and a construction that had to
    special-case it would get it wrong.
    """

    __slots__ = ()

    #: How tightly this node binds. Read only by :func:`_bracket`.
    precedence = _ATOM

    @property
    def children(self) -> Tuple["Node", ...]:
        """The sub-expressions, in written order. A leaf has none."""
        return ()

    @property
    def nullable(self) -> bool:
        """Whether the empty word is in this expression's language.

        Wanted by simplification -- ``ε|R`` is ``R`` when ``R`` already accepts
        the empty word -- and by anyone drawing the tree, since "can this part
        match nothing?" is the question behind most confusion about ``*``.
        """
        raise NotImplementedError

    @property
    def alphabet(self) -> FrozenSet[Symbol]:
        """Every symbol this expression mentions.

        Gathered from the literals, so ``∅a`` mentions ``a`` even though nothing
        can ever read it: this answers "what does the pattern talk about", not
        "what can the language contain". :func:`to_nfa` builds its machine over
        exactly this alphabet.
        """
        found: Set[Symbol] = set()
        pending: List[Node] = [self]
        while pending:
            node = pending.pop()
            if isinstance(node, Literal):
                found.add(node.symbol)
            pending.extend(node.children)
        return frozenset(found)

    def pattern(self) -> str:
        """This expression written back out, with brackets only where needed.

        Printing is a fixed point: ``parse(text).pattern()`` is ``text`` for
        anything this ever produced, so an expression the tool shows can be
        handed straight back to it.

        It is deliberately *not* an inverse of :func:`parse` on every tree.
        Brackets that say nothing are dropped -- ``(a|b)|c`` prints as
        ``a|b|c`` and parses back flat -- because the parser's job is to record
        what was typed and this one's is to write what the expression means.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return self.pattern()


def _bracket(node: Node, level: int) -> str:
    """``node`` written out, in brackets only if it binds more loosely than
    ``level`` requires. This one function is the whole of precedence on the way
    back out, as the four productions are on the way in."""
    text = node.pattern()
    return text if node.precedence >= level else f"({text})"


@dataclass(frozen=True, slots=True)
class EmptyLanguage(Node):
    """``∅``: no words at all. The identity of alternation."""

    @property
    def nullable(self) -> bool:
        return False

    def pattern(self) -> str:
        return EMPTY_LANGUAGE


@dataclass(frozen=True, slots=True)
class EmptyWord(Node):
    """``ε``: one word, the empty one. The identity of concatenation."""

    @property
    def nullable(self) -> bool:
        return True

    def pattern(self) -> str:
        return EMPTY_WORD


@dataclass(frozen=True, slots=True)
class Literal(Node):
    """A single symbol of the alphabet."""

    symbol: Symbol

    @property
    def nullable(self) -> bool:
        return False

    def pattern(self) -> str:
        # A reserved character goes back out escaped, so that printing a tree
        # and parsing it again is a round trip even for the awkward alphabets.
        if self.symbol in RESERVED:
            return ESCAPE + self.symbol
        return self.symbol


@dataclass(frozen=True, slots=True)
class Concat(Node):
    """Several expressions, one after another.

    N-ary rather than a binary tree leaning left or right: concatenation is
    associative, so a shape had to be chosen and any choice would be arbitrary
    and would have to be undone by every simplification rule. Zero parts is the
    empty word, which is what makes an empty pattern parse without a special
    case in the parser.
    """

    parts: Tuple[Node, ...]

    precedence = _CONCATENATION

    @property
    def children(self) -> Tuple[Node, ...]:
        return self.parts

    @property
    def nullable(self) -> bool:
        return all(part.nullable for part in self.parts)

    def pattern(self) -> str:
        if not self.parts:
            # Only a hand-built tree gets here; the parser writes EmptyWord
            # instead. Printing nothing would be an invisible expression, so the
            # identity is named out loud.
            return EMPTY_WORD
        return "".join(_bracket(part, _CONCATENATION) for part in self.parts)


@dataclass(frozen=True, slots=True)
class Alt(Node):
    """A choice between several expressions. Zero choices is ``∅``."""

    branches: Tuple[Node, ...]

    precedence = _ALTERNATION

    @property
    def children(self) -> Tuple[Node, ...]:
        return self.branches

    @property
    def nullable(self) -> bool:
        return any(branch.nullable for branch in self.branches)

    def pattern(self) -> str:
        if not self.branches:
            return EMPTY_LANGUAGE
        return "|".join(_bracket(branch, _ALTERNATION)
                        for branch in self.branches)


@dataclass(frozen=True, slots=True)
class Star(Node):
    """``R*``: any number of repetitions, including none."""

    child: Node

    precedence = _REPETITION

    @property
    def children(self) -> Tuple[Node, ...]:
        return (self.child,)

    @property
    def nullable(self) -> bool:
        return True

    def pattern(self) -> str:
        return _bracket(self.child, _REPETITION) + "*"


@dataclass(frozen=True, slots=True)
class Plus(Node):
    """``R+``: one repetition or more."""

    child: Node

    precedence = _REPETITION

    @property
    def children(self) -> Tuple[Node, ...]:
        return (self.child,)

    @property
    def nullable(self) -> bool:
        return self.child.nullable

    def pattern(self) -> str:
        return _bracket(self.child, _REPETITION) + "+"


@dataclass(frozen=True, slots=True)
class Question(Node):
    """``R?``: this expression or the empty word.

    Named for the operator rather than for what it means, because ``Optional``
    is taken: a node class of that name in a module that also writes
    ``Optional[Symbol]`` would be a trap for the next reader, and the type
    checker would not object.
    """

    child: Node

    precedence = _REPETITION

    @property
    def children(self) -> Tuple[Node, ...]:
        return (self.child,)

    @property
    def nullable(self) -> bool:
        return True

    def pattern(self) -> str:
        return _bracket(self.child, _REPETITION) + "?"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class _Parser:
    """Recursive descent, one method per production.

    The grammar in the module docstring and the four methods below are meant to
    be read side by side; if one changes, the other is wrong. Position is a
    single index into the pattern, and every error carries it, because the whole
    value of writing this by hand rather than generating it is the quality of
    the message a student gets back.
    """

    def __init__(self, pattern: str) -> None:
        self._source = pattern
        self._index = 0

    # -- machinery ----------------------------------------------------------

    def _peek(self) -> Optional[str]:
        """The character under the cursor, or ``None`` at the end."""
        if self._index >= len(self._source):
            return None
        return self._source[self._index]

    def _fail(self, message: str, position: int) -> RegexSyntaxError:
        return RegexSyntaxError(message, self._source, position)

    # -- the productions ----------------------------------------------------

    def parse(self) -> Node:
        node = self._alternation()
        remaining = self._peek()
        if remaining is not None:
            # _alternation stops at ')' and at nothing else, so anything left
            # over is a bracket that closes a group nobody opened.
            raise self._fail(
                f"'{remaining}' at position {self._index} closes a group that "
                f"was never opened", self._index)
        return node

    def _alternation(self) -> Node:
        branches = [self._concatenation()]
        while self._peek() == "|":
            self._index += 1
            branches.append(self._concatenation())
        if len(branches) == 1:
            return branches[0]
        return Alt(tuple(branches))

    def _concatenation(self) -> Node:
        parts: List[Node] = []
        while True:
            ahead = self._peek()
            if ahead is None or ahead in "|)":
                break
            parts.append(self._repetition())
        if not parts:
            # No factors at all: the empty word. This is the one line that makes
            # the empty pattern, `()` and the empty branch of `a|` all mean the
            # same thing, and it is a consequence of the grammar rather than
            # three special cases agreeing with each other.
            return EmptyWord()
        if len(parts) == 1:
            return parts[0]
        return Concat(tuple(parts))

    def _repetition(self) -> Node:
        # A *run* of postfix operators rather than at most one, which is what
        # makes `a**` legal and `Star(Star(a))`. Writing it is redundant, not
        # wrong, and simplify() is where redundancy is the topic; refusing it
        # here would be a rule to explain in an error message instead.
        node = self._atom()
        while True:
            ahead = self._peek()
            if ahead == "*":
                node = Star(node)
            elif ahead == "+":
                node = Plus(node)
            elif ahead == "?":
                node = Question(node)
            else:
                return node
            self._index += 1

    def _atom(self) -> Node:
        position = self._index
        char = self._peek()

        if char is None:
            # _concatenation stops at the end of the input, so nothing reaches
            # here through parse(). It stays because _atom is the only
            # production that indexes the text, and a parser whose atom rule can
            # read past the end raises IndexError at some future caller instead
            # of saying what is wrong. tests/test_regex.py calls this production
            # directly to prove the message exists.
            raise self._fail(
                f"the pattern ends at position {position}, where a symbol was "
                f"expected", position)

        if char == "(":
            self._index += 1
            inner = self._alternation()
            if self._peek() != ")":
                raise self._fail(
                    f"'(' at position {position} is never closed", position)
            self._index += 1
            return inner

        if char in "*+?":
            raise self._fail(
                f"'{char}' at position {position} has nothing to repeat: a "
                f"postfix operator needs an expression before it", position)

        if char == ESCAPE:
            self._index += 1
            escaped = self._peek()
            if escaped is None:
                raise self._fail(
                    f"the escape at position {position} has nothing after it",
                    position)
            if not is_legal_symbol(escaped):
                raise self._fail(
                    f"{escaped!r} at position {position + 1} cannot be a "
                    f"symbol, escaped or not: a symbol is one printable, "
                    f"non-whitespace character", position + 1)
            self._index += 1
            return Literal(escaped)

        self._index += 1
        if char == EMPTY_WORD:
            return EmptyWord()
        if char == EMPTY_LANGUAGE:
            return EmptyLanguage()
        if not is_legal_symbol(char):
            # Whitespace is the case that turns up: `a b` is somebody meaning
            # `ab` or meaning two patterns, and silently dropping the space
            # would pick one of those for them.
            raise self._fail(
                f"{char!r} at position {position} is not a symbol: a symbol is "
                f"one printable, non-whitespace character, and this expression "
                f"has no room for spaces", position)
        return Literal(char)


def parse(pattern: str) -> Node:
    """Parse ``pattern`` into a syntax tree.

    Faithful to what was typed: ``a**`` keeps both stars and ``a|a`` keeps both
    branches, because the tree is a thing to show the user and not only an
    intermediate value. :func:`simplify` is the separate step that tidies.

    The empty pattern is the empty word, ``ε`` -- see the module docstring for
    why that and not the empty language.

    Args:
        pattern: The expression, as text.

    Returns:
        The root of the tree, whose :meth:`Node.pattern` prints a pattern this
        function reads back identically. The tree itself survives that trip too
        unless brackets grouped an associative operator with itself: ``(a|b)|c``
        is kept as written here and printed as ``a|b|c``, which parses flat.
        The language is the same either way; the shape is the thing the printer
        has an opinion about.

    Raises:
        RegexSyntaxError: With the position and character that stopped it.
    """
    return _Parser(pattern).parse()


def alphabet_of(pattern: str) -> FrozenSet[Symbol]:
    """Every symbol ``pattern`` mentions.

    What a front end needs to widen an automaton's alphabet before dropping a
    machine built from this pattern next to another one.

    Raises:
        RegexSyntaxError: If the pattern does not parse.
    """
    return parse(pattern).alphabet


# ---------------------------------------------------------------------------
# Simplification
# ---------------------------------------------------------------------------
#
# Every rule below is an identity of regular languages, and each is applied by a
# smart constructor rather than by a pass over a finished tree. That is what
# keeps state elimination's intermediate labels from doubling in size at every
# rip: a label is simplified when it is built, so the next rip starts from
# something small. The unsound rules are the ones worth naming: `R+R+` is *not*
# `R+` (it needs two), and `R+R` is not either, so neither is here.


def _flatten(nodes: Iterable[Node], kind: type) -> List[Node]:
    """Splice any ``kind`` node's children into the list, to any depth.

    Associativity is why: ``(a|b)|c`` and ``a|(b|c)`` denote the same language
    and should be the same value, so one flat node is the normal form.
    """
    flat: List[Node] = []
    pending = list(nodes)
    while pending:
        node = pending.pop(0)
        if isinstance(node, kind):
            pending = list(node.children) + pending
        else:
            flat.append(node)
    return flat


def _alt(branches: Iterable[Node]) -> Node:
    """A choice, tidied: ``∅`` dropped, duplicates dropped, ``ε`` folded in."""
    unique: List[Node] = []
    for branch in _flatten(branches, Alt):
        if isinstance(branch, EmptyLanguage):
            continue  # the identity: choosing between R and nothing is R
        if branch not in unique:
            # `a|a` is `a`. Dedup keeps first-appearance order rather than
            # sorting: the branches of an alternation built during elimination
            # already come out in a stable order, and re-sorting whole
            # expressions would be a second ordering rule to keep true.
            unique.append(branch)

    empty_word = EmptyWord()
    if empty_word in unique and any(
            node.nullable for node in unique if node != empty_word):
        # Some other branch already accepts the empty word, so this one adds
        # nothing. `ε|a*` is `a*`.
        unique = [node for node in unique if node != empty_word]

    if not unique:
        return EmptyLanguage()
    if len(unique) == 1:
        return unique[0]

    if empty_word in unique:
        # `ε|R` is `R?`, which is both shorter and the way anyone would read it
        # aloud. Nothing in `rest` is nullable -- the rule above would have
        # removed the ε otherwise -- so `_question` will not undo this.
        rest = [node for node in unique if node != empty_word]
        return _question(rest[0] if len(rest) == 1 else Alt(tuple(rest)))

    return Alt(tuple(unique))


def _merge(left: Node, right: Node) -> Optional[Node]:
    """Two adjacent factors as one, or ``None`` if they do not combine.

    Only the identities involving a star are sound. ``R*R`` and ``RR*`` are both
    ``R+``; ``R*R*`` is ``R*``; ``R*R+`` and ``R+R*`` are ``R+``. ``R+R+`` looks
    like it belongs in that list and does not: it accepts two or more, which no
    node here can spell.
    """
    if isinstance(left, Star):
        if right == left.child:
            return Plus(left.child)
        if isinstance(right, (Star, Plus)) and right.child == left.child:
            return right if isinstance(right, Plus) else left
    if isinstance(right, Star):
        if left == right.child:
            return Plus(right.child)
        if isinstance(left, Plus) and left.child == right.child:
            return left
    return None


def _merge_adjacent(parts: List[Node]) -> List[Node]:
    """One left-to-right pass of :func:`_merge` over neighbouring factors."""
    merged: List[Node] = []
    for part in parts:
        combined = _merge(merged[-1], part) if merged else None
        if combined is None:
            merged.append(part)
        else:
            merged[-1] = combined
    return merged


def _fold_repetition(parts: List[Node]) -> Optional[List[Node]]:
    """``X X*`` or ``X* X`` as ``X+``, where ``X`` is several factors long.

    :func:`_merge` sees one factor at a time and so only catches the case where
    ``X`` is a single node. State elimination produces the longer form
    constantly -- ripping the last state of a loop leaves the whole body
    written out and then starred -- and folding it is the difference between
    ``ab(ab)*`` and ``(ab)+``, or between a screenful and a line.

    Returns the shortened list, or ``None`` if nothing folded. One fold per
    call, so the caller can re-run the cheaper rules over the result.
    """
    for index, node in enumerate(parts):
        if not isinstance(node, Star):
            continue
        block = (list(node.child.parts) if isinstance(node.child, Concat)
                 else [node.child])
        size = len(block)
        if size < 2:
            continue  # _merge already handles a one-factor body
        # Through the smart constructor, not `Plus(...)` directly. A nullable
        # body makes "one or more" the same as "any number", and `_plus` knows
        # that -- built raw, the fold emitted a node the very next pass
        # rewrote, so `simplify` was not a fixed point: `ba*b*(a*b*)*` settled
        # on `b(a*b*)+` once and `b(a*b*)*` twice.
        repeated = _plus(_concat(block))
        if index >= size and parts[index - size:index] == block:
            return parts[:index - size] + [repeated] + parts[index + 1:]
        if parts[index + 1:index + 1 + size] == block:
            return parts[:index] + [repeated] + parts[index + 1 + size:]
    return None


def _concat(parts: Iterable[Node]) -> Node:
    """A sequence, tidied: ``∅`` swallows it, ``ε`` disappears, stars merge."""
    kept: List[Node] = []
    for part in _flatten(parts, Concat):
        if isinstance(part, EmptyLanguage):
            # Nothing can be read here, so nothing can be read at all: one dead
            # factor kills the whole sequence. This is what makes a dead state
            # cost state elimination time and not output size.
            return EmptyLanguage()
        if isinstance(part, EmptyWord):
            continue  # the identity of concatenation
        kept.append(part)

    # Folding shortens the list by at least two every time it fires, so the loop
    # terminates however the two rules interleave.
    kept = _merge_adjacent(kept)
    folded = _fold_repetition(kept)
    while folded is not None:
        kept = _merge_adjacent(folded)
        folded = _fold_repetition(kept)

    if not kept:
        return EmptyWord()
    if len(kept) == 1:
        return kept[0]
    return Concat(tuple(kept))


def _star(node: Node) -> Node:
    """``R*``, tidied. ``∅*`` and ``ε*`` are both ``ε``."""
    if isinstance(node, (EmptyLanguage, EmptyWord)):
        return EmptyWord()
    if isinstance(node, Star):
        return node
    if isinstance(node, (Plus, Question)):
        # `(R+)*` and `(R?)*` are both `R*`: the inner operator only decides
        # whether one repetition is optional, which the star already answers.
        return Star(node.child)
    return Star(node)


def _plus(node: Node) -> Node:
    """``R+``, tidied."""
    if isinstance(node, (EmptyLanguage, EmptyWord)):
        return node  # nothing repeated is nothing; ε repeated is ε
    if isinstance(node, (Star, Plus)):
        return node
    if node.nullable:
        # One repetition is already optional, so "one or more" is "any number".
        return _star(node)
    return Plus(node)


def _question(node: Node) -> Node:
    """``R?``, tidied."""
    if isinstance(node, EmptyLanguage):
        return EmptyWord()  # nothing, or nothing at all: the empty word
    if isinstance(node, Plus):
        return Star(node.child)
    if node.nullable:
        return node  # it already accepts the empty word
    return Question(node)


def simplify(node: Node) -> Node:
    """An equivalent expression with the obvious redundancy removed.

    Bottom-up through the smart constructors above, so a rule that fires deep in
    the tree can enable one further up: ``(a*)*b|∅`` simplifies to ``a*b``
    because the star collapses before the alternation is built.

    This is *not* a minimal form and no such thing is being claimed -- deciding
    whether two regular expressions are equal is PSPACE-complete, and the answer
    to "are these the same language" in this codebase is
    :func:`fsa.equivalence.equivalent` on the machines, which decides it exactly.
    What this does is make an expression fit to read.
    """
    if isinstance(node, Alt):
        return _alt([simplify(branch) for branch in node.branches])
    if isinstance(node, Concat):
        return _concat([simplify(part) for part in node.parts])
    if isinstance(node, Star):
        return _star(simplify(node.child))
    if isinstance(node, Plus):
        return _plus(simplify(node.child))
    if isinstance(node, Question):
        return _question(simplify(node.child))
    return node


# ---------------------------------------------------------------------------
# Thompson's construction: expression -> NFA
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Fragment:
    """A part-built machine: one way in, one way out.

    The single accepting state is the invariant the whole construction rests on.
    It is why gluing two fragments together is one epsilon move rather than a
    loop over accepting states, and why every operator below is four lines.
    """

    start: StateId
    accept: StateId


class _Thompson:
    """The construction, carrying the state counter and the edges built so far.

    A class rather than a recursive function returning machines, because every
    fragment must draw its state ids from one counter: two fragments that both
    called their states ``q0`` would silently merge when the edges were put in
    one dictionary. Numbering is allocation order, and allocation is
    outside-in, so ``q0`` is always the machine's start state.
    """

    def __init__(self) -> None:
        self._states: List[StateId] = []
        self._edges: Dict[Tuple[StateId, Optional[Symbol]], Set[StateId]] = {}

    def _state(self) -> StateId:
        name = f"q{len(self._states)}"
        self._states.append(name)
        return name

    def _link(self, source: StateId, symbol: Optional[Symbol],
              target: StateId) -> None:
        self._edges.setdefault((source, symbol), set()).add(target)

    def machine(self, node: Node) -> NFA:
        """Build ``node``'s machine and close it into an :class:`NFA` value."""
        fragment = self.build(node)
        return NFA(
            states=frozenset(self._states),
            # Exactly the symbols the pattern mentions. Not the symbols that
            # label an edge: `∅a` mentions `a`, and a machine whose alphabet
            # quietly shrank would reject a word for the wrong reason
            # (REJECT_SYMBOL_NOT_IN_ALPHABET rather than REJECT_NO_TRANSITION).
            alphabet=node.alphabet,
            transitions={key: frozenset(targets)
                         for key, targets in self._edges.items()},
            initial=fragment.start,
            accept=frozenset({fragment.accept}),
        )

    def build(self, node: Node) -> _Fragment:
        """One epsilon-machine for ``node``, recursing into its children."""
        if isinstance(node, EmptyLanguage):
            # Two states and no edge between them: nothing can cross, which is
            # the empty language drawn rather than declared.
            start = self._state()
            return _Fragment(start, self._state())

        if isinstance(node, EmptyWord):
            start = self._state()
            accept = self._state()
            self._link(start, EPSILON, accept)
            return _Fragment(start, accept)

        if isinstance(node, Literal):
            start = self._state()
            accept = self._state()
            self._link(start, node.symbol, accept)
            return _Fragment(start, accept)

        if isinstance(node, Concat):
            if not node.parts:
                return self.build(EmptyWord())
            fragments = [self.build(part) for part in node.parts]
            for before, after in zip(fragments, fragments[1:]):
                self._link(before.accept, EPSILON, after.start)
            return _Fragment(fragments[0].start, fragments[-1].accept)

        if isinstance(node, Alt):
            start = self._state()
            accept = self._state()
            for branch in node.branches:
                inner = self.build(branch)
                self._link(start, EPSILON, inner.start)
                self._link(inner.accept, EPSILON, accept)
            # No branches means no path from start to accept, which is the empty
            # language -- the identity of alternation, falling out of the loop
            # running zero times rather than needing a case of its own.
            return _Fragment(start, accept)

        if isinstance(node, (Star, Plus, Question)):
            start = self._state()
            accept = self._state()
            inner = self.build(node.child)
            self._link(start, EPSILON, inner.start)
            self._link(inner.accept, EPSILON, accept)
            if not isinstance(node, Question):
                # The back edge is what makes repetition; it is also the epsilon
                # cycle that a naive closure spins on forever, which is why
                # NFA.epsilon_closure checks membership before enqueueing.
                self._link(inner.accept, EPSILON, inner.start)
            if not isinstance(node, Plus):
                self._link(start, EPSILON, accept)  # zero repetitions
            return _Fragment(start, accept)

        raise TypeError(f"not a regular expression node: {node!r}")


def thompson(node: Node) -> NFA:
    """The machine for a syntax tree, by Thompson's construction.

    Exposed beside :func:`to_nfa` because a front end that has already parsed --
    to draw the tree, or to report a syntax error with a caret -- should not
    have to parse the same text twice to get the machine.

    The result has one accepting state, no transition out of it, and about two
    states per node of the tree. It is nondeterministic and epsilon-ridden by
    design: :func:`fsa.subset.determinize` and :func:`fsa.minimize.minimize` are
    the steps that make it small, and keeping them separate is what lets a
    student see what each one did.
    """
    return _Thompson().machine(node)


def to_nfa(pattern: str) -> NFA:
    """Parse ``pattern`` and build its machine.

    Every word the expression denotes is accepted by the result and no other
    word is, over an alphabet of exactly the symbols the pattern mentions.

    Raises:
        RegexSyntaxError: If the pattern does not parse.
    """
    return thompson(parse(pattern))


# ---------------------------------------------------------------------------
# State elimination: automaton -> expression
# ---------------------------------------------------------------------------
#
# A GNFA is an automaton whose edges are labelled with whole regular expressions
# rather than single symbols. Every automaton is one already (each label a
# single literal), and ripping a state out preserves the language as long as
# every path through it is written onto the edges that bypassed it. Do that
# until only the two added ends remain and the label between them is the answer.


#: One edge of the GNFA: from, to. Every pair holds at most one label, because
#: two edges between the same two states are one alternation.
_Edge = Tuple[StateId, StateId]


def _fresh(name: StateId, taken: FrozenSet[StateId]) -> StateId:
    """``name``, or ``name`` with enough primes to be nobody else's."""
    candidate = name
    while candidate in taken:
        candidate += "'"
    return candidate


def _labels_of(machine: NFA) -> Dict[_Edge, Node]:
    """The machine's own edges, as GNFA labels.

    Several symbols between one pair of states become one alternation, in sorted
    symbol order -- ``sorted_transitions`` guarantees it -- so the expression
    this eventually produces does not depend on the order a set iterated in.
    """
    labels: Dict[_Edge, Node] = {}
    for source, symbol, targets in machine.sorted_transitions():
        piece: Node = EmptyWord() if symbol is None else Literal(symbol)
        for target in targets:
            edge = (source, target)
            existing = labels.get(edge)
            labels[edge] = piece if existing is None else _alt([existing, piece])
    return labels


def _next_to_rip(interior: List[StateId],
                 labels: Dict[_Edge, Node]) -> StateId:
    """Which state to remove next: the one with the fewest ways through it.

    Ripping a state writes one new label for every (predecessor, successor)
    pair, so the state whose product of in- and out-degree is smallest costs the
    least, and choosing greedily keeps the answer readable -- the difference
    between ``(a|b)*`` and a page of nested alternations on the same machine.
    The state id breaks ties, which is what makes the output the *same* page
    twice: a tie broken by set order would produce a different expression on
    different runs of the same program (docs/LESSONS.md).
    """
    incoming = {state: 0 for state in interior}
    outgoing = {state: 0 for state in interior}
    for source, target in labels:
        if source == target:
            continue  # a self-loop is not a way through, it is a way round
        if target in incoming:
            incoming[target] += 1
        if source in outgoing:
            outgoing[source] += 1
    return min(interior, key=lambda state: (incoming[state] * outgoing[state],
                                            state))


def _rip(state: StateId, labels: Dict[_Edge, Node]) -> None:
    """Remove ``state``, keeping every path that went through it.

    For each way in and each way out, the detour ``in · loop* · out`` is added
    to whatever already labelled that pair. The ``loop*`` is where the star
    comes from and the only place it does: a self-loop is the one thing a
    finite path cannot enumerate, so it is the reason regular expressions need
    an operator that automata express with a cycle.
    """
    loop = labels.get((state, state))
    repeat = EmptyWord() if loop is None else _star(loop)

    sources = sorted({source for source, target in labels
                      if target == state and source != state})
    targets = sorted({target for source, target in labels
                      if source == state and target != state})

    for source in sources:
        for target in targets:
            detour = _concat([labels[(source, state)], repeat,
                              labels[(state, target)]])
            existing = labels.get((source, target))
            labels[(source, target)] = (
                detour if existing is None else _alt([existing, detour]))

    for edge in [edge for edge in labels if state in edge]:
        del labels[edge]


def _eliminate(machine: NFA) -> Node:
    """The expression for ``machine``, by ripping every original state out."""
    if machine.initial is None:
        return EmptyLanguage()

    # Two ends are added so that the start state may be re-entered and an
    # accepting state may be left, both of which the ripping step assumes: it
    # never gives the new start an incoming edge or the new accept an outgoing
    # one, so those two survive to the end by construction rather than by being
    # skipped over.
    start = _fresh("<start>", machine.states)
    accept = _fresh("<accept>", machine.states | {start})

    labels = _labels_of(machine)
    labels[(start, machine.initial)] = EmptyWord()
    for state in sorted(machine.accept):
        labels[(state, accept)] = EmptyWord()

    interior = sorted(machine.states)
    while interior:
        state = _next_to_rip(interior, labels)
        interior.remove(state)
        _rip(state, labels)

    # No label left between the two ends means no path survived: the machine
    # accepts nothing. Every label was built through the smart constructors, so
    # what comes back is already simplified.
    return labels.get((start, accept), EmptyLanguage())


def from_automaton(automaton: AnyAutomaton) -> str:
    """A regular expression denoting the language ``automaton`` recognises.

    The other half of Kleene's theorem, and the harder half to trust: many
    implementations of it are correct about the language and produce an
    expression a page long. Every intermediate label here goes through
    :func:`simplify`, and the state with the fewest paths through it is removed
    first, which is what keeps the answer to a size a person can check.

    Takes either machine. A DFA is read as an NFA first -- the conversion is
    exact and never fails -- rather than writing the elimination twice, since
    the algorithm never asks whether the machine is deterministic and a second
    copy would be a second place to be wrong.

    Two cases are worth stating:

    * **No accepting state** (or none reachable) gives ``∅``, which is the
      honest answer: the language is empty.
    * **No initial state** also gives ``∅``. This is the one place the
      distinction the rest of the engine keeps -- ``initial=None`` means "no
      language defined yet", not "the empty language" -- cannot be preserved,
      because a regular expression can only denote a language and "undefined"
      is not one. A front end that cares should ask before calling.

    The result is stable: the same automaton produces the same expression on
    every run and in every process, so it can be written into a saved file, a
    test, or a diff.

    It is *not* the shortest expression for the language, and no algorithm here
    tries to be -- the answer tracks the machine it was given, so the way to a
    short expression is a small machine. ``determinize`` then ``minimize``
    first turns ``a*|(b|a+b)+a*`` into ``(a|b)*`` on the same language. That is
    left to the caller rather than done here, because determinising is
    exponential in the worst case and a function that sometimes takes a moment
    and sometimes takes a minute is one nobody can put behind a menu item.

    Args:
        automaton: The machine to describe. Never mutated.

    Returns:
        The expression as text, ready to be handed back to :func:`parse`. Round
        trips through the language: ``to_nfa(from_automaton(a))`` recognises
        exactly what ``a`` did.
    """
    machine = from_dfa(automaton) if isinstance(automaton, DFA) else automaton
    return _eliminate(machine).pattern()
