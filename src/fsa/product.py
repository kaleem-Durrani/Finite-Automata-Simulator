"""Product constructions: the boolean algebra of regular languages.

Union, intersection, difference and symmetric difference are one algorithm run
with four different accept tests, so there is one implementation --
:func:`product` -- and four one-line wrappers. Written out separately, three of
them get the alphabet handling right and the fourth does not.

Two decisions carry the weight.

**The product alphabet is the union of the two alphabets.** Anything smaller
throws words away: membership in this engine is :func:`fsa.simulate.accepts`,
which answers for *every* string -- a symbol outside an automaton's own alphabet
is a rejection (``REJECT_SYMBOL_NOT_IN_ALPHABET``), not an error. So each
operand denotes a set of strings, those sets can be unioned and intersected
whatever their alphabets, and the product has to reproduce the answer. A result
that could not even read ``b`` would not be the union of ``{a}`` and ``{b}``.

**A side that can no longer accept is tracked as ``None``**, a marker outside
its own state set. The union alphabet is what forces this: on a symbol the left
operand has never heard of, the left operand's delta is undefined -- the very
case a partial delta already presents when an arrow is simply missing. The two
are the same fact ("this side is finished") and get the same representation.
Refusing to represent it, and leaving the *pair's* transition undefined instead,
is the classic wrong implementation: the product would then die where only one
operand had died, and the union of ``{a}`` and ``{b}`` would accept neither
``a`` nor ``b``.

Because that finished-side pair is an ordinary state with ordinary edges, the
result is always complete, however partial the operands were -- so
:func:`complement` applies to any product without a completion step first.
"""

from collections import deque
from typing import Callable, Deque, Dict, Iterable, List, Optional, Set, Tuple

from fsa.analysis import missing_transitions
from fsa.automaton import DFA
from fsa.errors import IncompleteAutomatonError
from fsa.symbols import StateId, Symbol

AcceptWhen = Callable[[bool, bool], bool]
"""Decides acceptance from whether each side halted in an accepting state."""

# One side's position: a state of that operand, or None once it has failed.
_Side = Optional[StateId]

# A configuration of the two operands running in lockstep.
_Pair = Tuple[_Side, _Side]

# How a failed side is spelled in a pair's id. Any string is a legal state id,
# so this cannot be reserved; _name_pairs breaks the collision if a real state
# is called "-".
_FAILED = "-"


def product(left: DFA, right: DFA, accept_when: AcceptWhen) -> DFA:
    """Run two automata in lockstep and decide acceptance from the pair.

    The states of the result are the *reachable* pairs, found breadth-first
    from ``(left.initial, right.initial)``. The full cross product is
    ``|Q1| x |Q2|`` states and most of it is usually unreachable: a machine
    against itself -- what an equivalence check feeds to
    :func:`symmetric_difference` in the case that matters -- reaches only its
    own diagonal, ``|Q|`` states out of ``|Q|^2``.

    Delta is total on the result. Every pair has a successor on every symbol,
    because a side with nowhere to go moves to ``None`` and stays there rather
    than leaving the pair's transition undefined.

    Args:
        left: The first operand.
        right: The second operand.
        accept_when: Called with ``(left is accepting, right is accepting)``
            for each reachable pair; a side that has failed reports ``False``.
            Determines the accepting set and nothing else.

    Returns:
        A complete DFA over ``left.alphabet | right.alphabet`` whose states are
        named for the pairs they stand for, with no labels -- the id already
        spells the pair, and a second copy of that fact could only disagree
        with it. For every word ``w`` over the product alphabet::

            accepts(product(l, r, f), w) == f(accepts(l, w), accepts(r, w))

        For words containing symbols outside the product alphabet, no DFA over
        it accepts anything, so the identity survives only for an
        ``accept_when`` with ``f(False, False) == False`` -- true of all four
        boolean operations below.

        If either operand has no initial state, the result has none either: an
        operand that does not define a language yet cannot be combined into one
        that does. The result is then stateless, and carries only the alphabet.
        Returning the empty language instead would answer a question nobody
        asked, the same silent promotion :func:`fsa.ops.complete` refuses to
        make.
    """
    alphabet = left.alphabet | right.alphabet

    if left.initial is None or right.initial is None:
        return DFA(alphabet=alphabet)

    # Sorted, because the pairs are named in discovery order and set iteration
    # order is not stable across runs. Two structurally equal inputs must give
    # structurally equal results, or equality on the output means nothing.
    symbols = tuple(sorted(alphabet))

    start: _Pair = (left.initial, right.initial)
    order: List[_Pair] = [start]
    seen: Set[_Pair] = {start}
    edges: Dict[Tuple[_Pair, Symbol], _Pair] = {}

    queue: Deque[_Pair] = deque([start])
    while queue:
        pair = queue.popleft()
        for symbol in symbols:
            step = (_advance(left, pair[0], symbol),
                    _advance(right, pair[1], symbol))
            edges[(pair, symbol)] = step
            if step not in seen:
                seen.add(step)
                order.append(step)
                queue.append(step)

    names = _name_pairs(order)
    # One constructor call: every builder method re-validates the whole
    # automaton, and there are |pairs| x |Sigma| transitions to install.
    return DFA(
        states=frozenset(names.values()),
        alphabet=alphabet,
        transitions={
            (names[pair], symbol): names[step]
            for (pair, symbol), step in edges.items()
        },
        initial=names[start],
        accept=frozenset(
            names[pair] for pair in order
            if accept_when(_is_accepting(left, pair[0]),
                           _is_accepting(right, pair[1]))
        ),
    )


def union(left: DFA, right: DFA) -> DFA:
    """Words accepted by either operand."""
    return product(left, right, lambda in_left, in_right: in_left or in_right)


def intersection(left: DFA, right: DFA) -> DFA:
    """Words accepted by both operands."""
    return product(left, right, lambda in_left, in_right: in_left and in_right)


def difference(left: DFA, right: DFA) -> DFA:
    """Words the left operand accepts and the right one does not.

    Needs no completion of either operand, though the textbook identity is
    ``left & complement(right)`` and :func:`complement` insists on one. The
    finished-side marker already supplies what completion would supply: a run
    the right operand cannot continue is a run the right operand rejects, which
    is exactly when the difference should accept.
    """
    return product(left, right,
                   lambda in_left, in_right: in_left and not in_right)


def symmetric_difference(left: DFA, right: DFA) -> DFA:
    """Words exactly one operand accepts.

    Empty precisely when the two recognise the same language, so this is the
    construction an equivalence check is built on: any reachable accepting pair
    is a word the two disagree about, and the shortest path to one is the
    shortest such word.
    """
    return product(left, right, lambda in_left, in_right: in_left != in_right)


def complement(automaton: DFA) -> DFA:
    """Every word over Sigma the automaton rejects: delta kept, F flipped.

    Requires delta to be total, and refuses rather than guesses. On a partial
    automaton, flipping the accepting states does *not* complement the
    language: a word that stops for want of an arrow is rejected by the
    original and rejected again by the flipped copy, so it ends up in neither
    language and the two together no longer cover Sigma*. Completion is a real
    step here, not a formality, which is why the refusal is an exception and
    not a quiet correction.

    The complement is relative to Sigma*: a word containing a symbol outside
    the alphabet is not over Sigma at all, and is in neither language.

    An automaton with no initial state is complemented like any other. Its
    language was undefined and stays undefined -- there is nothing to get
    wrong, and refusing would make this the one operation here that treats a
    missing start state as an error.

    Raises:
        IncompleteAutomatonError: if delta is undefined anywhere.
    """
    missing = missing_transitions(automaton)
    if missing:
        shown = ", ".join(f"{state} on '{symbol}'" for state, symbol in missing[:4])
        if len(missing) > 4:
            shown += f", and {len(missing) - 4} more"
        raise IncompleteAutomatonError(
            f"cannot complement: delta is undefined for {len(missing)} pair(s) "
            f"({shown}). Flipping the accepting states of a partial automaton "
            f"does not complement its language -- a word that stops for want of "
            f"an arrow is rejected by the original and by the flipped copy "
            f"alike, so it lands in neither. Complete it first: "
            f"fsa.ops.complete adds a trap state without changing the "
            f"language, and complementing that result is correct."
        )

    return DFA(
        states=automaton.states,
        alphabet=automaton.alphabet,
        transitions=automaton.transitions,
        initial=automaton.initial,
        accept=automaton.states - automaton.accept,
        labels=automaton.labels,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _advance(automaton: DFA, state: _Side, symbol: Symbol) -> _Side:
    """Where one side goes on ``symbol``, or ``None`` if it is finished.

    ``None`` is absorbing, and three different situations arrive at it: the
    side had already failed, the symbol is not in this side's alphabet, or
    delta is undefined at this state. They mean the same thing to the product
    -- this side rejects the word and every extension of it -- so they are one
    branch. The middle case needs no test of its own: the constructor forbids
    transitions on symbols outside the alphabet, so ``target`` already returns
    ``None`` for a foreign symbol.
    """
    if state is None:
        return None
    return automaton.target(state, symbol)


def _is_accepting(automaton: DFA, state: _Side) -> bool:
    """Whether a side is in an accepting state. A finished side never is."""
    return state is not None and state in automaton.accept


def _name_pairs(pairs: Iterable[_Pair]) -> Dict[_Pair, StateId]:
    """Give each pair an id that says which pair it is.

    ``(q0,p1)`` reads as what it is, which is the whole point once the result
    is drawn on screen. It is not injective on its own, though: state ids are
    opaque strings the engine never parses, so ``("a,b", "c")`` and
    ``("a", "b,c")`` both spell ``(a,b,c)``. A collision escalates to
    ``(a,b,c)#1``, ``(a,b,c)#2``, ... -- the same "first name that is free"
    rule :func:`fsa.ops.complete` uses for its trap -- so a name always exists
    and, since the caller supplies the pairs in a deterministic order, is
    always the same name.
    """
    names: Dict[_Pair, StateId] = {}
    used: Set[StateId] = set()
    for pair in pairs:
        base = f"({_side_name(pair[0])},{_side_name(pair[1])})"
        name = base
        suffix = 1
        while name in used:
            name = f"{base}#{suffix}"
            suffix += 1
        names[pair] = name
        used.add(name)
    return names


def _side_name(state: _Side) -> str:
    """How one half of a pair is written in the pair's id."""
    return _FAILED if state is None else state
