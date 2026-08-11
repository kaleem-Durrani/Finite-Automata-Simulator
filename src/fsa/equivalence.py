"""Deciding whether two automata recognise the same language.

A "no" here has to be worth more than a "no". Being told the machine is wrong
teaches nothing; being told it disagrees with the reference on ``0110`` points
at the arrow to look at, and is what lets an exercise grade itself. So both
questions are answered by one search: :func:`counterexample` returns the
shortest word the two machines disagree about, and :func:`equivalent` is
exactly "there is no such word".

The search is a breadth-first walk over *pairs* of states -- the reachable part
of the symmetric-difference product, built as it is explored rather than
materialised as an automaton. Feeding one word to both machines at once means
the path that reached a pair *is* a word that reaches those two states, so the
first pair whose sides disagree about accepting hands back the distinguishing
word for free, and breadth-first order makes it the shortest one. (Ties within
a length are broken by symbol order, so the answer is deterministic -- two runs
on the same inputs cannot produce different counterexamples.)

The delicate part is that delta is partial. A missing transition is not "stay
put", and it is not a reason to refuse the comparison: it is a step into a sink
that accepts nothing and never comes back, which is what
:func:`fsa.simulate.run` already does when it stops and rejects. Modelling it
that way is what makes a word one machine can read and the other cannot come
out as distinguishing -- and it must, because the two machines genuinely
disagree about that word. ``None`` on a side of a pair is that sink.

Three consequences fall out of the same decision, and each is a case this
module is expected to answer rather than reject:

* An automaton with no initial state accepts nothing, so it starts in the sink
  and is equivalent to any machine recognising the empty language.
* The alphabets need not match. A word is compared over the union, because a
  symbol only one machine knows is rejected by the other for want of being in
  its alphabet -- indistinguishable, from outside, from a missing arrow.
* A symbol in neither alphabet is rejected by both whatever surrounds it, so it
  can never distinguish them and is never searched.
"""

from collections import deque
from typing import Deque, Dict, List, Mapping, Optional, Set, Tuple

from fsa.automaton import DFA
from fsa.symbols import StateId, Symbol

# One state from each machine, reached by reading the same word into both.
# ``None`` is the sink described in the module docstring: no acceptance, no
# outgoing edge, no way back.
Pair = Tuple[Optional[StateId], Optional[StateId]]


def _step(automaton: DFA, state: Optional[StateId],
          symbol: Symbol) -> Optional[StateId]:
    """Follow ``symbol`` from ``state``, reading undefined delta as the sink.

    The sink is absorbing: once a side has fallen out of its machine no later
    symbol can put it back. That is not an approximation -- a run that has
    stopped has stopped -- and it is the one rule that makes the rest correct.
    """
    if state is None:
        return None
    return automaton.target(state, symbol)


def _disagree(left: DFA, right: DFA, pair: Pair) -> bool:
    """Whether the two sides of ``pair`` differ about accepting.

    The sink accepts nothing on either side, which is precisely why it can be
    ``None`` instead of a state that would have to be invented and named.
    """
    left_state, right_state = pair
    return ((left_state is not None and left_state in left.accept)
            != (right_state is not None and right_state in right.accept))


def _word_reaching(pair: Pair, parents: Mapping[Pair, Tuple[Pair, Symbol]]) -> str:
    """Rebuild the word that first reached ``pair`` by walking back up the tree.

    Only the symbol and predecessor of each pair are stored, rather than a word
    per queue entry: the pair space is |Q_left| x |Q_right| at worst, and
    carrying a whole string along with each of those is the difference between
    linear and quadratic memory on the machines where this is slow at all.
    """
    symbols: List[Symbol] = []
    while pair in parents:
        pair, symbol = parents[pair]
        symbols.append(symbol)
    symbols.reverse()
    return "".join(symbols)


def counterexample(left: DFA, right: DFA) -> Optional[str]:
    """The shortest word the two automata disagree about, or ``None``.

    ``None`` means no such word exists -- the languages are equal, which is
    what :func:`equivalent` reports. When a word is returned,
    ``accepts(left, word) != accepts(right, word)`` holds, and no shorter word
    has that property.

    Note that ``""`` is a perfectly good counterexample: it is what comes back
    when the two machines already disagree about their start states. Callers
    must test the result against ``None``, not for truthiness, or they will
    read "they differ on the empty string" as "they agree".
    """
    # Sorted, so the choice among equally short counterexamples is stable
    # across runs and across dict orderings rather than merely arbitrary.
    symbols = sorted(left.alphabet | right.alphabet)

    start: Pair = (left.initial, right.initial)
    parents: Dict[Pair, Tuple[Pair, Symbol]] = {}
    seen: Set[Pair] = {start}
    queue: Deque[Pair] = deque([start])

    while queue:
        pair = queue.popleft()
        if _disagree(left, right, pair):
            return _word_reaching(pair, parents)

        # Both sides have fallen out of their machines: every continuation
        # leads back here, and this pair already agrees, so nothing below it
        # can ever distinguish them.
        if pair == (None, None):
            continue

        for symbol in symbols:
            successor: Pair = (_step(left, pair[0], symbol),
                               _step(right, pair[1], symbol))
            if successor not in seen:
                seen.add(successor)
                parents[successor] = (pair, symbol)
                queue.append(successor)

    # The pair space is finite -- at most (|Q_left| + 1) x (|Q_right| + 1),
    # counting each sink -- and no pair is enqueued twice, so exhausting the
    # queue is a proof, not a timeout.
    return None


def equivalent(left: DFA, right: DFA) -> bool:
    """Whether the two automata recognise the same language.

    Defined as, and computed as, "there is no word they disagree about", so
    this can never drift out of step with :func:`counterexample`: one is
    literally the other's answer tested against ``None``.
    """
    return counterexample(left, right) is None
