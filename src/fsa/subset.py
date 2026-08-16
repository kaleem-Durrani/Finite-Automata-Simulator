"""The subset construction: one DFA state per set of NFA states.

The theorem this implements is old and short -- every NFA has an equivalent DFA
-- and stating it is not the lesson. *Which* DFA, and where each of its states
came from, is; so this module is built around making that visible rather than
around producing the smallest machine it can. Minimisation is a separate step
with a separate answer (:func:`fsa.minimize.minimize`), and running the two
together would hide the states the construction actually invented.

Three decisions carry the module.

**Reachable subsets only.** The textbook definition takes Q' = P(Q), and for ten
NFA states that is 1024 DFA states of which a handful are ever entered. The walk
here starts at the epsilon closure of the start state and spreads outward,
breadth first, so it builds exactly the subsets some word can reach. That does
not dodge the exponential -- "the nth symbol from the end is an a" genuinely
needs 2^n states and gets every one of them -- it only stops us paying for it
where it is not real, which is also what keeps the result a diagram a person can
look at.

**A state is named after the set it came from.** ``{q0,q2}``, sorted. The name
*is* the explanation: it says "the NFA might be in q0 or in q2", and reading
that off the diagram is the whole exercise. Sorted because Python randomises
string hashing per process, so an unsorted name would come out differently on
different runs of the same program and two determinisations of one machine would
not compare equal (see docs/LESSONS.md).

**The empty subset is a state, not an absence.** It is the subset where every
branch has died, it is reached by an ordinary move like any other, and it moves
to itself on every symbol -- so writing it down is what makes the result
complete, with no special case anywhere in the loop. It is a genuine trap by
:func:`fsa.analysis.is_trap`'s derived definition, not a flagged one.

This is deliberately not a method on :class:`~fsa.nfa.NFA` and not part of
:mod:`fsa.ops`. It builds a machine with a different state set, which is exactly
why :func:`fsa.nfa.to_dfa` refuses a nondeterministic machine instead of quietly
doing this: a function that sometimes rewrites every state id and sometimes does
not is one whose result a caller cannot reason about.
"""

from typing import Dict, Final, FrozenSet, Iterable, List, Set, Tuple

from fsa.automaton import DFA
from fsa.nfa import NFA
from fsa.symbols import StateId, Symbol

#: One configuration of the NFA, and therefore one state of the DFA. The whole
#: construction is this alias taken seriously.
Subset = FrozenSet[StateId]


def subset_name(states: FrozenSet[StateId]) -> StateId:
    """The DFA state id for a set of NFA states: ``{q0,q2}``.

    Sorted, so that the name depends on the set and not on the order the set
    happens to iterate in -- which changes between processes. Ordinary
    lexicographic order, as everywhere else in the engine, so ``q10`` precedes
    ``q2``; a numeric sort here would be a second ordering rule for state ids
    and disagree with every sorted list the UI already shows.

    A singleton keeps its braces (``{q0}``, never ``q0``). The braces are the
    part that says this is a *set* of NFA states rather than the NFA state of
    the same name, and dropping them for the one-element case would make the
    two indistinguishable in a table -- precisely when a student is trying to
    work out which is which.
    """
    return "{" + ",".join(sorted(states)) + "}"


#: The trap: the subset where every branch has died. Spelled by the rule above
#: rather than written out again, so the two cannot drift apart.
EMPTY_SUBSET: Final[StateId] = subset_name(frozenset())

#: What the empty subset is called on screen. ``{}`` in a circle reads as a
#: state whose name failed to render; "trap" says what the state is for.
TRAP_LABEL: Final[str] = "trap"


def _labels(automaton: NFA, subsets: Iterable[Subset]) -> Dict[StateId, str]:
    """Display names for the subsets that deserve one.

    Two cases and no others. The empty subset is labelled, because its id alone
    does not say what it is. A subset with a labelled member is labelled from
    its members' display names -- ``{start,seen a}`` -- for the same reason
    :func:`fsa.minimize.quotient` writes ``q0+q2``: "which of my states is this
    one?" is the first question anyone asks of a derived machine.

    Everything else keeps no label at all. Its id already reads exactly as its
    label would, and writing that back would count as a real edit -- dirtying
    the file, spending an undo slot -- for a change nobody can see. See
    :meth:`fsa.automaton.DFA.with_label_removed`.

    Labels are cosmetic; the engine never reads them, so nothing here can change
    the language.
    """
    labels: Dict[StateId, str] = {}
    for subset in subsets:
        if not subset:
            labels[EMPTY_SUBSET] = TRAP_LABEL
        elif any(state in automaton.labels for state in subset):
            # Sorted by state id, not by label, so the label lists its members
            # in the same order the id does and the two read as one name.
            labels[subset_name(subset)] = "{" + ",".join(
                automaton.label_of(state) for state in sorted(subset)) + "}"
    return labels


def determinize(automaton: NFA) -> DFA:
    """The subset construction: an equivalent DFA, reachable subsets only.

    Every word gets the same verdict from the result as from ``automaton``,
    including words containing symbols outside the alphabet -- the alphabet is
    carried over untouched, so both machines reject those the same way, at the
    same position, for the same reason.

    The *reason* for the other rejections can differ, and that is not a defect
    to be fixed. An NFA that runs out of branches stops with
    :attr:`~fsa.simulate.Verdict.REJECT_NO_TRANSITION`; the DFA built here walks
    into the empty subset instead and reads the rest of the word sitting in it,
    finishing with :attr:`~fsa.simulate.Verdict.REJECT_NON_ACCEPTING`. Same
    verdict, different story -- the same trade :func:`fsa.ops.complete` makes,
    and the price of the guarantee below.

    Guarantees about the result:

    * **Complete.** delta is total: every state has a move on every symbol of
      the alphabet. The empty subset is what makes that true, and it is a real
      reachable state rather than a bolted-on trap -- it appears only when some
      reachable subset actually has a dead end, and it loops to itself forever
      after. (A machine with no states, from the case below, is complete
      vacuously.)
    * **Deterministic**, by type as well as by construction: one target per
      ``(state, symbol)``, and no epsilon moves, since every closure was already
      taken while building the subsets.
    * **Reachable.** Every state is reached by some word, because the walk only
      ever names subsets it has just arrived at.
    * **Not necessarily minimal.** Two subsets can recognise the same language
      from here on and both survive; :func:`fsa.minimize.minimize` is the step
      that merges them, and it is deliberately not folded in.

    An automaton with **no initial state** determinizes to an automaton with no
    states, keeping the alphabet. Nothing is reachable when there is nowhere to
    start, and both machines answer every word with
    :attr:`~fsa.simulate.Verdict.NO_INITIAL_STATE`. Inventing a start state here
    would turn "no language defined yet" into "the empty language", which this
    codebase reads as a different claim -- the same rule
    :func:`fsa.ops.trim` and :meth:`fsa.automaton.DFA.without_state` follow.

    Cost is one visit per reachable subset, each doing |Sigma| unions and
    closures. The number of reachable subsets is at worst 2^|Q|, which is a fact
    about the problem rather than about this implementation: the languages that
    reach the bound need every one of those states. A front end offering this on
    a large machine should expect that and say so before it runs.

    Args:
        automaton: The machine to determinize. Never mutated.

    Returns:
        A DFA over the same alphabet, whose state ids name the subsets they
        came from (see :func:`subset_name`).
    """
    if automaton.initial is None:
        return DFA(alphabet=automaton.alphabet)

    symbols = tuple(sorted(automaton.alphabet))
    start = automaton.epsilon_closure([automaton.initial])

    # `pending` is both the queue and the record of discovery order: the index
    # walks forward while the list grows behind it, which is a breadth-first
    # walk with no second structure to keep in step. Everything that leaves this
    # function is derived from `pending`, in that order -- `seen` is only ever
    # asked whether it holds something, never iterated, because a set's order is
    # not stable across processes.
    pending: List[Subset] = [start]
    seen: Set[Subset] = {start}
    transitions: Dict[Tuple[StateId, Symbol], StateId] = {}

    index = 0
    while index < len(pending):
        subset = pending[index]
        index += 1
        for symbol in symbols:
            moved: Set[StateId] = set()
            for state in sorted(subset):
                moved |= automaton.targets(state, symbol)
            # Closed on the way out, exactly as fsa.nfa.run does it, so a subset
            # is always the full set of states the machine could be standing in.
            target = automaton.epsilon_closure(moved)

            # Written unconditionally, the empty target included: that one line
            # is the whole of completeness. A subset with no move lands in {}
            # instead of leaving a hole in delta, and {} is then visited like
            # any other subset -- it has no members, so every symbol takes it
            # back to itself and it needs no special case to be a trap.
            transitions[(subset_name(subset), symbol)] = subset_name(target)
            if target not in seen:
                seen.add(target)
                pending.append(target)

    return DFA(
        states=frozenset(subset_name(subset) for subset in pending),
        alphabet=automaton.alphabet,
        transitions=transitions,
        initial=subset_name(start),
        # One accepting NFA state in the subset is enough -- accepting means
        # *some* branch survived, which is the definition the NFA simulator
        # uses, transplanted into the state set.
        accept=frozenset(subset_name(subset) for subset in pending
                         if subset & automaton.accept),
        labels=_labels(automaton, pending),
    )
