"""Minimisation: the fewest states that still recognise the language.

Moore's table-filling algorithm, and only Moore's. Hopcroft's is asymptotically
better and pedagogically worse -- its refinement queue is an optimisation
nobody can watch, while the marking table is a grid a student can fill in by
hand and then check against the tool. Shipping both would mean two answers to
one question, and two places for that answer to be wrong.

So the table is the primary artifact here. :func:`marking_table` returns it as
a value, and :func:`minimize` derives the smaller machine *from that same
table* rather than running a partition refinement of its own. The picture the
GUI draws and the automaton the tool hands back therefore cannot disagree --
the same discipline that makes ``grouped_transitions`` a derived view rather
than a second copy of delta.

Minimisation is defined for a *complete, reachable* automaton, and the automata
this program edits are routinely neither. Both public functions therefore
prepare their input first -- drop what nothing can reach, then complete what is
left -- and the table records what that preparation did, so a front end can
show the user the machine the table is actually about.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, FrozenSet, List, Mapping, Optional, Set, Tuple

from fsa.analysis import reachable
from fsa.automaton import DFA
from fsa.errors import UnknownStateError
from fsa.ops import complete
from fsa.symbols import StateId, Symbol

#: An unordered pair of distinct states, held in sorted order so that one pair
#: has exactly one spelling. Every table lookup canonicalises before comparing.
Pair = Tuple[StateId, StateId]

#: The only method implemented. See the module docstring for why there is one.
MOORE = "moore"

#: Longest merged class name worth showing. "q1+q2" says which states
#: became one, which is the reason the label exists at all; a compounded
#: set name -- what minimising the subset construction produces -- is
#: noise no canvas can fit, so the state keeps its own id instead.
MAX_MERGED_LABEL = 24


def _canonical(first: StateId, second: StateId) -> Pair:
    """The pair ``{first, second}`` written the one agreed way: sorted."""
    return (first, second) if first <= second else (second, first)


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Mark:
    """One filled cell: a distinguishable pair, and the reason it is one.

    ``round`` is also the length of the shortest word that tells the two states
    apart -- round 0 pairs are separated by the empty string (one accepts, the
    other does not), and a round *n* pair is separated by one symbol followed
    by a word of length *n* - 1. That equivalence is what makes the round
    number worth showing rather than an implementation detail of the loop.
    """

    pair: Pair
    round: int
    symbol: Optional[Symbol] = None
    """The symbol that separates them. ``None`` only for round 0, where no
    symbol is read at all."""

    successors: Optional[Pair] = None
    """Where ``symbol`` takes the two states: a pair marked one round earlier."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair", _canonical(*self.pair))
        if self.successors is not None:
            object.__setattr__(self, "successors", _canonical(*self.successors))

    def explain(self) -> str:
        """One sentence a student can check by hand."""
        left, right = self.pair
        if self.symbol is None or self.successors is None:
            return (f"{left} and {right} are separated by the empty string: "
                    f"one accepts and the other does not (round 0)")
        return (f"{left} and {right} are separated by '{self.symbol}', which "
                f"takes them to ({self.successors[0]}, {self.successors[1]}) "
                f"-- already separated in round {self.round - 1} (round "
                f"{self.round})")


@dataclass(frozen=True, slots=True)
class MarkingTable:
    """The result of table-filling: which pairs are distinguishable, and when.

    The automaton the table is *about* travels with it. That machine is not
    necessarily the one that was passed in -- unreachable states are gone and a
    trap may have been added -- and a front end that rendered the table's rows
    against the original's state list would draw a grid missing a column.
    Keeping them together makes that mistake unavailable.

    A pair absent from :attr:`marks` was never marked, which is the algorithm's
    conclusion that the two states are equivalent: no word tells them apart.
    """

    automaton: DFA
    """The prepared machine: reachable, and complete."""

    marks: Tuple[Mark, ...] = ()
    """Every distinguishable pair, ordered by round and then by pair."""

    unreachable: Tuple[StateId, ...] = ()
    """States dropped before filling the table. No word reaches them, so they
    cannot affect the language -- but left in they would be merged into some
    class and pollute the partition."""

    invented_trap: Optional[StateId] = None
    """The trap :func:`fsa.ops.complete` had to add, if delta was partial.
    ``None`` means the input was already complete. A front end should say which
    state it drew itself and which one the tool supplied."""

    def __post_init__(self) -> None:
        # Canonical order regardless of how the value was built, so two tables
        # for the same machine are equal and hashable as values.
        object.__setattr__(self, "marks", tuple(sorted(
            self.marks, key=lambda mark: (mark.round, mark.pair))))
        object.__setattr__(self, "unreachable", tuple(sorted(self.unreachable)))

    # ------------------------------------------------------------------
    # The grid
    # ------------------------------------------------------------------

    @property
    def states(self) -> Tuple[StateId, ...]:
        """The states the table covers, sorted: the axes of the grid."""
        return tuple(sorted(self.automaton.states))

    @property
    def pairs(self) -> Tuple[Pair, ...]:
        """Every unordered pair of distinct states, sorted.

        The whole lower triangle, marked or not -- a renderer needs the empty
        cells as much as the full ones.
        """
        states = self.states
        return tuple(
            (left, right)
            for index, left in enumerate(states)
            for right in states[index + 1:]
        )

    def by_pair(self) -> Mapping[Pair, Mark]:
        """The marks indexed by pair.

        Computed on demand rather than stored, for the same reason
        ``grouped_transitions`` is: a second stored copy is a second thing to
        keep in step. Build it once and index it when drawing a whole grid;
        :meth:`mark_of` is the right call for a single cell.
        """
        return MappingProxyType({mark.pair: mark for mark in self.marks})

    def mark_of(self, first: StateId, second: StateId) -> Optional[Mark]:
        """The mark for a pair in either order, or ``None`` if never marked.

        Raises :class:`UnknownStateError` for a state the table does not cover.
        Answering "equivalent" for a state that was dropped as unreachable
        would be a wrong answer dressed as a real one.
        """
        for state in (first, second):
            if state not in self.automaton.states:
                raise UnknownStateError(f"{state!r} is not in this table")
        if first == second:
            return None
        pair = _canonical(first, second)
        for mark in self.marks:
            if mark.pair == pair:
                return mark
        return None

    def is_distinguishable(self, first: StateId, second: StateId) -> bool:
        """Whether some word tells the two states apart. A state is never
        distinguishable from itself."""
        return self.mark_of(first, second) is not None

    # ------------------------------------------------------------------
    # Rounds
    # ------------------------------------------------------------------

    @property
    def rounds(self) -> int:
        """How many rounds marked something. 0 when nothing was ever marked."""
        return max((mark.round for mark in self.marks), default=-1) + 1

    def marks_in_round(self, number: int) -> Tuple[Mark, ...]:
        """The cells filled in one round -- one frame of the animation."""
        return tuple(mark for mark in self.marks if mark.round == number)

    # ------------------------------------------------------------------
    # The partition
    # ------------------------------------------------------------------

    @property
    def equivalent_pairs(self) -> Tuple[Pair, ...]:
        """The cells the algorithm left empty: pairs no word separates."""
        marked = {mark.pair for mark in self.marks}
        return tuple(pair for pair in self.pairs if pair not in marked)

    def equivalence_classes(self) -> Tuple[FrozenSet[StateId], ...]:
        """The states grouped by indistinguishability, in sorted order.

        Indistinguishability is an equivalence relation -- transitivity is what
        makes the empty cells a *partition* rather than a mere symmetric
        relation -- so a state joins a class if it is unmarked against that
        class's first member. There is no need to check it against the rest,
        and checking anyway would only hide a bug in the marking if one existed.
        """
        marked = {mark.pair for mark in self.marks}
        states = self.states
        assigned: Set[StateId] = set()
        classes: List[FrozenSet[StateId]] = []

        for index, left in enumerate(states):
            if left in assigned:
                continue
            members = {left}
            for right in states[index + 1:]:
                if right not in assigned and (left, right) not in marked:
                    members.add(right)
            assigned |= members
            classes.append(frozenset(members))
        return tuple(classes)


# ---------------------------------------------------------------------------
# Filling the table
# ---------------------------------------------------------------------------


def _prepare(automaton: DFA) -> Tuple[DFA, Tuple[StateId, ...], Optional[StateId]]:
    """Make an automaton fit to minimise: reachable first, then complete.

    The order is deliberate. Pruning first means completion only invents a trap
    when a state some word can actually reach needs one; completing first would
    let a state nothing can reach drag a trap into the result and inflate the
    machine the user gets back.

    An automaton with no initial state prunes to nothing, which is the honest
    answer: no word reaches any state, so there is no language to preserve.
    """
    kept = reachable(automaton)
    dropped = tuple(sorted(automaton.states - kept))

    pruned = automaton
    if dropped:
        # Filtering on the source alone is enough: a transition out of a
        # reachable state lands in a reachable state by the definition of
        # reachability. If that ever stopped holding the constructor would
        # raise UnknownStateError rather than quietly drop the edge.
        pruned = DFA(
            states=kept,
            alphabet=automaton.alphabet,
            transitions={
                (source, symbol): target
                for (source, symbol), target in automaton.transitions.items()
                if source in kept
            },
            initial=automaton.initial,
            accept=automaton.accept & kept,
            labels=automaton.labels,
        )

    prepared, trap = complete(pruned)
    return prepared, dropped, trap


def _fill(automaton: DFA) -> Tuple[Mark, ...]:
    """Moore's marking, round by round, over a complete automaton.

    Round 0 marks every pair split by acceptance. Round *n* marks every pair
    that some symbol carries to a pair marked in round *n* - 1. Insisting on
    the previous round rather than accepting any earlier one loses nothing --
    a pair pointing at an older mark was itself marked in the round after that
    one -- and it buys the invariant that a pair's round number is the length
    of the shortest word separating it, which is the fact worth putting on
    screen.

    Cost is O(rounds x |Q|^2 x |Sigma|). That is worse than Hopcroft on paper
    and irrelevant at the sizes a person draws by hand.
    """
    states = tuple(sorted(automaton.states))
    alphabet = tuple(sorted(automaton.alphabet))
    pairs = [
        (left, right)
        for index, left in enumerate(states)
        for right in states[index + 1:]
    ]

    marks: Dict[Pair, Mark] = {}
    frontier: List[Pair] = []
    for pair in pairs:
        left, right = pair
        if (left in automaton.accept) != (right in automaton.accept):
            marks[pair] = Mark(pair=pair, round=0)
            frontier.append(pair)

    number = 1
    while frontier:
        separated = set(frontier)
        newly: List[Pair] = []
        for pair in pairs:
            if pair in marks:
                continue
            left, right = pair
            for symbol in alphabet:
                left_target = automaton.target(left, symbol)
                right_target = automaton.target(right, symbol)
                if left_target is None or right_target is None:
                    # Cannot happen here -- the caller completes first -- and
                    # that completion is exactly why. On a partial machine,
                    # skipping is unsound (a symbol one state can read and the
                    # other cannot *does* separate them) and marking is wrong
                    # too. Completion turns the asymmetry into a real pair
                    # (target, trap), which this loop then handles properly.
                    continue
                if left_target == right_target:
                    continue  # a state is never distinguishable from itself
                successors = _canonical(left_target, right_target)
                if successors in separated:
                    marks[pair] = Mark(pair=pair, round=number,
                                       symbol=symbol, successors=successors)
                    newly.append(pair)
                    break
        frontier = newly
        number += 1

    return tuple(marks.values())


def marking_table(automaton: DFA) -> MarkingTable:
    """Fill Moore's table for ``automaton``.

    The input is prepared first: unreachable states are dropped and delta is
    completed. The returned table carries the prepared machine, the states that
    were dropped and the trap that was added, so the front end can render the
    grid and explain both edits.
    """
    prepared, dropped, trap = _prepare(automaton)
    return MarkingTable(
        automaton=prepared,
        marks=_fill(prepared),
        unreachable=dropped,
        invented_trap=trap,
    )


# ---------------------------------------------------------------------------
# Collapsing the table into an automaton
# ---------------------------------------------------------------------------


def quotient(table: MarkingTable) -> DFA:
    """Collapse each equivalence class of ``table`` into a single state.

    Public because a front end that animates the table wants to press "merge"
    at the end without filling the table a second time. :func:`minimize` is
    this applied to a freshly filled table.

    Naming: a class keeps the smallest of its members' ids, so a machine that
    was already minimal comes back with the very same ids (and the GUI keeps
    the coordinates of every state that survives). A class with more than one
    member also gets a label listing what went into it -- ``q0+q2`` -- because
    "where did q2 go?" is the first question anyone asks. Labels are cosmetic;
    the engine never reads them, so this cannot change the language.
    """
    prepared = table.automaton
    classes = list(table.equivalence_classes())

    if table.invented_trap is not None:
        # Delta was partial, so the trap in this table is ours, not the user's.
        # Dropping its class again is what keeps the promise that minimising
        # never adds states: transitions into it simply become undefined, and a
        # run that used to reject in the trap now rejects for want of an arrow.
        # Same verdict, different reason -- the exact trade complete() makes,
        # run backwards.
        trap_class = next(
            (members for members in classes if table.invented_trap in members),
            None)
        # Unless the initial state is in there: that means nothing accepting is
        # reachable at all, and an automaton with no start state reads as "no
        # language defined yet" in this codebase, not "accepts nothing".
        if trap_class is not None and prepared.initial not in trap_class:
            classes = [members for members in classes if members is not trap_class]

    representative: Dict[StateId, StateId] = {}
    for members in classes:
        chosen = min(members)
        for member in members:
            representative[member] = chosen

    states = frozenset(min(members) for members in classes)

    transitions: Dict[Tuple[StateId, Symbol], StateId] = {}
    for members in classes:
        source = min(members)
        for symbol in prepared.alphabet:
            target = prepared.target(source, symbol)
            if target is None:
                continue  # only reachable if a caller built the table by hand
            merged = representative.get(target)
            if merged is None:
                continue  # into the dropped trap class: delta stays undefined
            transitions[(source, symbol)] = merged

    # Round 0 splits accepting from non-accepting, so no class straddles the
    # two and the representative decides for all of its members.
    accept = frozenset(state for state in states if state in prepared.accept)

    labels: Dict[StateId, str] = {}
    for members in classes:
        chosen = min(members)
        if len(members) > 1:
            merged = "+".join(
                prepared.label_of(member) for member in sorted(members))
            # A name nobody can read teaches nothing. Minimising the output of
            # the subset construction compounds names that are already sets --
            # "{q0,q1,q2,q4,q6,q8}+{q1,q3,q5}" -- and on a canvas those run
            # into each other into one unreadable strip. "q1+q2" is worth
            # showing; that is not, so the state keeps its own id instead.
            if len(merged) <= MAX_MERGED_LABEL:
                labels[chosen] = merged
        elif chosen in prepared.labels:
            # Only carry a label that was really there. Writing the id back
            # would look identical on screen but count as an edit -- see
            # DFA.with_label_removed for why that distinction matters.
            labels[chosen] = prepared.labels[chosen]

    return DFA(
        states=states,
        alphabet=prepared.alphabet,
        transitions=transitions,
        initial=None if prepared.initial is None else representative[prepared.initial],
        accept=accept,
        labels=labels,
    )


def minimize(automaton: DFA, method: str = MOORE) -> DFA:
    """The smallest automaton recognising the same language.

    Every word gets the same verdict from the result as from the input --
    including words containing symbols outside the alphabet, since the alphabet
    is carried over unchanged.

    What it does with the awkward inputs, all of which are ordinary here:

    * **Unreachable states are dropped** first. No word reaches them, so they
      cannot change the language; left in, they would join some class and
      inflate it. A gap in delta that only an unreachable state had goes away
      with it -- dropped rather than filled -- so such an automaton can come
      back complete.
    * **A delta still partial after that is completed internally**, because
      table-filling is only sound on a total delta -- and then the invented
      trap's class is removed again, so a partial automaton minimises to a
      partial automaton and the result never has more states than the input.
      Rejections that used to end in that trap go back to ending for want of
      an arrow: the same verdict with the more informative reason
      (:attr:`~fsa.simulate.Verdict.REJECT_NO_TRANSITION`).
    * **A complete automaton minimises to a complete one.** Its trap, if it has
      one, is a state the user drew, and dropping it would answer a question
      nobody asked -- and hand the diagnostics panel a fresh complaint.
    * **The one exception**: when no accepting state is reachable, the whole
      machine collapses to a single non-accepting state that loops on every
      symbol. That state is kept even if it came from completion, because
      returning nothing at all would turn "accepts nothing" into "has no start
      state", which this codebase reads as a different claim.
    * **No initial state** gives back an empty automaton with the alphabet
      preserved: nothing is reachable, and both machines answer every word with
      :attr:`~fsa.simulate.Verdict.NO_INITIAL_STATE`.

    An automaton that is already minimal, reachable and complete comes back
    equal to the input, labels and all, so ``minimize(a) == a`` is a usable
    test for "nothing to do". Minimisation is idempotent in the same strong
    sense: ``minimize(minimize(a)) == minimize(a)``.

    Args:
        automaton: The automaton to minimise. Never mutated.
        method: Only ``"moore"``, matched case-insensitively. The argument
            exists because the CLI and the plan both name it, not because a
            second algorithm is coming.

    Raises:
        ValueError: If ``method`` is anything else.
    """
    if method.strip().lower() != MOORE:
        raise ValueError(
            f"unknown minimisation method {method!r}; this engine implements "
            f"{MOORE!r} only -- Hopcroft's is the faster algorithm and the "
            f"worse lesson, and a second implementation is a second thing to "
            f"be wrong")
    return quotient(marking_table(automaton))
