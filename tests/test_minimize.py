"""Minimisation, and the marking table it is derived from.

Two claims carry this module. The first is that minimising never changes the
language -- checked exhaustively over short words *and* exactly, by walking the
product of the two machines, so a distinguishing word longer than the bound
cannot slip through. The second is that the table is not decoration: the same
value the GUI will draw is what the minimised machine is built from, so every
test of one is a test of the other.

Imports no pygame and touches no display.
"""

from collections import deque
from typing import List, Optional, Tuple

import pytest

import fsa
from fsa import DFA
from fsa.language import words_up_to
from fsa.minimize import Mark, MarkingTable, marking_table, minimize, quotient
from fsa.symbols import StateId

# ---------------------------------------------------------------------------
# Fixtures by hand
# ---------------------------------------------------------------------------


def hopcroft_ullman() -> DFA:
    """The textbook eight-state machine, with its known answer.

    D is unreachable; A~E, B~H, and D~F, so the minimal machine has five
    states. Having one example whose answer is published means the property
    tests below cannot all be passing for the same wrong reason.
    """
    rows = {
        "A": ("B", "F"),
        "B": ("G", "C"),
        "C": ("A", "C"),
        "D": ("C", "G"),
        "E": ("H", "F"),
        "F": ("C", "G"),
        "G": ("G", "E"),
        "H": ("G", "C"),
    }
    automaton = DFA().with_states(sorted(rows))
    for source, (on_zero, on_one) in rows.items():
        automaton = (automaton
                     .with_transition(source, "0", on_zero)
                     .with_transition(source, "1", on_one))
    return automaton.with_initial("A").with_accept("C")


def partial_dfa() -> DFA:
    """a*b, with q1's arrows deliberately missing."""
    return (DFA()
            .with_states(["q0", "q1"])
            .with_transition("q0", "a", "q0")
            .with_transition("q0", "b", "q1")
            .with_accept("q1"))


def parity_with_a_redundant_state() -> DFA:
    """Even-length words over {a}, written with one state too many.

    q1 and q3 both mean "odd so far" and are indistinguishable; a minimiser
    that does nothing would still pass most other tests here.
    """
    return (DFA()
            .with_states(["q0", "q1", "q2", "q3"])
            .with_transition("q0", "a", "q1")
            .with_transition("q1", "a", "q2")
            .with_transition("q2", "a", "q3")
            .with_transition("q3", "a", "q2")
            .with_accept("q0")
            .with_accept("q2"))


# ---------------------------------------------------------------------------
# Exact language equality, so the property tests are proofs and not samples
# ---------------------------------------------------------------------------


def accepts_nothing(automaton: DFA) -> bool:
    """Whether no accepting state is reachable -- an exact emptiness test."""
    return not (fsa.reachable(automaton) & automaton.accept)


def distinguishing_word(first: DFA, second: DFA) -> Optional[str]:
    """The shortest word the two machines disagree on, or ``None``.

    Breadth-first over the product of their completions, which is exact: if the
    two languages differ at all, some pair of states reachable in the product
    disagrees on acceptance, and the path to it is a witness. Completing first
    is what makes the walk total -- and it cannot change either language.
    """
    left, _ = fsa.complete(first)
    right, _ = fsa.complete(second)
    assert left.alphabet == right.alphabet, "minimisation must keep the alphabet"

    if left.initial is None or right.initial is None:
        both_empty = accepts_nothing(left) and accepts_nothing(right)
        return None if both_empty else ""

    start = (left.initial, right.initial)
    seen = {start}
    queue: deque[Tuple[Tuple[StateId, StateId], str]] = deque([(start, "")])
    while queue:
        (here, there), word = queue.popleft()
        if (here in left.accept) != (there in right.accept):
            return word
        for symbol in sorted(left.alphabet):
            ahead = left.target(here, symbol)
            beyond = right.target(there, symbol)
            if ahead is None or beyond is None:
                continue  # both are complete, so this cannot fire
            if (ahead, beyond) not in seen:
                seen.add((ahead, beyond))
                queue.append(((ahead, beyond), word + symbol))
    return None


def assert_same_language(before: DFA, after: DFA) -> None:
    witness = distinguishing_word(before, after)
    assert witness is None, (
        f"{witness!r} is accepted by one and not the other: "
        f"{before!r} vs {after!r}")


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_round_zero_separates_accepting_from_non_accepting():
    table = marking_table(hopcroft_ullman())
    for other in table.states:
        if other == "C":
            continue
        mark = table.mark_of("C", other)
        assert mark is not None and mark.round == 0
        assert mark.symbol is None, "no symbol is read to separate on acceptance"
        assert "empty string" in mark.explain()


def test_a_later_round_names_the_symbol_and_the_pair_it_reduces_to():
    table = marking_table(hopcroft_ullman())
    mark = table.mark_of("B", "F")
    assert mark is not None
    assert mark.round == 1
    assert mark.symbol == "0", "B reads 0 to G, F reads 0 to C, split in round 0"
    assert mark.successors == ("C", "G")
    assert "round 0" in mark.explain()


def test_the_equivalent_pairs_are_exactly_the_textbook_answer():
    table = marking_table(hopcroft_ullman())
    assert table.equivalent_pairs == (("A", "E"), ("B", "H"))
    assert sorted(sorted(members) for members in table.equivalence_classes()) == [
        ["A", "E"], ["B", "H"], ["C"], ["F"], ["G"],
    ]


def test_unreachable_states_are_dropped_before_the_table_is_filled():
    table = marking_table(hopcroft_ullman())
    assert table.unreachable == ("D",)
    assert "D" not in table.states
    assert table.invented_trap is None, "the input was already complete"


def test_the_table_carries_the_machine_it_is_about():
    """Not the input: a renderer drawing the input's states against these rows
    would be missing a column and showing one that was dropped."""
    table = marking_table(partial_dfa())
    assert table.invented_trap == "trap"
    assert table.states == ("q0", "q1", "trap")
    assert fsa.is_complete(table.automaton)


def test_pairs_covers_the_whole_lower_triangle_in_sorted_order():
    table = marking_table(hopcroft_ullman())
    states = table.states
    assert len(table.pairs) == len(states) * (len(states) - 1) // 2
    assert list(table.pairs) == sorted(table.pairs)
    assert all(left < right for left, right in table.pairs)


def test_marks_are_ordered_by_round_then_pair():
    table = marking_table(hopcroft_ullman())
    keys = [(mark.round, mark.pair) for mark in table.marks]
    assert keys == sorted(keys)
    assert table.rounds == max(round_ for round_, _ in keys) + 1


def test_a_pair_reads_the_same_in_either_order():
    table = marking_table(hopcroft_ullman())
    assert table.mark_of("C", "G") == table.mark_of("G", "C")
    assert table.is_distinguishable("G", "C")
    assert not table.is_distinguishable("A", "E")


def test_a_state_is_not_distinguishable_from_itself():
    table = marking_table(hopcroft_ullman())
    assert table.mark_of("A", "A") is None
    assert not table.is_distinguishable("A", "A")


def test_asking_about_a_state_the_table_does_not_cover_is_an_error():
    """D was dropped as unreachable. Answering "equivalent" for it would be a
    wrong answer wearing the same clothes as a right one."""
    table = marking_table(hopcroft_ullman())
    with pytest.raises(fsa.UnknownStateError):
        table.mark_of("A", "D")
    with pytest.raises(fsa.UnknownStateError):
        table.mark_of("nope", "A")


def test_by_pair_agrees_with_mark_of():
    table = marking_table(hopcroft_ullman())
    indexed = table.by_pair()
    for pair in table.pairs:
        assert indexed.get(pair) == table.mark_of(*pair)


def test_marks_in_round_partitions_the_marks():
    table = marking_table(hopcroft_ullman())
    collected = []
    for number in range(table.rounds):
        chunk = table.marks_in_round(number)
        assert chunk, "a round that marked nothing ends the algorithm"
        collected.extend(chunk)
    assert sorted(collected, key=lambda mark: mark.pair) == sorted(
        table.marks, key=lambda mark: mark.pair)


def test_a_table_is_a_value():
    first = marking_table(hopcroft_ullman())
    second = marking_table(hopcroft_ullman())
    assert first == second
    assert hash(first) == hash(second)
    assert {first, second} == {first}


def test_a_pair_is_canonicalised_however_it_is_built():
    mark = Mark(pair=("q9", "q1"), round=2, symbol="a", successors=("q7", "q3"))
    assert mark.pair == ("q1", "q9")
    assert mark.successors == ("q3", "q7")


def test_a_table_with_nothing_in_it():
    table = MarkingTable(automaton=DFA())
    assert table.states == () and table.pairs == ()
    assert table.rounds == 0
    assert table.equivalence_classes() == ()


def test_the_round_is_the_length_of_the_shortest_separating_word():
    """The claim the round number makes, checked against brute force."""
    for automaton in (hopcroft_ullman(), partial_dfa(),
                      parity_with_a_redundant_state()):
        table = marking_table(automaton)
        prepared = table.automaton
        words = list(words_up_to(prepared.alphabet, 6))
        for left, right in table.pairs:
            separating = [
                word for word in words
                if fsa.accepts(prepared.with_initial(left), word)
                != fsa.accepts(prepared.with_initial(right), word)
            ]
            mark = table.mark_of(left, right)
            if mark is None:
                assert not separating, f"{left}/{right} unmarked but separable"
            else:
                assert separating, f"{left}/{right} marked but never separated"
                assert mark.round == min(len(word) for word in separating)


# ---------------------------------------------------------------------------
# Minimising: the known answer
# ---------------------------------------------------------------------------


def test_the_textbook_machine_minimises_to_five_states():
    smaller = minimize(hopcroft_ullman())
    assert len(smaller.states) == 5
    assert_same_language(hopcroft_ullman(), smaller)


def test_a_class_keeps_the_smallest_id_and_says_what_went_into_it():
    smaller = minimize(hopcroft_ullman())
    assert smaller.states == frozenset({"A", "B", "C", "F", "G"})
    assert smaller.label_of("A") == "A+E"
    assert smaller.label_of("B") == "B+H"
    assert smaller.label_of("C") == "C", "a class of one keeps its own name"
    assert "C" not in smaller.labels, "and gains no label it did not have"
    assert smaller.initial == "A"
    assert smaller.accept == frozenset({"C"})


def test_a_redundant_state_is_merged_away():
    smaller = minimize(parity_with_a_redundant_state())
    assert len(smaller.states) == 2
    assert_same_language(parity_with_a_redundant_state(), smaller)
    assert smaller.label_of("q1") == "q1+q3"


def test_an_existing_label_survives_a_class_of_one():
    """C is nobody's twin, so its own name is the whole truth about it."""
    automaton = hopcroft_ullman().with_label("C", "seen ab")
    smaller = minimize(automaton)
    assert smaller.label_of("C") == "seen ab"


def test_a_merged_class_is_labelled_from_the_labels_it_merged():
    automaton = (parity_with_a_redundant_state()
                 .with_label("q1", "odd")
                 .with_label("q3", "odd again"))
    assert minimize(automaton).label_of("q1") == "odd+odd again"


# ---------------------------------------------------------------------------
# The awkward inputs, which are the ordinary ones here
# ---------------------------------------------------------------------------


def test_a_partial_automaton_minimises_to_a_partial_one():
    """No trap appears in the result: the one completion invented is removed
    again, so minimising cannot hand back a bigger machine than it got."""
    before = partial_dfa()
    after = minimize(before)
    assert len(after.states) <= len(before.states)
    assert "trap" not in after.states
    assert not fsa.is_complete(after)
    assert_same_language(before, after)


def test_a_partial_rejection_keeps_its_more_informative_reason():
    after = minimize(partial_dfa())
    result = fsa.run(after, "ba")
    assert not result.accepted
    assert result.verdict is fsa.Verdict.REJECT_NO_TRANSITION


def test_a_complete_automaton_minimises_to_a_complete_one():
    """Its trap is one the user drew. Removing it would answer a question
    nobody asked and hand the diagnostics panel a fresh complaint."""
    before, trap = fsa.complete(partial_dfa())
    after = minimize(before)
    assert fsa.is_complete(after)
    assert trap is not None and trap in after.states
    assert_same_language(before, after)


def test_a_gap_only_an_unreachable_state_had_is_dropped_rather_than_filled():
    """So this partial machine minimises to a complete one, with no trap: the
    hole went away with the state that had it."""
    before = (partial_dfa()
              .with_transition("q1", "a", "q0")
              .with_transition("q1", "b", "q0")
              .with_state("q2"))
    assert not fsa.is_complete(before)
    assert marking_table(before).invented_trap is None
    after = minimize(before)
    assert fsa.is_complete(after)
    assert "q2" not in after.states
    assert_same_language(before, after)


def test_an_empty_language_collapses_to_one_rejecting_state():
    before = DFA().with_states(["q0", "q1"]).with_transition("q0", "a", "q1")
    after = minimize(before)
    assert len(after.states) == 1
    assert not after.accept
    assert after.initial is not None, (
        "'accepts nothing' must not become 'has no start state'")
    assert_same_language(before, after)


def test_an_automaton_with_no_start_state_minimises_to_nothing():
    before = hopcroft_ullman().with_initial(None)
    after = minimize(before)
    assert after.states == frozenset()
    assert after.initial is None
    assert after.alphabet == before.alphabet, "the alphabet is still known"
    assert fsa.run(after, "01").verdict is fsa.Verdict.NO_INITIAL_STATE


def test_an_empty_alphabet_leaves_only_the_start_state():
    before = DFA().with_states(["q0", "q1"]).with_accept("q0")
    after = minimize(before)
    assert after.states == frozenset({"q0"})
    assert after.accept == frozenset({"q0"})
    assert_same_language(before, after)


def test_an_empty_automaton_survives_the_trip():
    assert minimize(DFA()) == DFA()


def test_an_already_minimal_automaton_comes_back_equal():
    once = minimize(hopcroft_ullman())
    assert minimize(once) == once, "labels and ids included, not just the shape"


def test_the_input_is_not_mutated():
    before = hopcroft_ullman()
    snapshot = fsa.dumps(fsa.Document.of(before))
    minimize(before)
    assert fsa.dumps(fsa.Document.of(before)) == snapshot


def test_only_moore_is_implemented():
    assert minimize(hopcroft_ullman(), method="MOORE ") == minimize(hopcroft_ullman())
    with pytest.raises(ValueError, match="hopcroft"):
        minimize(hopcroft_ullman(), method="hopcroft")


def test_quotient_can_be_applied_to_a_table_the_caller_already_has():
    """What an animated table does when it reaches the last round."""
    table = marking_table(hopcroft_ullman())
    assert quotient(table) == minimize(hopcroft_ullman())


def test_quotient_does_not_assume_delta_is_total():
    """A table built by hand -- by a caller who skipped the preparation -- can
    be over a partial machine. Merging one is still defined: a transition that
    was undefined stays undefined rather than crashing the merge."""
    sketch = (DFA()
              .with_states(["q0", "q1"])
              .with_transition("q0", "a", "q1")
              .with_symbol("b"))
    merged = quotient(MarkingTable(automaton=sketch))
    assert merged.states == frozenset({"q0"}), "no marks means everything merges"
    assert merged.target("q0", "a") == "q0"
    assert merged.target("q0", "b") is None


# ---------------------------------------------------------------------------
# Properties, over 300 random machines
# ---------------------------------------------------------------------------


def random_dfa(rng) -> DFA:
    """Small, often partial, sometimes start-less, sometimes with a state whose
    behaviour is duplicated so that something is guaranteed to merge."""
    alphabet = rng.sample("ab01", rng.randrange(1, 4))
    ids = [f"q{i}" for i in range(rng.randrange(1, 8))]
    automaton = DFA().with_states(ids)
    for symbol in alphabet:
        automaton = automaton.with_symbol(symbol)
    for state in ids:
        for symbol in alphabet:
            if rng.random() < 0.75:
                automaton = automaton.with_transition(state, symbol, rng.choice(ids))
    for state in ids:
        if rng.random() < 0.4:
            automaton = automaton.with_accept(state)

    if rng.random() < 0.5:
        # Clone one state's whole row -- same targets, same acceptance -- and
        # divert an existing edge into the copy. The clone is reachable and
        # indistinguishable from its original by construction.
        original = rng.choice(ids)
        clone = "clone"
        automaton = automaton.with_state(clone)
        for symbol in alphabet:
            target = automaton.target(original, symbol)
            if target is not None:
                automaton = automaton.with_transition(clone, symbol, target)
        if original in automaton.accept:
            automaton = automaton.with_accept(clone)
        edges = sorted(automaton.transitions)
        if edges:
            source, symbol = rng.choice(edges)
            if source != clone:
                automaton = automaton.with_transition(source, symbol, clone)

    if rng.random() < 0.12:
        automaton = automaton.with_initial(None)
    return automaton


def random_machines(count: int) -> List[DFA]:
    import random
    rng = random.Random(90210)
    return [random_dfa(rng) for _ in range(count)]


MACHINES = random_machines(300)


def test_minimising_never_changes_the_language():
    """Exhaustively over every short word, and then exactly, in case the
    shortest witness is longer than the bound."""
    for before in MACHINES:
        after = minimize(before)
        for word in words_up_to(before.alphabet | set("xz"), 3):
            assert fsa.accepts(before, word) == fsa.accepts(after, word), \
                f"{word!r} changed verdict on {before!r}"
        assert_same_language(before, after)


def test_minimising_never_adds_states():
    for before in MACHINES:
        assert len(minimize(before).states) <= len(before.states)


def test_something_actually_shrinks():
    """Otherwise every property above could hold of a function that returns its
    argument."""
    shrunk = sum(1 for before in MACHINES
                 if len(minimize(before).states) < len(before.states))
    assert shrunk > len(MACHINES) // 3


def test_minimising_is_idempotent():
    for before in MACHINES:
        once = minimize(before)
        twice = minimize(once)
        assert len(twice.states) == len(once.states)
        assert twice == once, "same states, same ids, same labels"
        assert_same_language(once, twice)


def test_whether_delta_was_total_survives_minimisation():
    """The contract for the awkward inputs, over the whole corpus: a complete
    machine stays complete, a partial one stays partial, and the single
    exception is the empty language, where the last state standing is the trap
    itself and removing it would cost the machine its start state."""
    preserved = {"complete": 0, "partial": 0, "empty language": 0}
    for before in MACHINES:
        if before.initial is None:
            continue
        after = minimize(before)
        # It is the *reachable* part that has to be completed, and the table
        # says whether it was: a gap only an unreachable state had is dropped
        # with that state rather than filled, so such a machine legitimately
        # comes back complete.
        filled_a_gap = marking_table(before).invented_trap is not None
        if not after.accept:
            preserved["empty language"] += 1
            assert len(after.states) == 1 and after.initial is not None
        elif filled_a_gap:
            preserved["partial"] += 1
            assert not fsa.is_complete(after), \
                f"minimising {before!r} invented a trap and kept it"
        else:
            preserved["complete"] += 1
            assert fsa.is_complete(after)
    assert all(preserved.values()), f"a case was never exercised: {preserved}"


def test_the_result_is_reachable_and_has_nothing_left_to_merge():
    for before in MACHINES:
        after = minimize(before)
        assert fsa.reachable(after) == after.states
        assert marking_table(after).equivalent_pairs == (), \
            f"{after!r} still has a mergeable pair"


def test_indistinguishability_is_an_equivalence_relation():
    """Transitivity is what makes the empty cells a partition; without it,
    grouping states by the table would be meaningless."""
    for before in MACHINES:
        table = marking_table(before)
        equivalent = set(table.equivalent_pairs)
        states = table.states
        for left in states:
            for middle in states:
                for right in states:
                    linked = ((left, middle) in equivalent
                              and (middle, right) in equivalent)
                    if linked and left < right:
                        assert (left, right) in equivalent, \
                            f"{left}~{middle}~{right} but not {left}~{right}"


def test_every_class_is_wholly_accepting_or_wholly_rejecting():
    for before in MACHINES:
        table = marking_table(before)
        accept = table.automaton.accept
        for members in table.equivalence_classes():
            inside = {member in accept for member in members}
            assert len(inside) == 1, f"{sorted(members)} straddles acceptance"


def test_the_classes_partition_the_prepared_states():
    for before in MACHINES:
        table = marking_table(before)
        classes = table.equivalence_classes()
        seen = {}
        for index, members in enumerate(classes):
            for member in members:
                assert member not in seen, f"{member} is in two classes"
                seen[member] = index
        assert set(seen) == set(table.states)
