"""Automatic placement: ``Layout.auto``.

Two claims run through nearly every test here. First, *everything gets a
position* -- a state with no coordinates cannot be drawn, clicked or deleted,
so an automaton that comes back half-placed is worse than one not placed at
all. Second, *the same automaton always draws the same picture*, in this
process and the next; the way to break that is to iterate a set, which is why
one test spends a subprocess proving it.

Pure engine, so no display: nothing here imports pygame, and if it ever needs
to, the boundary has broken.
"""

import math
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from fsa import DFA, Layout
from fsa.layout import AUTO_SEPARATION, LAYER_ASPECT

# rendering.renderer.STATE_RADIUS. Copied rather than imported for the same
# reason the engine copies it: importing the renderer would drag pygame into a
# test of a module that must never depend on a display.
STATE_RADIUS = 30.0


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def chain() -> DFA:
    """q0 -a-> q1 -a-> q2, with a self-loop on b and no arrows out of q2."""
    return (DFA()
            .with_states(["q0", "q1", "q2"])
            .with_transition("q0", "a", "q1")
            .with_transition("q0", "b", "q0")
            .with_transition("q1", "a", "q2")
            .with_accept("q2"))


def generated(count: int = 12, seed: int = 20260811) -> DFA:
    """A machine of ``count`` states with randomly wired, partial transitions."""
    rng = random.Random(seed)
    names = [f"q{index}" for index in range(count)]
    automaton = DFA().with_states(names).with_initial("q0")
    for name in names:
        for symbol in "ab":
            # Left undefined a fifth of the time: delta is partial, and a
            # layout that only works on complete machines is a layout that
            # does not work on anything the user has drawn so far.
            if rng.random() < 0.8:
                automaton = automaton.with_transition(
                    name, symbol, rng.choice(names))
    return automaton.with_accept(rng.choice(names))


def separations(layout: Layout) -> list[float]:
    """Every centre-to-centre distance between two placed states."""
    points = [point for _, point in sorted(layout.positions.items())]
    return [math.dist(first, second)
            for index, first in enumerate(points)
            for second in points[index + 1:]]


def columns_of(layout: Layout) -> list[float]:
    """The distinct x coordinates used, left to right."""
    return sorted({point[0] for point in layout.positions.values()})


# ---------------------------------------------------------------------------
# Everything gets a position, and nothing else does
# ---------------------------------------------------------------------------


def test_every_state_is_placed_and_no_extras_are():
    automaton = chain()
    layout = Layout.auto(automaton)
    assert set(layout.positions) == set(automaton.states)


def test_an_empty_automaton_places_nothing():
    assert Layout.auto(DFA()) == Layout()


def test_the_result_carries_no_arc_offsets():
    """A drawing nobody has touched has no bowed edges to remember."""
    assert Layout.auto(chain()).arc_offsets == {}


def test_placing_does_not_change_the_automaton():
    automaton = generated()
    before = automaton
    Layout.auto(automaton)
    assert automaton == before


# ---------------------------------------------------------------------------
# Separation
# ---------------------------------------------------------------------------


def test_no_two_states_are_drawn_touching():
    """The exit criterion: never closer than two radii, so circles cannot overlap."""
    for automaton in (chain(), generated(), generated(count=2), chain().with_initial(None)):
        gaps = separations(Layout.auto(automaton))
        assert min(gaps) >= 2 * STATE_RADIUS


def test_the_separation_is_a_parameter():
    for requested in (2 * STATE_RADIUS, 140.0, 313.7):
        gaps = separations(Layout.auto(generated(), minimum_separation=requested))
        # Coordinates are rounded to a thousandth when the layout is built, so
        # the promise is kept to that precision rather than to the last bit.
        assert min(gaps) >= requested - 1e-6


def test_the_default_separation_clears_the_gui_radius():
    assert AUTO_SEPARATION >= 2 * STATE_RADIUS


def test_a_single_state_has_no_pair_to_separate():
    layout = Layout.auto(DFA().with_state("only"))
    assert list(layout.positions) == ["only"]


# ---------------------------------------------------------------------------
# The shape bfs_layers promises
# ---------------------------------------------------------------------------


def test_layers_run_left_to_right_by_distance():
    """A chain reads in the direction its arrows point."""
    layout = Layout.auto(chain())
    xs = [layout.position_of(state)[0] for state in ("q0", "q1", "q2")]
    assert xs[0] < xs[1] < xs[2]


def test_a_state_sits_in_the_column_of_its_shortest_path():
    """q3 is two steps away by one route and three by another; two wins."""
    automaton = (DFA()
                 .with_states(["q0", "q1", "q2", "q3"])
                 .with_transition("q0", "a", "q1")
                 .with_transition("q1", "a", "q3")
                 .with_transition("q0", "b", "q2")
                 .with_transition("q2", "b", "q1"))
    layout = Layout.auto(automaton)
    xs = columns_of(layout)
    assert layout.position_of("q3")[0] == xs[2]
    assert layout.position_of("q1")[0] == xs[1]
    assert layout.position_of("q2")[0] == xs[1]


def test_states_sharing_a_layer_share_a_column():
    automaton = (DFA()
                 .with_states(["q0", "a1", "b1"])
                 .with_transition("q0", "a", "a1")
                 .with_transition("q0", "b", "b1"))
    layout = Layout.auto(automaton)
    assert layout.position_of("a1")[0] == layout.position_of("b1")[0]
    assert layout.position_of("a1")[1] != layout.position_of("b1")[1]


def test_a_cycle_terminates_and_is_laid_out_once():
    """BFS on a machine with no exit still has to stop."""
    automaton = (DFA()
                 .with_states(["q0", "q1", "q2"])
                 .with_transition("q0", "a", "q1")
                 .with_transition("q1", "a", "q2")
                 .with_transition("q2", "a", "q0")
                 .with_accept("q0"))
    layout = Layout.auto(automaton)
    assert len(layout.positions) == 3
    assert len(columns_of(layout)) == 3


def test_missing_transitions_are_not_edges():
    """delta is partial, and an undefined pair must not place anything.

    A machine with a single arrow drawn so far is the normal state of an
    editor session, not an exceptional one.
    """
    automaton = (DFA()
                 .with_states(["q0", "q1"])
                 .with_transition("q0", "a", "q1")
                 .with_symbol("z"))
    layout = Layout.auto(automaton)
    assert layout.position_of("q0")[0] < layout.position_of("q1")[0]


def test_the_drawing_starts_at_the_origin():
    for origin in ((0.0, 0.0), (160.0, 160.0), (-40.5, 12.0)):
        layout = Layout.auto(generated(), origin=origin)
        box = layout.bounds()
        assert box is not None
        assert box[0] == pytest.approx(origin[0])
        assert box[1] == pytest.approx(origin[1])


# ---------------------------------------------------------------------------
# States BFS cannot reach
# ---------------------------------------------------------------------------


def test_unreachable_states_are_placed_after_the_machine():
    automaton = (chain()
                 .with_state("island")
                 .with_state("islet")
                 .with_transition("island", "a", "islet"))
    layout = Layout.auto(automaton)
    assert set(layout.positions) == set(automaton.states)

    reachable_right = max(layout.position_of(state)[0]
                          for state in ("q0", "q1", "q2"))
    for orphan in ("island", "islet"):
        assert layout.position_of(orphan)[0] > reachable_right


def test_a_gap_separates_the_orphans_from_the_spine():
    """One empty column, so the picture says these are not part of the flow."""
    automaton = chain().with_state("island")
    layout = Layout.auto(automaton)
    step = AUTO_SEPARATION * LAYER_ASPECT
    assert layout.position_of("island")[0] - layout.position_of("q2")[0] == pytest.approx(2 * step)


def test_no_initial_state_still_places_everything():
    """The editor is in this state whenever the start state has been deleted."""
    automaton = generated().with_initial(None)
    layout = Layout.auto(automaton)
    assert automaton.initial is None
    assert set(layout.positions) == set(automaton.states)
    assert min(separations(layout)) >= 2 * STATE_RADIUS


def test_a_dozen_homeless_states_do_not_form_one_long_column():
    """They would run off the bottom of any view; a square block does not."""
    layout = Layout.auto(generated().with_initial(None))
    tallest = max(sum(1 for point in layout.positions.values() if point[0] == x)
                  for x in columns_of(layout))
    assert tallest < 12
    assert len(columns_of(layout)) > 1


# ---------------------------------------------------------------------------
# A dozen states at once
# ---------------------------------------------------------------------------


def test_a_generated_twelve_state_machine_is_fully_and_safely_placed():
    automaton = generated(count=12)
    assert len(automaton.states) == 12
    layout = Layout.auto(automaton)
    assert set(layout.positions) == set(automaton.states)
    assert min(separations(layout)) >= 2 * STATE_RADIUS
    assert len(layout.positions) == 12, "no state placed twice, none forgotten"


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_generated_machines_are_all_placed_without_collisions(seed):
    layout = Layout.auto(generated(count=12, seed=seed))
    assert len(layout.positions) == 12
    assert min(separations(layout)) >= 2 * STATE_RADIUS


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_the_same_automaton_lays_out_the_same_way():
    automaton = generated()
    assert Layout.auto(automaton) == Layout.auto(automaton)


def test_layout_depends_on_the_automaton_and_not_on_how_it_was_built():
    """Equal values, built in different orders, must draw identically."""
    forwards = (DFA()
                .with_states(["q0", "q1", "q2"])
                .with_transition("q0", "a", "q1")
                .with_transition("q1", "a", "q2")
                .with_transition("q2", "b", "q0"))
    backwards = (DFA()
                 .with_states(["q2", "q1", "q0"])
                 .with_transition("q2", "b", "q0")
                 .with_transition("q1", "a", "q2")
                 .with_transition("q0", "a", "q1")
                 .with_initial("q0"))
    assert forwards == backwards
    assert Layout.auto(forwards) == Layout.auto(backwards)


_ACROSS_PROCESSES = """
from fsa import DFA, Layout

names = [f"q{i}" for i in range(12)]
a = DFA().with_states(names).with_initial("q0")
for index, name in enumerate(names):
    a = a.with_transition(name, "a", names[(index * 5 + 1) % 12])
    a = a.with_transition(name, "b", names[(index + 3) % 12])
a = a.with_state("stray").with_accept("q7")
print(sorted(Layout.auto(a).positions.items()))
"""


def test_the_layout_is_the_same_in_every_process():
    """The real test of "sort anything that comes out of a set".

    Python randomises string hashing per process, so a frozenset of state ids
    iterates in a different order every run. Repeating the call in one process
    cannot catch that -- the order is fixed for the life of the interpreter --
    so this runs the same layout under two hash seeds and compares.
    """
    src = Path(__file__).resolve().parent.parent / "src"
    drawings = []
    for seed in ("0", "1", "1000"):
        result = subprocess.run(
            [sys.executable, "-c", _ACROSS_PROCESSES],
            cwd=src, capture_output=True, text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        assert result.returncode == 0, result.stderr
        drawings.append(result.stdout)
    assert drawings[0] == drawings[1] == drawings[2]
    assert "stray" in drawings[0]


# ---------------------------------------------------------------------------
# Choosing an algorithm
# ---------------------------------------------------------------------------


def test_bfs_layers_is_the_default():
    assert Layout.auto(chain()) == Layout.auto(chain(), "bfs_layers")


def test_an_unknown_algorithm_is_refused():
    """Not a silent fallback: a caller that named a layout wants that layout."""
    with pytest.raises(ValueError) as raised:
        Layout.auto(chain(), algorithm="force_directed")
    assert "force_directed" in str(raised.value)


# ---------------------------------------------------------------------------
# Layout is about the graph, not about determinism
# ---------------------------------------------------------------------------


def test_auto_lays_out_a_nondeterministic_machine():
    """Regression. `_bfs_layers` walked delta with `automaton.target(...)`,
    which only a DFA has, so an NFA could not be drawn at all -- it raised
    AttributeError the first time anything tried. Layout only ever needed to
    know which states are one edge apart, and both types answer that.
    """
    from fsa.nfa import NFA

    automaton = NFA(states=frozenset({"q0", "q1", "q2", "q3"}),
                    alphabet=frozenset("ab")).with_initial("q0")
    for source, symbol, target in [
        ("q0", "a", "q0"), ("q0", "b", "q0"), ("q0", "a", "q1"),
        ("q1", "b", "q2"), ("q2", "b", "q3"),
    ]:
        automaton = automaton.with_transition(source, symbol, target)
    assert not automaton.is_deterministic()

    layout = Layout.auto(automaton)
    assert set(layout.positions) == set(automaton.states)
    placed = sorted(layout.positions.items())
    for index, (_a, first) in enumerate(placed):
        for _b, second in placed[index + 1:]:
            assert math.dist(first, second) >= 2 * 30.0


def test_auto_follows_epsilon_edges_when_layering():
    """An epsilon move is an edge: the states it joins are adjacent in the
    drawing, even though no symbol is read crossing it."""
    from fsa.nfa import EPSILON, NFA

    automaton = (NFA(states=frozenset({"q0", "q1"}), alphabet=frozenset("a"))
                 .with_initial("q0")
                 .with_transition("q0", EPSILON, "q1"))
    layout = Layout.auto(automaton)

    assert set(layout.positions) == {"q0", "q1"}
    # Reached in one step, so q1 sits in the next column, not the orphan block.
    assert layout.positions["q1"][0] > layout.positions["q0"][0]
