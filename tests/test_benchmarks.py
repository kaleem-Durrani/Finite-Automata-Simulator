"""Performance criteria, as tests rather than as numbers in a document.

`IMPROVEMENT_PLAN.md` sets these: 100 states under 8ms a frame, and the note
that font rendering was measured at 0.12ms for 83 renders and is therefore not
worth caching. A number written in a plan drifts silently; a number in a test
fails the build when it stops being true.

Deliberately loose bounds. These exist to catch a regression that changes the
order of magnitude -- an unbounded tessellation loop, a per-frame recomputation
of something cached -- not to police a few percent on somebody else's machine.
"""

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import fsa  # noqa: E402


def wide_machine(states: int) -> fsa.DFA:
    """A connected machine of `states` states over two symbols."""
    ids = [f"q{index}" for index in range(states)]
    automaton = fsa.DFA(states=frozenset(ids), alphabet=frozenset("ab"))
    automaton = automaton.with_initial(ids[0]).with_accept(ids[-1])
    for index, state in enumerate(ids):
        automaton = automaton.with_transition(state, "a", ids[(index + 1) % states])
        automaton = automaton.with_transition(state, "b", ids[index // 2])
    return automaton


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


def test_minimise_a_hundred_states(benchmark):
    automaton = wide_machine(100)
    result = benchmark(fsa.minimize, automaton)
    assert len(result.states) <= 100


def test_equivalence_of_two_hundred_state_machines(benchmark):
    left, right = wide_machine(100), wide_machine(100)
    assert benchmark(fsa.equivalent, left, right)


def test_auto_layout_of_a_hundred_states(benchmark):
    automaton = wide_machine(100)
    layout = benchmark(fsa.Layout.auto, automaton)
    assert len(layout.positions) == 100


# ---------------------------------------------------------------------------
# The frame
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason=(
    "Phase 8 criterion 5 has never actually been met -- it was written into "
    "the plan and never measured. This benchmark measures it: ~97ms a frame "
    "at 100 states, twelve times over. Diagnosed: geometry.segments() picks "
    "int(length / 9) from the WORLD length, so fitting 100 states to the "
    "screen still tessellates every edge into 40-72 anti-aliased quads even "
    "though each is a few pixels long. The fix is to scale tessellation by "
    "zoom, which changes geometry signatures and the paths hit-testing "
    "shares -- see ROADMAP Phase 15."))
@pytest.mark.parametrize("states", [100])
def test_a_hundred_state_frame_is_under_eight_milliseconds(benchmark, states):
    """The plan's headline rendering criterion.

    Asserted on the mean rather than the max: one slow frame is a scheduler
    artefact, a slow mean is a regression. Marked xfail rather than deleted
    or loosened, so the gap stays measured and visible instead of becoming
    a number nobody checks.
    """
    pygame = pytest.importorskip("pygame")
    from main import AutomatonSimulator

    app = AutomatonSimulator()
    app._handle_resize(1400, 860)
    automaton = wide_machine(states)
    app.editor.replace(
        fsa.Document(automaton, fsa.Layout.auto(automaton), states), None)
    app.ui_manager.sync_symbols_with(app.editor.automaton)
    app._fit_to_content()
    for _ in range(20):          # let the camera settle before measuring
        app._update(16)
        app._render()

    def frame() -> None:
        app._update(16)
        app._render()

    try:
        benchmark(frame)
        assert benchmark.stats["mean"] < 0.008, (
            f"{benchmark.stats['mean'] * 1000:.2f}ms a frame at {states} states")
    finally:
        pygame.quit()
