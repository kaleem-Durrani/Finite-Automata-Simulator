"""Smoke tests.

These exist so that CI is not vacuous. They assert only that the modules
import, that a DFA can be built and simulated, and that the demo automaton
recognises the language the README claims it does.

The real test suites — a conformance spec written against hand-computed
delta-hat, and headless event-replay tests for the app layer — arrive with
the next phase of work.
"""

import pygame
import pytest

from core.camera import Camera
from core.dfa import DFA
from core.state import State, StateType


@pytest.fixture(scope="module", autouse=True)
def _pygame_display():
    """Initialise a headless display once for the module."""
    pygame.init()
    pygame.display.set_mode((320, 240))
    yield
    pygame.quit()


@pytest.fixture
def demo() -> DFA:
    """The automaton the application creates on startup.

    Recognises a*b+ : any number of 'a's followed by at least one 'b'.
    """
    dfa = DFA()
    q0 = dfa.add_state((200, 200))
    q1 = dfa.add_state((400, 200))
    q2 = dfa.add_state((300, 350))
    dfa.set_state_type(q1, StateType.ACCEPT)
    dfa.set_state_type(q2, StateType.DEAD_END)
    dfa.add_transition(q0, q0, "a")
    dfa.add_transition(q0, q1, "b")
    dfa.add_transition(q1, q2, "a")
    dfa.add_transition(q1, q1, "b")
    dfa.add_transition(q2, q2, "a")
    dfa.add_transition(q2, q2, "b")
    return dfa


def test_modules_import():
    """The GUI layer imports without side effects beyond pygame init."""
    import main  # noqa: F401
    import rendering.renderer  # noqa: F401
    import ui.ui_manager  # noqa: F401


def test_core_is_pygame_free():
    """core/ must not depend on pygame.

    This is the boundary the project is being built around: the automata
    engine has to stay usable without a display. Guarding it with a test
    means the dependency cannot creep back in unnoticed.
    """
    import importlib
    import pathlib

    core_dir = pathlib.Path(importlib.import_module("core").__file__).parent
    offenders = [
        path.name
        for path in core_dir.glob("*.py")
        if "pygame" in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"pygame referenced in core/: {offenders}"


def test_add_state_sets_initial_state():
    dfa = DFA()
    first = dfa.add_state((0, 0))
    dfa.add_state((100, 0))
    assert dfa.initial_state == first


def test_state_hit_testing():
    state = State("q0", (100, 100))
    assert state.contains_point((100, 100))
    assert state.contains_point((100 + state.radius - 1, 100))
    assert not state.contains_point((100 + state.radius + 1, 100))


def test_camera_round_trips_screen_and_world():
    camera = Camera(800, 600)
    camera.pan(37, -14)
    camera.zoom_at((400, 300), 1.5)
    world = camera.screen_to_world((123, 456))
    screen = camera.world_to_screen(world)
    assert screen == pytest.approx((123, 456))


@pytest.mark.parametrize(
    ("word", "accepted"),
    [
        ("b", True),
        ("ab", True),
        ("aab", True),
        ("abb", True),
        ("bb", True),
        ("", False),
        ("a", False),
        ("aa", False),
        ("ba", False),
        ("aba", False),
    ],
)
def test_demo_recognises_a_star_b_plus(demo: DFA, word: str, accepted: bool):
    assert demo.process_string(word)[0] is accepted


def test_rejected_word_outside_alphabet(demo: DFA):
    accepted, path = demo.process_string("abz")
    assert accepted is False
    assert path[0] == demo.initial_state


def test_path_starts_at_initial_state(demo: DFA):
    _, path = demo.process_string("aab")
    assert path[0] == demo.initial_state
    assert len(path) == len("aab") + 1
