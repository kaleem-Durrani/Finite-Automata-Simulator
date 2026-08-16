"""Smoke tests.

The ground floor: the modules import, a document can be built and simulated,
and the camera round-trips. Deeper behaviour lives in the conformance spec, the
engine suite, the geometry suite and the app regressions.
"""

import pygame
import pytest

import fsa
from editor import EditorModel
from rendering.camera import Camera


@pytest.fixture(scope="module", autouse=True)
def _pygame_display():
    """Initialise a headless display once for the module."""
    pygame.init()
    pygame.display.set_mode((320, 240))
    yield
    pygame.quit()


@pytest.fixture
def demo():
    """The document the application opens with: a*b+."""
    import main
    return main.demo_document()


def test_modules_import():
    """The GUI layer imports without side effects beyond pygame init."""
    import main  # noqa: F401
    import rendering.renderer  # noqa: F401
    import ui.ui_manager  # noqa: F401


def test_engine_is_display_free():
    """fsa must not depend on pygame, and nothing may sneak it back in.

    Parses the imports rather than searching the text: the engine's own
    docstrings explain this rule, and a substring search flags them.
    """
    import ast
    import pathlib

    banned = {"pygame", "networkx", "numpy"}
    for module in pathlib.Path(fsa.__file__).parent.rglob("*.py"):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert name.split(".")[0] not in banned, f"{module.name}: {name}"


def test_camera_round_trips_screen_and_world():
    camera = Camera(800, 600)
    camera.pan(37, -14)
    camera.zoom_at((400, 300), 1.5)
    world = camera.screen_to_world((123, 456))
    assert camera.world_to_screen(world) == pytest.approx((123, 456))


def test_demo_recognises_a_star_b_plus(demo):
    accepted = ["b", "ab", "aab", "abb", "bb"]
    rejected = ["", "a", "aa", "ba", "aba"]
    for word in accepted:
        assert fsa.accepts(demo.as_dfa(), word), word
    for word in rejected:
        assert not fsa.accepts(demo.as_dfa(), word), word


def test_demo_has_a_layout(demo):
    """Every state must have coordinates, or it renders at the origin."""
    assert set(demo.layout.positions) == set(demo.automaton.states)


def test_editor_starts_clean(demo):
    editor = EditorModel(demo)
    assert editor.dirty is False
    assert editor.selection is None
    assert editor.path is None


def test_editing_marks_the_document_dirty(demo):
    editor = EditorModel(demo)
    editor.add_state((900.0, 900.0))
    assert editor.dirty is True
