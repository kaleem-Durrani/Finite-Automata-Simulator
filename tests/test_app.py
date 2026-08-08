"""Regression tests for the application layer.

Each of the first four tests replays a sequence that crashed the app before
this phase. They are written as event/method replays against a real
AutomatonSimulator running on the dummy SDL driver, so they exercise the same
code paths a user does -- including _render, which is where the worst of the
crashes lived (it fired every frame, so the process could not recover).
"""

import json
from pathlib import Path

import pygame
import pytest

import main as main_module
from core.state import StateType
from main import AutomatonSimulator


@pytest.fixture
def app(tmp_path, monkeypatch):
    """A fresh simulator whose file operations are confined to tmp_path."""
    monkeypatch.setattr(main_module, "PROJECT_DIR", str(tmp_path))
    simulator = AutomatonSimulator()
    yield simulator
    pygame.quit()


def pump(app: AutomatonSimulator, frames: int = 3):
    """Run the update/render loop without blocking on real time."""
    for _ in range(frames):
        app._update(16)
        app._render()


# ---------------------------------------------------------------------------
# Crashes
# ---------------------------------------------------------------------------


def test_context_menu_delete_clears_selection(app):
    """Right-click Delete State used to leave selected_state dangling.

    The next Q or W keypress then raised KeyError on a state that no longer
    existed.
    """
    app._select_state("q1")
    app._handle_context_menu_action("delete_state:q1")

    assert "q1" not in app.dfa.states
    assert app.selected_state is None

    app._toggle_accept_state()
    app._toggle_dead_end_state()
    pump(app)


def test_delete_while_transition_pending_does_not_crash_render(app):
    """Deleting the source of a half-drawn transition used to kill _render.

    _render looked up transition_start_state unguarded, so the KeyError
    repeated every frame and the process could never recover.
    """
    app._select_state("q1")
    app._start_transition("q1")
    assert app.creating_transition

    app._delete_selected_state()

    assert "q1" not in app.dfa.states
    pump(app, frames=10)
    assert not app.creating_transition


def test_delete_while_dragging_does_not_crash_on_release(app):
    """Press-and-hold a state, press Delete, release: _stop_dragging crashed."""
    app._select_state("q1")
    app._start_dragging("q1", (400, 200))
    assert app.dragging_state == "q1"

    app._delete_selected_state()
    assert app.dragging_state is None

    app._update_dragging((410, 210))
    app._handle_left_release((410, 210))
    pump(app)


def test_delete_state_in_active_trace_stops_execution(app):
    """A deleted state must not stay in the execution path being rendered."""
    app._test_string("ab")
    assert app.execution_active
    assert "q1" in app.execution_path

    app._select_state("q1")
    app._delete_selected_state()

    assert not app.execution_active
    assert app.execution_path == []
    pump(app)


def test_transition_to_deleted_state_reports_failure(app):
    """add_transition returning False used to be discarded silently."""
    app._start_transition("q0")
    app.dfa.remove_state("q2")
    app._complete_transition("q2")

    assert not app.creating_transition
    assert "Could not add transition" in app.message_text


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def test_save_then_load_round_trips_in_the_same_session(app, tmp_path):
    """Save and Load used to point at different files; nothing round-tripped."""
    before = app.dfa.to_dict()

    app._save_to_path("mine")
    saved = tmp_path / "mine.json"
    assert saved.exists(), "'.json' should be appended when no suffix is given"
    assert app.dirty is False
    assert app.current_filename == "mine.json"

    # Edit, then load the saved file back over the top.
    app._add_state_at_center()
    assert app.dirty is True
    assert len(app.dfa.states) == 4

    app._load_from_path("mine")

    assert app.dfa.to_dict() == before
    assert app.dirty is False


def test_serialisation_is_byte_stable(app):
    """The same automaton must always serialise to the same bytes.

    to_dict emits sets as lists, and Python randomises string hashing per
    process, so without sorting the output order varies from run to run --
    which makes saved files impossible to diff and comparisons flaky.
    """
    first = json.dumps(app.dfa.to_dict(), indent=2)

    app._save_to_path("stable.json")
    app._load_from_path("stable.json")

    assert json.dumps(app.dfa.to_dict(), indent=2) == first
    assert app.dfa.to_dict()["alphabet"] == sorted(app.dfa.to_dict()["alphabet"])


def test_load_replaces_the_rendered_edges(app, tmp_path):
    """The canvas used to keep drawing the previous automaton's arrows.

    Rendering reads transition_groups, which from_dict neither rebuilt nor
    cleared, so a loaded file showed the old machine's edges over the new
    states.
    """
    other = {
        "states": {
            "q0": {"position": [0, 0], "state_type": "normal"},
            "q1": {"position": [80, 0], "state_type": "accept"},
        },
        "transitions": {"q0": {"0": "q1"}, "q1": {"1": "q0"}},
        "alphabet": ["0", "1"],
        "initial_state": "q0",
        "accept_states": ["q1"],
        "dead_end_states": [],
        "next_state_id": 2,
    }
    (tmp_path / "other.json").write_text(json.dumps(other), encoding="utf-8")

    app._load_from_path("other.json")

    drawn = {key: sorted(group["symbols"]) for key, group in app.dfa.transition_groups.items()}
    assert drawn == {("q0", "q1"): ["0"], ("q1", "q0"): ["1"]}
    pump(app)


def test_save_load_preserves_arc_offsets(app):
    """Curve offsets are the user's work and used to be dropped on save."""
    app.dfa.add_transition("q0", "q2", "b", 55.0)
    app._save_to_path("curved.json")
    app._load_from_path("curved.json")

    assert app.dfa.transition_groups[("q0", "q2")]["arc_offset"] == 55.0


def test_load_of_missing_file_reports_on_screen(app):
    """Failures used to print to stdout, which nobody running a GUI reads."""
    app._load_from_path("does_not_exist.json")

    assert "Load failed" in app.message_text
    assert len(app.dfa.states) == 3, "the current automaton must survive a failed load"


def test_load_of_malformed_file_leaves_automaton_intact(app, tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    before = app.dfa.to_dict()

    app._load_from_path("broken.json")

    assert "Load failed" in app.message_text
    assert app.dfa.to_dict() == before


def test_load_clears_the_execution_trace(app):
    app._save_to_path("snap.json")
    app._test_string("ab")
    assert app.execution_active

    app._load_from_path("snap.json")
    assert not app.execution_active


def test_load_adds_unseen_symbols_to_the_palette(app, tmp_path):
    spec = {
        "states": {"q0": {"position": [0, 0], "state_type": "accept"}},
        "transitions": {"q0": {"x": "q0"}},
        "alphabet": ["x"],
        "initial_state": "q0",
        "accept_states": ["q0"],
        "dead_end_states": [],
        "next_state_id": 1,
    }
    (tmp_path / "x.json").write_text(json.dumps(spec), encoding="utf-8")

    app._load_from_path("x.json")
    assert "x" in app.ui_manager.available_symbols


def test_toolbar_save_and_load_drive_the_whole_flow(app, tmp_path):
    """Click Save, type a name, press Enter; then click Load and confirm."""

    def click(rect):
        event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rect.center)
        app._handle_mouse_down(event)
        app._process_ui_actions(app.ui_manager.handle_event(event))

    def key(code, char=""):
        event = pygame.event.Event(pygame.KEYDOWN, key=code, unicode=char)
        app._handle_key_down(event)
        app._process_ui_actions(app.ui_manager.handle_event(event))

    click(app.ui_manager.save_button_rect)
    assert app.ui_manager.file_prompt_mode == "save"

    for _ in range(40):
        key(pygame.K_BACKSPACE)
    for char in "demo":
        key(ord(char), char)
    key(pygame.K_RETURN, "\r")

    assert (tmp_path / "demo.json").exists()
    assert app.current_filename == "demo.json"

    key(pygame.K_SPACE, " ")
    assert len(app.dfa.states) == 4

    click(app.ui_manager.load_button_rect)
    assert app.ui_manager.confirm_intent == "load_after_confirm"
    key(pygame.K_y, "y")
    assert app.ui_manager.file_prompt_mode == "load"
    key(pygame.K_RETURN, "\r")

    assert len(app.dfa.states) == 3
    assert app.dirty is False


def test_ui_hit_tests_use_the_event_position(app):
    """Widgets used to be hit-tested against the live cursor, not the click.

    When the cursor moves between an event being queued and the queue being
    drained, those disagree: the click is either lost or applied to whatever
    the cursor has since moved over.
    """
    pygame.mouse.set_pos((0, 0))

    actions = app.ui_manager.handle_event(
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                           pos=app.ui_manager.test_button_rect.center))
    assert "test_string" in actions

    actions = app.ui_manager.handle_event(
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(5, 5)))
    assert "test_string" not in actions


def test_paths_resolve_against_the_project_not_the_cwd(app, tmp_path):
    """Relative names used to resolve against wherever python was launched."""
    assert Path(app._resolve_path("a.json")).parent == tmp_path
    assert Path(app._resolve_path("a")).name == "a.json"
    assert Path(app._resolve_path("")).name == AutomatonSimulator.DEFAULT_FILENAME


# ---------------------------------------------------------------------------
# Unsaved-work guards
# ---------------------------------------------------------------------------


def test_editing_marks_the_document_dirty(app):
    assert app.dirty is False
    app._add_state_at_center()
    assert app.dirty is True
    assert pygame.display.get_caption()[0].startswith("untitled*")


def test_quit_with_unsaved_changes_asks_first(app):
    app._add_state_at_center()

    pygame.event.post(pygame.event.Event(pygame.QUIT))
    app._handle_events()

    assert app.running is True, "quitting must not discard unsaved work silently"
    assert app.ui_manager.confirm_intent == "quit_after_confirm"

    # Confirming carries the quit through.
    app._process_ui_actions({"confirmed": "quit_after_confirm"})
    assert app.running is False


def test_quit_without_changes_does_not_ask(app):
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    app._handle_events()
    assert app.running is False


def test_load_with_unsaved_changes_asks_first(app):
    app._add_state_at_center()
    app._load_automaton()
    assert app.ui_manager.confirm_intent == "load_after_confirm"


# ---------------------------------------------------------------------------
# Keyboard focus
# ---------------------------------------------------------------------------


def test_typing_in_a_dialog_does_not_edit_the_automaton(app):
    """Editor shortcuts are bare letters and used to fire behind open dialogs."""
    app.ui_manager.adding_symbol = True
    states_before = len(app.dfa.states)

    for key, unicode_char in [(pygame.K_SPACE, " "), (pygame.K_q, "q"), (pygame.K_w, "w")]:
        app._handle_key_down(pygame.event.Event(pygame.KEYDOWN, key=key, unicode=unicode_char))

    assert len(app.dfa.states) == states_before


def test_filename_prompt_captures_the_keyboard(app):
    app.ui_manager.show_file_prompt("save", "")
    assert app.ui_manager.is_keyboard_captured()
    assert app.ui_manager.is_modal_active()

    for char in "notes":
        app.ui_manager.handle_event(
            pygame.event.Event(pygame.KEYDOWN, key=ord(char), unicode=char))
    assert app.ui_manager.file_prompt_text == "notes"

    actions = app.ui_manager.handle_event(
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, unicode="\r"))
    assert actions["save_to_path"] == "notes"
    assert not app.ui_manager.is_modal_active()


def test_escape_cancels_the_filename_prompt(app):
    app.ui_manager.show_file_prompt("load", "x.json")
    actions = app.ui_manager.handle_event(
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, unicode="\x1b"))
    assert actions.get("file_prompt_cancel") is True
    assert app.ui_manager.file_prompt_mode is None


def test_modal_swallows_canvas_clicks(app):
    app.ui_manager.show_confirm("Quit without saving?", "quit_after_confirm")
    states_before = len(app.dfa.states)

    app._handle_mouse_down(
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=(500, 400)))

    assert len(app.dfa.states) == states_before
    assert app.ui_manager.context_menu is None


def test_dialogs_render(app):
    """The modal frames must draw without error at the current window size."""
    app.ui_manager.show_file_prompt("save", "automaton.json")
    pump(app)
    app.ui_manager.hide_file_prompt()

    app.ui_manager.show_confirm("Quit without saving?", "quit_after_confirm")
    pump(app)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_set_initial_from_context_menu(app):
    app._handle_context_menu_action("set_initial:q2")
    assert app.dfa.initial_state == "q2"
    assert app.dirty is True


def test_set_initial_ignores_unknown_state(app):
    before = app.dfa.initial_state
    app._handle_context_menu_action("set_initial:nope")
    assert app.dfa.initial_state == before


def test_demo_state_types(app):
    assert app.dfa.states["q1"].state_type is StateType.ACCEPT
    assert app.dfa.states["q2"].state_type is StateType.DEAD_END
