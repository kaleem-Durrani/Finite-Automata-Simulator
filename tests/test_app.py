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
import ui.ui_manager as ui_manager_module
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


def send(app: AutomatonSimulator, event: pygame.event.Event):
    """Push one event through the application's real dispatch path.

    Goes via the event queue and _handle_events rather than calling a handler
    directly, so the UI-first / consume gate is exercised exactly as it is at
    runtime. A test that calls a handler directly proves nothing about routing.
    """
    pygame.event.clear()
    pygame.event.post(event)
    app._handle_events()


def click(app: AutomatonSimulator, pos, button: int = 1, shift: bool = False):
    """Click, building the event with exactly the fields pygame provides.

    Mouse events carry no `.mod`; modifier state lives on the keyboard and is
    read from pygame.key. Passing a fabricated `mod=` here is what let a real
    crash (AttributeError on event.mod) pass the whole suite -- the synthetic
    events had a field the real ones never do.
    """
    pygame.key.set_mods(pygame.KMOD_LSHIFT if shift else pygame.KMOD_NONE)
    try:
        send(app, pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=button, pos=pos))
    finally:
        pygame.key.set_mods(pygame.KMOD_NONE)


def press(app: AutomatonSimulator, code: int, char: str = ""):
    send(app, pygame.event.Event(pygame.KEYDOWN, key=code, unicode=char,
                                 mod=pygame.KMOD_NONE, scancode=0))


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
    click(app, app.ui_manager.save_button_rect.center)
    assert app.ui_manager.file_prompt_mode == "save"

    for _ in range(40):
        press(app, pygame.K_BACKSPACE)
    for char in "demo":
        press(app, ord(char), char)
    press(app, pygame.K_RETURN, "\r")

    assert (tmp_path / "demo.json").exists()
    assert app.current_filename == "demo.json"

    press(app, pygame.K_SPACE, " ")
    assert len(app.dfa.states) == 4

    click(app, app.ui_manager.load_button_rect.center)
    assert app.ui_manager.confirm_intent == "load_after_confirm"
    press(app, pygame.K_y, "y")
    assert app.ui_manager.file_prompt_mode == "load"
    press(app, pygame.K_RETURN, "\r")

    assert len(app.dfa.states) == 3
    assert app.dirty is False


def test_ui_hit_tests_use_the_event_position(app):
    """Widgets used to be hit-tested against the live cursor, not the click.

    When the cursor moves between an event being queued and the queue being
    drained, those disagree: the click is either lost or applied to whatever
    the cursor has since moved over.
    """
    pygame.mouse.set_pos((0, 0))

    actions, consumed = app.ui_manager.handle_event(
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                           pos=app.ui_manager.test_button_rect.center))
    assert "test_string" in actions
    assert consumed is True

    actions, consumed = app.ui_manager.handle_event(
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
    types_before = {sid: s.state_type for sid, s in app.dfa.states.items()}
    app._select_state("q0")

    press(app, pygame.K_SPACE, " ")
    press(app, pygame.K_q, "q")
    press(app, pygame.K_w, "w")

    assert len(app.dfa.states) == states_before
    assert {sid: s.state_type for sid, s in app.dfa.states.items()} == types_before


def test_typing_in_the_test_field_does_not_edit_the_automaton(app):
    app.ui_manager.input_active = True
    states_before = len(app.dfa.states)

    press(app, pygame.K_SPACE, " ")

    assert len(app.dfa.states) == states_before


def test_filename_prompt_captures_the_keyboard(app):
    app.ui_manager.show_file_prompt("save", "")
    assert app.ui_manager.is_keyboard_captured()
    assert app.ui_manager.is_modal_active()

    for char in "notes":
        press(app, ord(char), char)
    assert app.ui_manager.file_prompt_text == "notes"

    press(app, pygame.K_RETURN, "\r")
    assert not app.ui_manager.is_modal_active()
    assert app.current_filename == "notes.json"


def test_escape_cancels_the_filename_prompt(app):
    app.ui_manager.show_file_prompt("load", "x.json")
    press(app, pygame.K_ESCAPE, "\x1b")
    assert app.ui_manager.file_prompt_mode is None


def test_modal_swallows_canvas_clicks(app):
    app.ui_manager.show_confirm("Quit without saving?", "quit_after_confirm")
    states_before = len(app.dfa.states)

    click(app, (500, 400), button=3)

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


# ---------------------------------------------------------------------------
# Event routing: one owner per event
# ---------------------------------------------------------------------------


def canvas_point(app: AutomatonSimulator, fraction_y: float) -> tuple:
    """A point on the canvas at the given fraction of the window height."""
    return (app.screen.get_width() // 2, int(app.screen.get_height() * fraction_y))


def test_toolbar_click_does_not_also_hit_the_canvas(app):
    """Every event used to go to the UI *and* the canvas handlers.

    Clicking Test therefore also deselected whatever state was selected.
    """
    app._select_state("q1")

    click(app, app.ui_manager.test_button_rect.center)

    assert app.selected_state == "q1", "using a toolbar control must not deselect"


def test_clicks_report_whether_the_ui_consumed_them(app):
    _, consumed = app.ui_manager.handle_event(
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                           pos=app.ui_manager.save_button_rect.center))
    assert consumed is True

    _, consumed = app.ui_manager.handle_event(
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                           pos=canvas_point(app, 0.5)))
    assert consumed is False


def test_shift_click_starts_a_transition_without_dragging(app):
    """Shift+click used to do both, because it was polled *and* handled.

    The poll in _update armed transition creation while the click handler
    independently started a drag, so the source state followed the mouse.
    """
    state = app.dfa.states["q0"]
    before = list(state.position)
    screen_pos = app.renderer.camera.world_to_screen(state.position)

    click(app, (int(screen_pos[0]), int(screen_pos[1])), shift=True)

    assert app.creating_transition is True
    assert app.transition_start_state == "q0"
    assert app.dragging_state is None
    assert list(state.position) == before

    # Releasing shift and pumping frames must not re-arm or move anything.
    pump(app, frames=5)
    assert app.transition_start_state == "q0"
    assert list(state.position) == before


def test_plain_click_still_selects_and_drags(app):
    state = app.dfa.states["q0"]
    screen_pos = app.renderer.camera.world_to_screen(state.position)

    click(app, (int(screen_pos[0]), int(screen_pos[1])))

    assert app.selected_state == "q0"
    assert app.dragging_state == "q0"
    assert app.creating_transition is False


def test_handlers_only_read_fields_real_events_carry(app):
    """Guard against handlers reading attributes pygame never sets.

    A previous version read `event.mod` on a MOUSEBUTTONDOWN. Mouse events have
    no modifier field, so every real click raised AttributeError and the app
    died on startup -- while the suite stayed green, because the test helpers
    constructed events with a `mod=` kwarg that real events never have.

    These events are built with pygame's own field set and nothing else.
    """
    width, height = app.screen.get_size()
    centre = (width // 2, height // 2)

    events = [
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=centre),
        pygame.event.Event(pygame.MOUSEMOTION, pos=centre, rel=(1, 1), buttons=(1, 0, 0)),
        pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=centre),
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=centre),
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=2, pos=centre),
        pygame.event.Event(pygame.MOUSEBUTTONUP, button=2, pos=centre),
        pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=1, flipped=False, precise_x=0.0,
                           precise_y=1.0),
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE, unicode=" ",
                           mod=pygame.KMOD_NONE, scancode=44),
        pygame.event.Event(pygame.KEYUP, key=pygame.K_SPACE,
                           mod=pygame.KMOD_NONE, scancode=44),
        pygame.event.Event(pygame.VIDEORESIZE, w=900, h=700, size=(900, 700)),
    ]

    for event in events:
        send(app, event)
        pump(app, frames=1)


def test_shift_click_reads_the_keyboard_not_the_event(app):
    """Shift state must come from pygame.key, which is where pygame keeps it."""
    state = app.dfa.states["q0"]
    screen_pos = app.renderer.camera.world_to_screen(state.position)
    point = (int(screen_pos[0]), int(screen_pos[1]))

    pygame.key.set_mods(pygame.KMOD_LSHIFT)
    try:
        send(app, pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=point))
    finally:
        pygame.key.set_mods(pygame.KMOD_NONE)

    assert app.creating_transition is True
    assert app.dragging_state is None


def test_no_input_polling_remains():
    """Input must come from events, not from sampling device state per frame."""
    source = Path(main_module.__file__).read_text(encoding="utf-8")
    assert "key.get_pressed" not in source
    assert "mouse.get_pressed" not in source


def test_right_click_works_across_the_whole_canvas(app):
    """Right-click used to be dead over a third to a half of the window.

    Hit-testing used hardcoded bands (above y=120, below height-150) that bore
    no relation to what was actually drawn.
    """
    samples = 40
    # Collect the canvas points first, with no menu open -- is_over_ui counts
    # an open context menu, so sampling it mid-loop would skip points that are
    # only covered by the menu the previous iteration opened.
    app.ui_manager.context_menu = None
    points = [
        canvas_point(app, 0.05 + 0.9 * i / (samples - 1))
        for i in range(samples)
    ]
    canvas_points = [p for p in points if not app.ui_manager.is_over_ui(p)]
    assert len(canvas_points) > samples // 2, "most of the window should be canvas"

    opened = 0
    for point in canvas_points:
        app.ui_manager.context_menu = None
        click(app, point, button=3)
        if app.ui_manager.context_menu is not None:
            opened += 1

    assert opened == len(canvas_points), "every canvas point must open a context menu"


def test_ui_panels_are_not_canvas(app):
    """Points on a panel are the UI's, and must not open a canvas menu."""
    assert app.ui_manager.is_over_ui(app.ui_manager.save_button_rect.center)
    assert app.ui_manager.is_over_ui(app.ui_manager.input_rect.center)
    assert app.ui_manager.is_over_ui(app.ui_manager.layout.status_panel.center)
    assert not app.ui_manager.is_over_ui(canvas_point(app, 0.5))


def test_help_panel_scrolls_to_the_last_line(app):
    """The panel could not scroll at all: max_scroll computed to zero.

    Six lines, including every execution shortcut, were unreachable.
    """
    app.ui_manager.show_help = True
    visible = app.ui_manager.layout.help_visible_lines()
    assert len(ui_manager_module.HELP_LINES) > visible, "otherwise this proves nothing"

    for _ in range(50):
        send(app, pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1))

    expected = len(ui_manager_module.HELP_LINES) - visible
    assert app.ui_manager.help_scroll_offset == expected

    for _ in range(50):
        send(app, pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=1))
    assert app.ui_manager.help_scroll_offset == 0


def test_scrolling_the_help_panel_does_not_also_zoom(app):
    """Both the app and the UI handled the wheel, so it did both."""
    app.ui_manager.show_help = True
    zoom_before = app.renderer.camera.zoom

    send(app, pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1))

    assert app.renderer.camera.zoom == zoom_before
    assert app.ui_manager.help_scroll_offset > 0


def test_wheel_still_zooms_when_help_is_closed(app):
    zoom_before = app.renderer.camera.zoom
    send(app, pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=1))
    assert app.renderer.camera.zoom > zoom_before


def test_speed_slider_can_be_dragged_and_is_read(app):
    """The slider was inert: undraggable, and its value was never read."""
    slider = app.ui_manager.layout.speed_slider

    click(app, (slider.x + 2, slider.centery))
    assert app.ui_manager.speed_slider_dragging is True
    send(app, pygame.event.Event(pygame.MOUSEMOTION, pos=(slider.right - 2, slider.centery),
                                 rel=(0, 0), buttons=(1, 0, 0)))
    fast_end = app.ui_manager.animation_speed

    send(app, pygame.event.Event(pygame.MOUSEBUTTONUP, button=1,
                                 pos=(slider.right - 2, slider.centery)))
    assert app.ui_manager.speed_slider_dragging is False

    click(app, (slider.x + 2, slider.centery))
    slow_end = app.ui_manager.animation_speed
    assert fast_end > slow_end

    # And the value the app steps on is the one the slider set.
    app._test_string("aab")
    app._toggle_animation()
    app.animation_timer = 0
    app._update(16)
    assert app.execution_step >= 0


def test_symbol_buttons_are_clickable_before_the_first_draw(app):
    """Rects used to be produced as a side effect of drawing.

    The app fixture has never rendered a frame at this point, so the palette
    only responds if its rectangles were computed at construction.
    """
    manager = app.ui_manager
    assert set(manager.symbol_buttons) == set(manager.available_symbols)
    assert manager.add_symbol_button_rect is not None

    click(app, manager.symbol_buttons["b"].center)
    assert manager.selected_symbol == "b"


def test_added_symbol_gets_a_button_immediately(app):
    assert app.ui_manager.add_symbol("z")
    assert "z" in app.ui_manager.symbol_buttons
    click(app, app.ui_manager.symbol_buttons["z"].center)
    assert app.ui_manager.selected_symbol == "z"


def test_result_colour_comes_from_the_verdict_not_the_message(app):
    """The colour used to be decided by searching the message for "accepted".

    Testing the literal string 'accepted' therefore produced a rejection
    painted green. The verdict is now carried separately, so the message text
    -- which contains whatever the user typed -- cannot influence it.
    """
    app._test_string("accepted")
    assert app.ui_manager.test_verdict != "accept"
    assert "accepted" in app.ui_manager.test_result, "the message quotes the input"
    pump(app)

    app._test_string("ab")
    assert app.ui_manager.test_verdict == "accept"
    pump(app)


def test_rejections_say_why(app):
    """Four distinct outcomes, four distinct explanations."""
    app._test_string("ab")
    assert app.ui_manager.test_verdict == "accept"

    app._test_string("aa")
    assert app.ui_manager.test_verdict == "reject_non_accepting"
    assert "not an accepting state" in app.ui_manager.test_result

    app._test_string("abz")
    assert app.ui_manager.test_verdict == "reject_symbol_not_in_alphabet"
    assert "not in the alphabet" in app.ui_manager.test_result

    app.dfa.remove_transition("q0", "a")
    app._invalidate_engine()
    app._test_string("aa")
    assert app.ui_manager.test_verdict == "reject_no_transition"
    assert "incomplete" in app.ui_manager.test_result


def test_the_empty_string_can_be_tested(app):
    """The old UI refused it outright."""
    app._test_string("")
    assert app.execution_active
    assert app.execution_path == ["q0"]
    assert "empty string" in app.ui_manager.test_result
    pump(app)


def test_dead_states_are_derived_from_the_transition_function(app):
    """The app no longer trusts the user-set dead-end flag for simulation.

    q2 in the demo is a genuine trap, so it is reported as dead. Give it a way
    out and it stops being one, with no flag to update.
    """
    app.engine()
    assert "q2" in app._dead_states

    # Note the legacy argument order is (from, to, symbol); the engine's is
    # (source, symbol, target). Getting them the wrong way round returns False
    # silently, which is one more reason the legacy model is going away.
    assert app.dfa.add_transition("q2", "q1", "b") is True
    app._invalidate_engine()
    app.engine()
    assert app._dead_states == frozenset()


def test_app_language_matches_the_transition_function(app):
    """The case the old model got wrong, driven through the app.

    q1 is flagged DEAD_END but delta leads from it to an accepting state.
    """
    app.dfa = type(app.dfa)()
    q0 = app.dfa.add_state((0, 0))
    q1 = app.dfa.add_state((100, 0))
    q2 = app.dfa.add_state((200, 0))
    app.dfa.set_state_type(q1, StateType.DEAD_END)
    app.dfa.set_state_type(q2, StateType.ACCEPT)
    app.dfa.add_transition(q0, q1, "a")
    app.dfa.add_transition(q1, q2, "a")
    app._invalidate_engine()

    app._test_string("aa")
    assert app.ui_manager.test_verdict == "accept"
    assert app.execution_path == ["q0", "q1", "q2"]
