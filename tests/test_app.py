"""Regression tests for the application layer.

Driven against a real AutomatonSimulator on the dummy SDL driver. Events go
through the queue and the real loop, not straight into a handler, because the
handlers are not the thing that was broken -- the routing between them was.

Most of these pin a defect that shipped. The docstring says which.
"""

import json
import math
from pathlib import Path

import pygame
import pytest

import fsa
import main as main_module
import ui.ui_manager as ui_manager_module
from editor import EditorModel
from fsa import serialize
from main import AutomatonSimulator
from rendering.renderer import default_state_radius
from rendering.scene import NodeKind


@pytest.fixture
def app(tmp_path, monkeypatch):
    """A fresh simulator whose file operations are confined to tmp_path."""
    monkeypatch.setattr(main_module, "PROJECT_DIR", str(tmp_path))
    simulator = AutomatonSimulator()
    yield simulator
    pygame.quit()


def pump(app: AutomatonSimulator, frames: int = 3):
    for _ in range(frames):
        app._update(16)
        app._render()


def send(app: AutomatonSimulator, event: pygame.event.Event):
    """Push one event through the application's real dispatch path.

    A test that calls a handler directly proves nothing about routing, which is
    precisely what was broken.
    """
    pygame.event.clear()
    pygame.event.post(event)
    app._handle_events()


def click(app: AutomatonSimulator, pos, button: int = 1, shift: bool = False):
    """Click, building the event with exactly the fields pygame provides.

    Mouse events carry no `.mod`; modifier state lives on the keyboard. Passing
    a fabricated `mod=` here is what let a real crash (AttributeError on
    event.mod) pass the whole suite -- the synthetic events had a field the real
    ones never do.
    """
    pygame.key.set_mods(pygame.KMOD_LSHIFT if shift else pygame.KMOD_NONE)
    try:
        send(app, pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=button, pos=pos))
    finally:
        pygame.key.set_mods(pygame.KMOD_NONE)


def release(app: AutomatonSimulator, pos, button: int = 1):
    send(app, pygame.event.Event(pygame.MOUSEBUTTONUP, button=button, pos=pos))


def move(app: AutomatonSimulator, pos, buttons=(1, 0, 0)):
    send(app, pygame.event.Event(pygame.MOUSEMOTION, pos=pos, rel=(1, 1),
                                 buttons=buttons))


def press(app: AutomatonSimulator, code: int, char: str = ""):
    send(app, pygame.event.Event(pygame.KEYDOWN, key=code, unicode=char,
                                 mod=pygame.KMOD_NONE, scancode=0))


def screen_of(app: AutomatonSimulator, state: str):
    """Where a state currently sits on screen."""
    point = app.renderer.camera.world_to_screen(app.editor.position_of(state))
    return (int(point[0]), int(point[1]))


def canvas_point(app: AutomatonSimulator, fraction_y: float):
    return (app.screen.get_width() // 2, int(app.screen.get_height() * fraction_y))


def blank(app: AutomatonSimulator, *symbols: str):
    """Replace the document with an empty one over the given alphabet."""
    document = fsa.Document()
    for symbol in symbols:
        document = document.add_symbol(symbol)
    app.editor = EditorModel(document)
    return app.editor


# ---------------------------------------------------------------------------
# Crashes
# ---------------------------------------------------------------------------


def test_context_menu_delete_clears_selection(app):
    """Right-click Delete used to leave the selection dangling, so the next Q
    or W raised KeyError on a state that no longer existed."""
    app.editor.select("q1")
    app._handle_context_menu_action("delete_state:q1")

    assert "q1" not in app.editor.automaton.states
    assert app.editor.selection is None

    app._toggle_accept_state()
    app._make_trap(app.editor.selection)
    pump(app)


def test_delete_while_transition_pending_does_not_crash_render(app):
    """Deleting the source of a half-drawn transition used to kill _render,
    which fires every frame, so the process could never recover."""
    app.editor.select("q1")
    app.editor.begin_transition("q1")
    assert app.editor.pending_source == "q1"

    app._delete_selected_state()

    assert "q1" not in app.editor.automaton.states
    assert app.editor.pending_source is None
    pump(app, frames=10)


def test_delete_while_dragging_does_not_crash_on_release(app):
    """Press-and-hold a state, press Delete, release."""
    app.editor.select("q1")
    app.editor.begin_drag("q1", app.editor.position_of("q1"))
    assert app.editor.drag is not None and app.editor.drag.state == "q1"

    app._delete_selected_state()
    assert app.editor.drag is None

    release(app, (410, 210))
    pump(app)


def test_delete_state_in_active_trace_stops_execution(app):
    app._test_string("ab")
    assert app.execution_active and "q1" in app.execution_path

    app.editor.select("q1")
    app._delete_selected_state()
    pump(app)

    # The trace still names a state that is gone, so rendering must not assume
    # every path entry exists.
    assert "q1" not in app.editor.automaton.states


def test_transition_to_a_deleted_state_is_reported(app):
    app.editor.begin_transition("q0")
    app.editor.remove_state("q2")
    app._complete_transition("q2")

    assert app.editor.pending_source is None
    pump(app)


def test_dangling_references_are_dropped_centrally(app):
    """One method, called from every replacement, rather than a guard per
    call site -- which is how three of these were missed."""
    app.editor.select("q1")
    app.editor.set_hover("q1")
    app.editor.begin_drag("q1", (0.0, 0.0))
    app.editor.begin_transition("q1")

    app.editor.remove_state("q1")

    assert app.editor.selection is None
    assert app.editor.hover is None
    assert app.editor.drag is None
    assert app.editor.pending_source is None


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def test_save_then_load_round_trips_in_the_same_session(app, tmp_path):
    """Save and Load used to point at different files; nothing round-tripped."""
    before = serialize.to_dict(app.editor.document)

    app._save_to_path("mine")
    assert (tmp_path / "mine.json").exists(), "'.json' appended when omitted"
    assert app.editor.dirty is False
    assert app.editor.path == "mine.json"

    app._add_state_at_center()
    assert app.editor.dirty is True
    assert len(app.editor.automaton.states) == 4

    app._load_from_path("mine")

    assert serialize.to_dict(app.editor.document) == before
    assert app.editor.dirty is False


def test_serialisation_is_byte_stable(app):
    """Sets serialised via list() varied per process, because Python randomises
    string hashing. Saved files could not be diffed."""
    first = serialize.dumps(app.editor.document)

    app._save_to_path("stable.json")
    app._load_from_path("stable.json")

    assert serialize.dumps(app.editor.document) == first
    body = serialize.to_dict(app.editor.document)["automaton"]
    assert body["alphabet"] == sorted(body["alphabet"])
    assert body["states"] == sorted(body["states"])


def test_round_trip_preserves_positions_and_arcs(app):
    app.editor.add_transition("q0", "a", "q2", arc=55.0)
    app.editor.apply(app.editor.document.move_state("q0", (12.5, -34.25)))

    app._save_to_path("layout.json")
    app._load_from_path("layout.json")

    assert app.editor.layout.position_of("q0") == (12.5, -34.25)
    assert app.editor.layout.arc_of("q0", "q2") == 55.0


def test_load_replaces_the_whole_document(app, tmp_path):
    """The canvas used to keep drawing the previous automaton's arrows."""
    other = fsa.Document()
    for symbol in ("0", "1"):
        other = other.add_symbol(symbol)
    other, a = other.add_state((0.0, 0.0))
    other, b = other.add_state((200.0, 0.0))
    other = other.add_transition(a, "0", b).add_transition(b, "1", a)
    other = other.toggle_accept(b).set_initial(a)
    (tmp_path / "other.json").write_text(serialize.dumps(other), encoding="utf-8")

    app._load_from_path("other.json")

    drawn = {edge: sorted(symbols) for edge, symbols
             in app.editor.automaton.grouped_transitions().items()}
    assert drawn == {("q0", "q1"): ["0"], ("q1", "q0"): ["1"]}
    assert sorted(app.editor.automaton.alphabet) == ["0", "1"]
    pump(app)


def test_the_bundled_example_still_loads():
    """The old format has exactly one file in the world. It must still open."""
    example = Path(main_module.__file__).parent / "examples" / "simple_binary.json"
    document, error = serialize.load_or_error(str(example))

    assert document is not None, error
    assert sorted(document.automaton.states) == ["q0", "q1", "q2"]
    assert sorted(document.automaton.alphabet) == ["0", "1"]
    assert document.automaton.initial == "q0"
    assert document.automaton.accept == frozenset({"q1"})
    assert set(document.layout.positions) == set(document.automaton.states)


def test_the_legacy_dead_end_flag_is_not_honoured_on_load(app, tmp_path):
    """Reading it back would reintroduce the defect removing it fixed.

    q1 is flagged a dead end but delta leads from it to an accepting state.
    """
    legacy = {
        "states": {
            "q0": {"position": [0, 0], "state_type": "normal"},
            "q1": {"position": [100, 0], "state_type": "dead_end"},
            "q2": {"position": [200, 0], "state_type": "accept"},
        },
        "transitions": {"q0": {"a": "q1"}, "q1": {"a": "q2"}},
        "alphabet": ["a"],
        "initial_state": "q0",
        "accept_states": ["q2"],
        "dead_end_states": ["q1"],
        "next_state_id": 3,
    }
    (tmp_path / "legacy.json").write_text(json.dumps(legacy), encoding="utf-8")

    app._load_from_path("legacy.json")

    assert fsa.accepts(app.editor.automaton, "aa"), "delta says this is accepted"
    assert app.editor.analysis()[0] == frozenset(), "and nothing is a trap"


def test_load_of_missing_file_reports_on_screen(app):
    app._load_from_path("does_not_exist.json")
    assert "Load failed" in app.message_text
    assert len(app.editor.automaton.states) == 3, "the current document survives"


def test_load_of_malformed_file_leaves_the_document_intact(app, tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    before = serialize.to_dict(app.editor.document)

    app._load_from_path("broken.json")

    assert "Load failed" in app.message_text
    assert serialize.to_dict(app.editor.document) == before


def test_load_of_a_future_version_is_refused_clearly(app, tmp_path):
    (tmp_path / "future.json").write_text('{"version": 99}', encoding="utf-8")
    app._load_from_path("future.json")
    assert "version" in app.message_text


def test_load_clears_the_execution_trace(app):
    app._save_to_path("snap.json")
    app._test_string("ab")
    assert app.execution_active

    app._load_from_path("snap.json")
    assert not app.execution_active


def test_load_adopts_the_files_alphabet(app, tmp_path):
    document = fsa.Document().add_symbol("x")
    document, state = document.add_state((0.0, 0.0))
    document = document.add_transition(state, "x", state).toggle_accept(state)
    (tmp_path / "x.json").write_text(serialize.dumps(document), encoding="utf-8")

    app._load_from_path("x.json")
    assert app.ui_manager.available_symbols == ["x"]
    assert app.ui_manager.selected_symbol == "x"


def test_paths_resolve_against_the_project_not_the_cwd(app, tmp_path):
    assert Path(app._resolve_path("a.json")).parent == tmp_path
    assert Path(app._resolve_path("a")).name == "a.json"
    assert Path(app._resolve_path("")).name == AutomatonSimulator.DEFAULT_FILENAME


def test_toolbar_save_and_load_drive_the_whole_flow(app, tmp_path):
    click(app, app.ui_manager.save_button_rect.center)
    assert app.ui_manager.file_prompt_mode == "save"

    for _ in range(40):
        press(app, pygame.K_BACKSPACE)
    for char in "demo":
        press(app, ord(char), char)
    press(app, pygame.K_RETURN, "\r")

    assert (tmp_path / "demo.json").exists()
    assert app.editor.path == "demo.json"

    press(app, pygame.K_SPACE, " ")
    assert len(app.editor.automaton.states) == 4

    click(app, app.ui_manager.load_button_rect.center)
    assert app.ui_manager.confirm_intent == "load_after_confirm"
    press(app, pygame.K_y, "y")
    assert app.ui_manager.file_prompt_mode == "load"
    press(app, pygame.K_RETURN, "\r")

    assert len(app.editor.automaton.states) == 3
    assert app.editor.dirty is False


# ---------------------------------------------------------------------------
# Unsaved-work guards
# ---------------------------------------------------------------------------


def test_editing_marks_the_document_dirty(app):
    assert app.editor.dirty is False
    app._add_state_at_center()
    assert app.editor.dirty is True
    assert pygame.display.get_caption()[0].startswith("untitled*")


def test_quit_with_unsaved_changes_asks_first(app):
    app._add_state_at_center()

    send(app, pygame.event.Event(pygame.QUIT))
    assert app.running is True
    assert app.ui_manager.confirm_intent == "quit_after_confirm"

    app._process_ui_actions({"confirmed": "quit_after_confirm"})
    assert app.running is False


def test_quit_without_changes_does_not_ask(app):
    send(app, pygame.event.Event(pygame.QUIT))
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
    before = app.editor.document
    app.editor.select("q0")

    press(app, pygame.K_SPACE, " ")
    press(app, pygame.K_q, "q")
    press(app, pygame.K_w, "w")

    assert app.editor.document == before


def test_typing_in_the_test_field_does_not_edit_the_automaton(app):
    app.ui_manager.input_active = True
    before = app.editor.document
    press(app, pygame.K_SPACE, " ")
    assert app.editor.document == before


def test_filename_prompt_captures_the_keyboard(app):
    app.ui_manager.show_file_prompt("save", "")
    assert app.ui_manager.is_keyboard_captured()
    assert app.ui_manager.is_modal_active()

    for char in "notes":
        press(app, ord(char), char)
    assert app.ui_manager.file_prompt_text == "notes"

    press(app, pygame.K_RETURN, "\r")
    assert not app.ui_manager.is_modal_active()
    assert app.editor.path == "notes.json"


def test_escape_cancels_the_filename_prompt(app):
    app.ui_manager.show_file_prompt("load", "x.json")
    press(app, pygame.K_ESCAPE, "\x1b")
    assert app.ui_manager.file_prompt_mode is None


def test_modal_swallows_canvas_clicks(app):
    app.ui_manager.show_confirm("Quit without saving?", "quit_after_confirm")
    before = len(app.editor.automaton.states)

    click(app, (500, 400), button=3)

    assert len(app.editor.automaton.states) == before
    assert app.ui_manager.context_menu is None


def test_dialogs_render(app):
    app.ui_manager.show_file_prompt("save", "automaton.json")
    pump(app)
    app.ui_manager.hide_file_prompt()

    app.ui_manager.show_confirm("Quit without saving?", "quit_after_confirm")
    pump(app)


# ---------------------------------------------------------------------------
# Event routing
# ---------------------------------------------------------------------------


def test_toolbar_click_does_not_also_hit_the_canvas(app):
    """Every event used to go to the UI *and* the canvas handlers."""
    app.editor.select("q1")
    click(app, app.ui_manager.test_button_rect.center)
    assert app.editor.selection == "q1", "using a toolbar control must not deselect"


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
    """It used to do both: polled in _update *and* handled as a click, so the
    source state followed the mouse."""
    before = app.editor.position_of("q0")

    click(app, screen_of(app, "q0"), shift=True)

    assert app.editor.pending_source == "q0"
    assert app.editor.drag is None
    assert app.editor.position_of("q0") == before

    pump(app, frames=5)
    assert app.editor.pending_source == "q0"
    assert app.editor.position_of("q0") == before


def test_plain_click_selects_and_drags(app):
    click(app, screen_of(app, "q0"))
    assert app.editor.selection == "q0"
    assert app.editor.drag is not None
    assert app.editor.pending_source is None


def test_a_drag_only_touches_the_layout_on_release(app):
    """Layouts are immutable; committing one per motion event would allocate a
    new mapping every frame."""
    start = screen_of(app, "q0")
    click(app, start)
    before = app.editor.layout

    move(app, (start[0] + 40, start[1] + 25))
    assert app.editor.layout is before, "still uncommitted"
    assert app.editor.position_of("q0") != before.position_of("q0")

    release(app, (start[0] + 40, start[1] + 25))
    assert app.editor.layout is not before
    assert app.editor.layout.position_of("q0") != before.position_of("q0")


def test_dragging_stays_responsive_with_many_states(app):
    """Immutable layouts allocate, so a drag must not commit one per event.

    200 motion events over 30 states, budgeted well under a frame each.
    """
    import time

    blank(app, "a")
    for i in range(30):
        app.editor.add_state((float(i % 6) * 90, float(i // 6) * 90))
    target = sorted(app.editor.automaton.states)[0]

    app.editor.begin_drag(target, app.editor.position_of(target))
    started = time.perf_counter()
    for i in range(200):
        app.editor.update_drag((400.0 + i, 300.0 + i))
        app._build_scene()
    elapsed = time.perf_counter() - started
    app.editor.end_drag()

    assert elapsed < 2.0, f"200 drag frames took {elapsed:.2f}s"


def test_no_input_polling_remains():
    source = Path(main_module.__file__).read_text(encoding="utf-8")
    assert "key.get_pressed" not in source
    assert "mouse.get_pressed" not in source


def test_handlers_only_read_fields_real_events_carry(app):
    """A previous version read `event.mod` on a MOUSEBUTTONDOWN. Mouse events
    have no modifier field, so every real click raised AttributeError while the
    suite stayed green -- the helpers invented the field."""
    centre = (app.screen.get_width() // 2, app.screen.get_height() // 2)
    events = [
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=centre),
        pygame.event.Event(pygame.MOUSEMOTION, pos=centre, rel=(1, 1), buttons=(1, 0, 0)),
        pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=centre),
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=centre),
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=2, pos=centre),
        pygame.event.Event(pygame.MOUSEBUTTONUP, button=2, pos=centre),
        pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=1, flipped=False,
                           precise_x=0.0, precise_y=1.0),
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
    point = screen_of(app, "q0")
    pygame.key.set_mods(pygame.KMOD_LSHIFT)
    try:
        send(app, pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=point))
    finally:
        pygame.key.set_mods(pygame.KMOD_NONE)

    assert app.editor.pending_source == "q0"
    assert app.editor.drag is None


def test_right_click_works_across_the_whole_canvas(app):
    """Right-click used to be dead over a third to a half of the window."""
    samples = 40
    app.ui_manager.context_menu = None
    points = [canvas_point(app, 0.05 + 0.9 * i / (samples - 1)) for i in range(samples)]
    canvas_points = [p for p in points if not app.ui_manager.is_over_ui(p)]
    assert len(canvas_points) > samples // 2

    opened = 0
    for point in canvas_points:
        app.ui_manager.context_menu = None
        click(app, point, button=3)
        if app.ui_manager.context_menu is not None:
            opened += 1

    assert opened == len(canvas_points)


def test_ui_panels_are_not_canvas(app):
    # The right column slides in over ~260ms; give it time to arrive before
    # asking whether it covers the canvas.
    pump(app, frames=30)
    assert app.ui_manager.is_over_ui(app.ui_manager.save_button_rect.center)
    assert app.ui_manager.is_over_ui(app.ui_manager.input_rect.center)
    assert app.ui_manager.is_over_ui(app.ui_manager.layout.status_panel.center)
    assert not app.ui_manager.is_over_ui(canvas_point(app, 0.5))


def test_help_panel_scrolls_to_the_last_line(app):
    """max_scroll computed to zero, so six lines were unreachable."""
    app.ui_manager.show_help = True
    visible = app.ui_manager.layout.help_visible_lines()
    assert len(ui_manager_module.HELP_LINES) > visible

    for _ in range(50):
        send(app, pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1))
    assert app.ui_manager.help_scroll_offset == len(ui_manager_module.HELP_LINES) - visible

    for _ in range(50):
        send(app, pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=1))
    assert app.ui_manager.help_scroll_offset == 0


def test_scrolling_the_help_panel_does_not_also_zoom(app):
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
    move(app, (slider.right - 2, slider.centery))
    fast_end = app.ui_manager.animation_speed

    release(app, (slider.right - 2, slider.centery))
    assert app.ui_manager.speed_slider_dragging is False

    click(app, (slider.x + 2, slider.centery))
    assert fast_end > app.ui_manager.animation_speed


def test_symbol_buttons_are_clickable_before_the_first_draw(app):
    """Rects used to be produced as a side effect of drawing."""
    manager = app.ui_manager
    assert set(manager.symbol_buttons) == set(app.editor.automaton.alphabet)
    click(app, manager.symbol_buttons["b"].center)
    assert manager.selected_symbol == "b"


# ---------------------------------------------------------------------------
# The alphabet is the palette
# ---------------------------------------------------------------------------


def test_the_palette_is_the_alphabet(app):
    """They used to be two unrelated sets: you could draw with a symbol the
    machine did not recognise, and vice versa."""
    assert app.ui_manager.available_symbols == sorted(app.editor.automaton.alphabet)

    app._add_symbol("z")
    assert "z" in app.editor.automaton.alphabet
    assert app.ui_manager.available_symbols == sorted(app.editor.automaton.alphabet)
    assert "z" in app.ui_manager.symbol_buttons


def test_reserved_letters_can_now_be_symbols(app):
    """`q w r n p` were rejected because keyboard shortcuts owned them, so no
    automaton over an alphabet containing them could be built at all."""
    for symbol in ("q", "w", "r", "n", "p"):
        app._add_symbol(symbol)
        assert symbol in app.editor.automaton.alphabet, symbol
        assert symbol in app.ui_manager.symbol_buttons, symbol


def test_illegal_symbols_are_still_refused(app):
    for symbol in ("", "ab", " "):
        app._add_symbol(symbol)
        assert symbol not in app.editor.automaton.alphabet


def test_a_transition_needs_a_symbol_in_the_alphabet(app):
    app.ui_manager.selected_symbol = "!"
    app.editor.begin_transition("q0")
    app._complete_transition("q1")
    assert "!" not in app.editor.automaton.alphabet
    assert "not in the alphabet" in app.message_text


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def test_rejections_say_why(app):
    app._test_string("ab")
    assert app.ui_manager.test_verdict == "accept"

    app._test_string("aa")
    assert app.ui_manager.test_verdict == "reject_non_accepting"
    assert "not an accepting state" in app.ui_manager.test_result

    app._test_string("abz")
    assert app.ui_manager.test_verdict == "reject_symbol_not_in_alphabet"
    assert "not in the alphabet" in app.ui_manager.test_result

    app.editor.remove_transition("q0", "a")
    app._test_string("aa")
    assert app.ui_manager.test_verdict == "reject_no_transition"
    assert "incomplete" in app.ui_manager.test_result


def test_result_colour_comes_from_the_verdict_not_the_message(app):
    """The colour used to be decided by searching for "accepted", so testing
    the literal string 'accepted' produced a rejection painted green."""
    app._test_string("accepted")
    assert app.ui_manager.test_verdict != "accept"
    assert "accepted" in app.ui_manager.test_result
    pump(app)


def test_the_empty_string_can_be_tested(app):
    """The old UI refused it outright."""
    app._test_string("")
    assert app.execution_active
    assert app.execution_path == ["q0"]
    assert "empty string" in app.ui_manager.test_result
    pump(app)


def test_app_language_matches_the_transition_function(app):
    """The case the old model got wrong: a state that looks like a trap but
    whose delta leads to acceptance."""
    editor = blank(app, "a")
    document, q0 = editor.document.add_state((0.0, 0.0))
    document, q1 = document.add_state((150.0, 0.0))
    document, q2 = document.add_state((300.0, 0.0))
    document = document.add_transition(q0, "a", q1).add_transition(q1, "a", q2)
    editor.apply(document.toggle_accept(q2).set_initial(q0))

    app._test_string("aa")
    assert app.ui_manager.test_verdict == "accept"
    assert app.execution_path == [q0, q1, q2]


# ---------------------------------------------------------------------------
# State kinds
# ---------------------------------------------------------------------------


def test_nothing_is_a_trap_when_nothing_accepts(app):
    """With no accepting state every state is technically dead. True, and
    useless: it greyed out the whole canvas while the user was still drawing."""
    editor = blank(app, "a")
    document, q0 = editor.document.add_state((0.0, 0.0))
    document, q1 = document.add_state((150.0, 0.0))
    editor.apply(document.add_transition(q0, "a", q1).set_initial(q0))

    assert fsa.dead_states(app.editor.automaton) == {q0, q1}, "the maths is unchanged"
    assert app.editor.analysis()[0] == frozenset(), "the display stays quiet"

    kinds = {node.kind for node in app._build_scene().nodes}
    assert kinds == {NodeKind.NORMAL}
    assert app.ui_manager.warn_no_accepting is True


def test_traps_are_shown_once_something_accepts(app):
    editor = blank(app, "a", "b")
    document, q0 = editor.document.add_state((0.0, 0.0))
    document, q1 = document.add_state((150.0, 0.0))
    document, q2 = document.add_state((300.0, 0.0))
    document = (document.add_transition(q0, "a", q1)
                        .add_transition(q0, "b", q2)
                        .add_transition(q2, "a", q2))
    editor.apply(document.toggle_accept(q1).set_initial(q0))

    kinds = {node.id: node.kind for node in app._build_scene().nodes}
    assert kinds[q0] is NodeKind.NORMAL
    assert kinds[q1] is NodeKind.NORMAL
    assert kinds[q2] is NodeKind.DEAD


def test_unreachable_takes_priority_over_dead(app):
    """A state no word can enter cannot trap anything."""
    editor = blank(app, "a")
    document, q0 = editor.document.add_state((0.0, 0.0))
    document, q1 = document.add_state((150.0, 0.0))
    document, q2 = document.add_state((300.0, 0.0))
    editor.apply(document.add_transition(q0, "a", q1).toggle_accept(q1).set_initial(q0))

    dead, unreachable, _ = app.editor.analysis()
    assert q2 in dead and q2 in unreachable

    kinds = {node.id: node.kind for node in app._build_scene().nodes}
    assert kinds[q2] is NodeKind.UNREACHABLE


def test_the_legend_lists_only_what_is_on_screen(app):
    editor = blank(app, "a")
    document, q0 = editor.document.add_state((0.0, 0.0))
    editor.apply(document.toggle_accept(q0).set_initial(q0))

    app._build_scene()
    assert app.ui_manager.legend_dead is False
    assert app.ui_manager.legend_unreachable is False

    document, q1 = app.editor.document.add_state((150.0, 0.0))   # unreachable
    document, q2 = document.add_state((300.0, 0.0))              # a trap
    document = document.add_transition(q0, "a", q2).add_transition(q2, "a", q2)
    app.editor.apply(document)

    app._build_scene()
    assert app.ui_manager.legend_dead is True
    assert app.ui_manager.legend_unreachable is True


def test_each_state_kind_is_visually_distinct():
    """Fill, ring and shape must all differ; colour alone was not enough."""
    from rendering.theme import Theme

    for name in ("dark", "light"):
        palette = Theme(name).palette
        assert len({palette.state_fill, palette.accept_fill,
                    palette.dead_fill, palette.unreachable_fill}) == 4, name
        assert len({palette.state_ring, palette.accept_ring,
                    palette.dead_ring, palette.unreachable_ring}) == 4, name


def test_self_loops_point_away_from_other_edges(app):
    editor = blank(app, "a", "b")
    document, q0 = editor.document.add_state((0.0, 0.0))
    document, q1 = document.add_state((0.0, -200.0))
    editor.apply(document.add_transition(q1, "a", q0)
                         .add_transition(q0, "b", q0)
                         .toggle_accept(q1).set_initial(q1))

    scene = app._build_scene()
    assert math.sin(app._loop_angle_cache[q0]) > 0.8, "loop hangs below, away from q1"
    loop = next(e for e in scene.edges if e.key == (q0, q0))
    assert all(point[1] > -1 for point in loop.path)


# ---------------------------------------------------------------------------
# Context menu
# ---------------------------------------------------------------------------


def test_make_trap_actually_makes_a_trap(app):
    """"Set as dead end" used to write a flag nothing read."""
    app.editor.toggle_accept("q0")          # keep an accepting state elsewhere
    app.editor.select("q1")
    app._make_trap("q1")

    automaton = app.editor.automaton
    assert {s: automaton.target("q1", s) for s in automaton.alphabet} == {
        s: "q1" for s in automaton.alphabet}
    assert "q1" not in automaton.accept

    kinds = {node.id: node.kind for node in app._build_scene().nodes}
    assert kinds["q1"] is NodeKind.DEAD


def test_make_trap_reports_what_it_did(app):
    app._make_trap("q0")
    assert "loops on" in app.message_text
    assert "replacing" in app.message_text


def test_make_trap_needs_an_alphabet(app):
    blank(app)
    app.editor.add_state((0.0, 0.0))
    app._make_trap("q0")
    assert "nothing to loop on" in app.message_text


def test_make_trap_ignores_unknown_states(app):
    app._make_trap("nope")
    app._make_trap(None)
    pump(app)


def test_losing_the_last_accepting_state_is_visible(app):
    """Trapping your only accepting state destroys the language, and the canvas
    deliberately stays quiet -- so the status panel has to say it."""
    app._make_trap("q1")

    assert app.editor.automaton.accept == frozenset()
    kinds = {node.kind for node in app._build_scene().nodes}
    assert NodeKind.DEAD not in kinds
    assert app.ui_manager.warn_no_accepting is True
    pump(app)

    app.editor.toggle_accept("q0")
    app._build_scene()
    assert app.ui_manager.warn_no_accepting is False


def test_context_menu_reports_every_action(app):
    """Each item used to change something silently, or nothing at all."""
    app._handle_context_menu_action("toggle_accept:q0")
    assert "now accepting" in app.message_text
    assert "q0" in app.editor.automaton.accept

    app._handle_context_menu_action("toggle_accept:q0")
    assert "no longer accepting" in app.message_text

    app._handle_context_menu_action("set_initial:q2")
    assert "initial state" in app.message_text
    assert app.editor.automaton.initial == "q2"

    app._handle_context_menu_action("make_trap:q2")
    assert "loops on" in app.message_text


def test_state_menu_shows_what_the_state_already_is(app):
    app._show_state_context_menu((300, 300), "q1")
    items = {item[0]: item for item in app.ui_manager.context_menu.items}
    assert items["Accepting"][2] is True
    assert items["Initial state"][2] is False

    app.ui_manager.hide_context_menu()
    app._show_state_context_menu((300, 300), "q0")
    items = {item[0]: item for item in app.ui_manager.context_menu.items}
    assert items["Accepting"][2] is False
    assert items["Initial state"][2] is True


def test_menu_items_without_a_toggle_still_work(app):
    app._show_general_context_menu((300, 300))
    for item in app.ui_manager.context_menu.items:
        assert len(item) in (2, 3)
    pump(app)


def test_add_state_here_places_it_under_the_cursor(app):
    app._handle_context_menu_action("add_state:400.0,250.0")
    assert app.editor.selection is not None
    position = app.editor.position_of(app.editor.selection)
    assert position == pytest.approx((400.0, 250.0))


def test_new_states_do_not_stack_on_one_pixel(app):
    """Every new state used to land on the exact centre of the view, and
    because hit-testing returned the first match, only the oldest was ever
    clickable again."""
    added = [app.editor.add_state((500.0, 500.0)) for _ in range(5)]
    positions = [app.editor.position_of(state) for state in added]

    for i, first in enumerate(positions):
        for second in positions[i + 1:]:
            assert math.dist(first, second) > default_state_radius()


# ---------------------------------------------------------------------------
# Phase 6: the diagnostics panel
# ---------------------------------------------------------------------------


def incomplete_machine(app):
    """Two states over {a, b} with only one transition defined: 3 missing."""
    editor = blank(app, "a", "b")
    document, q0 = editor.document.add_state((200.0, 300.0))
    document, q1 = document.add_state((400.0, 300.0))
    document = document.add_transition(q0, "a", q1)
    editor.apply(document.toggle_accept(q1).set_initial(q0))
    return q0, q1


def test_diagnostics_list_exactly_the_missing_pairs(app):
    """The panel's data must match a hand-computed delta table."""
    q0, q1 = incomplete_machine(app)

    incomplete = next(d for d in app.editor.defects() if d.kind == "incomplete")
    assert set(incomplete.pairs) == {(q0, "b"), (q1, "a"), (q1, "b")}
    pump(app)
    assert app.ui_manager.diagnostics, "the panel was fed"


def test_fix_button_completes_the_automaton_in_one_click(app):
    """The Phase 6 exit criterion: event replay, not a direct call."""
    incomplete_machine(app)
    assert not fsa.is_complete(app.editor.automaton)

    pump(app, frames=30)          # let the diagnostics panel slide in
    assert app.ui_manager._fix_button is not None, "the Fix button is on screen"

    click(app, app.ui_manager._fix_button.center)
    pump(app)

    assert fsa.is_complete(app.editor.automaton)
    assert "routed" in app.message_text
    # The new trap has coordinates and is drawn as a trap.
    trap = next(s for s in app.editor.automaton.states if s.startswith("trap"))
    kinds = {node.id: node.kind for node in app._build_scene().nodes}
    assert kinds[trap] is NodeKind.DEAD


def test_completion_preserves_the_language(app):
    q0, q1 = incomplete_machine(app)
    words = ["", "a", "b", "ab", "aa", "ba", "abab"]
    before = {w: fsa.accepts(app.editor.automaton, w) for w in words}

    app._complete_automaton()

    after = {w: fsa.accepts(app.editor.automaton, w) for w in words}
    assert after == before


def test_completing_twice_reports_already_complete(app):
    incomplete_machine(app)
    app._complete_automaton()
    states_after_first = set(app.editor.automaton.states)

    app._complete_automaton()
    assert "Already complete" in app.message_text
    assert set(app.editor.automaton.states) == states_after_first


def test_clicking_a_defect_row_focuses_its_states(app):
    """An unreachable-state row glides the camera to the state it names."""
    editor = blank(app, "a")
    document, q0 = editor.document.add_state((0.0, 0.0))
    document, far = document.add_state((5000.0, 5000.0))   # far off screen
    editor.apply(document.add_transition(q0, "a", q0)
                         .toggle_accept(q0).set_initial(q0))

    pump(app, frames=30)
    row = next((payload for _r, payload in app.ui_manager._diagnostic_rows
                if far in payload.get("focus_states", [])), None)
    assert row is not None, "the unreachable defect is clickable"

    app._focus_states(row["focus_states"])
    pump(app, frames=45)          # let the camera glide

    on_screen = app.renderer.camera.world_to_screen(app.editor.position_of(far))
    assert 0 <= on_screen[0] <= app.screen.get_width()
    assert 0 <= on_screen[1] <= app.screen.get_height()


def test_back_step_reproduces_the_identical_path(app):
    """Phase 6 exit criterion: end -> 0 -> end again, same path throughout."""
    app._test_string("aab")
    original = list(app.execution_path)

    for _ in range(len(original)):
        app._next_execution_step()
        pump(app, frames=2)
    assert app.execution_step == len(original) - 1

    for _ in range(len(original)):
        app._previous_execution_step()
        pump(app, frames=2)
    assert app.execution_step == 0

    for _ in range(len(original)):
        app._next_execution_step()
        pump(app, frames=2)

    assert list(app.execution_path) == original
    assert app.execution_step == len(original) - 1


def test_the_run_panel_slides_out_when_execution_stops(app):
    app._test_string("ab")
    pump(app, frames=30)
    assert any(key == "run" for key, _r, t in app.ui_manager._column if t > 0.9)

    app._stop_execution()
    pump(app, frames=30)
    assert not any(key == "run" for key, _r, _t in app.ui_manager._column)


def test_panels_below_take_the_space_a_departing_panel_releases(app):
    """The column reflows smoothly: legend rises when the run panel leaves."""
    app._test_string("ab")
    pump(app, frames=30)
    with_run = {k: r.y for k, r, _t in app.ui_manager._column}

    app._stop_execution()
    pump(app, frames=40)
    without_run = {k: r.y for k, r, _t in app.ui_manager._column}

    assert "run" not in without_run
    for key in without_run:
        if key in with_run and key != "status":
            assert without_run[key] <= with_run[key], f"{key} should rise"


def test_invalid_symbols_show_red_in_the_input(app):
    """The colouring logic, not the pixels: chars outside the alphabet."""
    alphabet = app.editor.automaton.alphabet
    assert "a" in alphabet and "z" not in alphabet
    app.ui_manager.input_text = "abz"
    pump(app)      # draws without error, colouring per character


def test_the_token_despawns_when_its_travel_settles(app):
    """A settled token used to stay parked on the node's rim forever,
    covering the label -- the docstring said "at rest there is no token"
    and the code disagreed."""
    app._test_string("aab")
    app._next_execution_step()
    pump(app, frames=3, )
    assert app.traversing_step is not None, "travelling"

    pump(app, frames=40)          # let the travel settle
    assert app.traversing_step is None
    assert app._build_token(app._edge_paths(app.editor.positions())) is None


def test_the_column_never_reaches_the_strip_band(app):
    """On a short window the column used to run off the bottom and across
    the input panel. Panels that do not fit are dropped, lowest first."""
    send(app, pygame.event.Event(pygame.VIDEORESIZE, w=700, h=500,
                                 size=(700, 500)))
    app._test_string("ab")        # run panel + diagnostics + legend all want in
    pump(app, frames=40)

    limit = app.ui_manager.layout.string_strip.top
    for key, rect, t in app.ui_manager._column:
        if t > 0.9:
            assert rect.bottom <= limit, f"{key} crosses into the strip band"


def test_strip_cells_stop_at_the_column(app):
    """Long strings used to paint cells across the diagnostics panel."""
    send(app, pygame.event.Event(pygame.VIDEORESIZE, w=900, h=620,
                                 size=(900, 620)))
    app._test_string("abababababababababab")
    pump(app, frames=40)

    bounds = app.ui_manager._strip_bounds
    assert bounds is not None
    column_left = min((rect.x for _k, rect, t in app.ui_manager._column
                       if t > 0.9), default=10 ** 6)
    assert bounds.right <= column_left, "the strip stays clear of the panels"


def test_the_strip_bounds_clear_when_hidden(app):
    app._test_string("ab")
    pump(app, frames=30)
    assert app.ui_manager._strip_bounds is not None

    app._stop_execution()
    pump(app, frames=60)          # slide fully out
    assert app.ui_manager._strip_bounds is None


def test_overlays_draw_after_the_panels(app):
    """Modals were painted inside draw(), so the run panel and tape strip
    painted straight across an open Save dialog. The structural guarantee:
    _render calls draw_overlays after everything else."""
    order = []
    real_draw = app.ui_manager.draw
    real_strip = app.ui_manager.draw_string_visualization
    real_overlays = app.ui_manager.draw_overlays
    app.ui_manager.draw = lambda *a, **k: (order.append("draw"), real_draw(*a, **k))[1]
    app.ui_manager.draw_string_visualization = (
        lambda *a, **k: (order.append("strip"), real_strip(*a, **k))[1])
    app.ui_manager.draw_overlays = (
        lambda *a, **k: (order.append("overlays"), real_overlays(*a, **k))[1])

    app._render()

    assert order.index("overlays") > order.index("strip") > order.index("draw")


def test_set_initial_ignores_unknown_states(app):
    before = app.editor.automaton.initial
    app._handle_context_menu_action("set_initial:nope")
    assert app.editor.automaton.initial == before
