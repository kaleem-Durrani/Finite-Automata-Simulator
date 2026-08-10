"""
Finite Automata Simulator.

The application shell: owns the window, routes input, drives animation, and
turns the document into a scene for the renderer. It holds no automaton logic of
its own -- that is all in :mod:`fsa` -- and no drawing logic -- that is all in
:mod:`rendering`.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pygame

# Saved automata resolve against the project directory rather than the process
# working directory, so a file written in one session is findable in the next
# no matter where python was launched from.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# The engine lives under src/ so that it can be packaged and installed on its
# own. Running `python main.py` from a checkout has no install step, so put it
# on the path here rather than making the documented way to start the app
# depend on `pip install -e .` first.
_SRC = os.path.join(PROJECT_DIR, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ruff: noqa: E402
# Everything below imports after the path bootstrap above, deliberately.
import fsa
import fsa.document
from editor import EditorModel
from fsa import Document, geometry, serialize
from rendering.animation import Animated, AnimatedPoint, Track, ease_in_out, ease_out_back
from rendering.fonts import FontBook
from rendering.renderer import Renderer, default_state_radius
from rendering.scene import (
    EdgeVisual,
    GhostEdge,
    NodeKind,
    NodeVisual,
    Scene,
    StartMarker,
    TokenVisual,
)
from rendering.theme import Theme
from ui.ui_manager import UIManager

#: Symbols a brand-new document starts with, so the palette is never empty.
STARTING_ALPHABET = ("a", "b")


def demo_document() -> Document:
    """The automaton the application opens with: a*b+."""
    document = Document()
    for symbol in STARTING_ALPHABET:
        document = document.add_symbol(symbol)

    document, q0 = document.add_state((220.0, 220.0))
    document, q1 = document.add_state((470.0, 220.0))
    document, q2 = document.add_state((345.0, 400.0))

    document = document.add_transition(q0, "a", q0)
    document = document.add_transition(q0, "b", q1)
    document = document.add_transition(q1, "a", q2)
    document = document.add_transition(q1, "b", q1)
    document = document.add_transition(q2, "a", q2)
    document = document.add_transition(q2, "b", q2)
    return document.toggle_accept(q1).set_initial(q0)


class AutomatonSimulator:
    """The application."""

    DEFAULT_FILENAME = "automaton.json"

    def __init__(self) -> None:
        pygame.init()

        info = pygame.display.Info()
        self.screen = pygame.display.set_mode(
            (int(info.current_w * 0.75), int(info.current_h * 0.75)),
            pygame.RESIZABLE,
        )

        # Theme and fonts are shared, so switching palettes reaches every
        # surface at once.
        self.theme = Theme("dark")
        self.fonts = FontBook()
        self.editor = EditorModel(demo_document())
        self.renderer = Renderer(self.screen, self.theme, self.fonts)
        self.ui_manager = UIManager(self.screen, self.theme, self.fonts)

        self.running = True
        self.clock = pygame.time.Clock()
        self.fps = 60

        # Execution. The run comes from the engine, so a rejection carries a
        # reason and each step names the edge it took -- which is what the
        # travelling token animates along.
        self.execution_active = False
        self.execution_step = 0
        self.execution_string = ""
        self.execution_path: List[str] = []
        self.run_result: Optional[fsa.Run] = None

        self.animation_active = False
        self.animation_timer = 0
        self.animation_auto_advance = False

        # Animated view state. Each closes the gap to its target over time;
        # nothing in the interface snaps.
        self.node_active = Track(self.theme.motion.quick)
        self.node_selected = Track(self.theme.motion.instant)
        self.node_hover = Track(self.theme.motion.instant)
        self.node_settle = Track(self.theme.motion.normal)
        self.edge_active = Track(self.theme.motion.quick)
        self.token_travel = Animated(duration=self.theme.motion.step)
        self.traversing_step: Optional[int] = None

        self.cam_zoom = Animated(value=1.0, target=1.0,
                                 duration=self.theme.motion.normal,
                                 easing=ease_in_out)
        self.cam_offset = AnimatedPoint()
        self.cam_offset.jump_to((0.0, 0.0))

        self.message_text = ""
        self.message_timer = 0
        self.message_duration = 2400

        self.panning = False
        self.pan_start = (0, 0)
        self._right_press: Optional[Tuple[int, int]] = None
        self._right_dragged = False

        self._loop_angle_cache: Dict[str, float] = {}

        self.ui_manager.sync_symbols_with(self.editor.automaton)
        self._update_caption()

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(self.fps)
            self._handle_events()
            self._update(dt)
            self._render()
        pygame.quit()
        sys.exit()

    def _handle_events(self) -> None:
        """
        Process all pygame events.

        Each event is offered to the UI first. If the UI consumed it -- a click
        on a widget, a keypress into a text field, a wheel event over the open
        help panel -- the canvas never sees it. Previously every event went to
        both, so clicking Test also deselected the state behind it and scrolling
        the help panel also zoomed the camera.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._request_quit()
                continue

            if event.type == pygame.VIDEORESIZE:
                self._handle_resize(event.w, event.h)
                continue

            actions, consumed = self.ui_manager.handle_event(event)
            self._process_ui_actions(actions)
            if consumed:
                continue

            if event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_down(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                self._handle_mouse_up(event)
            elif event.type == pygame.MOUSEMOTION:
                self._handle_mouse_motion(event)
            elif event.type == pygame.MOUSEWHEEL:
                self._handle_mouse_wheel(event)
            elif event.type == pygame.KEYDOWN:
                self._handle_key_down(event)

    def _handle_resize(self, width: int, height: int) -> None:
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.renderer.update_screen_size(width, height)
        self.ui_manager.update_screen_size(width, height)

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    @staticmethod
    def _shift_held() -> bool:
        """
        Whether a shift key is down right now.

        Mouse events in pygame carry no modifier state -- only key events have
        a `.mod` field -- so a click has to ask the keyboard directly. Read once
        while handling the event, not sampled every frame, which is what made
        shift+click drag the state it was drawing from.
        """
        return bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)

    def _handle_mouse_down(self, event: pygame.event.Event) -> None:
        if event.button == 1:
            self._handle_left_click(event.pos, self._shift_held())
        elif event.button == 2:
            self.panning = True
            self.pan_start = event.pos
        elif event.button == 3:
            # Deferred: a right-drag pans (middle-button-only panning is
            # miserable on a trackpad), so the menu waits for release and only
            # opens if the button never travelled.
            self._right_press = event.pos
            self._right_dragged = False

    def _handle_mouse_up(self, event: pygame.event.Event) -> None:
        if event.button == 1:
            if self.editor.end_drag():
                self._update_caption()
        elif event.button == 2:
            self.panning = False
        elif event.button == 3:
            press, self._right_press = self._right_press, None
            if press is not None and not self._right_dragged:
                self._handle_right_click(press)
            self._right_dragged = False

    def _handle_mouse_motion(self, event: pygame.event.Event) -> None:
        if self._right_press is not None and event.buttons[2]:
            moved = (abs(event.pos[0] - self._right_press[0])
                     + abs(event.pos[1] - self._right_press[1]))
            if self._right_dragged or moved > 5:
                if not self._right_dragged:
                    self._right_dragged = True
                    self.pan_start = self._right_press
                dx = event.pos[0] - self.pan_start[0]
                dy = event.pos[1] - self.pan_start[1]
                self.renderer.camera.pan(dx, dy)
                self.pan_start = event.pos
                self._sync_camera_targets()
                return

        if self.panning:
            dx = event.pos[0] - self.pan_start[0]
            dy = event.pos[1] - self.pan_start[1]
            self.renderer.camera.pan(dx, dy)
            self.pan_start = event.pos
            # Direct manipulation wins: adopt the new position as the target so
            # the easing does not pull the view back where it was heading.
            self._sync_camera_targets()
        elif self.editor.drag is not None:
            self.editor.update_drag(self._world(event.pos))

        self.editor.set_hover(self._state_at(event.pos))

    def _handle_mouse_wheel(self, event: pygame.event.Event) -> None:
        """Zoom, or bend the transition being drawn.

        Help-panel scrolling is the UI's job; if it handled the event this is
        never reached. Both used to run, so scrolling also zoomed underneath.
        """
        if self.editor.pending_source is None:
            zoom_factor = 1.1 if event.y > 0 else 0.9
            self.renderer.camera.zoom_at(pygame.mouse.get_pos(), zoom_factor)
            self._sync_camera_targets()
        else:
            self.editor.bend_pending(event.y * 10)

    def _world(self, screen_point: Tuple[int, int]) -> Tuple[float, float]:
        return self.renderer.camera.screen_to_world(screen_point)

    def _state_at(self, screen_point: Tuple[int, int]) -> Optional[str]:
        return self.editor.state_at(self._world(screen_point),
                                    default_state_radius())

    def _handle_left_click(self, pos: Tuple[int, int], shift: bool = False) -> None:
        """
        Handle a left click on the canvas.

        Exactly one of {complete a transition, start a transition, select and
        drag, deselect} happens. Shift comes from the event's modifier state
        rather than being polled every frame, which is what used to make
        shift+click both start a transition *and* drag the source state.
        """
        clicked = self._state_at(pos)

        if self.editor.pending_source is not None:
            if clicked:
                self._complete_transition(clicked)
            else:
                self.editor.cancel_transition()
                self._show_message("Transition cancelled")
        elif shift and clicked:
            self.editor.select(clicked)
            self.editor.begin_transition(clicked)
            self._show_message(
                f"Drawing '{self.ui_manager.selected_symbol}' from {clicked}")
        elif clicked:
            self.editor.select(clicked)
            self.editor.begin_drag(clicked, self._world(pos))
        else:
            self.editor.select(None)

    def _handle_right_click(self, pos: Tuple[int, int]) -> None:
        clicked = self._state_at(pos)
        if clicked:
            self._show_state_context_menu(pos, clicked)
            return

        # An edge, if the click landed near enough to one of the drawn curves.
        world = self._world(pos)
        positions = self.editor.positions()
        paths = self._edge_paths(positions)
        edge = geometry.nearest_edge(world, paths,
                                     within=14.0 / max(0.4, self.renderer.camera.zoom))
        if edge is not None:
            self._show_edge_context_menu(pos, edge)
        else:
            self._show_general_context_menu(pos)

    def _show_edge_context_menu(self, pos: Tuple[int, int],
                                edge: Tuple[str, str]) -> None:
        """A menu for one drawn edge: remove any of its symbols, or straighten.

        Transitions used to be uneditable from the canvas at all -- the only
        way to remove one was to delete a state.
        """
        source, target = edge
        symbols = sorted(self.editor.automaton.grouped_transitions().get(edge, ()))
        items: List[Tuple[Any, ...]] = []
        for symbol in symbols:
            # The symbol is one character, so it travels first in the payload
            # and the rest is unambiguously the state id.
            items.append((f"Remove '{symbol}'", f"unedge:{symbol}{source}"))
        # A self-loop stores an arc that no renderer honours, so offering to
        # straighten one promises a change nothing can show.
        if source != target and self.editor.layout.arc_of(source, target):
            items.append(("---", ""))
            # The id's length travels with the payload. State ids are opaque,
            # so any character picked as a separator can occur inside one, and
            # splitting on it would quietly straighten a different edge.
            items.append(("Straighten",
                          f"straighten:{len(source)}:{source}{target}"))
        if items:
            self.ui_manager.show_context_menu(pos, items)

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def _handle_key_down(self, event: pygame.event.Event) -> None:
        """Handle key down events the UI did not consume."""
        # Chords first: a plain 'z' must not undo, and Ctrl+Z must not fall
        # through to any letter shortcut.
        if event.mod & pygame.KMOD_CTRL:
            if event.key == pygame.K_z and (event.mod & pygame.KMOD_SHIFT):
                self._redo()
            elif event.key == pygame.K_z:
                self._undo()
            elif event.key == pygame.K_y:
                self._redo()
            return

        if event.key == pygame.K_SPACE:
            self._add_state_at_center()
        elif event.key == pygame.K_DELETE:
            self._delete_selected_state()
        elif event.key == pygame.K_q:
            self._toggle_accept_state()
        elif event.key == pygame.K_w:
            self._make_trap(self.editor.selection)
        elif event.key == pygame.K_r:
            self._fit_to_content()
        elif event.key == pygame.K_n:
            self._next_execution_step()
        elif event.key == pygame.K_p:
            self._previous_execution_step()
        elif event.key == pygame.K_ESCAPE:
            if self.editor.pending_source is not None:
                self.editor.cancel_transition()
                self._show_message("Transition cancelled")
            else:
                self._stop_execution()
        elif event.key == pygame.K_TAB and self.execution_active:
            self._toggle_animation()
        elif event.unicode and event.unicode in self.editor.automaton.alphabet:
            self.ui_manager.selected_symbol = event.unicode
            self._show_message(f"Symbol '{event.unicode}' selected")

    # ------------------------------------------------------------------
    # UI actions
    # ------------------------------------------------------------------

    def _process_ui_actions(self, actions: Dict[str, Any]) -> None:
        for action, value in actions.items():
            if action == 'test_string':
                self._test_string(value)
            elif action == 'save_automaton':
                self._save_automaton()
            elif action == 'load_automaton':
                self._load_automaton()
            elif action == 'show_message':
                self._show_message(value)
            elif action == 'context_menu_action':
                self._handle_context_menu_action(value)
            elif action == 'save_to_path':
                self._save_to_path(value)
            elif action == 'load_to_path':
                self._load_from_path(value)
            elif action in ('file_prompt_cancel', 'confirm_cancel'):
                self._show_message("Cancelled")
            elif action == 'confirmed':
                self._handle_confirmed(value)
            elif action == 'toggle_theme':
                self._toggle_theme()
            elif action == 'complete_automaton':
                self._complete_automaton()
            elif action == 'focus_states':
                self._focus_states(value)
            elif action == 'rename_state':
                self._rename_state(*value)
            elif action == 'symbol_add':
                self._add_symbol(value)
            elif action == 'symbol_added':
                self._add_symbol(value)
            elif action == 'symbol_add_error':
                self._show_message(f"Error: {value}")
            elif action == 'symbol_dialog_cancel':
                self._show_message("Cancelled")

    def _handle_confirmed(self, intent: str) -> None:
        if intent == 'quit_after_confirm':
            self.running = False
        elif intent == 'load_after_confirm':
            self.ui_manager.show_file_prompt(
                'load', self.editor.path or self.DEFAULT_FILENAME)

    def _show_message(self, text: str) -> None:
        self.message_text = text
        self.message_timer = pygame.time.get_ticks()

    def _update_message(self) -> None:
        if (self.message_text
                and pygame.time.get_ticks() - self.message_timer > self.message_duration):
            self.message_text = ""

    def _toggle_theme(self) -> None:
        name = self.theme.toggle()
        self._show_message(f"{name.capitalize()} theme")

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def _after_edit(self) -> None:
        """Refresh anything that depends on the document having changed."""
        self.ui_manager.sync_symbols_with(self.editor.automaton)
        self._update_caption()

    def _add_state_at_center(self) -> None:
        centre = self._world((self.screen.get_width() // 2,
                              self.screen.get_height() // 2))
        state = self.editor.add_state(centre)
        self.editor.select(state)
        self._after_edit()
        self._show_message(f"Added {state}")

    def _delete_selected_state(self) -> None:
        state = self.editor.selection
        if state and self.editor.remove_state(state):
            self._after_edit()
            self._show_message(f"Deleted {state}")

    def _toggle_accept_state(self) -> None:
        state = self.editor.selection
        if state is None:
            return
        accepting = self.editor.toggle_accept(state)
        self._after_edit()
        self._show_message(
            f"{state} is {'now accepting' if accepting else 'no longer accepting'}")

    def _make_trap(self, state: Optional[str]) -> None:
        """Turn a state into a real trap: self-loop on every symbol.

        There used to be a "dead end" flag. It changed what the simulator
        accepted without changing any transition, so the tool computed a
        different language than the diagram showed. Trap-ness is derived now --
        which left a menu item writing a value nothing read. This does the thing
        the flag pretended to mean, so the state renders as a trap because it
        genuinely is one.
        """
        if state is None:
            return
        if not self.editor.automaton.alphabet:
            self._show_message("Add a symbol first: there is nothing to loop on")
            return

        ok, replaced = self.editor.make_trap(state)
        if not ok:
            return
        self._after_edit()
        alphabet = ", ".join(sorted(self.editor.automaton.alphabet))
        detail = (f", replacing {replaced} transition{'s' if replaced != 1 else ''}"
                  if replaced else "")
        self._show_message(f"{state} now loops on {alphabet}{detail}")

    def _rename_state(self, state: str, label: str) -> None:
        if not self.editor.rename(state, label):
            return
        self._after_edit()
        shown = label.strip() or state
        self._show_message(f"{state} is now labelled '{shown}'")

    def _undo(self) -> None:
        """Step the document back one edit, saying which one."""
        action = self.editor.undo()
        if action is None:
            self._show_message("Nothing to undo")
            return
        self._after_edit()
        self._show_message(f"Undid {action}")

    def _redo(self) -> None:
        action = self.editor.redo()
        if action is None:
            self._show_message("Nothing to redo")
            return
        self._after_edit()
        self._show_message(f"Redid {action}")

    def _complete_automaton(self) -> None:
        """One click from "your machine is incomplete" to a total automaton.

        Adds a trap state and routes every undefined (state, symbol) pair to
        it. The language is unchanged -- previously-undefined runs now die in
        the trap instead of halting -- which is precisely the lesson the
        diagnostics panel is teaching when it flags incompleteness.
        """
        before = len(fsa.missing_transitions(self.editor.automaton))
        document, trap = self.editor.document.complete()
        if trap is None:
            self._show_message("Already complete")
            return
        self.editor.apply(document)
        self._after_edit()
        self.node_settle.set(trap, 1.0, duration=self.theme.motion.quick,
                             easing=ease_out_back)
        self._show_message(
            f"Added {trap} and routed {before} missing "
            f"transition{'s' if before != 1 else ''} to it")

    def _focus_states(self, states: List[str]) -> None:
        """Glide the camera to the states a diagnostic names."""
        positions = [self.editor.position_of(s) for s in states
                     if s in self.editor.automaton.states]
        if not positions:
            return
        pad = default_state_radius() * 4
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zoom, offset = self.renderer.fit_to_bounds(
            (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad))
        self.cam_zoom.set(zoom, duration=self.theme.motion.slow, easing=ease_in_out)
        self.cam_offset.set(offset, duration=self.theme.motion.slow,
                            easing=ease_in_out)
        for state in states:
            self.node_settle.set(state, 1.0, duration=self.theme.motion.quick,
                                 easing=ease_out_back)

    def _add_symbol(self, symbol: str) -> None:
        if self.editor.add_symbol(symbol):
            self.ui_manager.selected_symbol = symbol
            self._after_edit()
            self._show_message(f"Added symbol '{symbol}'")
        else:
            self._show_message(f"Cannot add '{symbol}'")

    def _complete_transition(self, target: str) -> None:
        source = self.editor.pending_source
        if source is None:
            return
        symbol = self.ui_manager.selected_symbol
        arc = self.editor.pending_arc
        self.editor.cancel_transition()

        if symbol not in self.editor.automaton.alphabet:
            self._show_message(f"'{symbol}' is not in the alphabet")
            return

        replaced = self.editor.automaton.target(source, symbol)
        if not self.editor.add_transition(source, symbol, target, arc):
            self._show_message(f"Could not add transition from {source}")
            return
        self._after_edit()
        if replaced is not None and replaced != target:
            self._show_message(
                f"{source} --{symbol}--> {target}, replacing the edge to {replaced}")
        else:
            self._show_message(f"{source} --{symbol}--> {target}")

    # ------------------------------------------------------------------
    # Context menus
    # ------------------------------------------------------------------

    def _show_state_context_menu(self, pos: Tuple[int, int], state: str) -> None:
        # Toggles carry their current value, so the menu says what the state
        # already is instead of making the user guess and check.
        automaton = self.editor.automaton
        self.ui_manager.show_context_menu(pos, [
            ("Accepting", f"toggle_accept:{state}", state in automaton.accept),
            ("Initial state", f"set_initial:{state}", state == automaton.initial),
            ("---", ""),
            ("Rename...", f"rename_prompt:{state}"),
            ("Make a trap", f"make_trap:{state}"),
            ("---", ""),
            ("Delete state", f"delete_state:{state}"),
        ])

    def _show_general_context_menu(self, pos: Tuple[int, int]) -> None:
        world = self._world(pos)
        self.ui_manager.show_context_menu(pos, [
            ("Add state here", f"add_state:{world[0]},{world[1]}"),
            ("---", ""),
            ("Fit to content", "fit_view"),
        ])

    def _handle_context_menu_action(self, action: str) -> None:
        # partition rather than split, so a state id containing a colon cannot
        # truncate the payload.
        verb, _, payload = action.partition(":")

        if verb == "toggle_accept":
            if payload in self.editor.automaton.states:
                accepting = self.editor.toggle_accept(payload)
                self._after_edit()
                self._show_message(
                    f"{payload} is "
                    f"{'now accepting' if accepting else 'no longer accepting'}")
        elif verb == "make_trap":
            self._make_trap(payload)
        elif verb == "set_initial":
            if payload in self.editor.automaton.states:
                self.editor.set_initial(payload)
                self._after_edit()
                self._show_message(f"{payload} is now the initial state")
        elif verb == "delete_state":
            if self.editor.remove_state(payload):
                self._after_edit()
                self._show_message(f"Deleted {payload}")
        elif verb == "add_state":
            # The user picked this spot, so honour it unless it would overlap.
            x, _, y = payload.partition(",")
            state = self.editor.add_state((float(x), float(y)),
                                          minimum_gap=fsa.document.OVERLAP_GAP)
            self.editor.select(state)
            self._after_edit()
            self._show_message(f"Added {state}")
        elif verb == "rename_prompt":
            if payload in self.editor.automaton.states:
                self.ui_manager.show_rename_prompt(
                    payload, self.editor.automaton.label_of(payload))
        elif verb == "unedge":
            symbol, source = payload[0], payload[1:]
            target = self.editor.automaton.target(source, symbol)
            self.editor.remove_transition(source, symbol)
            self._after_edit()
            self._show_message(f"Removed {source} --{symbol}--> {target}")
        elif verb == "straighten":
            count, _, rest = payload.partition(":")
            source, target = rest[:int(count)], rest[int(count):]
            document = fsa.Document(
                self.editor.document.automaton,
                self.editor.document.layout.with_arc(source, target, 0.0),
                self.editor.document.next_id)
            self.editor.apply(document, action=f"straighten {source}->{target}")
            self._after_edit()
            # A two-way pair keeps an automatic bow so the two arrows stay
            # apart. Reporting "straightened" there would describe a line the
            # user can plainly see is still curved.
            twinned = (target, source) in self.editor.automaton.grouped_transitions()
            self._show_message("Manual bend cleared" if twinned
                               else "Edge straightened")
        elif verb == "fit_view":
            self._fit_to_content()

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    def _resolve_path(self, filename: str) -> str:
        """
        Resolve a user-supplied filename against the project directory.

        Relative paths used to resolve against the process working directory,
        so a file saved from one launch could be unreachable from the next.
        """
        filename = filename.strip() or self.DEFAULT_FILENAME
        if not os.path.splitext(filename)[1]:
            filename += ".json"
        if os.path.isabs(filename):
            return filename
        return os.path.normpath(os.path.join(PROJECT_DIR, filename))

    def _save_automaton(self) -> None:
        self.ui_manager.show_file_prompt(
            'save', self.editor.path or self.DEFAULT_FILENAME)

    def _load_automaton(self) -> None:
        if self.editor.dirty:
            self.ui_manager.show_confirm("Discard unsaved changes and load?",
                                         'load_after_confirm')
            return
        self.ui_manager.show_file_prompt(
            'load', self.editor.path or self.DEFAULT_FILENAME)

    def _save_to_path(self, filename: str) -> None:
        path = self._resolve_path(filename)
        ok, error = serialize.save_or_error(self.editor.document, path)
        if ok:
            self.editor.path = os.path.relpath(path, PROJECT_DIR)
            self.editor.dirty = False
            self._update_caption()
            self._show_message(f"Saved to {self.editor.path}")
        else:
            self._show_message(f"Save failed: {error}")

    def _load_from_path(self, filename: str) -> None:
        path = self._resolve_path(filename)
        document, error = serialize.load_or_error(path)
        if document is None:
            self._show_message(f"Load failed: {error}")
            return

        # A different machine: nothing about the previous one survives.
        self._stop_execution()
        self.editor.replace(document, os.path.relpath(path, PROJECT_DIR))
        self.ui_manager.test_result = ""
        self.ui_manager.test_verdict = ""
        self.ui_manager.sync_symbols_with(self.editor.automaton)
        for track in (self.node_selected, self.node_hover, self.node_settle,
                      self.node_active, self.edge_active):
            track.clear()

        self._update_caption()
        self._show_message(f"Loaded {self.editor.path}")

    def _update_caption(self) -> None:
        name = self.editor.path or "untitled"
        marker = "*" if self.editor.dirty else ""
        pygame.display.set_caption(f"{name}{marker} - Finite Automata Simulator")

    def _request_quit(self) -> None:
        if self.editor.dirty:
            self.ui_manager.show_confirm("Quit without saving?", 'quit_after_confirm')
        else:
            self.running = False

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _test_string(self, test_string: str) -> None:
        """Run a string and start the visualisation.

        The empty string is a legal and pedagogically important input. The old
        version refused it outright.
        """
        result = fsa.run(self.editor.automaton, test_string)
        self.run_result = result

        self.ui_manager.test_result = result.explain()
        self.ui_manager.test_verdict = result.verdict.value

        self.execution_active = True
        self.execution_string = test_string
        self.execution_path = list(result.path)
        self.execution_step = 0
        self.traversing_step = None
        self.token_travel.jump_to(0.0)

        if self.execution_path:
            self.node_settle.set(self.execution_path[0], 1.0,
                                 duration=self.theme.motion.quick,
                                 easing=ease_out_back)
            self.node_settle.set(self.execution_path[0], 0.0,
                                 duration=self.theme.motion.normal)

    def _goto_execution_step(self, index: int, animate: bool = True) -> None:
        """Move the visualisation to a position in the run.

        Animates the token along the edge connecting the two positions, in
        whichever direction it is travelling, so stepping backwards reads as the
        machine reversing rather than teleporting.
        """
        if not self.execution_active or not self.execution_path:
            return

        index = max(0, min(len(self.execution_path) - 1, index))
        if index == self.execution_step:
            return

        forward = index > self.execution_step
        step_index = self.execution_step if forward else index
        self.execution_step = index

        steps = self.run_result.steps if self.run_result else ()
        if animate and 0 <= step_index < len(steps):
            self.traversing_step = step_index
            self.token_travel.jump_to(0.0 if forward else 1.0)
            self.token_travel.set(1.0 if forward else 0.0,
                                  duration=self.theme.motion.step)
            step = steps[step_index]
            self.edge_active.set(f"{step.source}|{step.target}", 1.0,
                                 duration=self.theme.motion.instant)
        else:
            self.traversing_step = None

        self.node_settle.set(self.execution_path[index], 1.0,
                             duration=self.theme.motion.quick,
                             easing=ease_out_back)

    def _next_execution_step(self) -> None:
        if self.execution_active and self.execution_step < len(self.execution_path) - 1:
            self._goto_execution_step(self.execution_step + 1)

    def _previous_execution_step(self) -> None:
        if self.execution_active and self.execution_step > 0:
            self._goto_execution_step(self.execution_step - 1)

    def _stop_execution(self) -> None:
        self.execution_active = False
        self.execution_step = 0
        self.execution_string = ""
        self.execution_path = []
        self.run_result = None
        self.traversing_step = None
        self.animation_active = False
        self.animation_auto_advance = False
        self.node_active.clear()
        self.edge_active.clear()
        self.token_travel.jump_to(0.0)

    def _toggle_animation(self) -> None:
        if not self.execution_active:
            return
        self.animation_active = not self.animation_active
        self.animation_auto_advance = self.animation_active
        self.animation_timer = pygame.time.get_ticks()
        self._show_message("Playing" if self.animation_active else "Paused")

    def _advance_playback(self) -> None:
        if not (self.animation_active and self.animation_auto_advance
                and self.execution_active):
            return

        now = pygame.time.get_ticks()
        interval = max(self.ui_manager.animation_speed,
                       self.theme.motion.step + 60)
        if now - self.animation_timer < interval:
            return

        if self.execution_step < len(self.execution_path) - 1:
            self._goto_execution_step(self.execution_step + 1)
            self.animation_timer = now
        else:
            self.animation_auto_advance = False
            self.animation_active = False
            self._show_message("Playback finished")

    def _current_execution_state(self) -> Optional[str]:
        if not self.execution_active or not self.execution_path:
            return None
        return self.execution_path[min(self.execution_step,
                                       len(self.execution_path) - 1)]

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _fit_to_content(self) -> None:
        """Frame the whole automaton, easing rather than snapping.

        `camera.reset()` pinned the world origin to the top-left corner and
        could leave the graph entirely off screen -- a "reset view" that lost
        your work rather than finding it.
        """
        bounds = self.editor.layout.bounds(default_state_radius())
        if bounds is None:
            self.cam_zoom.set(1.0, duration=self.theme.motion.slow)
            self.cam_offset.set((0.0, 0.0), duration=self.theme.motion.slow)
            self._show_message("View reset")
            return

        zoom, offset = self.renderer.fit_to_bounds(bounds)
        self.cam_zoom.set(zoom, duration=self.theme.motion.slow, easing=ease_in_out)
        self.cam_offset.set(offset, duration=self.theme.motion.slow,
                            easing=ease_in_out)
        self._show_message("Fitted to content")

    def _sync_camera_targets(self) -> None:
        """Adopt the camera's current values as the animation's targets."""
        self.cam_zoom.jump_to(self.renderer.camera.zoom)
        self.cam_offset.jump_to((self.renderer.camera.offset_x,
                                 self.renderer.camera.offset_y))

    def _update_camera(self, dt: float) -> None:
        self.cam_zoom.update(dt)
        self.cam_offset.update(dt)
        if not (self.cam_zoom.is_settled and self.cam_offset.is_settled):
            self.renderer.camera.zoom = self.cam_zoom.value
            self.renderer.camera.offset_x = self.cam_offset.value[0]
            self.renderer.camera.offset_y = self.cam_offset.value[1]

    # ------------------------------------------------------------------
    # Frame update
    # ------------------------------------------------------------------

    def _update(self, dt: float) -> None:
        self.ui_manager.update(dt)
        self._update_message()
        self._update_camera(dt)
        self._update_animation_targets()
        self._advance_playback()

        for track in (self.node_active, self.node_selected, self.node_hover,
                      self.node_settle, self.edge_active):
            track.update(dt)
        self.token_travel.update(dt)

    def _update_animation_targets(self) -> None:
        """Point every animated value where it should be, then let it travel."""
        states = self.editor.automaton.states
        for track in (self.node_active, self.node_selected, self.node_hover,
                      self.node_settle):
            track.drop_missing(states)

        current = self._current_execution_state()
        for state in states:
            self.node_selected.set(state, 1.0 if state == self.editor.selection else 0.0)
            self.node_hover.set(state, 1.0 if state == self.editor.hover else 0.0)
            self.node_settle.set(state, 0.0, duration=self.theme.motion.normal)
            self.node_active.set(state, 1.0 if state == current else 0.0)

        if self.token_travel.is_settled:
            # The token exists only while travelling; parked on a node's rim it
            # covered the label and contradicted the glow that already marks
            # the current state.
            self.traversing_step = None
            for edge in self.editor.automaton.grouped_transitions():
                self.edge_active.set(f"{edge[0]}|{edge[1]}", 0.0,
                                     duration=self.theme.motion.normal)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def _symbol_index(self, symbol: str) -> int:
        """Position in the sorted alphabet, for edge colouring.

        Keyed on position rather than the literal characters 'a' and 'b', which
        rendered the shipped {0,1} example entirely in one colour.
        """
        alphabet = sorted(self.editor.automaton.alphabet)
        return alphabet.index(symbol) if symbol in alphabet else len(alphabet)

    def _loop_angles(self, positions: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Where each state's self-loop should point.

        Away from the average direction of its other edges, so a loop does not
        sit on top of an arrow arriving at the same node.
        """
        neighbours: Dict[str, List[Tuple[float, float]]] = {}
        for (source, target) in self.editor.automaton.grouped_transitions():
            if source == target:
                continue
            for a, b in ((source, target), (target, source)):
                if a in positions and b in positions:
                    neighbours.setdefault(a, []).append(positions[b])

        # The start marker enters the initial state from the left; treating it
        # as a phantom neighbour keeps that state's self-loop from being drawn
        # straight through the arrow.
        initial = self.editor.automaton.initial
        if initial in positions:
            point = positions[initial]
            neighbours.setdefault(initial, []).append((point[0] - 220.0, point[1]))

        return {
            state: geometry.quietest_direction(point, neighbours.get(state, []))
            for state, point in positions.items()
        }

    def _edge_paths(self, positions: Dict[str, Tuple[float, float]],
                    loop_angles: Optional[Dict[str, float]] = None):
        """The drawn path of every transition group, in world coordinates.

        Takes the loop angles rather than reading a field another method
        happened to fill in: relying on that ordering meant calling this alone
        raised KeyError.
        """
        if loop_angles is None:
            loop_angles = self._loop_angles(positions)
        radius = default_state_radius()
        automaton = self.editor.automaton
        layout = self.editor.layout
        grouped = automaton.grouped_transitions()
        paths = {}

        for (source, target) in grouped:
            if source not in positions or target not in positions:
                continue

            if source == target:
                paths[(source, target)] = geometry.self_loop_path(
                    positions[source], radius,
                    angle=loop_angles[source])
                continue

            arc = layout.arc_of(source, target)
            if not arc:
                arc = geometry.auto_arc(source, target,
                                        (target, source) in grouped)
            paths[(source, target)] = geometry.edge_path(
                positions[source], positions[target], radius, radius, arc)

        return paths

    def _build_scene(self) -> Scene:
        """Describe this frame as geometry, with no reference to pixels."""
        automaton = self.editor.automaton
        dead, unreachable, accepts_anything = self.editor.analysis()
        radius = default_state_radius()
        positions = self.editor.positions()

        self._loop_angle_cache = self._loop_angles(positions)
        scene = Scene()

        self.ui_manager.legend_dead = False
        self.ui_manager.legend_unreachable = False
        self.ui_manager.warn_no_accepting = not accepts_anything

        edge_paths = self._edge_paths(positions, self._loop_angle_cache)
        grouped = automaton.grouped_transitions()
        for edge, path in edge_paths.items():
            symbols = sorted(grouped[edge])
            label_at = None
            if edge[0] == edge[1]:
                label_at = geometry.self_loop_label_anchor(
                    positions[edge[0]], radius, self._loop_angle_cache[edge[0]])
            scene.edges.append(EdgeVisual(
                key=edge,
                path=path,
                label=", ".join(symbols),
                label_at=label_at,
                color_index=self._symbol_index(symbols[0]) if symbols else 0,
                active=self.edge_active.get(f"{edge[0]}|{edge[1]}"),
            ))

        for state in automaton.states:
            # Unreachable wins over dead. A state no word can enter cannot trap
            # anything, so "you can never get here" is the more useful fact.
            if state in unreachable:
                kind = NodeKind.UNREACHABLE
            elif state in dead:
                kind = NodeKind.DEAD
            else:
                kind = NodeKind.NORMAL

            self.ui_manager.legend_dead |= kind is NodeKind.DEAD
            self.ui_manager.legend_unreachable |= kind is NodeKind.UNREACHABLE

            scene.nodes.append(NodeVisual(
                id=state,
                position=positions[state],
                radius=radius,
                label=automaton.label_of(state),
                kind=kind,
                is_accept=state in automaton.accept,
                selected=self.node_selected.get(state),
                hover=self.node_hover.get(state),
                active=self.node_active.get(state),
                settle=self.node_settle.get(state),
            ))

        if automaton.initial and automaton.initial in positions:
            scene.start_marker = StartMarker(
                geometry.start_marker_path(positions[automaton.initial], radius))

        pending = self.editor.pending_source
        if pending is not None and pending in positions:
            world_mouse = self._world(pygame.mouse.get_pos())
            scene.ghost_edge = GhostEdge(
                path=geometry.edge_path(positions[pending], world_mouse,
                                        radius, 0.0, self.editor.pending_arc),
                label=self.ui_manager.selected_symbol,
                valid=self.ui_manager.selected_symbol in automaton.alphabet,
            )

        scene.token = self._build_token(edge_paths)
        return scene

    def _build_token(self, edge_paths) -> Optional[TokenVisual]:
        """The read head, positioned along the edge it is crossing.

        This is what replaces a text label teleporting between states: the
        marker moves along the real drawn path, at constant speed in screen
        distance rather than in curve parameter.

        At rest there is no token -- the active state's glow already says where
        the machine is, and a marker parked on a node covers its label.
        """
        if not self.execution_active or not self.run_result:
            return None
        if self.traversing_step is None:
            return None

        steps = self.run_result.steps
        if self.traversing_step >= len(steps):
            return None

        step = steps[self.traversing_step]
        path = edge_paths.get((step.source, step.target))
        if not path:
            return None

        travel = self.token_travel.value
        trail_start = max(0.0, travel - 0.22)
        trail = [geometry.point_at(path, trail_start + (travel - trail_start) * i / 6)
                 for i in range(7)]
        return TokenVisual(position=geometry.point_at(path, travel),
                           radius=7.0, trail=trail, intensity=1.0)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        """Compose one frame.

        Deliberately short: clear, hand a scene to the renderer, let the UI draw
        itself. Every colour and geometry decision lives elsewhere.
        """
        self.renderer.clear()
        self.renderer.draw_scene(self._build_scene())

        # The diagnostics panel reads the editor's cached analysis; feeding it
        # here keeps the UI a consumer of facts rather than a computer of them.
        self.ui_manager.diagnostics = self.editor.defects()

        self.ui_manager.draw(self.editor.automaton, self.ui_manager.test_result,
                             self.animation_active, self.execution_active)
        self.ui_manager.draw_execution_status(
            self.execution_active, self.execution_step,
            self.execution_string, self.execution_path, self.run_result)
        self.ui_manager.draw_legend(self.editor.automaton)
        # Called even when inactive: the strip animates itself out.
        self.ui_manager.draw_string_visualization(
            self.execution_string, self.execution_step, self.run_result)

        # Help, menus and modals paint over every panel; the toast over
        # everything. When overlays lived inside draw(), the run panel and
        # tape strip painted straight across an open Save dialog.
        self.ui_manager.draw_overlays()

        if self.message_text:
            self._draw_message()

        pygame.display.flip()

    def _draw_message(self) -> None:
        """The transient toast, bottom right, fading over its last third."""
        from rendering import primitives

        palette = self.theme.palette
        elapsed = pygame.time.get_ticks() - self.message_timer
        remaining = max(0, self.message_duration - elapsed)
        fade = min(1.0, remaining / (self.message_duration * 0.34))

        surface = self.fonts.ui("body").render(self.message_text, True, palette.text)
        pad_x, pad_y = self.theme.space.md, self.theme.space.sm
        rect = pygame.Rect(0, 0, surface.get_width() + pad_x * 2,
                           surface.get_height() + pad_y * 2)
        rect.bottomright = (self.screen.get_width() - self.theme.space.lg,
                            self.screen.get_height() - self.theme.space.lg)

        primitives.translucent_panel(
            self.screen, rect, (*palette.panel_raised, int(238 * fade)),
            radius=self.theme.radius.md,
            border=(*palette.border, int(255 * fade)))
        surface.set_alpha(int(255 * fade))
        self.screen.blit(surface, surface.get_rect(center=rect.center))


if __name__ == "__main__":
    AutomatonSimulator().run()
