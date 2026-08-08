"""
Main application file for the Finite Automata Simulator.

This is the entry point that coordinates all components and handles
the main game loop, event processing, and application state.
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
from core.dfa import DFA
from core.state import StateType
from rendering import geometry
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


class AutomatonSimulator:
    """
    Main application class that coordinates all components.
    
    This class manages the main game loop, event handling, and
    coordinates between the DFA, UI, rendering, and camera systems.
    """
    
    DEFAULT_FILENAME = "automaton.json"

    def __init__(self):
        """Initialize the automaton simulator."""
        # Initialize Pygame
        pygame.init()
        
        # Get screen info for responsive sizing
        info = pygame.display.Info()
        screen_width = int(info.current_w * 0.75)  # 75% of screen width
        screen_height = int(info.current_h * 0.75)  # 75% of screen height
        
        # Create resizable window
        self.screen = pygame.display.set_mode(
            (screen_width, screen_height), 
            pygame.RESIZABLE
        )
        pygame.display.set_caption("Finite Automata Simulator")
        
        # Initialize components. Theme and fonts are shared, so switching
        # palettes reaches every surface at once.
        self.theme = Theme("dark")
        self.fonts = FontBook()
        self.dfa = DFA()
        self.renderer = Renderer(self.screen, self.theme, self.fonts)
        self.ui_manager = UIManager(self.screen, self.theme, self.fonts)
        
        # Application state
        self.running = True
        self.clock = pygame.time.Clock()
        self.fps = 60
        
        # Interaction state
        self.selected_state: Optional[str] = None
        self.dragging_state: Optional[str] = None
        self.drag_offset = (0, 0)
        self.creating_transition = False
        self.transition_start_state: Optional[str] = None
        self.transition_arc_offset = 0.0  # For curved arrows
        
        # Execution visualization. The run comes from the engine, so a
        # rejection carries a reason and each step names the edge it took --
        # which is what the travelling token animates along.
        self.execution_active = False
        self.execution_step = 0
        self.execution_string = ""
        self.execution_path: List[str] = []
        self.run_result: Optional[fsa.Run] = None

        # Animation system. The step interval lives on the UI manager, next to
        # the slider that sets it -- there is one owner, not two.
        self.animation_active = False
        self.animation_timer = 0
        self.animation_auto_advance = False

        # Animated view state. Each of these closes the gap to its target over
        # time; nothing in the interface snaps.
        self.node_active = Track(self.theme.motion.quick)
        self.node_selected = Track(self.theme.motion.instant)
        self.node_hover = Track(self.theme.motion.instant)
        self.node_settle = Track(self.theme.motion.normal)
        self.edge_active = Track(self.theme.motion.quick)
        self.token_travel = Animated(duration=self.theme.motion.step)
        self.traversing_step: Optional[int] = None

        # Camera easing, so zoom, reset and fit glide instead of jumping.
        self.cam_zoom = Animated(value=1.0, target=1.0,
                                 duration=self.theme.motion.normal,
                                 easing=ease_in_out)
        self.cam_offset = AnimatedPoint()
        self.cam_offset.jump_to((0.0, 0.0))

        # The engine view of the automaton, rebuilt only when the structure
        # changes. Dragging a state does not invalidate it.
        self._engine: Optional[fsa.DFA] = None
        self._dead_states: frozenset = frozenset()
        self._unreachable_states: frozenset = frozenset()

        # Message system for on-screen feedback
        self.message_text = ""
        self.message_timer = 0
        self.message_duration = 2000  # 2 seconds
        
        # Camera controls
        self.panning = False
        self.pan_start = (0, 0)

        # File state
        self.current_filename: Optional[str] = None
        self.dirty = False

        # Create initial demo automaton
        self._create_demo_automaton()
        self._update_caption()
        
    def _create_demo_automaton(self):
        """Create a simple demo automaton for testing."""
        # Add states
        q0 = self.dfa.add_state((200, 200))
        q1 = self.dfa.add_state((400, 200))
        q2 = self.dfa.add_state((300, 350))
        
        # Set state types
        self.dfa.set_state_type(q1, StateType.ACCEPT)
        self.dfa.set_state_type(q2, StateType.DEAD_END)
        
        # Add transitions
        self.dfa.add_transition(q0, q0, 'a', 0.0)
        self.dfa.add_transition(q0, q1, 'b', 0.0)
        self.dfa.add_transition(q1, q2, 'a', 0.0)
        self.dfa.add_transition(q1, q1, 'b', 0.0)
        self.dfa.add_transition(q2, q2, 'a', 0.0)
        self.dfa.add_transition(q2, q2, 'b', 0.0)
        
    def run(self):
        """Main application loop."""
        while self.running:
            dt = self.clock.tick(self.fps)
            
            # Handle events
            self._handle_events()
            
            # Update
            self._update(dt)
            
            # Render
            self._render()
            
        pygame.quit()
        sys.exit()

    def _handle_events(self):
        """
        Process all pygame events.

        Each event is offered to the UI first. If the UI consumed it -- a click
        on a widget, a keypress into a text field, a wheel event over the open
        help panel -- the canvas never sees it. Previously every event was given
        to both, so clicking the Test button also deselected the state behind
        it and scrolling the help panel also zoomed the camera.
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

    def _handle_resize(self, width: int, height: int):
        """Handle window resize events."""
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.renderer.update_screen_size(width, height)
        self.ui_manager.update_screen_size(width, height)

    def _handle_mouse_down(self, event):
        """Handle mouse button down events that the UI did not consume."""
        if event.button == 1:  # Left click
            self._handle_left_click(event.pos, self._shift_held())
        elif event.button == 2:  # Middle click
            self._start_panning(event.pos)
        elif event.button == 3:  # Right click
            self._handle_right_click(event.pos)

    @staticmethod
    def _shift_held() -> bool:
        """
        Whether a shift key is down right now.

        Mouse events in pygame carry no modifier state -- only key events have
        a `.mod` field -- so a click has to ask the keyboard directly. This is
        read once while handling the event, not sampled every frame, which is
        what made shift+click drag the state it was drawing from.
        """
        return bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)

    def _handle_mouse_up(self, event):
        """Handle mouse button up events."""
        if event.button == 1:  # Left click
            self._handle_left_release(event.pos)
        elif event.button == 2:  # Middle click
            self._stop_panning()

    def _handle_mouse_motion(self, event):
        """Handle mouse motion events."""
        if self.panning:
            self._update_panning(event.pos)
        elif self.dragging_state:
            self._update_dragging(event.pos)

        # Update hover states
        self._update_hover_states(event.pos)

    def _handle_mouse_wheel(self, event):
        """Zoom, or bend the transition being drawn.

        Help-panel scrolling is the UI's job; if it handled the event this is
        never reached. Both used to run, so scrolling the panel also zoomed the
        camera underneath it.
        """
        if not self.creating_transition:
            zoom_factor = 1.1 if event.y > 0 else 0.9
            self.renderer.camera.zoom_at(pygame.mouse.get_pos(), zoom_factor)
            self._sync_camera_targets()
        else:
            # Adjust transition arc when creating transition
            self.transition_arc_offset += event.y * 10
            self.transition_arc_offset = max(-100, min(100, self.transition_arc_offset))

    def _handle_key_down(self, event):
        """Handle key down events that the UI did not consume."""

        # Handle system keys first (these take priority)
        if event.key == pygame.K_SPACE:
            self._add_state_at_center()
            return
        elif event.key == pygame.K_DELETE:
            self._delete_selected_state()
            return
        elif event.key == pygame.K_q:
            self._toggle_accept_state()
            return
        elif event.key == pygame.K_w:
            self._toggle_dead_end_state()
            return
        elif event.key == pygame.K_r:
            self.renderer.camera.reset()
            return
        elif event.key == pygame.K_n:
            self._next_execution_step()
            return
        elif event.key == pygame.K_p:
            self._previous_execution_step()
            return
        elif event.key == pygame.K_ESCAPE:
            self._stop_execution()
            return
        elif event.key == pygame.K_TAB and self.execution_active:
            # Toggle animation mode (changed from 'a' to TAB to avoid conflict)
            self._toggle_animation()
            return

        # Only check for symbol keys if it's not a system key
        symbol = event.unicode
        if symbol and symbol in self.ui_manager.available_symbols:
            self.ui_manager.selected_symbol = symbol
            self._show_message(f"Selected symbol: {symbol}")

    def _handle_key_up(self, event):
        """Handle key up events."""
        pass  # Currently no key up handling needed

    def _process_ui_actions(self, actions: Dict[str, Any]):
        """Process actions returned by the UI manager."""
        for action, value in actions.items():
            if action == 'test_string':
                self._test_string(value)
            elif action == 'save_automaton':
                self._save_automaton()
            elif action == 'load_automaton':
                self._load_automaton()
            elif action == 'symbol_selected':
                # Symbol already set by UI manager
                pass
            elif action == 'show_message':
                self._show_message(value)
            elif action == 'add_symbol':
                self._show_add_symbol_dialog()
            elif action == 'context_menu_action':
                self._handle_context_menu_action(value)
            elif action == 'save_to_path':
                self._save_to_path(value)
            elif action == 'load_to_path':
                self._load_from_path(value)
            elif action == 'file_prompt_cancel':
                self._show_message("Cancelled")
            elif action == 'confirmed':
                self._handle_confirmed(value)
            elif action == 'confirm_cancel':
                self._show_message("Cancelled")
            elif action == 'toggle_theme':
                self._toggle_theme()
            elif action == 'symbol_added':
                self._show_message(f"Added symbol: {value}")
            elif action == 'symbol_add_error':
                self._show_message(f"Error: {value}")
            elif action == 'symbol_dialog_cancel':
                self._show_message("Symbol addition cancelled")
            elif action == 'symbol_add':
                if self._add_symbol(value):
                    self._show_message(f"Added symbol: {value}")
                else:
                    self._show_message(f"Error: Cannot add symbol '{value}'")

    def _handle_confirmed(self, intent: str):
        """Carry out the action a confirmation dialog was guarding."""
        if intent == 'quit_after_confirm':
            self.running = False
        elif intent == 'load_after_confirm':
            self.ui_manager.show_file_prompt(
                'load', self.current_filename or self.DEFAULT_FILENAME)

    def _show_message(self, text: str):
        """Show a temporary message on screen."""
        self.message_text = text
        self.message_timer = pygame.time.get_ticks()

    def _update_message(self):
        """Update message display timer."""
        if self.message_text and pygame.time.get_ticks() - self.message_timer > self.message_duration:
            self.message_text = ""

    def _add_symbol(self, symbol: str) -> bool:
        """Add a new symbol to the available symbols."""
        # Use the UI manager's add_symbol method which has proper validation
        return self.ui_manager.add_symbol(symbol)

    def _handle_left_click(self, pos: Tuple[int, int], shift: bool = False):
        """
        Handle a left click on the canvas.

        Exactly one of {complete a transition, start a transition, select and
        drag, deselect} happens. Shift is read from the event's modifier state
        rather than polled from the keyboard each frame, which is what used to
        make shift+click both start a transition *and* drag the source state.
        """
        world_pos = self.renderer.camera.screen_to_world(pos)
        clicked_state = self._get_state_at_position(world_pos)

        if self.creating_transition:
            if clicked_state:
                self._complete_transition(clicked_state)
            else:
                self._cancel_transition()
                self._show_message("Transition cancelled")
        elif shift and clicked_state:
            self._select_state(clicked_state)
            self._start_transition(clicked_state)
        elif clicked_state:
            self._select_state(clicked_state)
            self._start_dragging(clicked_state, pos)
        else:
            self._deselect_all()

    def _handle_right_click(self, pos: Tuple[int, int]):
        """Handle right mouse click on the canvas."""
        world_pos = self.renderer.camera.screen_to_world(pos)
        clicked_state = self._get_state_at_position(world_pos)

        if clicked_state:
            self._show_state_context_menu(pos, clicked_state)
        else:
            self._show_general_context_menu(pos)

    def _handle_left_release(self, _pos: Tuple[int, int]):
        """Handle left mouse button release."""
        if self.dragging_state:
            self._stop_dragging()

    def _start_panning(self, pos: Tuple[int, int]):
        """Start panning the camera."""
        self.panning = True
        self.pan_start = pos

    def _stop_panning(self):
        """Stop panning the camera."""
        self.panning = False

    def _update_panning(self, pos: Tuple[int, int]):
        """Update camera panning."""
        if self.panning:
            dx = pos[0] - self.pan_start[0]
            dy = pos[1] - self.pan_start[1]
            self.renderer.camera.pan(dx, dy)
            self.pan_start = pos
            # Direct manipulation wins: adopt the new position as the target so
            # the easing does not pull the view back where it was heading.
            self._sync_camera_targets()

    def _update_dragging(self, pos: Tuple[int, int]):
        """Update state dragging."""
        state = self.dfa.states.get(self.dragging_state) if self.dragging_state else None
        if state is not None:
            world_pos = self.renderer.camera.screen_to_world(pos)
            state.position[0] = world_pos[0] - self.drag_offset[0]
            state.position[1] = world_pos[1] - self.drag_offset[1]
            self._mark_dirty()

    def _update_hover_states(self, pos: Tuple[int, int]):
        """Update hover states for visual feedback."""
        world_pos = self.renderer.camera.screen_to_world(pos)

        # Clear all hover states
        for state in self.dfa.states.values():
            state.hover = False

        # Set hover for state under mouse
        hovered_state = self._get_state_at_position(world_pos)
        if hovered_state:
            self.dfa.states[hovered_state].hover = True

    def _get_state_at_position(self, world_pos: Tuple[float, float]) -> Optional[str]:
        """Get the state at the given world position."""
        for state_id, state in self.dfa.states.items():
            if state.contains_point(world_pos):
                return state_id
        return None

    def _select_state(self, state_id: str):
        """Select a state."""
        # Deselect all first
        self._deselect_all()

        # Select the clicked state
        self.selected_state = state_id
        self.dfa.states[state_id].selected = True

    def _deselect_all(self):
        """Deselect all states."""
        self.selected_state = None
        for state in self.dfa.states.values():
            state.selected = False

    def _start_dragging(self, state_id: str, screen_pos: Tuple[int, int]):
        """Start dragging a state."""
        world_pos = self.renderer.camera.screen_to_world(screen_pos)
        state = self.dfa.states[state_id]

        self.dragging_state = state_id
        self.drag_offset = (world_pos[0] - state.position[0],
                           world_pos[1] - state.position[1])
        state.being_dragged = True

    def _stop_dragging(self):
        """Stop dragging the current state."""
        if self.dragging_state:
            state = self.dfa.states.get(self.dragging_state)
            if state is not None:
                state.being_dragged = False
            self.dragging_state = None
            self.drag_offset = (0, 0)

    def _forget_state(self, state_id: str):
        """
        Drop every app-level reference to a state that no longer exists.

        The DFA owns the automaton, but the app holds its own pointers into it:
        the selection, the state being dragged, the source of a half-drawn
        transition, and the execution trace. If any of those outlive the state
        they name, the next lookup raises KeyError -- and for the transition
        pointer that lookup happens in _render, so it fires every frame and
        kills the process.

        Called from every path that removes a state, so there is one place to
        keep correct rather than one per caller.
        """
        if self.selected_state == state_id:
            self.selected_state = None

        if self.dragging_state == state_id:
            self.dragging_state = None
            self.drag_offset = (0, 0)

        if self.transition_start_state == state_id:
            self._cancel_transition()

        if state_id in self.execution_path:
            self._stop_execution()

    def _remove_state(self, state_id: str):
        """Remove a state and clear every reference to it."""
        if self.dfa.remove_state(state_id):
            self._forget_state(state_id)
            self._structural_change()
            self._show_message(f"Deleted state {state_id}")

    def _add_state_at_center(self):
        """Add a new state at the center of the current view."""
        center_screen = (self.screen.get_width() // 2, self.screen.get_height() // 2)
        center_world = self.renderer.camera.screen_to_world(center_screen)
        state_id = self.dfa.add_state(center_world)
        self._structural_change()
        self._show_message(f"Added state {state_id}")

    def _delete_selected_state(self):
        """Delete the currently selected state."""
        if self.selected_state:
            self._remove_state(self.selected_state)

    def _toggle_accept_state(self):
        """Toggle the selected state between normal and accept."""
        state = self.dfa.states.get(self.selected_state) if self.selected_state else None
        if state is not None:
            if state.state_type == StateType.ACCEPT:
                self.dfa.set_state_type(self.selected_state, StateType.NORMAL)
            else:
                self.dfa.set_state_type(self.selected_state, StateType.ACCEPT)
            self._structural_change()

    def _toggle_dead_end_state(self):
        """Toggle the selected state between normal and dead end."""
        state = self.dfa.states.get(self.selected_state) if self.selected_state else None
        if state is not None:
            if state.state_type == StateType.DEAD_END:
                self.dfa.set_state_type(self.selected_state, StateType.NORMAL)
            else:
                self.dfa.set_state_type(self.selected_state, StateType.DEAD_END)
            self._structural_change()

    # ------------------------------------------------------------------
    # The engine
    # ------------------------------------------------------------------

    def _invalidate_engine(self):
        """Mark the engine view stale after a structural change.

        Separate from the dirty flag: dragging a state changes the document but
        not the automaton, and rebuilding on every mouse-motion event would be
        wasteful.

        The derived sets are cleared too, so nothing can read a stale answer
        between here and the next rebuild.
        """
        self._engine = None
        self._dead_states = frozenset()
        self._unreachable_states = frozenset()

    def engine(self) -> fsa.DFA:
        """The automaton as the engine sees it.

        The editor still owns a mutable model; this is the bridge. Simulation
        and analysis go through here, so the app computes the language the
        transition function actually describes rather than the one the old
        dead-end flag implied.
        """
        if self._engine is None:
            automaton = fsa.DFA(alphabet=frozenset(self.dfa.alphabet))
            automaton = automaton.with_states(list(self.dfa.states))
            for source, symbol_map in self.dfa.transitions.items():
                for symbol, target in symbol_map.items():
                    automaton = automaton.with_transition(source, symbol, target)
            for state_id in self.dfa.accept_states:
                automaton = automaton.with_accept(state_id)
            initial = self.dfa.initial_state
            automaton = automaton.with_initial(
                initial if initial in self.dfa.states else None)

            self._engine = automaton
            self._unreachable_states = fsa.unreachable_states(automaton)

            # With no accepting state, *every* state is technically a trap:
            # nothing can reach acceptance because there is nothing to reach.
            # True, and useless -- it greys out the whole canvas while the user
            # is still drawing, before they have marked anything accepting.
            # The real problem in that case is "no accepting states", which
            # analysis already reports on its own.
            if automaton.accept:
                self._dead_states = fsa.dead_states(automaton)
            else:
                self._dead_states = frozenset()
        return self._engine

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _test_string(self, test_string: str):
        """Run a string through the automaton and start the visualisation.

        The empty string is a legal, and pedagogically important, input. The
        old version refused it.
        """
        result = fsa.run(self.engine(), test_string)
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

    def _goto_execution_step(self, index: int, animate: bool = True):
        """Move the visualisation to a position in the run.

        Animates the token along the edge that connects the two positions, in
        whichever direction it is travelling, so stepping backwards reads as
        the machine reversing rather than teleporting.
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

        state_id = self.execution_path[index]
        self.node_settle.set(state_id, 1.0, duration=self.theme.motion.quick,
                             easing=ease_out_back)

    def _next_execution_step(self):
        """Advance one transition."""
        if self.execution_active and self.execution_step < len(self.execution_path) - 1:
            self._goto_execution_step(self.execution_step + 1)

    def _previous_execution_step(self):
        """Go back one transition."""
        if self.execution_active and self.execution_step > 0:
            self._goto_execution_step(self.execution_step - 1)

    def _stop_execution(self):
        """Stop execution visualization."""
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

    def _toggle_animation(self):
        """Toggle animation mode for execution."""
        if self.execution_active:
            self.animation_active = not self.animation_active
            if self.animation_active:
                self.animation_auto_advance = True
                self.animation_timer = pygame.time.get_ticks()
                self._show_message("Animation mode ON - automatic stepping")
            else:
                self.animation_auto_advance = False
                self._show_message("Animation mode OFF - manual stepping")

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------

    def _resolve_path(self, filename: str) -> str:
        """
        Resolve a user-supplied filename against the project directory.

        Relative paths used to resolve against the process working directory,
        which meant a file saved from one launch could be unreachable from the
        next. Anchoring to the project directory makes save and load agree
        regardless of where python was started from.
        """
        filename = filename.strip()
        if not filename:
            filename = self.DEFAULT_FILENAME
        if not os.path.splitext(filename)[1]:
            filename += '.json'
        if os.path.isabs(filename):
            return filename
        return os.path.normpath(os.path.join(PROJECT_DIR, filename))

    def _save_automaton(self):
        """Prompt for a filename and save the current automaton."""
        self.ui_manager.show_file_prompt('save', self.current_filename or self.DEFAULT_FILENAME)

    def _load_automaton(self):
        """Prompt for a filename and load an automaton, guarding unsaved work."""
        if self.dirty:
            self.ui_manager.show_confirm(
                "Discard unsaved changes and load?", 'load_after_confirm')
            return
        self.ui_manager.show_file_prompt('load', self.current_filename or self.DEFAULT_FILENAME)

    def _save_to_path(self, filename: str):
        """Write the automaton to disk and report the outcome on screen."""
        path = self._resolve_path(filename)
        ok, error = self.dfa.save_to_file(path)
        if ok:
            self.current_filename = os.path.relpath(path, PROJECT_DIR)
            self.dirty = False
            self._update_caption()
            self._show_message(f"Saved to {self.current_filename}")
        else:
            self._show_message(f"Save failed: {error}")

    def _load_from_path(self, filename: str):
        """Read an automaton from disk and report the outcome on screen."""
        path = self._resolve_path(filename)
        ok, error = self.dfa.load_from_file(path)
        if not ok:
            self._show_message(f"Load failed: {error}")
            return

        # The loaded automaton is a different machine; nothing about the
        # previous one -- selection, drag, half-drawn edge, trace -- survives.
        self._stop_execution()
        self._cancel_transition()
        self._deselect_all()
        self.dragging_state = None
        self.drag_offset = (0, 0)
        self.ui_manager.test_result = ""
        self.ui_manager.sync_symbols_with(self.dfa)

        self._invalidate_engine()
        self.node_selected.clear()
        self.node_hover.clear()
        self.node_settle.clear()

        self.current_filename = os.path.relpath(path, PROJECT_DIR)
        self.dirty = False
        self._update_caption()
        self._show_message(f"Loaded {self.current_filename}")

    def _mark_dirty(self):
        """Record that the document has unsaved changes."""
        if not self.dirty:
            self.dirty = True
            self._update_caption()

    def _structural_change(self):
        """Record an edit that changes the automaton, not just its layout.

        Moving a state is a document change but not a structural one, so
        dragging does not force the engine to be rebuilt on every motion event.
        """
        self._mark_dirty()
        self._invalidate_engine()

    def _update_caption(self):
        """Reflect the current file and unsaved state in the window title."""
        name = self.current_filename or "untitled"
        marker = "*" if self.dirty else ""
        pygame.display.set_caption(f"{name}{marker} - Finite Automata Simulator")

    def _request_quit(self):
        """Quit, guarding unsaved work."""
        if self.dirty:
            self.ui_manager.show_confirm("Quit without saving?", 'quit_after_confirm')
        else:
            self.running = False

    def _show_add_symbol_dialog(self):
        """Show dialog to add a new symbol."""
        # The dialog is now handled by the UI manager
        pass

    def _show_state_context_menu(self, pos: Tuple[int, int], state_id: str):
        """Show context menu for a state."""
        items = [
            ("Set as Accept State", f"set_accept:{state_id}"),
            ("Set as Dead End", f"set_dead_end:{state_id}"),
            ("Set as Normal", f"set_normal:{state_id}"),
            ("---", ""),
            ("Set as Initial", f"set_initial:{state_id}"),
            ("---", ""),
            ("Delete State", f"delete_state:{state_id}")
        ]

        self.ui_manager.show_context_menu(pos, items)

    def _show_general_context_menu(self, pos: Tuple[int, int]):
        """Show general context menu."""
        world_pos = self.renderer.camera.screen_to_world(pos)

        items = [
            ("Add State Here", f"add_state:{world_pos[0]},{world_pos[1]}"),
            ("---", ""),
            ("Reset View", "reset_view")
        ]

        self.ui_manager.show_context_menu(pos, items)

    def _handle_context_menu_action(self, action: str):
        """Handle context menu actions."""
        # split(":", 1) rather than split(":") so that a state id containing a
        # colon cannot silently truncate the payload.
        verb, _, payload = action.partition(":")

        if verb == "set_accept":
            self.dfa.set_state_type(payload, StateType.ACCEPT)
            self._structural_change()
        elif verb == "set_dead_end":
            self.dfa.set_state_type(payload, StateType.DEAD_END)
            self._structural_change()
        elif verb == "set_normal":
            self.dfa.set_state_type(payload, StateType.NORMAL)
            self._structural_change()
        elif verb == "set_initial":
            if payload in self.dfa.states:
                self.dfa.initial_state = payload
                self._structural_change()
                self._show_message(f"{payload} is now the initial state")
        elif verb == "delete_state":
            # Goes through _remove_state so the app's own references to this
            # state are cleared too. Calling dfa.remove_state directly here was
            # what left selected_state/transition_start_state dangling.
            self._remove_state(payload)
        elif verb == "add_state":
            coords = payload.split(",")
            pos = (float(coords[0]), float(coords[1]))
            state_id = self.dfa.add_state(pos)
            self._structural_change()
            self._show_message(f"Added state {state_id}")
        elif verb == "reset_view":
            self._fit_to_content()

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _fit_to_content(self):
        """Frame the whole automaton, easing rather than snapping.

        `reset()` pinned the world origin to the top-left corner, which could
        leave the graph entirely off screen -- a "reset view" that lost your
        work rather than finding it.
        """
        scene = self._build_scene()
        bounds = scene.bounds()
        if bounds is None:
            self.cam_zoom.set(1.0, duration=self.theme.motion.slow)
            self.cam_offset.set((0.0, 0.0), duration=self.theme.motion.slow)
            self._show_message("View reset")
            return

        zoom, offset = self.renderer.fit_to_bounds(bounds)
        self.cam_zoom.set(zoom, duration=self.theme.motion.slow, easing=ease_in_out)
        self.cam_offset.set(offset, duration=self.theme.motion.slow, easing=ease_in_out)
        self._show_message("Fitted to content")

    def _sync_camera_targets(self):
        """Adopt the camera's current values as the animation's targets.

        Direct manipulation -- panning, wheel zoom -- writes straight to the
        camera. Without this the easing would drag it back to where the last
        animated move left it.
        """
        self.cam_zoom.jump_to(self.renderer.camera.zoom)
        self.cam_offset.jump_to((self.renderer.camera.offset_x,
                                 self.renderer.camera.offset_y))

    def _update_camera(self, dt: float):
        self.cam_zoom.update(dt)
        self.cam_offset.update(dt)
        if not (self.cam_zoom.is_settled and self.cam_offset.is_settled):
            self.renderer.camera.zoom = self.cam_zoom.value
            self.renderer.camera.offset_x = self.cam_offset.value[0]
            self.renderer.camera.offset_y = self.cam_offset.value[1]

    # ------------------------------------------------------------------
    # Frame update
    # ------------------------------------------------------------------

    def _update(self, dt: float):
        """Update application state."""
        self.ui_manager.update(dt)
        self._update_message()
        self._update_camera(dt)
        self._update_animation_targets()
        self._advance_playback()

        for track in (self.node_active, self.node_selected, self.node_hover,
                      self.node_settle, self.edge_active):
            track.update(dt)
        self.token_travel.update(dt)

    def _update_animation_targets(self):
        """Point every animated value at where it should be, then let it travel."""
        live = set(self.dfa.states)
        for track in (self.node_active, self.node_selected, self.node_hover,
                      self.node_settle):
            track.drop_missing(live)

        for state_id, state in self.dfa.states.items():
            self.node_selected.set(state_id, 1.0 if state.selected else 0.0)
            self.node_hover.set(state_id, 1.0 if state.hover else 0.0)
            self.node_settle.set(state_id, 0.0, duration=self.theme.motion.normal)

        current = self._current_execution_state()
        for state_id in self.dfa.states:
            self.node_active.set(state_id, 1.0 if state_id == current else 0.0)

        # Traversal highlight fades once the token has arrived.
        if self.token_travel.is_settled:
            for key in list(self.dfa.transition_groups):
                self.edge_active.set(f"{key[0]}|{key[1]}", 0.0,
                                     duration=self.theme.motion.normal)

    def _current_execution_state(self) -> Optional[str]:
        if not self.execution_active or not self.execution_path:
            return None
        index = min(self.execution_step, len(self.execution_path) - 1)
        return self.execution_path[index]

    def _advance_playback(self):
        """Step the run forward on a timer while playback is running."""
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

    def _start_transition(self, from_state: str):
        """Start creating a transition from the given state."""
        self.creating_transition = True
        self.transition_start_state = from_state
        self.transition_arc_offset = 0.0
        self._show_message(f"Creating transition for '{self.ui_manager.selected_symbol}' from {from_state}")

    def _complete_transition(self, to_state: str):
        """Complete the transition to the given state."""
        if self.creating_transition and self.transition_start_state:
            from_state = self.transition_start_state
            symbol = self.ui_manager.selected_symbol
            added = self.dfa.add_transition(
                from_state, to_state, symbol, self.transition_arc_offset)
            if added:
                self._structural_change()
                self._show_message(f"Added transition: {from_state} --{symbol}--> {to_state}")
            else:
                # add_transition only reports failure by returning False; without
                # this the gesture silently does nothing.
                self._show_message(f"Could not add transition from {from_state}")
            self._cancel_transition()

    def _cancel_transition(self):
        """Cancel the current transition creation."""
        self.creating_transition = False
        self.transition_start_state = None
        self.transition_arc_offset = 0.0

    # ------------------------------------------------------------------
    # Building the scene
    # ------------------------------------------------------------------

    def _symbol_index(self, symbol: str) -> int:
        """Position of a symbol in the sorted alphabet, for edge colouring.

        Keyed on position rather than on the literal characters 'a' and 'b',
        which rendered the shipped {0,1} example entirely in one colour.
        """
        alphabet = sorted(self.dfa.alphabet)
        try:
            return alphabet.index(symbol)
        except ValueError:
            return len(alphabet)

    def _loop_angles(self) -> Dict[str, float]:
        """Where each state's self-loop should point.

        Away from the average direction of that state's other edges, so a loop
        does not sit on top of an arrow arriving at the same node.
        """
        neighbours: Dict[str, List[Tuple[float, float]]] = {}
        for (source_id, target_id) in self.dfa.transition_groups:
            if source_id == target_id:
                continue
            for a, b in ((source_id, target_id), (target_id, source_id)):
                other = self.dfa.states.get(b)
                if a in self.dfa.states and other is not None:
                    neighbours.setdefault(a, []).append(tuple(other.position))

        return {
            state_id: geometry.quietest_direction(
                tuple(state.position), neighbours.get(state_id, []))
            for state_id, state in self.dfa.states.items()
        }

    def _edge_paths(self) -> Dict[Tuple[str, str], List[Tuple[float, float]]]:
        """The drawn path of every transition group, in world coordinates."""
        radius = default_state_radius()
        paths: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        self._loop_angle_cache = self._loop_angles()

        for key, group in self.dfa.transition_groups.items():
            source_id, target_id = key
            source = self.dfa.states.get(source_id)
            target = self.dfa.states.get(target_id)
            if source is None or target is None:
                continue

            if source_id == target_id:
                paths[key] = geometry.self_loop_path(
                    tuple(source.position), radius,
                    angle=self._loop_angle_cache[source_id])
                continue

            arc = float(group.get('arc_offset', 0.0))
            if abs(arc) < 0.01:
                arc = geometry.auto_arc(
                    source_id, target_id,
                    (target_id, source_id) in self.dfa.transition_groups)

            paths[key] = geometry.edge_path(
                tuple(source.position), tuple(target.position),
                radius, radius, arc)

        return paths

    def _build_scene(self) -> Scene:
        """Describe this frame as geometry, with no reference to pixels.

        Everything the renderer needs is produced here. When the editor moves
        onto the engine's immutable document, only this method changes.
        """
        self.engine()  # refreshes the derived dead/unreachable sets
        radius = default_state_radius()
        scene = Scene()
        self.ui_manager.legend_dead = False
        self.ui_manager.legend_unreachable = False

        edge_paths = self._edge_paths()
        for key, path in edge_paths.items():
            group = self.dfa.transition_groups[key]
            symbols = sorted(group['symbols'])
            label_at = None
            if key[0] == key[1]:
                state = self.dfa.states[key[0]]
                label_at = geometry.self_loop_label_anchor(
                    tuple(state.position), radius,
                    self._loop_angle_cache[key[0]])
            scene.edges.append(EdgeVisual(
                key=key,
                path=path,
                label=", ".join(symbols),
                label_at=label_at,
                color_index=self._symbol_index(symbols[0]) if symbols else 0,
                active=self.edge_active.get(f"{key[0]}|{key[1]}"),
            ))

        for state_id, state in self.dfa.states.items():
            # Unreachable wins over dead. A state no word can enter cannot
            # trap anything, so "you can never get here" is the more useful of
            # the two facts.
            if state_id in self._unreachable_states:
                kind = NodeKind.UNREACHABLE
            elif state_id in self._dead_states:
                kind = NodeKind.DEAD
            else:
                kind = NodeKind.NORMAL

            self.ui_manager.legend_dead |= kind is NodeKind.DEAD
            self.ui_manager.legend_unreachable |= kind is NodeKind.UNREACHABLE

            scene.nodes.append(NodeVisual(
                id=state_id,
                position=tuple(state.position),
                radius=radius,
                label=state_id,
                kind=kind,
                is_accept=state_id in self.dfa.accept_states,
                selected=self.node_selected.get(state_id),
                hover=self.node_hover.get(state_id),
                active=self.node_active.get(state_id),
                settle=self.node_settle.get(state_id),
            ))

        initial = self.dfa.initial_state
        if initial and initial in self.dfa.states:
            scene.start_marker = StartMarker(geometry.start_marker_path(
                tuple(self.dfa.states[initial].position), radius))

        if self.creating_transition and self.transition_start_state:
            source = self.dfa.states.get(self.transition_start_state)
            if source is None:
                # The source was deleted mid-gesture. An unguarded lookup here
                # used to raise KeyError every frame and kill the process.
                self._cancel_transition()
            else:
                world_mouse = self.renderer.camera.screen_to_world(
                    pygame.mouse.get_pos())
                scene.ghost_edge = GhostEdge(
                    path=geometry.edge_path(tuple(source.position), world_mouse,
                                            radius, 0.0,
                                            self.transition_arc_offset),
                    label=self.ui_manager.selected_symbol,
                )

        scene.token = self._build_token(edge_paths)
        return scene

    def _build_token(self, edge_paths) -> Optional[TokenVisual]:
        """The read head, positioned along the edge it is currently crossing.

        This is what replaces a text label teleporting between states: the
        marker moves continuously along the real drawn path, at a constant
        speed in screen distance rather than in curve parameter.
        """
        if not self.execution_active or not self.run_result:
            return None

        steps = self.run_result.steps
        travel = self.token_travel.value

        if self.traversing_step is not None and self.traversing_step < len(steps):
            step = steps[self.traversing_step]
            path = edge_paths.get((step.source, step.target))
            if path:
                trail_start = max(0.0, travel - 0.22)
                trail = [geometry.point_at(path, trail_start + (travel - trail_start) * i / 6)
                         for i in range(7)]
                return TokenVisual(
                    position=geometry.point_at(path, travel),
                    radius=7.0,
                    trail=trail,
                    intensity=1.0,
                )

        # At rest there is no token. The active state's glow already says where
        # the machine is, and a marker parked on the node covers its label.
        return None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self):
        """Compose one frame.

        Deliberately short: it clears, hands a scene to the renderer, and lets
        the UI draw itself. Every colour and geometry decision lives elsewhere.
        """
        self.renderer.clear()
        self.renderer.draw_scene(self._build_scene())

        self.ui_manager.draw(self.dfa, self.ui_manager.test_result,
                             self.animation_active)
        self.ui_manager.draw_execution_status(
            self.execution_active,
            self.execution_step,
            self.execution_string,
            self.execution_path,
            self.run_result,
        )
        if self.execution_active:
            self.ui_manager.draw_string_visualization(
                self.execution_string, self.execution_step, self.run_result)

        # Draw temporary message
        if self.message_text:
            self._draw_message()

        pygame.display.flip()

    def _draw_message(self):
        """Draw the transient toast, bottom right.

        Fades out over its last third rather than vanishing, and uses the
        cached font instead of constructing a new one every frame.
        """
        palette = self.theme.palette
        elapsed = pygame.time.get_ticks() - self.message_timer
        remaining = max(0, self.message_duration - elapsed)
        fade = min(1.0, remaining / (self.message_duration * 0.34))

        font = self.fonts.ui("body")
        text_surface = font.render(self.message_text, True, palette.text)

        pad_x, pad_y = self.theme.space.md, self.theme.space.sm
        rect = pygame.Rect(0, 0,
                           text_surface.get_width() + pad_x * 2,
                           text_surface.get_height() + pad_y * 2)
        rect.bottomright = (self.screen.get_width() - self.theme.space.lg,
                            self.screen.get_height() - self.theme.space.lg)

        from rendering import primitives
        primitives.translucent_panel(
            self.screen, rect,
            (*palette.panel_raised, int(238 * fade)),
            radius=self.theme.radius.md,
            border=(*palette.border, int(255 * fade)))

        text_surface.set_alpha(int(255 * fade))
        self.screen.blit(text_surface, text_surface.get_rect(center=rect.center))

    def _toggle_theme(self):
        """Switch between the dark and light palettes."""
        name = self.theme.toggle()
        self._show_message(f"{name.capitalize()} theme")


if __name__ == "__main__":
    app = AutomatonSimulator()
    app.run()
