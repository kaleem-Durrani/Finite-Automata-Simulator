"""
Main application file for the Finite Automata Simulator.

This is the entry point that coordinates all components and handles
the main game loop, event processing, and application state.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pygame

# Import our modules
from core.dfa import DFA
from core.state import StateType
from rendering.renderer import Renderer
from ui.ui_manager import UIManager

# Saved automata resolve against the project directory rather than the process
# working directory, so a file written in one session is findable in the next
# no matter where python was launched from.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


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
        
        # Initialize components
        self.dfa = DFA()
        self.renderer = Renderer(self.screen)
        self.ui_manager = UIManager(self.screen)
        
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
        
        # Execution visualization
        self.execution_active = False
        self.execution_step = 0
        self.execution_string = ""
        self.execution_path: List[str] = []

        # Animation system. The step interval lives on the UI manager, next to
        # the slider that sets it -- there is one owner, not two.
        self.animation_active = False
        self.animation_timer = 0
        self.animation_auto_advance = False

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
            self._handle_left_click(event.pos, bool(event.mod & pygame.KMOD_SHIFT))
        elif event.button == 2:  # Middle click
            self._start_panning(event.pos)
        elif event.button == 3:  # Right click
            self._handle_right_click(event.pos)

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
            self._mark_dirty()
            self._show_message(f"Deleted state {state_id}")

    def _add_state_at_center(self):
        """Add a new state at the center of the current view."""
        center_screen = (self.screen.get_width() // 2, self.screen.get_height() // 2)
        center_world = self.renderer.camera.screen_to_world(center_screen)
        state_id = self.dfa.add_state(center_world)
        self._mark_dirty()
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
            self._mark_dirty()

    def _toggle_dead_end_state(self):
        """Toggle the selected state between normal and dead end."""
        state = self.dfa.states.get(self.selected_state) if self.selected_state else None
        if state is not None:
            if state.state_type == StateType.DEAD_END:
                self.dfa.set_state_type(self.selected_state, StateType.NORMAL)
            else:
                self.dfa.set_state_type(self.selected_state, StateType.DEAD_END)
            self._mark_dirty()

    def _test_string(self, test_string: str):
        """Test a string against the automaton."""
        if not test_string:
            self.ui_manager.test_result = "Enter a string to test"
            return

        accepted, path = self.dfa.process_string(test_string)

        if accepted:
            self.ui_manager.test_result = f"String '{test_string}' ACCEPTED"
        else:
            self.ui_manager.test_result = f"String '{test_string}' REJECTED"

        # Start execution visualization
        self.execution_active = True
        self.execution_step = 0
        self.execution_string = test_string
        self.execution_path = path

    def _next_execution_step(self):
        """Move to the next step in execution visualization."""
        if self.execution_active and self.execution_step < len(self.execution_path) - 1:
            self.execution_step += 1
            self._show_message(f"Step {self.execution_step + 1}/{len(self.execution_path)}")

    def _previous_execution_step(self):
        """Move to the previous step in execution visualization."""
        if self.execution_active and self.execution_step > 0:
            self.execution_step -= 1
            self._show_message(f"Step {self.execution_step + 1}/{len(self.execution_path)}")

    def _stop_execution(self):
        """Stop execution visualization."""
        self.execution_active = False
        self.execution_step = 0
        self.execution_string = ""
        self.execution_path = []
        self.animation_active = False
        self.animation_auto_advance = False

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

        self.current_filename = os.path.relpath(path, PROJECT_DIR)
        self.dirty = False
        self._update_caption()
        self._show_message(f"Loaded {self.current_filename}")

    def _mark_dirty(self):
        """Record that the automaton has unsaved changes."""
        if not self.dirty:
            self.dirty = True
            self._update_caption()

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
            self._mark_dirty()
        elif verb == "set_dead_end":
            self.dfa.set_state_type(payload, StateType.DEAD_END)
            self._mark_dirty()
        elif verb == "set_normal":
            self.dfa.set_state_type(payload, StateType.NORMAL)
            self._mark_dirty()
        elif verb == "set_initial":
            if payload in self.dfa.states:
                self.dfa.initial_state = payload
                self._mark_dirty()
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
            self._mark_dirty()
            self._show_message(f"Added state {state_id}")
        elif verb == "reset_view":
            self.renderer.camera.reset()

    def _update(self, dt: float):
        """Update application state."""
        # Update UI manager
        self.ui_manager.update(dt)

        # Update message system
        self._update_message()

        # Handle animation auto-advance. The interval is owned by the UI, which
        # is where the slider that sets it lives; keeping a second copy here
        # meant dragging the slider changed a value nothing ever read.
        if (self.animation_active and self.animation_auto_advance and
            self.execution_active):
            current_time = pygame.time.get_ticks()
            if current_time - self.animation_timer >= self.ui_manager.animation_speed:
                if self.execution_step < len(self.execution_path) - 1:
                    self.execution_step += 1
                    self.animation_timer = current_time
                else:
                    # Animation finished
                    self.animation_auto_advance = False

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
                self._mark_dirty()
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

    def _render(self):
        """Render the entire application."""
        # Clear screen
        self.renderer.clear()

        # Draw initial state arrow
        if self.dfa.initial_state and self.dfa.initial_state in self.dfa.states:
            initial_state = self.dfa.states[self.dfa.initial_state]
            self.renderer.draw_initial_state_arrow(initial_state.position)

        # Draw transitions using transition groups
        for (from_state_id, to_state_id), group_data in self.dfa.transition_groups.items():
            if from_state_id not in self.dfa.states or to_state_id not in self.dfa.states:
                continue

            from_state = self.dfa.states[from_state_id]
            to_state = self.dfa.states[to_state_id]

            symbols = sorted(list(group_data['symbols']))
            arc_offset = group_data['arc_offset']

            # Choose color based on first symbol (for consistency)
            first_symbol = symbols[0] if symbols else 'a'
            if first_symbol == 'a':
                color = self.renderer.colors['transition_a']
            elif first_symbol == 'b':
                color = self.renderer.colors['transition_b']
            else:
                color = self.renderer.colors['transition_other']

            # Check if this is a self-loop
            is_self_loop = from_state_id == to_state_id

            # If no arc offset set, calculate default for bidirectional arrows
            if arc_offset == 0.0 and not is_self_loop:
                reverse_key = (to_state_id, from_state_id)
                if reverse_key in self.dfa.transition_groups:
                    # Create curved arrows - one curves up, one curves down
                    if from_state_id < to_state_id:  # Lexicographic order
                        arc_offset = 30.0  # This arrow curves up
                    else:
                        arc_offset = -30.0  # This arrow curves down

            # Combine symbols for label
            label = ','.join(symbols)

            self.renderer.draw_arrow(
                from_state.position,
                to_state.position,
                color,
                label,
                is_self_loop,
                arc_offset
            )

        # Draw transition being created
        if self.creating_transition and self.transition_start_state:
            start_state = self.dfa.states.get(self.transition_start_state)
            if start_state is None:
                # The source state was deleted mid-gesture. An unguarded lookup
                # here raised KeyError every frame, which killed the process.
                self._cancel_transition()
            else:
                mouse_pos = pygame.mouse.get_pos()
                world_mouse = self.renderer.camera.screen_to_world(mouse_pos)
                self.renderer.draw_arrow(
                    start_state.position,
                    world_mouse,
                    self.renderer.colors['transition_creating'],
                    self.ui_manager.selected_symbol,
                    False,
                    self.transition_arc_offset
                )

        # Draw states with execution highlighting
        for state_id, state in self.dfa.states.items():
            # Check if this state is currently being executed
            is_executing = (self.execution_active and
                          self.execution_step < len(self.execution_path) and
                          self.execution_path[self.execution_step] == state_id)

            self.renderer.draw_state(state, is_executing)

        # Draw UI
        self.ui_manager.draw(self.dfa, self.ui_manager.test_result, self.animation_active)

        # Draw execution status
        self.ui_manager.draw_execution_status(
            self.execution_active,
            self.execution_step,
            self.execution_string,
            self.execution_path
        )
        if self.execution_active:

            # Draw string visualization
            self.ui_manager.draw_string_visualization(self.execution_string, self.execution_step)

            # Draw current state indicator
            if self.execution_step < len(self.execution_path):
                current_state_id = self.execution_path[self.execution_step]
                if current_state_id in self.dfa.states:
                    current_state = self.dfa.states[current_state_id]
                    self.renderer.draw_current_state_indicator(current_state.position)

        # Draw temporary message
        if self.message_text:
            self._draw_message()

        pygame.display.flip()

    def _draw_message(self):
        """Draw temporary message on screen."""
        font = pygame.font.Font(None, 24)  # Smaller font
        text_surface = font.render(self.message_text, True, (255, 255, 255))

        # Create background - position at bottom right
        padding = 10
        bg_width = text_surface.get_width() + padding * 2
        bg_height = text_surface.get_height() + padding * 2
        bg_x = self.screen.get_width() - bg_width - 20
        bg_y = self.screen.get_height() - bg_height - 20

        # Draw background
        bg_rect = pygame.Rect(bg_x, bg_y, bg_width, bg_height)
        pygame.draw.rect(self.screen, (0, 0, 0, 180), bg_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), bg_rect, 2)

        # Draw text
        text_rect = text_surface.get_rect(center=bg_rect.center)
        self.screen.blit(text_surface, text_rect)


if __name__ == "__main__":
    app = AutomatonSimulator()
    app.run()
