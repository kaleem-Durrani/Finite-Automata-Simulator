"""
UI Manager module for handling user interface elements.

This module manages all UI components including toolbars, input fields,
context menus, and help panels.
"""

from typing import Any, Dict, List, Optional, Tuple

import pygame

import fsa
from rendering.animation import Animated, Timer, ease_out
from rendering.fonts import FontBook
from rendering.theme import Theme
from ui import column_layout, context_menu, dialogs, events, layout_spec

# The help text lives at module level so that the code which scrolls it and the
# code which draws it agree on how many lines there are. They used to be two
# independent guesses, and they disagreed badly enough that the maximum scroll
# offset came out as zero.
# Re-exported so callers keep one import site for the interface layer even
# though the pieces now live in their own modules.
from ui.context_menu import (  # noqa: E402
    ContextMenu,
    MenuItem,
)
from ui.events import UiEvent
from ui.layout_spec import (
    SPEED_MAX_MS,
    SPEED_MIN_MS,
    LayoutSpec,
)
from ui.panels import (
    column,
    diagnostics,
    help_panel,
    palette,
    run_panel,
    tape_strip,
    test_panel,
    toolbar,
)
from ui.panels.help_panel import HELP_LINES  # noqa: E402
from ui.widgets import Chrome

#: Longest display name a state may be given. In the same spirit as the 50
#: the test string gets: enough for "even number of bs", short of a caption.
RENAME_LABEL_LIMIT = 24

#: What every handler in this module returns: the events the application
#: must act on, and nothing about how the interface looks.
Events = List[UiEvent]


class UIManager:
    """
    Manages the user interface elements and interactions.
    
    Handles input fields, buttons, context menus, help panels,
    and all user interface state management.
    """
    
    def __init__(self, screen: pygame.Surface, theme: Optional[Theme] = None,
                 fonts: Optional[FontBook] = None):
        """
        Initialize the UI manager.

        Args:
            screen: Pygame surface for rendering UI elements
            theme: Shared design tokens. Shared rather than owned so that
                switching palettes reaches the canvas and the panels together.
            fonts: Shared, cached font faces.
        """
        self.screen = screen
        self.screen_width = screen.get_width()
        self.screen_height = screen.get_height()
        self.theme = theme or Theme("dark")
        self.fonts = fonts or FontBook()

        # UI state management
        self.show_help = False
        self.input_text = ""
        self.input_active = False
        self.test_result = ""
        self.test_verdict = ""
        self.selected_symbol = 'a'
        self.context_menu: Optional[ContextMenu] = None
        
        # The palette *is* the automaton's alphabet, kept in sync by
        # sync_symbols_with. It used to be a separate hardcoded list, so the
        # symbols you could draw with and the symbols the machine recognised
        # were two unrelated sets.
        self.available_symbols: List[str] = []

        # Symbol addition dialog state
        self.adding_symbol = False
        self.new_symbol_input = ""

        # Filename prompt state ('save' | 'load' | 'rename' | None)
        self.file_prompt_mode: Optional[str] = None
        self.file_prompt_text = ""
        self.rename_target = ""

        # Confirmation dialog state; confirm_intent names the action being
        # guarded so the app layer decides what "yes" means.
        self.confirm_message = ""
        self.confirm_intent: Optional[str] = None
        
        # UI element rectangles
        self._setup_ui_elements()
        
        # Input handling for backspace
        self.backspace_timer = 0
        self.backspace_repeat_delay = 500  # ms before repeat starts
        self.backspace_repeat_rate = 50    # ms between repeats

        # Help panel scrolling
        self.help_scroll_offset = 0

        # Animation controls
        self.animation_speed = 1000  # ms per step
        self.speed_slider_dragging = False

        # Whether the execution panel is on screen; it is only a hit target
        # while it is actually drawn.
        self.execution_panel_visible = False

        # Which state kinds are present, so the legend lists only what is on
        # screen. Set by the app, which is what computes them.
        self.legend_dead = False
        self.legend_unreachable = False

        # Structural problems with the automaton, set by the app each frame
        # from the editor's cached analysis. Drives the diagnostics panel.
        self.diagnostics: Tuple[Any, ...] = ()

        # Side panels slide in and out rather than popping. Each panel key maps
        # to an Animated 0..1; 1 is fully on screen. The column layout also
        # scales each panel's occupied height by its slide value, so when one
        # leaves, the panels below glide up instead of jumping.
        self.column_state = column_layout.ColumnState()

        # Collapsed panels keep their header and drop their body. The header is
        # the notch: it still names what is inside, so folding a panel away
        # never costs the user the knowledge that it exists. Each panel also
        # eases between the two heights rather than snapping.


        # The test panel folds down to a labelled pill. It is the least often
        # needed thing on screen and it was the largest, so it starts folded
        # and opens on a click or as soon as a run reports a verdict.
        self.input_expanded = False

        # The current tool: "pointer", "pan" or "transition". One name rather
        # than a flag per tool, because the tools are exclusive and two
        # booleans can both be true. Drawing a transition used to require
        # holding shift -- a hidden modifier nothing on screen mentioned -- so
        # it is a visible tool now, with shift kept as the shortcut.
        self.tool = "pointer"

        # This frame's right-column rects, for hit-testing. Recomputed at the
        # top of draw(); events read the previous frame's values, which is
        # harmless because a sliding panel takes ~260ms to arrive -- nothing
        # can be clicked on the frame it first exists, unlike the modal-dialog
        # bug this pattern replaced.
        self._column: List[Tuple[str, pygame.Rect, float]] = []
        self._diagnostic_rows: List[Tuple[pygame.Rect, Dict[str, Any]]] = []
        self._fix_button: Optional[pygame.Rect] = None

        # Which widget rect is currently held down, for the pressed visual.
        self._pressed_rect: Optional[pygame.Rect] = None

        # Tape strip motion: the scroll glides and the current cell pops.
        self.strip_scroll = Animated(duration=self.theme.motion.normal,
                                     easing=ease_out)
        self.strip_pop = Timer(duration=self.theme.motion.quick)
        self._tape = tape_strip.TapeState()
        self._strip_bounds: Optional[pygame.Rect] = None

        # No accepting state means no string can ever be accepted. The canvas
        # deliberately stays quiet about it -- marking every state dead would
        # be true and useless -- so the status panel has to say it instead.
        self.warn_no_accepting = False

    # Fonts and colours are read through the shared theme, so there is exactly
    # one definition of "the colour of a border" in the application.
    def _setup_ui_elements(self):
        """Recompute every UI rectangle for the current window size."""
        self.layout = LayoutSpec.for_size(self.screen_width, self.screen_height)
        self._recompute_symbol_buttons()

    # ------------------------------------------------------------------
    # Small drawing helpers
    # ------------------------------------------------------------------

    def _recompute_symbol_buttons(self):
        """
        Position the symbol palette.

        Computed here rather than while drawing, so that a button can be
        clicked on the first frame it exists instead of only after it has been
        painted once.
        """
        count = len(self.available_symbols) + 1  # the add button shares the grid
        self.symbol_buttons = {
            symbol: self.layout.symbol_chip(index, count)
            for index, symbol in enumerate(self.available_symbols)
        }
        self.add_symbol_button_rect = self.layout.symbol_chip(count - 1, count)

    # Named rectangles are read straight from the layout so that drawing and
    # hit-testing cannot drift apart.
    @property
    def symbol_card_rect(self) -> pygame.Rect:
        """The floating palette card, sized to the current alphabet."""
        return self.layout.symbol_card(len(self.available_symbols) + 1)

    @property
    def input_panel_rect(self) -> pygame.Rect:
        """The test panel in whichever form it is currently in."""
        return self.layout.input_panel(expanded=self.input_expanded,
                                       has_verdict=bool(self.test_result))

    @property
    def input_rect(self) -> pygame.Rect:
        return self.layout.input_field(self.input_panel_rect)

    @property
    def test_button_rect(self) -> pygame.Rect:
        return self.layout.test_button(self.input_panel_rect)

    @property
    def input_collapse_rect(self) -> pygame.Rect:
        return self.layout.input_collapse_button(self.input_panel_rect)

    @property
    def pan_button_rect(self) -> pygame.Rect:
        return self.layout.pan_button

    @property
    def transition_button_rect(self) -> pygame.Rect:
        return self.layout.transition_button

    @property
    def pan_tool(self) -> bool:
        return self.tool == "pan"

    @property
    def transition_tool(self) -> bool:
        return self.tool == "transition"

    def select_tool(self, tool: str) -> None:
        """Choose a tool, or click the active one again to put it back.

        Exclusive by construction: there is one name, so turning a tool on
        cannot leave another one also on.
        """
        self.tool = "pointer" if self.tool == tool else tool

    @property
    def help_button_rect(self) -> pygame.Rect:
        return self.layout.help_button

    @property
    def save_button_rect(self) -> pygame.Rect:
        return self.layout.save_button

    @property
    def load_button_rect(self) -> pygame.Rect:
        return self.layout.load_button

    @property
    def theme_button_rect(self) -> pygame.Rect:
        return self.layout.theme_button

    @property
    def speed_slider_rect(self) -> Optional[pygame.Rect]:
        """The slider tracks the run panel wherever it has slid to.

        The playback controls belong to a run, so they exist only while one
        does. They used to be drawn in the status panel at all times, which is
        most of what made that panel look half empty: a paused readout and a
        speed slider for an animation that was not playing.
        """
        for key, rect, _t in self._column:
            if key == "run" and not self.column_state.collapsed.get("run"):
                return self.layout.speed_slider(rect)
        return None

    @property
    def chrome(self) -> Chrome:
        """What the panels draw through. Rebuilt per access rather than cached,
        because the surface is replaced on every resize and a stale one would
        paint into a window that no longer exists."""
        return Chrome(screen=self.screen, theme=self.theme, fonts=self.fonts)

    def show_context_menu(self, position: Tuple[int, int],
                          items: List[MenuItem]) -> None:
        """Open a menu, nudged to fit on screen."""
        self.context_menu = context_menu.place(
            position=position, items=items,
            screen_width=self.screen_width, screen_height=self.screen_height)

    def hide_context_menu(self) -> None:
        self.context_menu = None

    def draw_legend(self, automaton: "fsa.DFA") -> None:
        """Explain the state styles, showing only the kinds actually present."""
        rect = next((r for key, r, _t in self._column if key == "legend"), None)
        column.draw_legend(self.chrome, rect=rect, automaton=automaton,
                           legend_dead=self.legend_dead,
                           legend_unreachable=self.legend_unreachable,
                           collapsed=bool(self.column_state.collapsed.get("legend")),
                           layout=self.layout,
                           mouse_pos=pygame.mouse.get_pos())

    def draw_execution_status(self, execution_active: bool, execution_step: int,
                              _execution_string: str, execution_path: List[str],
                              run: Optional[Any] = None) -> None:
        """Draw the execution trace panel."""
        panel = next((r for key, r, _t in self._column if key == "run"), None)
        run_panel.draw_run_panel(
            self.chrome, panel=panel, layout=self.layout,
            execution_active=execution_active, execution_step=execution_step,
            execution_path=execution_path, run=run,
            animation_active=getattr(self, "_animation_active", False),
            animation_speed=self.animation_speed,
            slider=self.speed_slider_rect,
            collapsed=bool(self.column_state.collapsed.get("run")),
            pressed_rect=self._pressed_rect,
            mouse_pos=pygame.mouse.get_pos())

    def draw_string_visualization(self, test_string: str, current_step: int,
                                  run: Optional[Any] = None) -> None:
        """Draw the input tape: sliding in and out, scrolling, popping."""
        strip = self.column_state.slides.get("strip")
        was_hidden = strip is None or strip.value <= 0.01
        slide = self.column_state.slide("strip",
                                        self.execution_panel_visible, self.theme)
        self._strip_bounds = tape_strip.draw(
            self.chrome, test_string=test_string, current_step=current_step,
            visible=self.execution_panel_visible, slide=slide,
            was_hidden=was_hidden, layout=self.layout, column=self._column,
            scroll=self.strip_scroll, pop=self.strip_pop,
            state=self._tape, run=run)

    def update_screen_size(self, width: int, height: int):
        """
        Update UI element positions when screen size changes.

        Args:
            width: New screen width
            height: New screen height
        """
        self.screen_width = width
        self.screen_height = height
        self._setup_ui_elements()
    
    def handle_event(self, event: pygame.event.Event) -> Tuple[Events, bool]:
        """
        Handle a UI event.

        Args:
            event: Pygame event to process

        Returns:
            (events, consumed). `consumed` is True when this event belonged to
            the UI and must not also be interpreted as a click on the canvas.
            The application layer runs its own handlers only when it is False,
            which is what stops one click from doing two contradictory things.
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            return self._handle_mouse_down(event)
        if event.type == pygame.MOUSEBUTTONUP:
            return self._handle_mouse_up(event)
        if event.type == pygame.MOUSEMOTION:
            return self._handle_mouse_motion(event)
        if event.type == pygame.KEYDOWN:
            # A modifier chord is never text. Every text field here ignores
            # Ctrl combinations anyway (their unicode is a control character,
            # which is not printable), so reporting one as captured only ever
            # loses it: typing a string and pressing enter leaves the field
            # focused, which used to make Ctrl+Z -- the app's only undo -- do
            # nothing at all, silently, in the most ordinary flow there is.
            captured = (self.is_keyboard_captured()
                        and not event.mod & pygame.KMOD_CTRL)
            return self._handle_key_down(event), captured
        if event.type == pygame.KEYUP:
            return self._handle_key_up(event), False
        if event.type == pygame.MOUSEWHEEL:
            return self._handle_mouse_wheel(event)

        return [], False

    # ------------------------------------------------------------------
    # Hit testing
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # The sliding right column
    # ------------------------------------------------------------------

    def opaque_panels(self) -> List[pygame.Rect]:
        """The UI regions currently painted over the canvas.

        The right column contributes its panels only once they are mostly on
        screen, so a panel sliding away releases the canvas underneath it.
        """
        panels = [self.layout.toolbar, self.symbol_card_rect,
                  self.input_panel_rect]
        panels += [rect for _key, rect, t in self._column if t > 0.5]
        # The strip's opaque region is what was actually drawn, not the
        # full-width band it lives in -- clicks beside the cells belong to
        # the canvas.
        if self._strip_bounds is not None:
            panels.append(self._strip_bounds)
        if self.show_help:
            panels.append(self.layout.help_panel)
        return panels

    def is_over_ui(self, pos: Tuple[int, int]) -> bool:
        """Whether a point lands on a UI panel rather than the canvas."""
        if self.is_modal_active():
            return True
        if self.context_menu and self.context_menu.visible:
            if context_menu.bounds(menu=self.context_menu).collidepoint(pos):
                return True
        return any(panel.collidepoint(pos) for panel in self.opaque_panels())

    def _widget_hits(self):
        """
        Interactive widgets, topmost first.

        Returned as (rect, handler) so the click loop can stop at the first
        match. Testing every widget instead meant a single click could fire two
        conflicting actions.
        """
        hits = [
            (self.layout.help_button, self._on_help_button),
            (self.layout.save_button, self._on_save_button),
            (self.layout.load_button, self._on_load_button),
            (self.layout.theme_button, self._on_theme_button),
            (self.layout.pan_button, self._on_pan_button),
            (self.layout.transition_button, self._on_transition_button),
            (self.add_symbol_button_rect, self._on_add_symbol_button),
        ]
        run_panel = next((r for key, r, t in self._column
                          if key == "run" and t > 0.5), None)
        if run_panel is not None and not self.column_state.collapsed.get("run"):
            back, play, forward, stop = self.layout.run_buttons(run_panel)
            hits += [(back, self._on_step_previous), (play, self._on_play_pause),
                     (forward, self._on_step_next), (stop, self._on_stop_run)]
        if self.input_expanded:
            # The collapse chevron sits inside the panel, so it has to be
            # tested before the panel's own widgets.
            hits.append((self.input_collapse_rect, self._on_input_collapse))
            hits.append((self.test_button_rect, self._on_test_button))
            hits.append((self.input_rect, self._on_input_field))
        else:
            hits.append((self.input_panel_rect, self._on_input_expand))
        slider = self.speed_slider_rect
        if slider is not None:
            hits.append((slider, self._on_speed_slider))
        # Panel headers toggle their panel. Added before the body widgets so a
        # header click never falls through to whatever is drawn beneath it.
        for key, rect, t in self._column:
            if t > 0.5:
                hits.append((self.layout.panel_header(rect),
                             self._panel_header_handler(key)))
        if self._fix_button is not None:
            hits.append((self._fix_button, self._on_fix_button))
        for row_rect, payload in self._diagnostic_rows:
            hits.append((row_rect, self._diagnostic_handler(payload)))
        for symbol, rect in self.symbol_buttons.items():
            hits.append((rect, self._symbol_handler(symbol)))
        return [(rect, handler) for rect, handler in hits if rect is not None]

    # -- widget handlers ------------------------------------------------

    # Handlers return the events the application must act on. Anything that
    # only changes how the interface looks -- focus, folding, the help panel --
    # is done here and announced to nobody, because nobody was listening.

    def _on_help_button(self, _pos) -> Events:
        self.show_help = not self.show_help
        self.help_scroll_offset = 0
        return []

    def _on_pan_button(self, _pos) -> Events:
        self.select_tool("pan")
        return [events.ToolSelected(self.tool)]

    def _on_transition_button(self, _pos) -> Events:
        self.select_tool("transition")
        return [events.ToolSelected(self.tool)]

    def _on_step_next(self, _pos) -> Events:
        return [events.StepForward()]

    def _on_step_previous(self, _pos) -> Events:
        return [events.StepBack()]

    def _on_play_pause(self, _pos) -> Events:
        return [events.ToggleAnimation()]

    def _on_stop_run(self, _pos) -> Events:
        return [events.StopExecution()]

    def _on_confirm_yes(self, _pos) -> Events:
        intent = self.confirm_intent
        self.hide_confirm()
        return [events.Confirmed(intent)] if intent else []

    def _on_confirm_no(self, _pos) -> Events:
        self.hide_confirm()
        return []

    def _on_input_collapse(self, _pos) -> Events:
        self.input_expanded = False
        self.input_active = False
        return []

    def _on_input_expand(self, _pos) -> Events:
        self.input_expanded = True
        self.input_active = True
        return []

    def _panel_header_handler(self, key: str):
        def handler(_pos) -> Events:
            self.column_state.toggle(key)
            return []
        return handler

    def _on_save_button(self, _pos) -> Events:
        return [events.SaveRequested()]

    def _on_load_button(self, _pos) -> Events:
        return [events.LoadRequested()]

    def _on_theme_button(self, _pos) -> Events:
        return [events.ToggleTheme()]

    def _on_test_button(self, _pos) -> Events:
        return [events.TestString(self.input_text)]

    def _on_input_field(self, _pos) -> Events:
        self.input_active = True
        return []

    def _on_add_symbol_button(self, _pos) -> Events:
        self.adding_symbol = True
        self.new_symbol_input = ""
        return []

    def _on_fix_button(self, _pos) -> Events:
        return [events.CompleteAutomaton()]

    def _diagnostic_handler(self, states: Tuple[str, ...]):
        def handler(_pos) -> Events:
            return [events.FocusStates(states)]
        return handler

    def _symbol_handler(self, symbol: str):
        def handler(_pos) -> Events:
            self.selected_symbol = symbol
            return [events.SymbolSelected(symbol)]
        return handler

    def _on_speed_slider(self, pos) -> Events:
        self.speed_slider_dragging = True
        self._set_speed_from_x(pos[0])
        return []

    def _set_speed_from_x(self, x: int) -> None:
        """Map an x coordinate on the slider to an animation speed.

        The speed lives on the manager and the application reads it there, so
        moving the slider is not news anyone has to be told.
        """
        slider = self.speed_slider_rect
        if slider is None:
            return
        ratio = (x - slider.x) / slider.width
        ratio = max(0.0, min(1.0, ratio))
        self.animation_speed = int(
            SPEED_MIN_MS + ratio * (SPEED_MAX_MS - SPEED_MIN_MS))

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def _handle_mouse_down(self, event) -> Tuple[Events, bool]:
        """Route a mouse press to exactly one owner, topmost first."""
        # Hit-test against where the click happened, not where the cursor is
        # now. Those differ whenever the mouse moves between the event being
        # queued and the queue being drained, which loses clicks and lets a
        # click register against whatever the cursor has since moved over.
        pos = event.pos

        # A modal dialog owns everything beneath it.
        if self.is_modal_active():
            return self._handle_modal_click(pos), True

        # The add-symbol dialog is modal too, but has its own buttons.
        if self.adding_symbol:
            return self._handle_symbol_dialog_click(pos), True

        # The context menu is above every other widget, and any click while it
        # is open belongs to it -- either choosing an item or dismissing it.
        if self.context_menu and self.context_menu.visible:
            chosen = context_menu.event_at(menu=self.context_menu, position=pos)
            self.hide_context_menu()
            return ([chosen] if chosen is not None else []), True

        if event.button != 1:
            # Right and middle clicks belong to the canvas unless they land on
            # a panel.
            return [], self.is_over_ui(pos)

        for rect, handler in self._widget_hits():
            if rect.collidepoint(pos):
                # Focus belongs to the field alone, so clicking any other
                # widget drops it and the two handlers that own the field set
                # it back. Identity against a layout rect used to decide this,
                # which stopped working once rects became computed values.
                self.input_active = False
                self._pressed_rect = pygame.Rect(rect)
                return handler(pos), True

        # Not a widget: clicking anywhere else drops text focus.
        self.input_active = False

        # A click on a panel with no widget under it is still the UI's.
        return [], self.is_over_ui(pos)

    def _handle_mouse_up(self, event) -> Tuple[Events, bool]:
        """Release the speed slider and the pressed-button visual."""
        self._pressed_rect = None
        if event.button == 1 and self.speed_slider_dragging:
            self.speed_slider_dragging = False
            return [], True
        return [], False

    def _handle_mouse_motion(self, event) -> Tuple[Events, bool]:
        """Drag the speed slider."""
        if self.speed_slider_dragging:
            self._set_speed_from_x(event.pos[0])
            return [], True
        return [], False

    def _handle_modal_click(self, pos) -> Events:
        """Route a click inside a modal dialog to its buttons.

        Every click used to be swallowed here, which meant the confirmation
        dialog and the file prompt could only be answered with the keyboard --
        the buttons a mouse user goes looking for did not exist, and the keys
        were named in small grey print along the bottom edge.
        """
        if self.confirm_intent:
            cancel, confirm = self.layout.confirm_buttons(
                dialogs.confirm_rect(self.screen_width, self.screen_height))
            if confirm.collidepoint(pos):
                return self._on_confirm_yes(pos)
            if cancel.collidepoint(pos):
                return self._on_confirm_no(pos)
            return []

        if self.file_prompt_mode:
            cancel, confirm = self.layout.confirm_buttons(
                dialogs.file_prompt_rect(self.screen_width, self.screen_height))
            if confirm.collidepoint(pos):
                return self._submit_file_prompt()
            if cancel.collidepoint(pos):
                self.hide_file_prompt()
        return []

    def _handle_symbol_dialog_click(self, pos) -> Events:
        """Handle the add-symbol dialog's own buttons."""
        cancel, add = dialogs.symbol_dialog_buttons(self.screen_width,
                                                    self.screen_height)

        if cancel.collidepoint(pos):
            self.adding_symbol = False
            self.new_symbol_input = ""
        elif add.collidepoint(pos) and self.new_symbol_input:
            symbol = self.new_symbol_input
            self.adding_symbol = False
            self.new_symbol_input = ""
            if self.can_add_symbol(symbol):
                return [events.SymbolAdded(symbol)]
            return [events.SymbolRejected(
                "Not a symbol, or already in the alphabet")]

        return []

    def _handle_mouse_wheel(self, event) -> Tuple[Events, bool]:
        """Scroll the help panel. The canvas zooms only when this declines."""
        if not self.show_help:
            return [], False

        # Scroll bounds come from the same constants the drawing code uses.
        # They used to be independent guesses that disagreed with the content,
        # producing a maximum scroll of zero -- so the panel could not scroll at
        # all and its last six lines, including every execution shortcut, were
        # unreachable.
        max_scroll = max(0, len(HELP_LINES) - self.layout.help_visible_lines())
        self.help_scroll_offset -= event.y * 3
        self.help_scroll_offset = max(0, min(max_scroll, self.help_scroll_offset))
        return [], True
    
    def is_keyboard_captured(self) -> bool:
        """
        Whether a text field or dialog currently owns the keyboard.

        Editor shortcuts are bare letters, so the app must not act on a keypress
        that is meant for a text field. This is the single place that answers
        "is someone typing?".
        """
        return bool(
            self.input_active
            or self.adding_symbol
            or self.file_prompt_mode
            or self.confirm_intent
        )

    def is_modal_active(self) -> bool:
        """Whether a modal dialog is open and should absorb all input."""
        return bool(self.file_prompt_mode or self.confirm_intent)

    def show_file_prompt(self, mode: str, default_name: str = ""):
        """Open the filename prompt in 'save' or 'load' mode."""
        self.file_prompt_mode = mode
        self.file_prompt_text = default_name
        self.input_active = False

    def show_rename_prompt(self, state: str, current_label: str):
        """Open the text prompt to rename a state.

        Rides the filename prompt's machinery -- same modal frame, same keys --
        because a rename is the same interaction with a different verb.
        """
        self.file_prompt_mode = "rename"
        self.rename_target = state
        self.file_prompt_text = "" if current_label == state else current_label
        self.input_active = False

    def hide_file_prompt(self):
        """Close the filename prompt."""
        self.file_prompt_mode = None
        self.file_prompt_text = ""

    def show_confirm(self, message: str, intent: str):
        """Open a yes/no dialog guarding `intent`."""
        self.confirm_message = message
        self.confirm_intent = intent
        self.input_active = False

    def hide_confirm(self):
        """Close the confirmation dialog."""
        self.confirm_message = ""
        self.confirm_intent = None

    def sync_symbols_with(self, automaton: "fsa.DFA"):
        """Adopt the automaton's alphabet as the palette.

        One source of truth: what you can draw with is exactly what the machine
        recognises. Previously these were two unrelated sets, so a loaded file's
        symbols could be unusable and a palette symbol could be outside the
        alphabet entirely.
        """
        self.available_symbols = sorted(automaton.alphabet)
        if self.available_symbols and self.selected_symbol not in self.available_symbols:
            self.selected_symbol = self.available_symbols[0]
        self._recompute_symbol_buttons()

    def _submit_file_prompt(self) -> Events:
        """Accept whatever the prompt currently holds.

        Shared by the Enter key and the confirm button, so the two cannot come
        to different conclusions about what the prompt meant.
        """
        mode = self.file_prompt_mode
        name = self.file_prompt_text.strip()
        target = self.rename_target
        self.hide_file_prompt()
        if mode == 'rename':
            # An empty name is a deliberate reset to the state's own id, so it
            # is not a cancel.
            return [events.RenameState(target, name)]
        if not name:
            return []
        return [events.SaveToPath(name) if mode == 'save'
                else events.LoadFromPath(name)]

    def _handle_file_prompt_key(self, event) -> Events:
        """Handle keys while the filename prompt is open."""
        if event.key == pygame.K_ESCAPE:
            self.hide_file_prompt()
        elif event.key == pygame.K_RETURN:
            return self._submit_file_prompt()
        elif event.key == pygame.K_BACKSPACE:
            self.file_prompt_text = self.file_prompt_text[:-1]
        elif event.unicode.isprintable():
            # A path may reasonably be long; a state's display name has a
            # circle to fit inside, and the renderer will elide what does not.
            limit = RENAME_LABEL_LIMIT if self.file_prompt_mode == "rename" else 120
            if len(self.file_prompt_text) < limit:
                self.file_prompt_text += event.unicode

        return []

    def _handle_confirm_key(self, event) -> Events:
        """Handle keys while the confirmation dialog is open."""
        if event.key in (pygame.K_y, pygame.K_RETURN):
            return self._on_confirm_yes(None)
        if event.key in (pygame.K_n, pygame.K_ESCAPE):
            return self._on_confirm_no(None)
        return []

    def _handle_key_down(self, event) -> Events:
        """Handle key down events."""
        # Dialogs are modal, and they are checked before anything else so that
        # keys reach the topmost one only.
        if self.confirm_intent:
            return self._handle_confirm_key(event)

        if self.file_prompt_mode:
            return self._handle_file_prompt_key(event)

        if self.adding_symbol:
            if event.key == pygame.K_ESCAPE:
                self.adding_symbol = False
                self.new_symbol_input = ""
            elif event.key == pygame.K_RETURN:
                symbol = self.new_symbol_input
                if symbol and self.can_add_symbol(symbol):
                    self.adding_symbol = False
                    self.new_symbol_input = ""
                    return [events.SymbolAdded(symbol)]
                return [events.SymbolRejected(
                    "Not a symbol, or already in the alphabet")]
            elif event.key == pygame.K_BACKSPACE:
                if self.new_symbol_input:
                    self.new_symbol_input = self.new_symbol_input[:-1]
            elif event.unicode.isprintable() and len(event.unicode) == 1:
                # One character only, so a second keystroke replaces the first.
                self.new_symbol_input = event.unicode
        elif self.input_active:
            if event.key == pygame.K_BACKSPACE:
                if self.input_text:
                    self.input_text = self.input_text[:-1]
                self.backspace_timer = pygame.time.get_ticks()
            elif event.key == pygame.K_RETURN:
                return [events.TestString(self.input_text)]
            elif event.unicode.isprintable() and len(self.input_text) < 50:
                # Only characters that could plausibly be in an alphabet.
                if event.unicode.isalnum() or event.unicode in '+-*/.()[]{}|&!~^':
                    self.input_text += event.unicode

        return []

    def _handle_key_up(self, event) -> Events:
        """Key releases change nothing the application needs to hear about."""
        del event
        return []
    
    def update(self, dt: float):
        """
        Update UI state (called every frame).
        
        Args:
            dt: Delta time since last frame in milliseconds
        """
        # Handle backspace repeat
        if self.input_active and pygame.key.get_pressed()[pygame.K_BACKSPACE]:
            current_time = pygame.time.get_ticks()
            if (current_time - self.backspace_timer > self.backspace_repeat_delay):
                # Start repeating
                if (current_time - self.backspace_timer) % self.backspace_repeat_rate < dt:
                    if self.input_text:
                        self.input_text = self.input_text[:-1]

        # Panel slides, folds and tape motion close toward their targets every
        # frame.
        self.column_state.advance(dt)
        self.strip_scroll.update(dt)
        self.strip_pop.update(dt)

    def draw(self, automaton: "fsa.DFA", test_result: str = "",
             animation_active: bool = False, execution_active: bool = False):
        """
        Draw all UI elements.

        Args:
            automaton: The current automaton, for the status panel and palette.
            test_result: Result of the last string test.
            animation_active: Whether playback is running. Passed in rather than
                pushed onto the manager after draw() has already run, which made
                the indicator show the previous frame's state.
            execution_active: Whether a run is being visualised, which decides
                whether the run panel occupies its slot in the column.
        """
        self._animation_active = animation_active
        self.execution_panel_visible = execution_active

        legend_rows = 1
        legend_rows += 1 if automaton.accept else 0
        legend_rows += 1 if self.legend_dead else 0
        legend_rows += 1 if self.legend_unreachable else 0

        # The diagnostics panel measures its own wrapped text, so the
        # column cannot lay it out one size and the panel paint another.
        self._column = column_layout.compute(
            self.column_state, layout=self.layout, theme=self.theme,
            execution_active=self.execution_panel_visible,
            legend_rows=legend_rows,
            diagnostics_height=diagnostics.body_height(
                self.chrome, self.diagnostics, layout_spec.PANEL_WIDTH),
            warn_no_accepting=self.warn_no_accepting)
        self._diagnostic_rows = []
        self._fix_button = None

        chrome = self.chrome
        mouse_pos = pygame.mouse.get_pos()

        toolbar.draw_toolbar(chrome, layout=self.layout, tool=self.tool,
                             show_help=self.show_help,
                             pressed_rect=self._pressed_rect,
                             mouse_pos=mouse_pos)
        test_panel.draw(chrome, panel=self.input_panel_rect,
                        field=self.input_rect,
                        test_button=self.test_button_rect,
                        collapse_button=self.input_collapse_rect,
                        input_text=self.input_text,
                        input_active=self.input_active,
                        input_expanded=self.input_expanded,
                        test_result=test_result,
                        test_verdict=self.test_verdict,
                        automaton=automaton, mouse_pos=mouse_pos,
                        pressed_rect=self._pressed_rect)
        palette.draw(chrome, card_rect=self.symbol_card_rect,
                     symbol_buttons=self.symbol_buttons,
                     add_button_rect=self.add_symbol_button_rect,
                     selected_symbol=self.selected_symbol,
                     mouse_pos=mouse_pos, pressed_rect=self._pressed_rect)

        for key, rect, _t in self._column:
            if key == "status":
                column.draw_status(chrome, rect=rect, automaton=automaton,
                                   warn_no_accepting=self.warn_no_accepting,
                                   collapsed=bool(self.column_state.collapsed.get(key)),
                                   layout=self.layout, mouse_pos=mouse_pos)
            elif key == "diagnostics":
                hits = diagnostics.draw_diagnostics(
                    chrome, rect=rect, diagnostics=self.diagnostics,
                    collapsed=bool(self.column_state.collapsed.get(key)),
                    layout=self.layout, pressed_rect=self._pressed_rect,
                    mouse_pos=mouse_pos)
                self._diagnostic_rows = list(hits.rows)
                self._fix_button = hits.fix_button

    def draw_overlays(self) -> None:
        """Everything that must sit above every panel: help, menus, modals.

        A separate pass the application calls after the run panel, legend and
        tape strip. When these were painted inside draw(), those later calls
        painted over them -- the tape strip cut a row of cells straight through
        the help text, and the run panel clipped the Save dialog while the rest
        of the screen sat dimmed around them.
        """
        chrome = self.chrome
        mouse_pos = pygame.mouse.get_pos()

        if self.show_help:
            help_panel.draw(chrome, layout=self.layout,
                            scroll_offset=self.help_scroll_offset)

        if self.context_menu and self.context_menu.visible:
            context_menu.draw(chrome, menu=self.context_menu,
                              mouse_pos=mouse_pos)

        if self.adding_symbol:
            dialogs.draw_add_symbol_dialog(
                chrome, screen_width=self.screen_width,
                screen_height=self.screen_height,
                new_symbol_input=self.new_symbol_input,
                pressed_rect=self._pressed_rect, mouse_pos=mouse_pos)

        if self.file_prompt_mode:
            dialogs.draw_file_prompt(
                chrome, layout=self.layout, screen_width=self.screen_width,
                screen_height=self.screen_height,
                file_prompt_mode=self.file_prompt_mode,
                file_prompt_text=self.file_prompt_text,
                rename_target=self.rename_target,
                pressed_rect=self._pressed_rect, mouse_pos=mouse_pos)

        if self.confirm_intent:
            dialogs.draw_confirm_dialog(
                chrome, layout=self.layout, screen_width=self.screen_width,
                screen_height=self.screen_height,
                confirm_intent=self.confirm_intent,
                confirm_message=self.confirm_message,
                pressed_rect=self._pressed_rect, mouse_pos=mouse_pos)

    def can_add_symbol(self, symbol: str) -> bool:
        """Whether a symbol could be added to the alphabet.

        Delegates to the engine rather than keeping its own rules, and no
        longer reserves letters. `q`, `w`, `r`, `n` and `p` used to be rejected
        because keyboard shortcuts owned them, which meant no automaton over an
        alphabet containing those letters could be built at all.
        """
        return (fsa.is_legal_symbol(symbol)
                and symbol not in self.available_symbols)
