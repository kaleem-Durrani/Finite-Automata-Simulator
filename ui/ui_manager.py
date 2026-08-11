"""
UI Manager module for handling user interface elements.

This module manages all UI components including toolbars, input fields,
context menus, and help panels.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pygame

import fsa
from rendering import primitives
from rendering.animation import Animated, Timer, ease_out
from rendering.fonts import FontBook
from rendering.theme import Theme
from ui.layout_spec import (
    HELP_LINE_HEIGHT,
    HELP_TITLE_HEIGHT,
    PANEL_GAP,
    PANEL_HEADER_HEIGHT,
    PANEL_MARGIN,
    PANEL_WIDTH,
    SPEED_MAX_MS,
    SPEED_MIN_MS,
    LayoutSpec,
)

# The help text lives at module level so that the code which scrolls it and the
# code which draws it agree on how many lines there are. They used to be two
# independent guesses, and they disagreed badly enough that the maximum scroll
# offset came out as zero.
HELP_LINES = [
    "Mouse Controls:",
    "- Left Click: Select states/UI",
    "- Shift+Click: Start transition",
    "- Right Click: Menu for a state",
    "  or a transition arrow",
    "- Space+Drag: Pan the view",
    "  (or switch on the hand tool)",
    "- Right/Middle Drag: Pan too",
    "- Scroll Wheel: Zoom",
    "",
    "Panels:",
    "- Click a panel's title to fold",
    "  it away; the title stays as",
    "  the notch that reopens it",
    "- The test box folds down to",
    "  a pill in the bottom corner",
    "",
    "Keyboard Shortcuts:",
    "- Ctrl+Z / Ctrl+Y: Undo / Redo",
    "- Space: Tap to add a state,",
    "  hold to pan the view",
    "- Delete: Remove selected state",
    "- Q: Toggle accepting",
    "- W: Make a trap (loop all",
    "  symbols back to itself)",
    "- R: Fit view to automaton",
    "",
    "Creating Transitions:",
    "- Select symbol from toolbar",
    "- Shift+click source state",
    "- Click target state",
    "",
    "Editing:",
    "- Right-click a state to",
    "  rename it (a display label;",
    "  the id does not change)",
    "- Right-click an arrow to",
    "  remove one of its symbols,",
    "  or straighten a curve",
    "",
    "Testing Strings:",
    "- Enter string in input field",
    "- Click Test or press Enter",
    "",
    "Execution Visualization:",
    "- N: Next step",
    "- P: Previous step",
    "- TAB: Toggle animation",
    "- ESC: Stop visualization",
    "",
    "File Operations:",
    "- Save/Load buttons in toolbar",
    "- Filenames are relative to",
    "  the project folder",
]


#: What each right-hand panel is called. A collapsed panel shows only this, so
#: it doubles as the notch's label: folding a panel away must not cost the user
#: the knowledge of what is inside it.
PANEL_TITLES = {
    "status": "Automaton",
    "run": "Run",
    "diagnostics": "Diagnostics",
    "legend": "Legend",
}

CONTEXT_MENU_WIDTH = 168
CONTEXT_MENU_ITEM_HEIGHT = 27
# Left inset for item labels, leaving room for the toggle marker.
CONTEXT_MENU_GUTTER = 28

#: Breathing room between a nudged context menu and the window edge.
CONTEXT_MENU_MARGIN = 8

#: Longest display name a state may be given. In the same spirit as the 50 the
#: test string gets: enough for "even number of bs", short of a caption.
RENAME_LABEL_LIMIT = 24


@dataclass
class ContextMenu:
    """A context menu: a position and a list of items.

    An item is ``(label, action)``, or ``(label, action, checked)`` for a
    toggle. Carrying the current value means the menu can show what a state
    already is, rather than making the user pick something and look to find
    out whether it changed anything.
    """
    position: Tuple[int, int]
    items: List[Tuple[Any, ...]]
    visible: bool = True
    selected_index: int = -1


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
        self.show_tutorial = False
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
        self.slides: Dict[str, Animated] = {}

        # Collapsed panels keep their header and drop their body. The header is
        # the notch: it still names what is inside, so folding a panel away
        # never costs the user the knowledge that it exists. Each panel also
        # eases between the two heights rather than snapping.
        self.collapsed: Dict[str, bool] = {}
        self.opens: Dict[str, Animated] = {}

        # The test panel folds down to a labelled pill. It is the least often
        # needed thing on screen and it was the largest, so it starts folded
        # and opens on a click or as soon as a run reports a verdict.
        self.input_expanded = False

        # Hand tool: while it is on, dragging anywhere on the canvas pans.
        # Holding space does the same without leaving the pointer tool, which
        # is the shortcut every drawing program uses. The space half lives in
        # the application, which owns canvas interaction; this is only the
        # toolbar's toggle.
        self.pan_tool = False

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
        self._strip_last_step = -1
        self._strip_cache: Optional[Tuple[str, int, Any]] = None
        self._strip_bounds: Optional[pygame.Rect] = None

        # No accepting state means no string can ever be accepted. The canvas
        # deliberately stays quiet about it -- marking every state dead would
        # be true and useless -- so the status panel has to say it instead.
        self.warn_no_accepting = False

    # Fonts and colours are read through the shared theme, so there is exactly
    # one definition of "the colour of a border" in the application.
    @property
    def font_small(self) -> pygame.font.Font:
        return self.fonts.ui("small")

    @property
    def font_medium(self) -> pygame.font.Font:
        return self.fonts.ui("body")

    @property
    def font_large(self) -> pygame.font.Font:
        return self.fonts.ui("title")

    @property
    def colors(self) -> Dict[str, Any]:
        """Semantic names used by the drawing code, resolved from the theme.

        Kept as a mapping so the call sites read the same as before, but with
        no values of its own: changing a palette changes this.
        """
        palette = self.theme.palette
        return {
            'ui_bg': palette.panel,
            'ui_raised': palette.panel_raised,
            'ui_border': palette.border,
            'ui_border_strong': palette.border_strong,
            'button_normal': palette.control,
            'button_hover': palette.control_hover,
            'button_active': palette.control_active,
            'text': palette.text,
            'text_muted': palette.text_muted,
            'text_faint': palette.text_faint,
            'text_on_accent': palette.text_on_accent,
            'accent': palette.accent,
            'input_active': palette.field,
            'input_inactive': palette.panel_raised,
            'success': palette.success,
            'error': palette.error,
            'warning': palette.warning,
        }

    def _setup_ui_elements(self):
        """Recompute every UI rectangle for the current window size."""
        self.layout = LayoutSpec.for_size(self.screen_width, self.screen_height)
        self._recompute_symbol_buttons()

    # ------------------------------------------------------------------
    # Small drawing helpers
    # ------------------------------------------------------------------

    def _button(self, rect: pygame.Rect, label: str, *, active: bool = False,
                hovered: bool = False, accent: bool = False) -> None:
        """Draw a labelled button with real depth.

        Raised at rest, and pressed -- shadow gone, bevels inverted, label
        nudged down a pixel -- while the mouse is held on it. The nudge is what
        makes a click feel like a click.
        """
        palette = self.theme.palette
        pressed = self._pressed_rect == rect
        if accent or active:
            fill = palette.accent
            text_color = palette.text_on_accent
            border = palette.accent
        elif hovered:
            fill = palette.control_hover
            text_color = palette.text
            border = palette.border_strong
        else:
            fill = palette.control
            text_color = palette.text
            border = palette.border

        nudge = primitives.raised_button(
            self.screen, rect, fill, radius=self.theme.radius.md, border=border,
            bevel_light=palette.bevel_light, bevel_dark=palette.bevel_dark,
            pressed=pressed, shadow=palette.shadow)
        surface = self.fonts.ui("body_strong").render(label, True, text_color)
        self.screen.blit(surface, surface.get_rect(
            center=(rect.centerx, rect.centery + nudge)))

    def _section_label(self, text: str, position: Tuple[int, int]) -> None:
        """A small uppercase caption above a group of controls."""
        surface = self.fonts.ui("small_strong").render(
            text.upper(), True, self.theme.palette.text_faint)
        self.screen.blit(surface, position)

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
            if key == "run" and not self.collapsed.get("run"):
                return self.layout.speed_slider(rect)
        return None

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
    
    def handle_event(self, event: pygame.event.Event) -> Tuple[Dict[str, Any], bool]:
        """
        Handle a UI event.

        Args:
            event: Pygame event to process

        Returns:
            (actions, consumed). `consumed` is True when this event belonged to
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

        return {}, False

    # ------------------------------------------------------------------
    # Hit testing
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # The sliding right column
    # ------------------------------------------------------------------

    def _slide(self, key: str, visible: bool) -> float:
        """The 0..1 slide progress for a panel, easing toward its target."""
        entry = self.slides.get(key)
        if entry is None:
            entry = Animated(value=0.0, target=0.0,
                             duration=self.theme.motion.normal, easing=ease_out)
            self.slides[key] = entry
        entry.set(1.0 if visible else 0.0)
        return entry.value

    def _diagnostic_height(self) -> int:
        """How tall the diagnostics body wants to be for its current rows."""
        rows = min(len(self.diagnostics), 4)
        return rows * 40 + 6

    def _open(self, key: str) -> float:
        """The 0..1 openness of a panel, easing toward collapsed or expanded."""
        entry = self.opens.get(key)
        if entry is None:
            entry = Animated(value=1.0, target=1.0,
                             duration=self.theme.motion.quick, easing=ease_out)
            self.opens[key] = entry
        entry.set(0.0 if self.collapsed.get(key) else 1.0)
        return entry.value

    def toggle_panel(self, key: str) -> None:
        """Fold a right-hand panel away, or bring it back."""
        self.collapsed[key] = not self.collapsed.get(key, False)

    def compute_right_column(self, execution_active: bool,
                             legend_rows: int) -> List[Tuple[str, pygame.Rect, float]]:
        """Lay out the right-hand panels for this frame.

        Deterministic from current state -- panel heights depend only on
        content counts known before drawing. Each panel slides horizontally by
        its own progress, and the space it occupies scales with that progress,
        so panels below glide up as one above departs instead of jumping.

        A panel's height is now its header plus however much body it currently
        wants, eased between the two so collapsing glides. Bodies are sized
        from their content rather than from a fixed constant, which is what
        left every panel looking half empty.
        """
        width, margin, gap = PANEL_WIDTH, PANEL_MARGIN, PANEL_GAP
        home_x = self.layout.column_home_x()
        y = float(self.layout.column_top())
        limit = self.layout.column_limit()

        wanted = [
            ("status", True, self._status_body_height()),
            ("run", execution_active, 146),
            ("diagnostics", bool(self.diagnostics), self._diagnostic_height()),
            ("legend", legend_rows >= 2, legend_rows * 22 + 6),
        ]

        column: List[Tuple[str, pygame.Rect, float]] = []
        for key, visible, body in wanted:
            height = PANEL_HEADER_HEIGHT + int(body * self._open(key))
            # A collapsed panel is a header, and a header always fits, so only
            # the expanded height can push a panel off the bottom.
            fits = y + PANEL_HEADER_HEIGHT <= limit
            t = self._slide(key, visible and fits)
            if t <= 0.01:
                continue
            height = min(height, max(PANEL_HEADER_HEIGHT, int(limit - y)))
            offset = (1.0 - t) * (width + margin * 2)
            rect = pygame.Rect(int(home_x + offset), int(y), width, height)
            column.append((key, rect, t))
            y += (height + gap) * t
        return column

    def _status_body_height(self) -> int:
        """Four rows, plus the warning line only when there is one."""
        return 4 * 19 + (18 if self.warn_no_accepting else 0) + 12

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
            if self._context_menu_rect().collidepoint(pos):
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
            (self.add_symbol_button_rect, self._on_add_symbol_button),
        ]
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

    def _on_help_button(self, _pos) -> Dict[str, Any]:
        self.show_help = not self.show_help
        self.help_scroll_offset = 0
        return {'toggle_help': True}

    def _on_pan_button(self, _pos) -> Dict[str, Any]:
        self.pan_tool = not self.pan_tool
        return {'pan_tool': self.pan_tool}

    def _on_input_collapse(self, _pos) -> Dict[str, Any]:
        self.input_expanded = False
        self.input_active = False
        return {'input_collapsed': True}

    def _on_input_expand(self, _pos) -> Dict[str, Any]:
        self.input_expanded = True
        self.input_active = True
        return {'input_expanded': True}

    def _panel_header_handler(self, key: str):
        def handler(_pos) -> Dict[str, Any]:
            self.toggle_panel(key)
            return {'panel_toggled': key}
        return handler

    def _on_save_button(self, _pos) -> Dict[str, Any]:
        return {'save_automaton': True}

    def _on_load_button(self, _pos) -> Dict[str, Any]:
        return {'load_automaton': True}

    def _on_theme_button(self, _pos) -> Dict[str, Any]:
        return {'toggle_theme': True}

    def _on_test_button(self, _pos) -> Dict[str, Any]:
        return {'test_string': self.input_text}

    def _on_input_field(self, _pos) -> Dict[str, Any]:
        self.input_active = True
        return {'input_focus': True}

    def _on_add_symbol_button(self, _pos) -> Dict[str, Any]:
        self.adding_symbol = True
        self.new_symbol_input = ""
        return {'add_symbol': True}

    def _on_fix_button(self, _pos) -> Dict[str, Any]:
        return {'complete_automaton': True}

    def _diagnostic_handler(self, payload: Dict[str, Any]):
        def handler(_pos) -> Dict[str, Any]:
            return payload
        return handler

    def _symbol_handler(self, symbol: str):
        def handler(_pos) -> Dict[str, Any]:
            self.selected_symbol = symbol
            return {'symbol_selected': symbol,
                    'show_message': f"Selected symbol: {symbol}"}
        return handler

    def _on_speed_slider(self, pos) -> Dict[str, Any]:
        self.speed_slider_dragging = True
        return self._set_speed_from_x(pos[0])

    def _set_speed_from_x(self, x: int) -> Dict[str, Any]:
        """Map an x coordinate on the slider to an animation speed."""
        slider = self.speed_slider_rect
        if slider is None:
            return {}
        ratio = (x - slider.x) / slider.width
        ratio = max(0.0, min(1.0, ratio))
        self.animation_speed = int(
            SPEED_MIN_MS + ratio * (SPEED_MAX_MS - SPEED_MIN_MS))
        return {'speed_changed': self.animation_speed}

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def _handle_mouse_down(self, event) -> Tuple[Dict[str, Any], bool]:
        """Route a mouse press to exactly one owner, topmost first."""
        actions: Dict[str, Any] = {}
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
            menu_action = self._handle_context_menu_click(pos)
            self.context_menu = None
            if menu_action:
                actions['context_menu_action'] = menu_action
            return actions, True

        if event.button != 1:
            # Right and middle clicks belong to the canvas unless they land on
            # a panel.
            return actions, self.is_over_ui(pos)

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
        return actions, self.is_over_ui(pos)

    def _handle_mouse_up(self, event) -> Tuple[Dict[str, Any], bool]:
        """Release the speed slider and the pressed-button visual."""
        self._pressed_rect = None
        if event.button == 1 and self.speed_slider_dragging:
            self.speed_slider_dragging = False
            return {}, True
        return {}, False

    def _handle_mouse_motion(self, event) -> Tuple[Dict[str, Any], bool]:
        """Drag the speed slider."""
        if self.speed_slider_dragging:
            return self._set_speed_from_x(event.pos[0]), True
        return {}, False

    def _handle_modal_click(self, pos) -> Dict[str, Any]:
        """Clicks while a modal dialog is open are swallowed."""
        del pos
        return {}

    def _handle_symbol_dialog_click(self, pos) -> Dict[str, Any]:
        """Handle the add-symbol dialog's own buttons."""
        actions: Dict[str, Any] = {}
        cancel, add = self._symbol_dialog_buttons()

        if cancel.collidepoint(pos):
            self.adding_symbol = False
            self.new_symbol_input = ""
            actions['symbol_dialog_cancel'] = True
        elif add.collidepoint(pos) and self.new_symbol_input:
            actions['symbol_add'] = self.new_symbol_input
            self.adding_symbol = False
            self.new_symbol_input = ""

        return actions

    def _handle_mouse_wheel(self, event) -> Tuple[Dict[str, Any], bool]:
        """Scroll the help panel. The canvas zooms only when this declines."""
        if not self.show_help:
            return {}, False

        # Scroll bounds come from the same constants the drawing code uses.
        # They used to be independent guesses that disagreed with the content,
        # producing a maximum scroll of zero -- so the panel could not scroll at
        # all and its last six lines, including every execution shortcut, were
        # unreachable.
        max_scroll = max(0, len(HELP_LINES) - self.layout.help_visible_lines())
        self.help_scroll_offset -= event.y * 3
        self.help_scroll_offset = max(0, min(max_scroll, self.help_scroll_offset))
        return {}, True
    
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

    def _handle_file_prompt_key(self, event) -> Dict[str, Any]:
        """Handle keys while the filename prompt is open."""
        actions: Dict[str, Any] = {}

        if event.key == pygame.K_ESCAPE:
            self.hide_file_prompt()
            actions['file_prompt_cancel'] = True
        elif event.key == pygame.K_RETURN:
            mode = self.file_prompt_mode
            name = self.file_prompt_text.strip()
            self.hide_file_prompt()
            if mode == 'rename':
                # An empty name is a deliberate reset to the state's own id,
                # so it is not a cancel.
                actions['rename_state'] = (self.rename_target, name)
            elif name:
                actions['save_to_path' if mode == 'save' else 'load_to_path'] = name
            else:
                actions['file_prompt_cancel'] = True
        elif event.key == pygame.K_BACKSPACE:
            self.file_prompt_text = self.file_prompt_text[:-1]
        elif event.unicode.isprintable():
            # A path may reasonably be long; a state's display name has a
            # circle to fit inside, and the renderer will elide what does not.
            limit = RENAME_LABEL_LIMIT if self.file_prompt_mode == "rename" else 120
            if len(self.file_prompt_text) < limit:
                self.file_prompt_text += event.unicode

        return actions

    def _handle_confirm_key(self, event) -> Dict[str, Any]:
        """Handle keys while the confirmation dialog is open."""
        actions: Dict[str, Any] = {}

        if event.key in (pygame.K_y, pygame.K_RETURN):
            intent = self.confirm_intent
            self.hide_confirm()
            actions['confirmed'] = intent
        elif event.key in (pygame.K_n, pygame.K_ESCAPE):
            self.hide_confirm()
            actions['confirm_cancel'] = True

        return actions

    def _handle_key_down(self, event) -> Dict[str, Any]:
        """Handle key down events."""
        actions: Dict[str, Any] = {}

        # Dialogs are modal, and they are checked before anything else so that
        # keys reach the topmost one only.
        if self.confirm_intent:
            return self._handle_confirm_key(event)

        if self.file_prompt_mode:
            return self._handle_file_prompt_key(event)

        if self.adding_symbol:
            # Handle symbol addition dialog
            if event.key == pygame.K_ESCAPE:
                self.adding_symbol = False
                self.new_symbol_input = ""
            elif event.key == pygame.K_RETURN:
                if self.new_symbol_input and self.can_add_symbol(self.new_symbol_input):
                    actions['symbol_added'] = self.new_symbol_input
                    self.adding_symbol = False
                    self.new_symbol_input = ""
                else:
                    actions['symbol_add_error'] = "Not a symbol, or already in the alphabet"
            elif event.key == pygame.K_BACKSPACE:
                if self.new_symbol_input:
                    self.new_symbol_input = self.new_symbol_input[:-1]
            elif event.unicode.isprintable() and len(event.unicode) == 1:
                # Replace the input with the new character (only one character allowed)
                self.new_symbol_input = event.unicode
        elif self.input_active:
            # Handle input field events
            if event.key == pygame.K_BACKSPACE:
                if self.input_text:
                    self.input_text = self.input_text[:-1]
                self.backspace_timer = pygame.time.get_ticks()
                actions['backspace_start'] = True
            elif event.key == pygame.K_RETURN:
                actions['test_string'] = self.input_text
            elif event.unicode.isprintable() and len(self.input_text) < 50:
                # Only allow characters that make sense for automaton input
                if event.unicode.isalnum() or event.unicode in '+-*/.()[]{}|&!~^':
                    self.input_text += event.unicode

        return actions
    
    def _handle_key_up(self, event) -> Dict[str, Any]:
        """Handle key up events."""
        actions: Dict[str, Any] = {}
        
        if event.key == pygame.K_BACKSPACE:
            actions['backspace_stop'] = True
            
        return actions
    
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
        for slide in self.slides.values():
            slide.update(dt)
        for opening in self.opens.values():
            opening.update(dt)
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

        self._column = self.compute_right_column(self.execution_panel_visible,
                                                 legend_rows)
        self._diagnostic_rows = []
        self._fix_button = None

        self._draw_toolbar()
        self._draw_input_area(automaton, test_result)
        self._draw_alphabet_selector()
        for key, rect, _t in self._column:
            if key == "status":
                self._draw_status_info(automaton, rect)
            elif key == "diagnostics":
                self._draw_diagnostics(rect)

    def draw_overlays(self) -> None:
        """Everything that must sit above every panel: help, menus, modals.

        A separate pass the application calls after the run panel, legend and
        tape strip. When these were painted inside draw(), those later calls
        painted over them -- the tape strip cut a row of cells straight through
        the help text, and the run panel clipped the Save dialog while the rest
        of the screen sat dimmed around them.
        """
        if self.show_help:
            self._draw_help_panel()

        if self.context_menu and self.context_menu.visible:
            self._draw_context_menu()

        if self.adding_symbol:
            self._draw_add_symbol_dialog()

        if self.file_prompt_mode:
            self._draw_file_prompt()

        if self.confirm_intent:
            self._draw_confirm_dialog()

    def _draw_modal_frame(self, width: int, height: int, title: str) -> pygame.Rect:
        """Dim the screen and draw an empty centred dialog box, returning it."""
        palette = self.theme.palette
        primitives.dim(self.screen, (0, 0, 0, 150 if palette.is_dark else 90))

        rect = pygame.Rect(
            (self.screen_width - width) // 2,
            (self.screen_height - height) // 2,
            width,
            height,
        )
        primitives.elevated_panel(self.screen, rect, palette.panel_raised,
                                  radius=self.theme.radius.lg,
                                  border=palette.border_strong,
                                  shadow=palette.shadow, lift=7,
                                  bevel_light=palette.bevel_light,
                                  bevel_dark=palette.bevel_dark)

        title_surface = self.fonts.ui("heading").render(title, True, palette.text)
        self.screen.blit(title_surface, title_surface.get_rect(
            centerx=rect.centerx, y=rect.y + self.theme.space.lg))

        return rect

    def _draw_file_prompt(self):
        """Draw the filename prompt (or its rename variant)."""
        titles = {"save": "Save as", "load": "Load file",
                  "rename": f"Rename {getattr(self, 'rename_target', '')}"}
        rect = self._draw_modal_frame(440, 160,
                                      titles.get(self.file_prompt_mode, ""))

        palette = self.theme.palette
        if self.file_prompt_mode == "rename":
            hint_text = "A display label. Leave empty to use the state's id."
        else:
            hint_text = "Relative to the project folder. '.json' is added if omitted."
        hint = self.fonts.ui("small").render(hint_text, True, palette.text_muted)
        self.screen.blit(hint, (rect.x + self.theme.space.lg, rect.y + 50))

        field = pygame.Rect(rect.x + self.theme.space.lg, rect.y + 74,
                            rect.width - self.theme.space.lg * 2, 34)
        primitives.panel(self.screen, field, palette.field,
                         radius=self.theme.radius.md, border=palette.accent,
                         border_width=2)

        # Show the tail of the text so the caret stays visible on long paths.
        shown = self.file_prompt_text[-40:]
        text_surface = self.fonts.mono("input").render(shown, True, palette.text)
        text_rect = text_surface.get_rect(midleft=(field.left + 6, field.centery))
        self.screen.blit(text_surface, text_rect)

        if pygame.time.get_ticks() % 1100 < 560:
            caret_x = min(text_rect.right + 2, field.right - 5)
            pygame.draw.line(self.screen, palette.accent,
                             (caret_x, field.top + 7), (caret_x, field.bottom - 7), 2)

        footer = self.fonts.ui("small").render("Enter to confirm, Escape to cancel",
                                               True, palette.text_faint)
        self.screen.blit(footer, footer.get_rect(centerx=rect.centerx,
                                                 y=rect.bottom - 26))

    def _draw_confirm_dialog(self):
        """Draw the yes/no confirmation dialog."""
        rect = self._draw_modal_frame(420, 130, "Unsaved changes")

        palette = self.theme.palette
        message = self.fonts.ui("body").render(self.confirm_message, True,
                                               palette.text_muted)
        self.screen.blit(message, message.get_rect(centerx=rect.centerx, y=rect.y + 56))

        footer = self.fonts.ui("small").render(
            "Y or Enter to confirm, N or Escape to cancel", True, palette.text_faint)
        self.screen.blit(footer, footer.get_rect(centerx=rect.centerx,
                                                 y=rect.bottom - 30))

    def _draw_toolbar(self):
        """Draw the main toolbar at the top of the screen."""
        palette = self.theme.palette
        toolbar = self.layout.toolbar
        pygame.draw.rect(self.screen, palette.panel, toolbar)
        pygame.draw.line(self.screen, palette.border,
                         (0, toolbar.bottom - 1), (toolbar.right, toolbar.bottom - 1))

        title = self.fonts.ui("title").render("Finite Automata", True, palette.text)
        self.screen.blit(title, title.get_rect(
            midleft=(self.theme.space.lg, toolbar.centery)))

        subtitle = self.fonts.ui("small").render(
            "simulator", True, palette.text_faint)
        self.screen.blit(subtitle, subtitle.get_rect(
            midleft=(self.theme.space.lg + title.get_width() + 8,
                     toolbar.centery + 1)))

        mouse_pos = pygame.mouse.get_pos()
        pan = self.pan_button_rect
        self._button(pan, "", active=self.pan_tool,
                     hovered=pan.collidepoint(mouse_pos))
        self._hand_icon(pan, palette.text_on_accent if self.pan_tool
                        else palette.text_muted)
        self._button(self.theme_button_rect,
                     "Light" if self.theme.is_dark else "Dark",
                     hovered=self.theme_button_rect.collidepoint(mouse_pos))
        self._button(self.load_button_rect, "Load",
                     hovered=self.load_button_rect.collidepoint(mouse_pos))
        self._button(self.save_button_rect, "Save",
                     hovered=self.save_button_rect.collidepoint(mouse_pos))
        self._button(self.help_button_rect, "Help", active=self.show_help,
                     hovered=self.help_button_rect.collidepoint(mouse_pos))

    def _draw_input_area(self, automaton: "fsa.DFA", test_result: str):
        """Draw the input area for testing strings."""
        palette = self.theme.palette
        panel = self.input_panel_rect
        mouse_pos = pygame.mouse.get_pos()

        if not self.input_expanded:
            # Folded: a pill that still says what it opens. This panel used to
            # be 600x118 of permanently reserved canvas for one text field.
            hovered = panel.collidepoint(mouse_pos)
            primitives.elevated_panel(
                self.screen, panel,
                palette.control_hover if hovered else palette.panel,
                radius=self.theme.radius.lg, border=palette.border,
                shadow=palette.shadow, bevel_light=palette.bevel_light,
                bevel_dark=palette.bevel_dark)
            label = self.fonts.ui("small_strong").render(
                "Test a string", True, palette.text)
            self.screen.blit(label, label.get_rect(
                midleft=(panel.x + self.theme.space.md, panel.centery)))
            self._chevron(pygame.Rect(panel.right - 30, panel.centery - 9, 18, 18),
                          palette.text_muted, pointing="up")
            return

        primitives.elevated_panel(self.screen, panel, palette.panel,
                                  radius=self.theme.radius.lg,
                                  border=palette.border, shadow=palette.shadow,
                                  bevel_light=palette.bevel_light,
                                  bevel_dark=palette.bevel_dark)

        collapse = self.input_collapse_rect
        self._chevron(collapse, palette.accent if collapse.collidepoint(mouse_pos)
                      else palette.text_muted, pointing="down")

        # The field is sunken -- the one recessed surface on a screen of raised
        # ones, which is what makes it read as "type here" without a label.
        field = self.input_rect
        primitives.sunken_well(self.screen, field, palette.field,
                               radius=self.theme.radius.md,
                               border=(palette.accent if self.input_active
                                       else palette.border),
                               well_shadow=palette.well_shadow)

        font = self.fonts.mono("input")
        display_text = self.input_text if len(self.input_text) <= 22 else self.input_text[-22:]
        if display_text:
            # Character by character, so a symbol the alphabet does not contain
            # shows red as it is typed -- the mistake is visible before Test is
            # pressed, at the exact position it was made.
            x = field.left + self.theme.space.sm
            for char in display_text:
                valid = char in automaton.alphabet
                glyph = font.render(char, True,
                                    palette.text if valid else palette.error)
                self.screen.blit(glyph, glyph.get_rect(
                    midleft=(x, field.centery)))
                x += glyph.get_width()
            caret_x = min(x + 2, field.right - 5)
        else:
            hint = font.render("epsilon" if self.input_active else "type here",
                               True, palette.text_faint)
            self.screen.blit(hint, hint.get_rect(
                midleft=(field.left + self.theme.space.sm, field.centery)))
            caret_x = field.left + self.theme.space.sm

        if self.input_active and pygame.time.get_ticks() % 1100 < 560:
            pygame.draw.line(self.screen, palette.accent,
                             (caret_x, field.top + 6), (caret_x, field.bottom - 6), 2)

        self._button(self.test_button_rect, "Test", accent=True,
                     hovered=self.test_button_rect.collidepoint(mouse_pos))

        if test_result:
            self._draw_verdict(test_result, panel)

    def _hand_icon(self, rect: pygame.Rect, colour) -> None:
        """A small hand, for the pan tool. Drawn rather than typed: the glyph
        for it is not in every font, and a missing glyph is a blank button."""
        cx, cy = rect.centerx, rect.centery
        palm = pygame.Rect(cx - 5, cy - 1, 10, 8)
        pygame.draw.rect(self.screen, colour, palm, border_radius=3)
        for index in range(3):
            finger = pygame.Rect(cx - 5 + index * 4, cy - 7, 2, 7)
            pygame.draw.rect(self.screen, colour, finger, border_radius=1)
        thumb = pygame.Rect(cx + 4, cy - 3, 3, 5)
        pygame.draw.rect(self.screen, colour, thumb, border_radius=1)

    def _chevron(self, rect: pygame.Rect, colour, *, pointing: str) -> None:
        """A small caret, drawn as three points rather than a glyph so it does
        not depend on a font having the character."""
        mid_x, mid_y = rect.centerx, rect.centery
        reach, drop = 5, 3
        if pointing == "up":
            points = [(mid_x - reach, mid_y + drop), (mid_x, mid_y - drop),
                      (mid_x + reach, mid_y + drop)]
        elif pointing == "down":
            points = [(mid_x - reach, mid_y - drop), (mid_x, mid_y + drop),
                      (mid_x + reach, mid_y - drop)]
        else:  # "right", for a folded side panel
            points = [(mid_x - drop, mid_y - reach), (mid_x + drop, mid_y),
                      (mid_x - drop, mid_y + reach)]
        pygame.draw.lines(self.screen, colour, False, points, 2)

    def _draw_verdict(self, message: str, panel: pygame.Rect) -> None:
        """Show the result of the last run, coloured by its verdict.

        The colour comes from the verdict the engine reported, not from
        searching the message for the word "accepted" -- which used to paint a
        rejection green whenever that word appeared in the user's own input.
        """
        palette = self.theme.palette
        if self.test_verdict == "accept":
            color, mark = palette.success, "ACCEPTED"
        elif self.test_verdict == "no_initial_state":
            color, mark = palette.warning, "NO START STATE"
        elif self.test_verdict:
            color, mark = palette.error, "REJECTED"
        else:
            color, mark = palette.text_muted, ""

        y = self.input_rect.bottom + 10
        if mark:
            badge_font = self.fonts.ui("small_strong")
            badge_text = badge_font.render(mark, True, palette.text_on_accent)
            badge = pygame.Rect(panel.x + self.theme.space.md, y,
                                badge_text.get_width() + 14,
                                badge_text.get_height() + 6)
            primitives.panel(self.screen, badge, color, radius=self.theme.radius.sm)
            self.screen.blit(badge_text, badge_text.get_rect(center=badge.center))
            detail_x = badge.right + self.theme.space.sm
        else:
            detail_x = panel.x + self.theme.space.md

        # The engine's own sentence, trimmed to the panel.
        detail = message
        surface = self.fonts.ui("small").render(detail, True, palette.text_muted)
        available = panel.right - detail_x - self.theme.space.md
        while surface.get_width() > available and len(detail) > 12:
            detail = detail[:-4] + "..."
            surface = self.fonts.ui("small").render(detail, True, palette.text_muted)
        self.screen.blit(surface, (detail_x, y + 2))

    def _draw_alphabet_selector(self):
        """Draw the alphabet selector with available symbols.

        Reads the rectangles computed in _recompute_symbol_buttons rather than
        producing them as a side effect of drawing.
        """
        palette = self.theme.palette
        card = self.symbol_card_rect
        primitives.elevated_panel(self.screen, card, palette.panel,
                                  radius=self.theme.radius.lg,
                                  border=palette.border, shadow=palette.shadow,
                                  bevel_light=palette.bevel_light,
                                  bevel_dark=palette.bevel_dark)

        # The caption rides inside the card, down its left edge, so the palette
        # explains itself without a heading above it claiming another row of
        # height. The chips themselves are the point: what you can draw with is
        # visible at all times, which is why this never collapses.
        caption = self.fonts.ui("small_strong").render(
            "SYMBOL", True, palette.text_faint)
        self.screen.blit(caption, caption.get_rect(
            midleft=(card.x + self.theme.space.sm, card.centery)))

        mouse_pos = pygame.mouse.get_pos()
        mono = self.fonts.mono("input")

        for index, (symbol, button_rect) in enumerate(self.symbol_buttons.items()):
            selected = symbol == self.selected_symbol
            hovered = button_rect.collidepoint(mouse_pos)

            if selected:
                fill, text_color, border = (palette.accent, palette.text_on_accent,
                                            palette.accent)
            elif hovered:
                fill, text_color, border = (palette.control_hover, palette.text,
                                            palette.border_strong)
            else:
                fill, text_color, border = (palette.control, palette.text,
                                            palette.border)

            primitives.panel(self.screen, button_rect, fill,
                             radius=self.theme.radius.md, border=border)
            surface = mono.render(symbol, True, text_color)
            self.screen.blit(surface, surface.get_rect(center=button_rect.center))

            # A hairline in this symbol's edge colour, tying the palette to the
            # arrows it draws.
            swatch = pygame.Rect(button_rect.x + 7, button_rect.bottom - 5,
                                 button_rect.width - 14, 2)
            pygame.draw.rect(self.screen, self.theme.edge_color(index), swatch,
                             border_radius=1)

        add_rect = self.add_symbol_button_rect
        self._button(add_rect, "+", hovered=add_rect.collidepoint(mouse_pos))

    def _draw_panel_frame(self, key: str, rect: pygame.Rect) -> Optional[pygame.Rect]:
        """Draw a right-column panel's card and its header.

        Returns the rectangle left over for the body, or ``None`` when the
        panel is folded down to its header. The header is always drawn and is
        always the click target, so a collapsed panel is a labelled notch
        rather than a thing that has vanished.
        """
        palette = self.theme.palette
        primitives.elevated_panel(self.screen, rect, palette.panel,
                                  radius=self.theme.radius.lg,
                                  border=palette.border, shadow=palette.shadow,
                                  bevel_light=palette.bevel_light,
                                  bevel_dark=palette.bevel_dark)

        header = self.layout.panel_header(rect)
        hovered = header.collidepoint(pygame.mouse.get_pos())
        title = self.fonts.ui("small_strong").render(
            PANEL_TITLES.get(key, key).upper(), True,
            palette.text if hovered else palette.text_faint)
        self.screen.blit(title, title.get_rect(
            midleft=(header.x + self.theme.space.md, header.centery)))

        self._chevron(pygame.Rect(header.right - 28, header.y + 8, 18, 18),
                      palette.text if hovered else palette.text_muted,
                      pointing="right" if self.collapsed.get(key) else "down")

        body = pygame.Rect(rect.x, header.bottom, rect.width,
                           rect.bottom - header.bottom)
        return body if body.height > 6 else None

    def _draw_status_info(self, automaton: "fsa.DFA", panel_rect: pygame.Rect):
        """Draw status information about the current automaton."""
        palette = self.theme.palette
        body = self._draw_panel_frame("status", panel_rect)
        if body is None:
            return

        info_x = body.x + self.theme.space.md
        value_x = info_x + 92

        rows = [
            ("States", str(len(automaton.states)), False),
            ("Alphabet", ", ".join(sorted(automaton.alphabet)) if automaton.alphabet
             else "empty", not automaton.alphabet),
            ("Start", automaton.initial or "none", automaton.initial is None),
            ("Accepting", str(len(automaton.accept)), self.warn_no_accepting),
        ]

        label_font = self.fonts.ui("small")
        value_font = self.fonts.ui("small_strong")
        row_y = body.y + 4
        for label, value, warn in rows:
            self.screen.blit(label_font.render(label, True, palette.text_muted),
                             (info_x, row_y))
            colour = palette.text if value not in ("none", "empty") else palette.text_faint
            if warn:
                colour = palette.warning
            surface = value_font.render(value, True, colour)
            available = body.right - value_x - self.theme.space.md
            while surface.get_width() > available and len(value) > 4:
                value = value[:-4] + "..."
                surface = value_font.render(value, True, colour)
            self.screen.blit(surface, (value_x, row_y))
            row_y += 19

        if self.warn_no_accepting:
            self.screen.blit(
                self.fonts.ui("small").render("No string can be accepted", True,
                                              palette.warning),
                (info_x, row_y))

    def draw_legend(self, automaton: "fsa.DFA") -> None:
        """Explain the state styles, showing only the kinds actually present.

        A diagram that dims some states and hatches others is only useful if
        the reader knows what those mean. Listing every kind all the time would
        be noise, so entries appear as the automaton acquires them.

        Lives in the sliding right column, so it can never collide with the
        other panels -- each takes the space the one above releases.
        """
        rect = next((r for key, r, _t in self._column if key == "legend"), None)
        if rect is None:
            return
        palette = self.theme.palette
        entries = [("normal", palette.state_fill, palette.state_ring, "plain")]

        if automaton.accept:
            entries.append(("accepting", palette.accept_fill,
                            palette.accept_ring, "double"))
        if self.legend_dead:
            entries.append(("trap", palette.dead_fill, palette.dead_ring, "hatch"))
        if self.legend_unreachable:
            entries.append(("unreachable", palette.unreachable_fill,
                            palette.unreachable_ring, "dashed"))

        if len(entries) < 2:
            return

        row_h = 22
        body = self._draw_panel_frame("legend", rect)
        if body is None:
            return

        font = self.fonts.ui("small")
        y = body.y + 3
        for label, fill, ring, style in entries:
            if y + row_h > body.bottom:
                break
            centre = (body.x + self.theme.space.md + 9, y + 7)
            primitives.filled_circle(self.screen, centre, 9, fill)
            if style == "hatch":
                primitives.hatch_circle(self.screen, centre, 8,
                                        palette.dead_hatch, spacing=4, width=1)
            if style == "dashed":
                primitives.dashed_ring(self.screen, centre, 9, 2, ring, dashes=8)
            else:
                primitives.ring(self.screen, centre, 9, 2, ring)
            if style == "double":
                primitives.ring(self.screen, centre, 6, 1, ring)

            self.screen.blit(font.render(label, True, palette.text_muted),
                             (body.x + self.theme.space.md + 26, y))
            y += row_h

    def _draw_diagnostics(self, rect: pygame.Rect) -> None:
        """Structural problems with the automaton, each one actionable.

        Rows with named states can be clicked to jump the camera to them; the
        "incomplete" row carries a Fix button that adds a trap state and routes
        every missing transition to it in one click. The visual feedback *is*
        the lesson: a rejection for want of an arrow is a different problem
        from a wrong language, and this panel is where that becomes concrete.
        """
        palette = self.theme.palette
        body = self._draw_panel_frame("diagnostics", rect)
        if body is None:
            return

        font = self.fonts.ui("tiny")
        row_y = body.y + 2
        for defect in self.diagnostics[:4]:
            if row_y + 36 > body.bottom:
                break
            row = pygame.Rect(body.x + 6, row_y, body.width - 12, 36)

            colour = palette.error if defect.is_blocking else palette.warning
            if defect.kind == "unreachable_states":
                colour = palette.unreachable_ring
            primitives.filled_circle(self.screen, (row.x + 10, row.y + 10), 4,
                                     colour)

            has_fix = defect.kind == "incomplete"
            budget = row.width - 28 - (44 if has_fix else 0)

            # Two wrapped lines rather than one truncated one: a single line
            # always cut exactly the part that named the states and symbols,
            # which is the panel's entire teaching content.
            words = defect.message.split()
            lines: List[str] = []
            current = ""
            for word in words:
                trial = (current + " " + word).strip()
                if font.size(trial)[0] <= budget or not current:
                    current = trial
                else:
                    lines.append(current)
                    current = word
                    if len(lines) == 2:
                        break
            if current and len(lines) < 2:
                lines.append(current)
            if len(lines) == 2 and font.size(lines[1])[0] > budget:
                while lines[1] and font.size(lines[1] + "...")[0] > budget:
                    lines[1] = lines[1][:-2]
                lines[1] += "..."

            for j, text in enumerate(lines):
                self.screen.blit(font.render(text, True, palette.text_muted),
                                 (row.x + 22, row.y + 3 + j * 14))

            if has_fix:
                fix = pygame.Rect(row.right - 40, row.y + 6, 36, 24)
                self._button(fix, "Fix", accent=True,
                             hovered=fix.collidepoint(pygame.mouse.get_pos()))
                self._fix_button = fix
            elif defect.states:
                self._diagnostic_rows.append(
                    (row, {"focus_states": list(defect.states)}))

            row_y += 40

    def _draw_playback_controls(self, x: int, y: int):
        """Playback state and speed, inside the run panel.

        These used to live in the status panel and were drawn whether or not
        anything was running -- a "Paused" dot and a speed slider for an
        animation that did not exist, which is most of what made that panel
        look half empty.
        """
        palette = self.theme.palette
        slider = self.speed_slider_rect
        if slider is None:
            return
        animating = getattr(self, '_animation_active', False)

        dot_colour = palette.success if animating else palette.text_faint
        primitives.filled_circle(self.screen, (x + 4, y + 7), 4, dot_colour)
        label = "Playing" if animating else "Paused"
        self.screen.blit(
            self.fonts.ui("small").render(label, True, palette.text_muted),
            (x + 15, y))

        speed_ratio = (self.animation_speed - SPEED_MIN_MS) / (SPEED_MAX_MS - SPEED_MIN_MS)
        speed_ratio = max(0.0, min(1.0, speed_ratio))
        self.screen.blit(
            self.fonts.ui("small").render(f"{self.animation_speed} ms", True,
                                          palette.text_faint),
            (x + 78, y))

        # Track, filled portion, then handle, following the slid panel.
        track = pygame.Rect(slider.x, slider.centery - 2, slider.width, 4)
        primitives.panel(self.screen, track, palette.control, radius=2)
        filled = pygame.Rect(track.x, track.y, int(track.width * speed_ratio), 4)
        primitives.panel(self.screen, filled, palette.accent, radius=2)

        handle_x = slider.x + speed_ratio * slider.width
        primitives.filled_circle(self.screen, (handle_x, slider.centery), 7,
                                 palette.panel_raised)
        primitives.ring(self.screen, (handle_x, slider.centery), 7, 2,
                        palette.accent)

    def draw_string_visualization(self, test_string: str, current_step: int,
                                  run: Optional[Any] = None) -> None:
        """Draw the input tape: sliding in and out, scrolling, popping.

        The strip glides up from the bottom edge when a run starts and back
        down when it stops -- which needs the *previous* run's content for the
        exit animation, so the last drawn state is cached. The scroll between
        cells eases rather than jumping, and the cell under the read head pops
        briefly when the position changes.
        """
        was_hidden = (self.slides.get("strip") is None
                      or self.slides["strip"].value <= 0.01)
        t = self._slide("strip", self.execution_panel_visible)
        if self.execution_panel_visible:
            self._strip_cache = (test_string, current_step, run)
        elif self._strip_cache is not None:
            test_string, current_step, run = self._strip_cache
        if t <= 0.01:
            self._strip_cache = None
            self._strip_bounds = None
            return

        palette = self.theme.palette
        strip = self.layout.string_strip
        cell_w, cell_h = 34, 40
        gap = 6
        step = cell_w + gap
        count = max(1, len(test_string))
        total = count * step - gap

        # Slide offset: fully below the bottom edge at t=0, in place at t=1.
        # The travel spans the real distance to the window edge; a fixed 74px
        # meant the exit animation stopped mid-screen and the strip blinked out.
        rise = (1.0 - t) * (self.screen_height - strip.y + 8)
        top = strip.y + int(rise)

        # The pop restarts whenever the read head moves.
        if current_step != self._strip_last_step:
            self._strip_last_step = current_step
            self.strip_pop.start()

        # The drawable span stops where the right column begins, so long
        # strings scroll instead of painting cells across the diagnostics
        # panel.
        left_bound = 40
        right_bound = self.screen_width - 40
        if self._column:
            right_bound = min(right_bound,
                              min(rect.x for _k, rect, _t in self._column) - 12)

        centre_x = (left_bound + right_bound) // 2
        span = right_bound - left_bound
        if total <= span:
            target_x = centre_x - total // 2
        else:
            wanted = centre_x - int((current_step + 0.5) * step)
            target_x = max(right_bound - total, min(left_bound, wanted))

        # Ease toward the target, but jump on the frame the strip first
        # appears. The old guard tested t < 0.05, which one 16ms update has
        # already passed, so the strip visibly travelled in from x=0.
        if was_hidden:
            self.strip_scroll.jump_to(float(target_x))
        else:
            self.strip_scroll.set(float(target_x))
        start_x = int(self.strip_scroll.value)

        stopped_at = getattr(run, "stopped_at", len(test_string))

        if not test_string:
            rect = pygame.Rect(centre_x - 30, top, 60, cell_h)
            primitives.panel(self.screen, rect, palette.strip_cell,
                             radius=self.theme.radius.md, border=palette.border)
            glyph = self.fonts.ui("body").render("ε", True, palette.text_muted)
            self.screen.blit(glyph, glyph.get_rect(center=rect.center))
            return

        self._strip_bounds = pygame.Rect(
            max(left_bound - 8, min(start_x, right_bound) - 8), top - 12,
            min(total, span) + 16, cell_h + 24)

        mono = self.fonts.mono("strip")
        pop = 1.0 - self.strip_pop.progress
        for i, char in enumerate(test_string):
            x = start_x + i * step
            if x < -step or x + cell_w > right_bound + cell_w // 2:
                continue

            consumed = i < current_step
            unreached = i >= stopped_at and stopped_at < len(test_string)
            is_current = i == current_step

            rect = pygame.Rect(x, top, cell_w, cell_h)
            if is_current and pop > 0.01:
                grow = int(4 * pop)
                rect = rect.inflate(grow, grow)

            if is_current:
                fill, text_color = palette.strip_cell_current, palette.text_on_accent
            elif consumed:
                fill, text_color = palette.strip_cell_done, palette.strip_text_done
            else:
                fill, text_color = palette.strip_cell, palette.strip_text
            if unreached and not is_current:
                text_color = palette.text_faint

            primitives.elevated_panel(
                self.screen, rect, fill, radius=self.theme.radius.md,
                border=palette.accent if is_current else palette.border,
                shadow=palette.shadow, lift=3 if is_current else 2,
                bevel_light=palette.bevel_light, bevel_dark=palette.bevel_dark)
            glyph = mono.render(char, True, text_color)
            self.screen.blit(glyph, glyph.get_rect(center=rect.center))

            if is_current:
                # The read head: a marker above the cell, pointing at it.
                primitives.pointer(self.screen, (rect.centerx, rect.top - 4),
                                   8, palette.accent, direction="down")

            if consumed:
                pygame.draw.line(self.screen, palette.success,
                                 (rect.x + 8, rect.bottom + 3),
                                 (rect.right - 8, rect.bottom + 3), 2)

        if stopped_at < len(test_string):
            marker_x = start_x + stopped_at * step - gap // 2
            pygame.draw.line(self.screen, palette.error,
                             (marker_x, top - 4), (marker_x, top + cell_h + 4), 2)

    def _draw_help_panel(self):
        """Draw the help panel with scrollable content.

        Both the geometry and the line count come from the layout and from
        HELP_LINES, which is also what the scroll handler reads. They used to be
        independent constants that disagreed.
        """
        palette = self.theme.palette
        panel_rect = self.layout.help_panel
        primitives.elevated_panel(self.screen, panel_rect, palette.panel_raised,
                                  radius=self.theme.radius.lg,
                                  border=palette.border_strong,
                                  shadow=palette.shadow, lift=6,
                                  bevel_light=palette.bevel_light,
                                  bevel_dark=palette.bevel_dark)

        # Title
        title_text = self.fonts.ui("heading").render("Controls", True, palette.text)
        title_rect = title_text.get_rect(centerx=panel_rect.centerx, y=panel_rect.y + 10)
        self.screen.blit(title_text, title_rect)

        content_y = panel_rect.y + HELP_TITLE_HEIGHT
        visible_lines = self.layout.help_visible_lines()

        start_line = max(0, min(self.help_scroll_offset,
                                max(0, len(HELP_LINES) - visible_lines)))
        end_line = min(len(HELP_LINES), start_line + visible_lines)

        for i in range(start_line, end_line):
            line = HELP_LINES[i]
            display_y = content_y + (i - start_line) * HELP_LINE_HEIGHT
            # Bullets and their continuation lines share one inset; headers sit
            # left of both. Continuations used to lose their leading spaces to
            # lstrip and outdent past their own bullet.
            if line.startswith("-") or line.startswith(" "):
                text_x = panel_rect.x + 28
                text = line.lstrip("- ").strip()
            else:
                text_x = panel_rect.x + 16
                text = line
            is_heading = line.endswith(":")
            font = self.fonts.ui("small_strong" if is_heading else "small")
            colour = palette.text if is_heading else palette.text_muted
            text_surface = font.render(text, True, colour)
            self.screen.blit(text_surface, (text_x, display_y))

        # Scrollbar
        if len(HELP_LINES) > visible_lines:
            scrollbar_x = panel_rect.right - 15
            scrollbar_height = visible_lines * HELP_LINE_HEIGHT
            scrollbar_rect = pygame.Rect(scrollbar_x, content_y, 6, scrollbar_height)
            primitives.panel(self.screen, scrollbar_rect, palette.control, radius=3)

            thumb_height = max(20, int(scrollbar_height * visible_lines / len(HELP_LINES)))
            thumb_y = content_y + int((scrollbar_height - thumb_height) * start_line
                                      / (len(HELP_LINES) - visible_lines))
            thumb_rect = pygame.Rect(scrollbar_x, thumb_y, 6, thumb_height)
            primitives.panel(self.screen, thumb_rect, palette.border_strong, radius=3)

            footer = self.fonts.ui("small").render("Scroll for more", True,
                                                   palette.text_faint)
            self.screen.blit(footer, footer.get_rect(centerx=panel_rect.centerx,
                                                     y=panel_rect.bottom - 20))

    def _draw_context_menu(self):
        """Draw the context menu if visible."""
        if not self.context_menu or not self.context_menu.visible:
            return

        menu_x, menu_y = self.context_menu.position
        item_height = CONTEXT_MENU_ITEM_HEIGHT
        menu_width = CONTEXT_MENU_WIDTH

        palette = self.theme.palette
        menu_rect = self._context_menu_rect()
        # A rectangular shadow under a rectangular menu. The circular
        # soft_shadow used before bulged out below the bottom edge as a dark
        # disc.
        primitives.elevated_panel(self.screen, menu_rect, palette.panel_raised,
                                  radius=self.theme.radius.md,
                                  border=palette.border_strong,
                                  shadow=palette.shadow, lift=5,
                                  bevel_light=palette.bevel_light,
                                  bevel_dark=palette.bevel_dark)

        # Menu items
        mouse_pos = pygame.mouse.get_pos()
        for i, item in enumerate(self.context_menu.items):
            label, _action = item[0], item[1]
            checked = item[2] if len(item) > 2 else None
            item_y = menu_y + i * item_height
            item_rect = pygame.Rect(menu_x, item_y, menu_width, item_height)

            # Highlight hovered item
            if item_rect.collidepoint(mouse_pos) and label != "---":
                primitives.panel(self.screen, item_rect.inflate(-6, -2),
                                 palette.control_hover, radius=self.theme.radius.sm)
                self.context_menu.selected_index = i

            # Separator line
            if label == "---":
                line_y = item_y + item_height // 2
                pygame.draw.line(self.screen, palette.border,
                               (menu_x + 10, line_y), (menu_x + menu_width - 10, line_y))
            else:
                colour = palette.error if label.startswith("Delete") else palette.text
                text_surface = self.fonts.ui("small").render(label, True, colour)
                text_rect = text_surface.get_rect(
                    midleft=(menu_x + CONTEXT_MENU_GUTTER, item_y + item_height // 2))
                self.screen.blit(text_surface, text_rect)

                if checked is not None:
                    centre = (menu_x + 15, item_y + item_height // 2)
                    if checked:
                        primitives.filled_circle(self.screen, centre, 5, palette.accent)
                        primitives.filled_circle(self.screen, centre, 2,
                                                 palette.panel_raised)
                    else:
                        primitives.ring(self.screen, centre, 5, 1, palette.border_strong)

    def _handle_context_menu_click(self, mouse_pos: Tuple[int, int]) -> Optional[str]:
        """Handle clicks on context menu items."""
        if not self.context_menu:
            return None

        menu_x, menu_y = self.context_menu.position

        for i, item in enumerate(self.context_menu.items):
            label, action = item[0], item[1]
            item_y = menu_y + i * CONTEXT_MENU_ITEM_HEIGHT
            item_rect = pygame.Rect(menu_x, item_y, CONTEXT_MENU_WIDTH,
                                    CONTEXT_MENU_ITEM_HEIGHT)

            if item_rect.collidepoint(mouse_pos) and label != "---":
                return action

        return None

    def show_context_menu(self, position: Tuple[int, int],
                          items: List[Tuple[Any, ...]]):
        """
        Show a context menu at the specified position, nudged to fit on screen.

        Drawing and hit-testing both derive their rows from this one stored
        position, so a row pushed past the bottom edge is not merely invisible:
        no mouse position can ever land on it, and there is no keyboard
        fallback. Right-clicking low on the canvas would otherwise lose the
        last items of the menu -- "Delete state" among them.

        Args:
            position: (x, y) position to show the menu
            items: List of (label, action) tuples
        """
        height = len(items) * CONTEXT_MENU_ITEM_HEIGHT
        margin = CONTEXT_MENU_MARGIN
        x, y = position
        # Clamped rather than flipped: the menu stays under the pointer, and a
        # menu taller than the window still starts at the top, showing as much
        # as there is room for instead of hanging off both ends. The margin
        # keeps a nudged menu from sitting flush against the window edge.
        x = max(margin, min(x, self.screen_width - CONTEXT_MENU_WIDTH - margin))
        y = max(margin, min(y, self.screen_height - height - margin))
        self.context_menu = ContextMenu((x, y), items)

    def hide_context_menu(self):
        """Hide the context menu."""
        self.context_menu = None

    def _context_menu_rect(self) -> pygame.Rect:
        """Bounding box of the open context menu."""
        if not self.context_menu:
            return pygame.Rect(0, 0, 0, 0)
        menu_x, menu_y = self.context_menu.position
        return pygame.Rect(menu_x, menu_y, CONTEXT_MENU_WIDTH,
                           len(self.context_menu.items) * CONTEXT_MENU_ITEM_HEIGHT)

    def _symbol_dialog_buttons(self) -> Tuple[pygame.Rect, pygame.Rect]:
        """Cancel and Add rectangles for the add-symbol dialog.

        Computed rather than recorded during drawing, so the buttons are live on
        the frame the dialog opens instead of the frame after.
        """
        width, height = 300, 180
        x = (self.screen_width - width) // 2
        y = (self.screen_height - height) // 2
        return (pygame.Rect(x + 50, y + 130, 80, 25),
                pygame.Rect(x + 170, y + 130, 80, 25))

    def can_add_symbol(self, symbol: str) -> bool:
        """Whether a symbol could be added to the alphabet.

        Delegates to the engine rather than keeping its own rules, and no
        longer reserves letters. `q`, `w`, `r`, `n` and `p` used to be rejected
        because keyboard shortcuts owned them, which meant no automaton over an
        alphabet containing those letters could be built at all.
        """
        return (fsa.is_legal_symbol(symbol)
                and symbol not in self.available_symbols)

    def draw_execution_status(self, execution_active: bool, execution_step: int,
                              _execution_string: str, execution_path: List[str],
                              run: Optional[Any] = None):
        """
        Draw the execution trace panel.

        Positions are counted in *transitions taken*, against the length of the
        run. The old panel counted the current index against the length of the
        input string, so it reported "Step 3/5" for a run that had halted after
        two symbols, and never said why it stopped.

        Args:
            execution_active: Whether execution visualization is active
            execution_step: Position in the run
            execution_string: String being processed
            execution_path: States visited
            run: The engine's record of the run, if there is one
        """
        if not execution_active:
            return
        panel_rect = next((r for key, r, _t in self._column if key == "run"), None)
        if panel_rect is None:
            return

        palette = self.theme.palette
        body = self._draw_panel_frame("run", panel_rect)
        if body is None:
            return

        x = body.x + self.theme.space.md
        y = body.y + 2

        total_steps = max(0, len(execution_path) - 1)
        position = f"{execution_step} / {total_steps}"
        pos_surface = self.fonts.ui("small_strong").render(position, True,
                                                           palette.text_muted)
        self.screen.blit(pos_surface, (body.right - self.theme.space.md
                                       - pos_surface.get_width(), y))

        # Progress bar across the run.
        track = pygame.Rect(x, y + 20, body.width - self.theme.space.md * 2, 4)
        primitives.panel(self.screen, track, palette.control, radius=2)
        if total_steps:
            done = pygame.Rect(track.x, track.y,
                               int(track.width * execution_step / total_steps), 4)
            primitives.panel(self.screen, done, palette.accent, radius=2)

        current_state = (execution_path[execution_step]
                         if execution_step < len(execution_path) else "-")
        state_line = self.fonts.ui("body_strong").render(
            f"in {current_state}", True, palette.text)
        self.screen.blit(state_line, (x, y + 32))

        steps = getattr(run, "steps", ()) or ()
        verdict = getattr(run, "verdict", None)
        if execution_step < len(steps):
            step = steps[execution_step]
            detail = f"next: read '{step.symbol}' to {step.target}"
        elif verdict is not None:
            detail = str(verdict.value).replace("_", " ")
        else:
            detail = "run complete"
        self.screen.blit(
            self.fonts.ui("small").render(detail, True, palette.text_muted),
            (x, y + 54))

        hint = "N next   P back   Tab play   Esc stop"
        self.screen.blit(
            self.fonts.ui("small").render(hint, True, palette.text_faint),
            (x, y + 74))

        # Below the hint, not on top of it: both were positioned from opposite
        # ends of the panel and met in the middle.
        self._draw_playback_controls(x, y + 96)

    def _draw_add_symbol_dialog(self):
        """The add-symbol dialog, on the same modal frame as its siblings.

        It predated the elevation pass and looked like a different application
        next to the Save dialog: square corners, hard borders, flat buttons,
        and no dimmed backdrop despite being modal to the keyboard.
        """
        palette = self.theme.palette
        rect = self._draw_modal_frame(300, 180, "Add a symbol")

        hint = self.fonts.ui("small").render(
            "One printable character.", True, palette.text_muted)
        self.screen.blit(hint, (rect.x + 20, rect.y + 48))

        field = pygame.Rect(rect.x + 20, rect.y + 68, 260, 32)
        primitives.sunken_well(self.screen, field, palette.field,
                               radius=self.theme.radius.md,
                               border=palette.accent,
                               well_shadow=palette.well_shadow)

        if self.new_symbol_input:
            glyph = self.fonts.mono("input").render(
                self.new_symbol_input, True, palette.text)
            self.screen.blit(glyph, glyph.get_rect(
                midleft=(field.left + 8, field.centery)))
            caret_x = field.left + 8 + glyph.get_width() + 2
        else:
            caret_x = field.left + 8

        if pygame.time.get_ticks() % 1100 < 560:
            pygame.draw.line(self.screen, palette.accent,
                             (caret_x, field.top + 6),
                             (caret_x, field.bottom - 6), 2)

        cancel, add = self._symbol_dialog_buttons()
        mouse_pos = pygame.mouse.get_pos()
        self._button(cancel, "Cancel",
                     hovered=cancel.collidepoint(mouse_pos))
        self._button(add, "Add", accent=True,
                     hovered=add.collidepoint(mouse_pos))

        footer = self.fonts.ui("small").render(
            "Enter to add, Escape to cancel", True, palette.text_faint)
        self.screen.blit(footer, footer.get_rect(
            centerx=rect.centerx, y=rect.bottom - 24))
