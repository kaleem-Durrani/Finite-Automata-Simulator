"""
UI Manager module for handling user interface elements.

This module manages all UI components including toolbars, input fields,
context menus, and help panels.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pygame

from core.dfa import DFA
from rendering import primitives
from rendering.fonts import FontBook
from rendering.theme import Theme
from ui.layout_spec import (
    HELP_LINE_HEIGHT,
    HELP_TITLE_HEIGHT,
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
    "- Right Click: Context menu",
    "- Middle Click+Drag: Pan view",
    "- Scroll Wheel: Zoom",
    "",
    "Keyboard Shortcuts:",
    "- Space: Add state at center",
    "- Delete: Remove selected state",
    "- Q: Toggle accept state",
    "- W: Toggle dead end state",
    "- R: Reset camera view",
    "",
    "Creating Transitions:",
    "- Select symbol from toolbar",
    "- Shift+click source state",
    "- Click target state",
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


CONTEXT_MENU_WIDTH = 150
CONTEXT_MENU_ITEM_HEIGHT = 25


@dataclass
class ContextMenu:
    """Represents a context menu with items and position."""
    position: Tuple[int, int]
    items: List[Tuple[str, str]]  # (label, action)
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
        
        # Available symbols for transitions (can be modified by user)
        self.available_symbols = ['a', 'b', '0', '1']
        self.system_keys = {'n', 'p', 'escape', 'space', 'delete', 'q', 'w', 'r'}  # Reserved keys

        # Symbol addition dialog state
        self.adding_symbol = False
        self.new_symbol_input = ""

        # Filename prompt state ('save' | 'load' | None)
        self.file_prompt_mode: Optional[str] = None
        self.file_prompt_text = ""

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
        """Draw a labelled button in one of its three states."""
        palette = self.theme.palette
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

        primitives.panel(self.screen, rect, fill, radius=self.theme.radius.md,
                         border=border)
        surface = self.fonts.ui("body_strong").render(label, True, text_color)
        self.screen.blit(surface, surface.get_rect(center=rect.center))

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
        self.symbol_buttons = {
            symbol: self.layout.symbol_button(index)
            for index, symbol in enumerate(self.available_symbols)
        }
        self.add_symbol_button_rect = self.layout.symbol_button(len(self.available_symbols))

    # Named rectangles are read straight from the layout so that drawing and
    # hit-testing cannot drift apart.
    @property
    def input_rect(self) -> pygame.Rect:
        return self.layout.input_field

    @property
    def test_button_rect(self) -> pygame.Rect:
        return self.layout.test_button

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
    def speed_slider_rect(self) -> pygame.Rect:
        return self.layout.speed_slider

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
            return self._handle_key_down(event), self.is_keyboard_captured()
        if event.type == pygame.KEYUP:
            return self._handle_key_up(event), False
        if event.type == pygame.MOUSEWHEEL:
            return self._handle_mouse_wheel(event)

        return {}, False

    # ------------------------------------------------------------------
    # Hit testing
    # ------------------------------------------------------------------

    def opaque_panels(self) -> List[pygame.Rect]:
        """The UI regions currently painted over the canvas."""
        return self.layout.opaque_panels(
            execution_active=self.execution_panel_visible,
            help_open=self.show_help,
        )

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
            (self.layout.test_button, self._on_test_button),
            (self.layout.input_field, self._on_input_field),
            (self.layout.speed_slider, self._on_speed_slider),
            (self.add_symbol_button_rect, self._on_add_symbol_button),
        ]
        for symbol, rect in self.symbol_buttons.items():
            hits.append((rect, self._symbol_handler(symbol)))
        return [(rect, handler) for rect, handler in hits if rect is not None]

    # -- widget handlers ------------------------------------------------

    def _on_help_button(self, _pos) -> Dict[str, Any]:
        self.show_help = not self.show_help
        self.help_scroll_offset = 0
        return {'toggle_help': True}

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
        slider = self.layout.speed_slider
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
                self.input_active = rect is self.layout.input_field
                return handler(pos), True

        # Not a widget: clicking anywhere else drops text focus.
        self.input_active = False

        # A click on a panel with no widget under it is still the UI's.
        return actions, self.is_over_ui(pos)

    def _handle_mouse_up(self, event) -> Tuple[Dict[str, Any], bool]:
        """Release the speed slider."""
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

    def sync_symbols_with(self, dfa: DFA):
        """
        Make sure every symbol used by the automaton appears in the palette.

        The palette and the automaton's alphabet are still two separate things
        (they are unified in a later phase), so loading a file over a different
        alphabet would otherwise leave its symbols undrawable.
        """
        for symbol in sorted(dfa.alphabet):
            if symbol not in self.available_symbols:
                self.available_symbols.append(symbol)
        if self.selected_symbol not in self.available_symbols and self.available_symbols:
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
            if name:
                actions['save_to_path' if mode == 'save' else 'load_to_path'] = name
            else:
                actions['file_prompt_cancel'] = True
        elif event.key == pygame.K_BACKSPACE:
            self.file_prompt_text = self.file_prompt_text[:-1]
        elif event.unicode.isprintable() and len(self.file_prompt_text) < 120:
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
                # Try to add the symbol
                if self.new_symbol_input and self.add_symbol(self.new_symbol_input):
                    actions['symbol_added'] = self.new_symbol_input
                    self.adding_symbol = False
                    self.new_symbol_input = ""
                else:
                    actions['symbol_add_error'] = "Invalid symbol or already exists"
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

    def draw(self, dfa: DFA, test_result: str = "", animation_active: bool = False):
        """
        Draw all UI elements.

        Args:
            dfa: The current DFA for displaying information
            test_result: Result of the last string test
            animation_active: Whether playback is running. Passed in rather than
                pushed onto the manager after draw() has already run, which made
                the indicator show the previous frame's state.
        """
        self._animation_active = animation_active
        self._draw_toolbar()
        self._draw_input_area(test_result)
        self._draw_alphabet_selector(dfa)
        self._draw_status_info(dfa)

        if self.show_help:
            self._draw_help_panel()

        if self.context_menu and self.context_menu.visible:
            self._draw_context_menu()

        if self.adding_symbol:
            self._draw_add_symbol_dialog()

        # Modal dialogs paint last so nothing is drawn over them.
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
        primitives.soft_shadow(self.screen, rect.center, rect.width / 2,
                               palette.shadow, layers=4, spread=10)
        primitives.panel(self.screen, rect, palette.panel_raised,
                         radius=self.theme.radius.lg, border=palette.border_strong)

        title_surface = self.fonts.ui("heading").render(title, True, palette.text)
        self.screen.blit(title_surface, title_surface.get_rect(
            centerx=rect.centerx, y=rect.y + self.theme.space.lg))

        return rect

    def _draw_file_prompt(self):
        """Draw the filename prompt."""
        verb = "Save as" if self.file_prompt_mode == 'save' else "Load file"
        rect = self._draw_modal_frame(440, 160, verb)

        palette = self.theme.palette
        hint = self.fonts.ui("small").render(
            "Relative to the project folder. '.json' is added if omitted.",
            True, palette.text_muted)
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
        self._button(self.theme_button_rect,
                     "Light" if self.theme.is_dark else "Dark",
                     hovered=self.theme_button_rect.collidepoint(mouse_pos))
        self._button(self.load_button_rect, "Load",
                     hovered=self.load_button_rect.collidepoint(mouse_pos))
        self._button(self.save_button_rect, "Save",
                     hovered=self.save_button_rect.collidepoint(mouse_pos))
        self._button(self.help_button_rect, "Help", active=self.show_help,
                     hovered=self.help_button_rect.collidepoint(mouse_pos))

    def _draw_input_area(self, test_result: str):
        """Draw the input area for testing strings."""
        palette = self.theme.palette
        panel = self.layout.input_panel
        primitives.panel(self.screen, panel, palette.panel,
                         radius=self.theme.radius.lg, border=palette.border)

        self._section_label("Test a string",
                            (panel.x + self.theme.space.md, panel.y + 10))

        # Input field. Focus is shown with an accent ring rather than a colour
        # change, which reads at a glance without moving anything.
        field = self.input_rect
        primitives.panel(self.screen, field, palette.field,
                         radius=self.theme.radius.md,
                         border=palette.accent if self.input_active else palette.border,
                         border_width=2 if self.input_active else 1)

        font = self.fonts.mono("input")
        display_text = self.input_text if len(self.input_text) <= 22 else self.input_text[-22:]
        if display_text:
            text_surface = font.render(display_text, True, palette.text)
        else:
            text_surface = font.render("epsilon" if self.input_active else "type here",
                                       True, palette.text_faint)
        text_rect = text_surface.get_rect(
            midleft=(field.left + self.theme.space.sm, field.centery))
        self.screen.blit(text_surface, text_rect)

        if self.input_active and pygame.time.get_ticks() % 1100 < 560:
            caret_x = (text_rect.right + 2) if display_text else (field.left + self.theme.space.sm)
            pygame.draw.line(self.screen, palette.accent,
                             (caret_x, field.top + 6), (caret_x, field.bottom - 6), 2)

        mouse_pos = pygame.mouse.get_pos()
        self._button(self.test_button_rect, "Test", accent=True,
                     hovered=self.test_button_rect.collidepoint(mouse_pos))

        if test_result:
            self._draw_verdict(test_result, panel)

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

    def _draw_alphabet_selector(self, _dfa: DFA):
        """Draw the alphabet selector with available symbols.

        Reads the rectangles computed in _recompute_symbol_buttons rather than
        producing them as a side effect of drawing.
        """
        palette = self.theme.palette
        panel = self.layout.symbol_panel
        pygame.draw.rect(self.screen, palette.panel, panel)
        pygame.draw.line(self.screen, palette.border,
                         (0, panel.bottom - 1), (panel.right, panel.bottom - 1))

        self._section_label("Transition symbol",
                            (self.layout.symbol_row_origin[0], panel.y + 8))

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

    def _draw_status_info(self, dfa: DFA):
        """Draw status information about the current automaton."""
        palette = self.theme.palette
        panel_rect = self.layout.status_panel
        primitives.panel(self.screen, panel_rect, palette.panel,
                         radius=self.theme.radius.lg, border=palette.border)

        info_x = panel_rect.x + self.theme.space.md
        info_y = panel_rect.y + self.theme.space.md
        value_x = info_x + 92

        self._section_label("Automaton", (info_x, info_y))

        rows = [
            ("States", str(len(dfa.states))),
            ("Alphabet", ", ".join(sorted(dfa.alphabet)) if dfa.alphabet else "empty"),
            ("Start", dfa.initial_state or "none"),
            ("Accepting", str(len(dfa.accept_states))),
        ]

        label_font = self.fonts.ui("small")
        value_font = self.fonts.ui("small_strong")
        row_y = info_y + 20
        for label, value in rows:
            self.screen.blit(label_font.render(label, True, palette.text_muted),
                             (info_x, row_y))
            colour = palette.text if value not in ("none", "empty") else palette.text_faint
            surface = value_font.render(value, True, colour)
            available = panel_rect.right - value_x - self.theme.space.md
            while surface.get_width() > available and len(value) > 4:
                value = value[:-4] + "..."
                surface = value_font.render(value, True, colour)
            self.screen.blit(surface, (value_x, row_y))
            row_y += 19

        self._draw_animation_controls_in_status(info_x, row_y + 8)
        self._draw_legend(dfa, panel_rect)

    def _draw_legend(self, dfa: DFA, above: pygame.Rect) -> None:
        """Explain the state styles, showing only the kinds actually present.

        A diagram that dims some states and hatches others is only useful if
        the reader knows what those mean. Listing every kind all the time would
        be noise, so entries appear as the automaton acquires them.
        """
        palette = self.theme.palette
        entries = [("normal", palette.state_fill, palette.state_ring, "plain")]

        if dfa.accept_states:
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
        rect = pygame.Rect(above.x, above.bottom + 10, above.width,
                           row_h * len(entries) + 30)
        primitives.panel(self.screen, rect, palette.panel,
                         radius=self.theme.radius.lg, border=palette.border)
        self._section_label("Legend", (rect.x + self.theme.space.md,
                                       rect.y + self.theme.space.md))

        font = self.fonts.ui("small")
        y = rect.y + 32
        for label, fill, ring, style in entries:
            centre = (rect.x + self.theme.space.md + 9, y + 7)
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
                             (rect.x + self.theme.space.md + 26, y))
            y += row_h

    def _draw_animation_controls_in_status(self, x: int, y: int):
        """Playback speed, inside the status panel."""
        palette = self.theme.palette
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

        # Track, filled portion, then handle. The rectangle comes from the
        # layout so the handle can be grabbed before the panel is first drawn.
        slider = self.layout.speed_slider
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
                                  run: Optional[Any] = None):
        """Draw the input tape, with the read head under the current symbol.

        The head slides between cells rather than jumping, and symbols past the
        point where the run stopped are dimmed and marked, so a rejection shows
        both where it happened and how much of the word was never reached.
        """
        palette = self.theme.palette
        strip = self.layout.string_strip

        cell_w, cell_h = 34, 40
        gap = 4
        step = cell_w + gap
        count = max(1, len(test_string))
        total = count * step - gap

        centre_x = self.screen_width // 2
        if total <= self.screen_width - 80:
            start_x = centre_x - total // 2
        else:
            target = centre_x - int((current_step + 0.5) * step)
            start_x = max(self.screen_width - total - 40, min(40, target))

        top = strip.y
        stopped_at = getattr(run, "stopped_at", len(test_string))

        # The empty word still deserves a cell, labelled.
        if not test_string:
            rect = pygame.Rect(centre_x - 30, top, 60, cell_h)
            primitives.panel(self.screen, rect, palette.strip_cell,
                             radius=self.theme.radius.md, border=palette.border)
            surface = self.fonts.ui("body").render("ε", True, palette.text_muted)
            self.screen.blit(surface, surface.get_rect(center=rect.center))
            return

        mono = self.fonts.mono("strip")
        for i, char in enumerate(test_string):
            x = start_x + i * step
            if x < -step or x > self.screen_width:
                continue

            consumed = i < current_step
            unreached = i >= stopped_at and stopped_at < len(test_string)
            is_current = i == current_step

            rect = pygame.Rect(x, top, cell_w, cell_h)
            if is_current:
                fill, text_color = palette.strip_cell_current, palette.text_on_accent
            elif consumed:
                fill, text_color = palette.strip_cell_done, palette.strip_text_done
            else:
                fill, text_color = palette.strip_cell, palette.strip_text

            if unreached and not is_current:
                text_color = palette.text_faint

            primitives.panel(self.screen, rect, fill, radius=self.theme.radius.md,
                             border=palette.accent if is_current else palette.border)
            surface = mono.render(char, True, text_color)
            self.screen.blit(surface, surface.get_rect(center=rect.center))

            # A tick under everything the machine actually read.
            if consumed:
                pygame.draw.line(self.screen, palette.success,
                                 (rect.x + 8, rect.bottom + 3),
                                 (rect.right - 8, rect.bottom + 3), 2)

        # The point the run halted, if it stopped short.
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
        primitives.panel(self.screen, panel_rect, palette.panel_raised,
                         radius=self.theme.radius.lg, border=palette.border_strong)

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
            text_x = panel_rect.x + (28 if line.startswith("-") else 16)
            is_heading = line.endswith(":")
            font = self.fonts.ui("small_strong" if is_heading else "small")
            colour = palette.text if is_heading else palette.text_muted
            text_surface = font.render(line.lstrip("- "), True, colour)
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
        primitives.soft_shadow(self.screen, menu_rect.center, menu_rect.width / 2,
                               palette.shadow, layers=3, spread=6)
        primitives.panel(self.screen, menu_rect, palette.panel_raised,
                         radius=self.theme.radius.md, border=palette.border_strong)

        # Menu items
        mouse_pos = pygame.mouse.get_pos()
        for i, (label, action) in enumerate(self.context_menu.items):
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
                text_rect = text_surface.get_rect(midleft=(menu_x + 10, item_y + item_height // 2))
                self.screen.blit(text_surface, text_rect)

    def _handle_context_menu_click(self, mouse_pos: Tuple[int, int]) -> Optional[str]:
        """Handle clicks on context menu items."""
        if not self.context_menu:
            return None

        menu_x, menu_y = self.context_menu.position

        for i, (label, action) in enumerate(self.context_menu.items):
            item_y = menu_y + i * CONTEXT_MENU_ITEM_HEIGHT
            item_rect = pygame.Rect(menu_x, item_y, CONTEXT_MENU_WIDTH,
                                    CONTEXT_MENU_ITEM_HEIGHT)

            if item_rect.collidepoint(mouse_pos) and label != "---":
                return action

        return None

    def show_context_menu(self, position: Tuple[int, int], items: List[Tuple[str, str]]):
        """
        Show a context menu at the specified position.

        Args:
            position: (x, y) position to show the menu
            items: List of (label, action) tuples
        """
        self.context_menu = ContextMenu(position, items)

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

    def add_symbol(self, symbol: str) -> bool:
        """
        Add a new symbol to the available symbols.

        Args:
            symbol: Single character symbol to add

        Returns:
            True if added successfully, False if invalid or already exists
        """
        # Validate symbol
        if (len(symbol) != 1 or
            symbol in self.available_symbols or
            symbol.lower() in self.system_keys or
            symbol.upper() in self.system_keys or
            not symbol.isprintable() or
            symbol.isspace()):
            return False

        self.available_symbols.append(symbol)
        self._recompute_symbol_buttons()
        return True

    def remove_symbol(self, symbol: str) -> bool:
        """
        Remove a symbol from available symbols.

        Args:
            symbol: Symbol to remove

        Returns:
            True if removed successfully, False if not found
        """
        if symbol in self.available_symbols and len(self.available_symbols) > 1:
            self.available_symbols.remove(symbol)
            # Change selected symbol if it was removed
            if self.selected_symbol == symbol:
                self.selected_symbol = self.available_symbols[0]
            return True
        return False

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
        self.execution_panel_visible = execution_active
        if not execution_active:
            return

        palette = self.theme.palette
        panel_rect = self.layout.execution_panel
        primitives.panel(self.screen, panel_rect, palette.panel,
                         radius=self.theme.radius.lg, border=palette.border)

        x = panel_rect.x + self.theme.space.md
        y = panel_rect.y + self.theme.space.md
        self._section_label("Run", (x, y))

        total_steps = max(0, len(execution_path) - 1)
        position = f"{execution_step} / {total_steps}"
        pos_surface = self.fonts.ui("small_strong").render(position, True,
                                                           palette.text_muted)
        self.screen.blit(pos_surface, (panel_rect.right - self.theme.space.md
                                       - pos_surface.get_width(), y))

        # Progress bar across the run.
        track = pygame.Rect(x, y + 20, panel_rect.width - self.theme.space.md * 2, 4)
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
            (x, panel_rect.bottom - 24))

    def _draw_add_symbol_dialog(self):
        """Draw the add symbol dialog."""
        # Dialog dimensions - increased height to prevent overlap
        dialog_width = 300
        dialog_height = 180
        dialog_x = (self.screen_width - dialog_width) // 2
        dialog_y = (self.screen_height - dialog_height) // 2

        # Background
        dialog_rect = pygame.Rect(dialog_x, dialog_y, dialog_width, dialog_height)
        pygame.draw.rect(self.screen, self.colors['ui_bg'], dialog_rect)
        pygame.draw.rect(self.screen, self.colors['ui_border'], dialog_rect, 3)

        # Title
        title_text = self.font_medium.render("Add New Symbol", True, self.colors['text'])
        title_rect = title_text.get_rect(centerx=dialog_x + dialog_width // 2, y=dialog_y + 15)
        self.screen.blit(title_text, title_rect)

        # Instructions
        instruction_text = "Enter a single character:"
        instruction_surface = self.font_small.render(instruction_text, True, self.colors['text'])
        self.screen.blit(instruction_surface, (dialog_x + 20, dialog_y + 45))

        # Input field
        input_rect = pygame.Rect(dialog_x + 20, dialog_y + 65, 260, 30)
        pygame.draw.rect(self.screen, self.colors['input_active'], input_rect)
        pygame.draw.rect(self.screen, self.colors['ui_border'], input_rect, 2)

        # Input text
        if self.new_symbol_input:
            text_surface = self.font_medium.render(self.new_symbol_input, True, self.colors['text'])
            text_rect = text_surface.get_rect(midleft=(input_rect.left + 5, input_rect.centery))
            self.screen.blit(text_surface, text_rect)

        # Cursor (blinking)
        if pygame.time.get_ticks() % 1000 < 500:
            cursor_x = input_rect.left + 5 + (len(self.new_symbol_input) * 12)
            cursor_y1 = input_rect.top + 5
            cursor_y2 = input_rect.bottom - 5
            pygame.draw.line(self.screen, self.colors['text'], (cursor_x, cursor_y1), (cursor_x, cursor_y2), 2)

        # Buttons come from _symbol_dialog_buttons so that clicking and drawing
        # use one definition, and so the buttons work on the dialog's first
        # frame rather than only after it has been painted once.
        cancel_button, add_button = self._symbol_dialog_buttons()

        # Cancel button
        pygame.draw.rect(self.screen, self.colors['button_normal'], cancel_button)
        pygame.draw.rect(self.screen, self.colors['ui_border'], cancel_button, 2)
        cancel_text = self.font_small.render("Cancel", True, self.colors['text'])
        cancel_text_rect = cancel_text.get_rect(center=cancel_button.center)
        self.screen.blit(cancel_text, cancel_text_rect)

        # Add button
        pygame.draw.rect(self.screen, self.colors['button_normal'], add_button)
        pygame.draw.rect(self.screen, self.colors['ui_border'], add_button, 2)
        add_text = self.font_small.render("Add", True, self.colors['text'])
        add_text_rect = add_text.get_rect(center=add_button.center)
        self.screen.blit(add_text, add_text_rect)

        # Instructions at bottom
        help_text = "Press Enter to add, Escape to cancel"
        help_surface = self.font_small.render(help_text, True, self.colors['text'])
        help_rect = help_surface.get_rect(centerx=dialog_x + dialog_width // 2, y=dialog_y + dialog_height - 20)
        self.screen.blit(help_surface, help_rect)
