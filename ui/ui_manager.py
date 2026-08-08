"""
UI Manager module for handling user interface elements.

This module manages all UI components including toolbars, input fields,
context menus, and help panels.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pygame

from core.dfa import DFA


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
    
    def __init__(self, screen: pygame.Surface):
        """
        Initialize the UI manager.
        
        Args:
            screen: Pygame surface for rendering UI elements
        """
        self.screen = screen
        self.screen_width = screen.get_width()
        self.screen_height = screen.get_height()
        
        # UI state management
        self.show_help = False
        self.show_tutorial = False
        self.input_text = ""
        self.input_active = False
        self.test_result = ""
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
        
        # Fonts for different text sizes
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 32)
        
        # Color scheme
        self.colors = {
            'ui_bg': (240, 240, 240),
            'ui_border': (100, 100, 100),
            'button_normal': (200, 200, 200),
            'button_hover': (220, 220, 220),
            'button_active': (180, 180, 180),
            'text': (0, 0, 0),
            'input_active': (255, 255, 255),
            'input_inactive': (240, 240, 240),
            'success': (0, 150, 0),
            'error': (200, 0, 0)
        }
        
        # Input handling for backspace
        self.backspace_timer = 0
        self.backspace_repeat_delay = 500  # ms before repeat starts
        self.backspace_repeat_rate = 50    # ms between repeats

        # Help panel scrolling
        self.help_scroll_offset = 0

        # Animation controls
        self.animation_speed = 1000  # ms per step
        self.speed_slider_dragging = False
        
    def _setup_ui_elements(self):
        """Initialize UI element positions and sizes."""
        # Main input and test area
        self.input_rect = pygame.Rect(20, self.screen_height - 100, 200, 30)
        self.test_button_rect = pygame.Rect(230, self.screen_height - 100, 80, 30)
        
        # Toolbar buttons
        self.help_button_rect = pygame.Rect(self.screen_width - 100, 20, 80, 30)
        self.save_button_rect = pygame.Rect(self.screen_width - 190, 20, 80, 30)
        self.load_button_rect = pygame.Rect(self.screen_width - 280, 20, 80, 30)
        
        # Symbol buttons (will be populated dynamically)
        self.symbol_buttons = {}
        
        # Add symbol button
        self.add_symbol_button_rect = None  # Will be set dynamically
        
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
    
    def handle_event(self, event: pygame.event.Event) -> Dict[str, Any]:
        """
        Handle UI events and return actions for the main application.
        
        Args:
            event: Pygame event to process
            
        Returns:
            Dictionary of actions to be handled by the main application
        """
        actions: Dict[str, Any] = {}
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            actions.update(self._handle_mouse_down(event))
        elif event.type == pygame.KEYDOWN:
            actions.update(self._handle_key_down(event))
        elif event.type == pygame.KEYUP:
            actions.update(self._handle_key_up(event))
        elif event.type == pygame.MOUSEWHEEL:
            actions.update(self._handle_mouse_wheel(event))
        
        return actions

    def _handle_mouse_wheel(self, event) -> Dict[str, Any]:
        """Handle mouse wheel events."""
        actions: Dict[str, Any] = {}

        # Check if scrolling in help panel
        if self.show_help:
            # Scroll help panel
            self.help_scroll_offset -= event.y * 3  # Scroll speed

            # Calculate max scroll based on content
            help_lines_count = 20  # Approximate number of help lines
            visible_lines = 25  # Approximate visible lines
            max_scroll = max(0, help_lines_count - visible_lines)

            # Clamp scroll offset
            self.help_scroll_offset = max(0, min(max_scroll, self.help_scroll_offset))

        return actions
    
    def _handle_mouse_down(self, event) -> Dict[str, Any]:
        """Handle mouse button down events."""
        actions: Dict[str, Any] = {}

        # A modal dialog swallows clicks; the widgets behind it are not live.
        if self.is_modal_active():
            return actions

        # Only handle left clicks in UI
        if event.button != 1:
            return actions

        # Hit-test against where the click happened, not where the cursor is
        # now. Those differ whenever the mouse moves between the event being
        # queued and the queue being drained, which loses clicks and lets a
        # click register against whatever the cursor has since moved over.
        mouse_pos = event.pos

        # Check input field
        if self.input_rect.collidepoint(mouse_pos):
            self.input_active = True
            actions['input_focus'] = True
        else:
            self.input_active = False
            
        # Check test button
        if self.test_button_rect.collidepoint(mouse_pos):
            actions['test_string'] = self.input_text
            
        # Check toolbar buttons
        if self.help_button_rect.collidepoint(mouse_pos):
            self.show_help = not self.show_help
            actions['toggle_help'] = True
            
        if self.save_button_rect.collidepoint(mouse_pos):
            actions['save_automaton'] = True
            
        if self.load_button_rect.collidepoint(mouse_pos):
            actions['load_automaton'] = True
        
        # Check symbol buttons
        for symbol, button_rect in self.symbol_buttons.items():
            if button_rect.collidepoint(mouse_pos):
                self.selected_symbol = symbol
                actions['symbol_selected'] = symbol
                actions['show_message'] = f"Selected symbol: {symbol}"
                break
        
        # Check add symbol button
        if (self.add_symbol_button_rect and
            self.add_symbol_button_rect.collidepoint(mouse_pos)):
            self.adding_symbol = True
            self.new_symbol_input = ""
            actions['add_symbol'] = True

        # Check symbol dialog buttons
        if self.adding_symbol:
            if hasattr(self, 'symbol_dialog_cancel_rect') and self.symbol_dialog_cancel_rect.collidepoint(mouse_pos):
                self.adding_symbol = False
                self.new_symbol_input = ""
                actions['symbol_dialog_cancel'] = True
            elif hasattr(self, 'symbol_dialog_add_rect') and self.symbol_dialog_add_rect.collidepoint(mouse_pos):
                if self.new_symbol_input:
                    actions['symbol_add'] = self.new_symbol_input
                    self.adding_symbol = False
                    self.new_symbol_input = ""

        # Check speed slider
        if hasattr(self, 'speed_slider_rect') and self.speed_slider_rect.collidepoint(mouse_pos):
            self.speed_slider_dragging = True
            # Calculate new speed based on mouse position
            relative_x = mouse_pos[0] - self.speed_slider_rect.x
            ratio = relative_x / self.speed_slider_rect.width
            ratio = max(0, min(1, ratio))

            min_speed, max_speed = 500, 3000
            self.animation_speed = int(min_speed + ratio * (max_speed - min_speed))
            actions['speed_changed'] = self.animation_speed
        
        # Check context menu
        if self.context_menu and self.context_menu.visible:
            menu_action = self._handle_context_menu_click(mouse_pos)
            if menu_action:
                actions['context_menu_action'] = menu_action
                self.context_menu = None
                return actions  # Return early to prevent other actions
            else:
                # Click outside menu - close it
                self.context_menu = None
                return actions  # Return early to prevent other actions
                
        return actions
    
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

    def draw(self, dfa: DFA, test_result: str = ""):
        """
        Draw all UI elements.

        Args:
            dfa: The current DFA for displaying information
            test_result: Result of the last string test
        """
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
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))

        rect = pygame.Rect(
            (self.screen_width - width) // 2,
            (self.screen_height - height) // 2,
            width,
            height,
        )
        pygame.draw.rect(self.screen, self.colors['ui_bg'], rect)
        pygame.draw.rect(self.screen, self.colors['ui_border'], rect, 3)

        title_surface = self.font_medium.render(title, True, self.colors['text'])
        title_rect = title_surface.get_rect(centerx=rect.centerx, y=rect.y + 15)
        self.screen.blit(title_surface, title_rect)

        return rect

    def _draw_file_prompt(self):
        """Draw the filename prompt."""
        verb = "Save as" if self.file_prompt_mode == 'save' else "Load file"
        rect = self._draw_modal_frame(440, 160, verb)

        hint = self.font_small.render(
            "Relative to the project folder. '.json' is added if omitted.",
            True, self.colors['text'])
        self.screen.blit(hint, (rect.x + 20, rect.y + 48))

        field = pygame.Rect(rect.x + 20, rect.y + 72, rect.width - 40, 30)
        pygame.draw.rect(self.screen, self.colors['input_active'], field)
        pygame.draw.rect(self.screen, self.colors['ui_border'], field, 2)

        # Show the tail of the text so the caret stays visible on long paths.
        shown = self.file_prompt_text[-40:]
        text_surface = self.font_medium.render(shown, True, self.colors['text'])
        text_rect = text_surface.get_rect(midleft=(field.left + 6, field.centery))
        self.screen.blit(text_surface, text_rect)

        if pygame.time.get_ticks() % 1000 < 500:
            caret_x = min(text_rect.right + 2, field.right - 4)
            pygame.draw.line(self.screen, self.colors['text'],
                             (caret_x, field.top + 5), (caret_x, field.bottom - 5), 2)

        footer = self.font_small.render("Enter to confirm, Escape to cancel",
                                        True, self.colors['text'])
        self.screen.blit(footer, footer.get_rect(centerx=rect.centerx,
                                                 y=rect.bottom - 26))

    def _draw_confirm_dialog(self):
        """Draw the yes/no confirmation dialog."""
        rect = self._draw_modal_frame(420, 130, "Unsaved changes")

        message = self.font_medium.render(self.confirm_message, True, self.colors['text'])
        self.screen.blit(message, message.get_rect(centerx=rect.centerx, y=rect.y + 55))

        footer = self.font_small.render("Y or Enter to confirm, N or Escape to cancel",
                                        True, self.colors['text'])
        self.screen.blit(footer, footer.get_rect(centerx=rect.centerx,
                                                 y=rect.bottom - 30))

    def _draw_toolbar(self):
        """Draw the main toolbar at the top of the screen."""
        toolbar_rect = pygame.Rect(0, 0, self.screen_width, 50)
        pygame.draw.rect(self.screen, self.colors['ui_bg'], toolbar_rect)
        pygame.draw.rect(self.screen, self.colors['ui_border'], toolbar_rect, 2)

        # Title
        title_text = self.font_large.render("Finite Automata Simulator", True, self.colors['text'])
        self.screen.blit(title_text, (20, 15))

        # Save button
        pygame.draw.rect(self.screen, self.colors['button_normal'], self.save_button_rect)
        pygame.draw.rect(self.screen, self.colors['ui_border'], self.save_button_rect, 2)
        save_text = self.font_medium.render("Save", True, self.colors['text'])
        save_text_rect = save_text.get_rect(center=self.save_button_rect.center)
        self.screen.blit(save_text, save_text_rect)

        # Load button
        pygame.draw.rect(self.screen, self.colors['button_normal'], self.load_button_rect)
        pygame.draw.rect(self.screen, self.colors['ui_border'], self.load_button_rect, 2)
        load_text = self.font_medium.render("Load", True, self.colors['text'])
        load_text_rect = load_text.get_rect(center=self.load_button_rect.center)
        self.screen.blit(load_text, load_text_rect)

        # Help button
        help_color = self.colors['button_active'] if self.show_help else self.colors['button_normal']
        pygame.draw.rect(self.screen, help_color, self.help_button_rect)
        pygame.draw.rect(self.screen, self.colors['ui_border'], self.help_button_rect, 2)

        help_text = self.font_medium.render("Help", True, self.colors['text'])
        help_text_rect = help_text.get_rect(center=self.help_button_rect.center)
        self.screen.blit(help_text, help_text_rect)

    def _draw_input_area(self, test_result: str):
        """Draw the input area for testing strings."""
        # Input field
        input_color = self.colors['input_active'] if self.input_active else self.colors['input_inactive']
        pygame.draw.rect(self.screen, input_color, self.input_rect)
        pygame.draw.rect(self.screen, self.colors['ui_border'], self.input_rect, 2)

        # Input text (show last 25 characters if too long)
        display_text = self.input_text if len(self.input_text) <= 25 else self.input_text[-25:]
        text_surface = self.font_medium.render(display_text, True, self.colors['text'])
        text_rect = text_surface.get_rect(midleft=(self.input_rect.left + 5, self.input_rect.centery))
        self.screen.blit(text_surface, text_rect)

        # Cursor (blinking)
        if self.input_active and pygame.time.get_ticks() % 1000 < 500:
            cursor_x = text_rect.right + 2
            cursor_y1 = self.input_rect.top + 5
            cursor_y2 = self.input_rect.bottom - 5
            pygame.draw.line(self.screen, self.colors['text'], (cursor_x, cursor_y1), (cursor_x, cursor_y2), 2)

        # Test button with hover effect
        mouse_pos = pygame.mouse.get_pos()
        button_color = (self.colors['button_hover'] if self.test_button_rect.collidepoint(mouse_pos)
                        else self.colors['button_normal'])
        pygame.draw.rect(self.screen, button_color, self.test_button_rect)
        pygame.draw.rect(self.screen, self.colors['ui_border'], self.test_button_rect, 2)

        test_text = self.font_medium.render("Test", True, self.colors['text'])
        test_text_rect = test_text.get_rect(center=self.test_button_rect.center)
        self.screen.blit(test_text, test_text_rect)

        # Label
        label_text = self.font_medium.render("Test String:", True, self.colors['text'])
        self.screen.blit(label_text, (20, self.screen_height - 130))

        # Test result
        if test_result:
            result_color = self.colors['success'] if "accepted" in test_result.lower() else self.colors['error']
            result_text = self.font_medium.render(test_result, True, result_color)
            self.screen.blit(result_text, (320, self.screen_height - 95))

    def _draw_alphabet_selector(self, _dfa: DFA):
        """Draw the alphabet selector with available symbols."""
        # Clear previous button rects
        self.symbol_buttons.clear()

        # Title
        title_text = self.font_medium.render("Transition Symbols:", True, self.colors['text'])
        self.screen.blit(title_text, (20, 60))

        # Draw symbol buttons
        start_x = 20
        start_y = 85
        button_width = 40
        button_height = 30
        spacing = 5

        mouse_pos = pygame.mouse.get_pos()

        for i, symbol in enumerate(self.available_symbols):
            button_x = start_x + i * (button_width + spacing)
            button_rect = pygame.Rect(button_x, start_y, button_width, button_height)
            self.symbol_buttons[symbol] = button_rect

            # Button color based on selection and hover
            if symbol == self.selected_symbol:
                button_color = self.colors['button_active']
            elif button_rect.collidepoint(mouse_pos):
                button_color = self.colors['button_hover']
            else:
                button_color = self.colors['button_normal']

            pygame.draw.rect(self.screen, button_color, button_rect)
            pygame.draw.rect(self.screen, self.colors['ui_border'], button_rect, 2)

            # Symbol text
            symbol_text = self.font_medium.render(symbol, True, self.colors['text'])
            symbol_text_rect = symbol_text.get_rect(center=button_rect.center)
            self.screen.blit(symbol_text, symbol_text_rect)

        # Add symbol button
        add_button_x = start_x + len(self.available_symbols) * (button_width + spacing)
        self.add_symbol_button_rect = pygame.Rect(add_button_x, start_y, button_width, button_height)

        add_button_color = (self.colors['button_hover'] if
                           self.add_symbol_button_rect.collidepoint(mouse_pos) else
                           self.colors['button_normal'])

        pygame.draw.rect(self.screen, add_button_color, self.add_symbol_button_rect)
        pygame.draw.rect(self.screen, self.colors['ui_border'], self.add_symbol_button_rect, 2)

        add_text = self.font_medium.render("+", True, self.colors['text'])
        add_text_rect = add_text.get_rect(center=self.add_symbol_button_rect.center)
        self.screen.blit(add_text, add_text_rect)

    def _draw_status_info(self, dfa: DFA):
        """Draw status information about the current automaton."""
        info_x = self.screen_width - 300
        info_y = 60

        # Background panel
        panel_rect = pygame.Rect(info_x - 10, info_y - 10, 290, 100)
        pygame.draw.rect(self.screen, self.colors['ui_bg'], panel_rect)
        pygame.draw.rect(self.screen, self.colors['ui_border'], panel_rect, 2)

        # Status information
        status_lines = [
            f"States: {len(dfa.states)}",
            f"Alphabet: {', '.join(sorted(dfa.alphabet)) if dfa.alphabet else 'None'}",
            f"Initial: {dfa.initial_state or 'None'}",
            f"Accept: {len(dfa.accept_states)}"
        ]

        for i, line in enumerate(status_lines):
            text_surface = self.font_small.render(line, True, self.colors['text'])
            self.screen.blit(text_surface, (info_x, info_y + i * 20))

        # Add animation controls to the status box
        self._draw_animation_controls_in_status(info_x, info_y + len(status_lines) * 20 + 10)

    def _draw_animation_controls_in_status(self, x: int, y: int):
        """Draw animation controls inside the status box."""
        # Animation status
        animating = getattr(self, '_animation_active', False)
        status_text = "Animation: ON" if animating else "Animation: OFF"
        status_color = self.colors['success'] if animating else self.colors['error']
        status_surface = self.font_small.render(status_text, True, status_color)
        self.screen.blit(status_surface, (x, y))

        # Speed slider
        slider_y = y + 20
        slider_width = 120
        slider_height = 15
        slider_rect = pygame.Rect(x, slider_y, slider_width, slider_height)

        # Draw slider background
        pygame.draw.rect(self.screen, self.colors['ui_border'], slider_rect)
        pygame.draw.rect(self.screen, self.colors['input_inactive'], slider_rect.inflate(-2, -2))

        # Calculate slider position (500ms to 3000ms range)
        min_speed, max_speed = 500, 3000
        speed_ratio = (self.animation_speed - min_speed) / (max_speed - min_speed)
        speed_ratio = max(0, min(1, speed_ratio))

        # Draw slider handle
        handle_x = x + int(speed_ratio * (slider_width - 8))
        handle_rect = pygame.Rect(handle_x, slider_y - 1, 8, slider_height + 2)
        pygame.draw.rect(self.screen, self.colors['button_normal'], handle_rect)
        pygame.draw.rect(self.screen, self.colors['ui_border'], handle_rect, 1)

        # Speed label
        speed_label = f"Speed: {self.animation_speed}ms"
        speed_surface = self.font_small.render(speed_label, True, self.colors['text'])
        self.screen.blit(speed_surface, (x, slider_y + 20))

        # Store slider rect for mouse handling
        self.speed_slider_rect = slider_rect

    def draw_animation_controls(self, animation_active: bool):
        """Draw animation status and speed controls."""
        # Position below the status info
        control_x = self.screen_width - 290
        control_y = 180

        # Animation status
        status_text = "Animation: ON" if animation_active else "Animation: OFF"
        status_color = self.colors['success'] if animation_active else self.colors['error']
        status_surface = self.font_small.render(status_text, True, status_color)
        self.screen.blit(status_surface, (control_x, control_y))

        # Speed slider
        slider_y = control_y + 25
        slider_width = 150
        slider_height = 20
        slider_rect = pygame.Rect(control_x, slider_y, slider_width, slider_height)

        # Draw slider background
        pygame.draw.rect(self.screen, self.colors['ui_border'], slider_rect)
        pygame.draw.rect(self.screen, self.colors['input_inactive'], slider_rect.inflate(-2, -2))

        # Calculate slider position (500ms to 3000ms range)
        min_speed, max_speed = 500, 3000
        speed_ratio = (self.animation_speed - min_speed) / (max_speed - min_speed)
        speed_ratio = max(0, min(1, speed_ratio))

        # Draw slider handle
        handle_x = control_x + int(speed_ratio * (slider_width - 10))
        handle_rect = pygame.Rect(handle_x, slider_y - 2, 10, slider_height + 4)
        pygame.draw.rect(self.screen, self.colors['button_normal'], handle_rect)
        pygame.draw.rect(self.screen, self.colors['ui_border'], handle_rect, 2)

        # Speed label
        speed_label = f"Speed: {self.animation_speed}ms"
        speed_surface = self.font_small.render(speed_label, True, self.colors['text'])
        self.screen.blit(speed_surface, (control_x, slider_y + 25))

        return slider_rect  # Return for mouse handling

    def draw_string_visualization(self, test_string: str, current_step: int):
        """Draw the test string with current position highlighted."""
        if not test_string:
            return

        # Position at bottom of screen
        string_y = self.screen_height - 80
        char_width = 30
        char_height = 40

        # Calculate total width needed
        total_width = len(test_string) * char_width

        # Calculate starting position for centering or scrolling
        center_x = self.screen_width // 2

        if total_width <= self.screen_width - 40:
            # String fits on screen - center it
            start_x = center_x - total_width // 2
        else:
            # String doesn't fit - implement scrolling
            # Keep current character in center when possible
            target_x = center_x - current_step * char_width

            # Clamp to screen bounds
            min_x = self.screen_width - total_width - 20
            max_x = 20
            start_x = max(min_x, min(max_x, target_x))

        # Draw each character
        for i, char in enumerate(test_string):
            char_x = start_x + i * char_width

            # Skip characters that are off-screen
            if char_x < -char_width or char_x > self.screen_width:
                continue

            # Create character rectangle
            char_rect = pygame.Rect(char_x, string_y, char_width - 2, char_height)

            # Choose colors based on position
            if i == current_step:
                # Current character - bright highlight
                bg_color = self.colors['success']
                text_color = (0, 0, 0)
                border_color = (255, 255, 255)
            elif i < current_step:
                # Processed characters - dim
                bg_color = (100, 100, 100)
                text_color = (200, 200, 200)
                border_color = (150, 150, 150)
            else:
                # Unprocessed characters - normal
                bg_color = (50, 50, 50)
                text_color = (255, 255, 255)
                border_color = (100, 100, 100)

            # Draw character background and border
            pygame.draw.rect(self.screen, bg_color, char_rect)
            pygame.draw.rect(self.screen, border_color, char_rect, 2)

            # Draw character text
            char_surface = self.font_medium.render(char, True, text_color)
            char_text_rect = char_surface.get_rect(center=char_rect.center)
            self.screen.blit(char_surface, char_text_rect)

    def _draw_help_panel(self):
        """Draw the help panel with scrollable content."""
        panel_width = 400
        panel_height = 500
        panel_x = (self.screen_width - panel_width) // 2
        panel_y = (self.screen_height - panel_height) // 2

        # Background
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, self.colors['ui_bg'], panel_rect)
        pygame.draw.rect(self.screen, self.colors['ui_border'], panel_rect, 3)

        # Title
        title_text = self.font_large.render("Help & Controls", True, self.colors['text'])
        title_rect = title_text.get_rect(centerx=panel_x + panel_width // 2, y=panel_y + 10)
        self.screen.blit(title_text, title_rect)

        # Help content (scrollable area)
        content_y = panel_y + 50
        content_height = panel_height - 60

        help_lines = [
            "Mouse Controls:",
            "• Left Click: Select states/UI",
            "• Shift+Click: Start transition",
            "• Right Click: Context menu",
            "• Middle Click+Drag: Pan view",
            "• Scroll Wheel: Zoom",
            "",
            "Keyboard Shortcuts:",
            "• Space: Add state at center",
            "• Delete: Remove selected state",
            "• Q: Toggle accept state",
            "• W: Toggle dead end state",
            "• R: Reset camera view",
            "",
            "Creating Transitions:",
            "• Select symbol from toolbar",
            "• Shift+click source state",
            "• Click target state",
            "",
            "Testing Strings:",
            "• Enter string in input field",
            "• Click Test or press Enter",
            "",
            "Execution Visualization:",
            "• N: Next step",
            "• P: Previous step",
            "• ESC: Stop visualization",
            "",
            "File Operations:",
            "• Save/Load buttons in toolbar"
        ]

        # Draw help text with scrolling
        line_height = 18
        visible_lines = content_height // line_height

        # Calculate which lines to show based on scroll offset
        start_line = max(0, self.help_scroll_offset)
        end_line = min(len(help_lines), start_line + visible_lines)

        for i in range(start_line, end_line):
            line = help_lines[i]
            display_y = content_y + (i - start_line) * line_height

            # Skip if outside visible area
            if display_y < content_y or display_y > content_y + content_height:
                continue

            if line.startswith("•"):
                # Indent bullet points
                text_x = panel_x + 30
            elif line.endswith(":"):
                # Headers
                text_x = panel_x + 15
            else:
                text_x = panel_x + 15

            text_surface = self.font_small.render(line, True, self.colors['text'])

            # Only draw if within the content area
            if display_y >= content_y and display_y + line_height <= content_y + content_height:
                self.screen.blit(text_surface, (text_x, display_y))

        # Draw scroll indicator if needed
        if len(help_lines) > visible_lines:
            # Scroll bar
            scrollbar_x = panel_x + panel_width - 15
            scrollbar_height = content_height
            scrollbar_rect = pygame.Rect(scrollbar_x, content_y, 10, scrollbar_height)
            pygame.draw.rect(self.screen, self.colors['ui_border'], scrollbar_rect)

            # Scroll thumb
            thumb_height = max(20, int(scrollbar_height * visible_lines / len(help_lines)))
            thumb_y = content_y + int((scrollbar_height - thumb_height) * start_line
                                      / (len(help_lines) - visible_lines))
            thumb_rect = pygame.Rect(scrollbar_x + 1, thumb_y, 8, thumb_height)
            pygame.draw.rect(self.screen, self.colors['button_normal'], thumb_rect)

            # Scroll instructions
            scroll_text = "Use mouse wheel to scroll"
            scroll_surface = self.font_small.render(scroll_text, True, self.colors['text'])
            scroll_rect = scroll_surface.get_rect(centerx=panel_x + panel_width // 2, y=panel_y + panel_height - 15)
            self.screen.blit(scroll_surface, scroll_rect)

    def _draw_context_menu(self):
        """Draw the context menu if visible."""
        if not self.context_menu or not self.context_menu.visible:
            return

        menu_x, menu_y = self.context_menu.position
        item_height = 25
        menu_width = 150
        menu_height = len(self.context_menu.items) * item_height

        # Background
        menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)
        pygame.draw.rect(self.screen, self.colors['ui_bg'], menu_rect)
        pygame.draw.rect(self.screen, self.colors['ui_border'], menu_rect, 2)

        # Menu items
        mouse_pos = pygame.mouse.get_pos()
        for i, (label, action) in enumerate(self.context_menu.items):
            item_y = menu_y + i * item_height
            item_rect = pygame.Rect(menu_x, item_y, menu_width, item_height)

            # Highlight hovered item
            if item_rect.collidepoint(mouse_pos):
                pygame.draw.rect(self.screen, self.colors['button_hover'], item_rect)
                self.context_menu.selected_index = i

            # Separator line
            if label == "---":
                line_y = item_y + item_height // 2
                pygame.draw.line(self.screen, self.colors['ui_border'],
                               (menu_x + 5, line_y), (menu_x + menu_width - 5, line_y))
            else:
                # Draw text
                text_surface = self.font_small.render(label, True, self.colors['text'])
                text_rect = text_surface.get_rect(midleft=(menu_x + 10, item_y + item_height // 2))
                self.screen.blit(text_surface, text_rect)

    def _handle_context_menu_click(self, mouse_pos: Tuple[int, int]) -> Optional[str]:
        """Handle clicks on context menu items."""
        if not self.context_menu:
            return None

        menu_x, menu_y = self.context_menu.position
        item_height = 25
        menu_width = 150

        for i, (label, action) in enumerate(self.context_menu.items):
            item_y = menu_y + i * item_height
            item_rect = pygame.Rect(menu_x, item_y, menu_width, item_height)

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
                            execution_string: str, execution_path: List[str]):
        """
        Draw execution status with step information.

        Args:
            execution_active: Whether execution visualization is active
            execution_step: Current step in execution
            execution_string: String being processed
            execution_path: Path of states visited
        """
        if not execution_active:
            return

        # Execution panel
        panel_width = 300
        panel_height = 120
        panel_x = self.screen_width - panel_width - 20
        panel_y = 180

        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, self.colors['ui_bg'], panel_rect)
        pygame.draw.rect(self.screen, self.colors['ui_border'], panel_rect, 2)

        # Title
        title_text = self.font_medium.render("Execution Trace", True, self.colors['text'])
        self.screen.blit(title_text, (panel_x + 10, panel_y + 10))

        # Current step info
        if execution_step < len(execution_string):
            current_char = execution_string[execution_step]
            step_text = f"Step {execution_step + 1}/{len(execution_string)}: Reading '{current_char}'"
        else:
            step_text = "Finished processing string"

        step_surface = self.font_small.render(step_text, True, self.colors['text'])
        self.screen.blit(step_surface, (panel_x + 10, panel_y + 35))

        # Current state
        if execution_step < len(execution_path):
            current_state = execution_path[execution_step]
            state_text = f"Current state: {current_state}"
        else:
            state_text = "Execution complete"

        state_surface = self.font_small.render(state_text, True, self.colors['text'])
        self.screen.blit(state_surface, (panel_x + 10, panel_y + 55))

        # Controls
        controls_text = "N: Next, P: Previous, TAB: Animation, ESC: Stop"
        controls_surface = self.font_small.render(controls_text, True, self.colors['text'])
        self.screen.blit(controls_surface, (panel_x + 10, panel_y + 85))

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

        # Buttons - moved down to prevent overlap
        button_width = 80
        button_height = 25
        cancel_button = pygame.Rect(dialog_x + 50, dialog_y + 130, button_width, button_height)
        add_button = pygame.Rect(dialog_x + 170, dialog_y + 130, button_width, button_height)

        # Store button rects for click detection
        self.symbol_dialog_cancel_rect = cancel_button
        self.symbol_dialog_add_rect = add_button

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
