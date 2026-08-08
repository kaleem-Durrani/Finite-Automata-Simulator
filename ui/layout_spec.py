"""Window layout.

Every UI region is computed here, once, from the window size. Both the drawing
code and the hit-testing code read the same rectangles, so a panel cannot be
drawn in one place and clicked in another.

Before this existed, hit-testing used hardcoded bands ("anything above y=120 is
the toolbar; anything below height-150 is the input area"). Those bands did not
match what was actually drawn: they covered a third to a half of the window,
which killed right-click over most of the canvas, while missing the status
panel and the execution panel entirely.
"""

from dataclasses import dataclass
from typing import List

import pygame

# Toolbar
TOOLBAR_HEIGHT = 50
TOOLBAR_BUTTON = (80, 30)
TOOLBAR_BUTTON_Y = 10
TOOLBAR_BUTTON_GAP = 10

# Symbol palette
SYMBOL_PANEL_TOP = TOOLBAR_HEIGHT
SYMBOL_PANEL_HEIGHT = 70
SYMBOL_BUTTON = (40, 30)
SYMBOL_BUTTON_GAP = 5
SYMBOL_ROW_ORIGIN = (20, TOOLBAR_HEIGHT + 35)

# Status panel (top right)
STATUS_WIDTH = 300
STATUS_HEIGHT = 170
STATUS_MARGIN = 12

# Speed slider, inside the status panel
SLIDER_SIZE = (120, 15)
SLIDER_OFFSET = (10, 110)
SPEED_MIN_MS = 500
SPEED_MAX_MS = 3000

# Execution panel, below the status panel
EXECUTION_WIDTH = 300
EXECUTION_HEIGHT = 120

# Input area (bottom left)
INPUT_PANEL_SIZE = (560, 86)
INPUT_PANEL_MARGIN = 12

# Help panel (centred)
HELP_PANEL_SIZE = (420, 500)
HELP_TITLE_HEIGHT = 50
HELP_LINE_HEIGHT = 18
HELP_FOOTER_HEIGHT = 22


@dataclass(frozen=True)
class LayoutSpec:
    """Named regions for one window size."""

    width: int
    height: int

    toolbar: pygame.Rect
    load_button: pygame.Rect
    save_button: pygame.Rect
    help_button: pygame.Rect

    symbol_panel: pygame.Rect
    symbol_row_origin: tuple

    status_panel: pygame.Rect
    speed_slider: pygame.Rect

    execution_panel: pygame.Rect

    input_panel: pygame.Rect
    input_field: pygame.Rect
    test_button: pygame.Rect

    help_panel: pygame.Rect

    string_strip: pygame.Rect

    @classmethod
    def for_size(cls, width: int, height: int) -> "LayoutSpec":
        """Build the layout for a window of the given size."""
        button_w, button_h = TOOLBAR_BUTTON
        step = button_w + TOOLBAR_BUTTON_GAP

        help_button = pygame.Rect(width - step, TOOLBAR_BUTTON_Y, button_w, button_h)
        save_button = pygame.Rect(width - step * 2, TOOLBAR_BUTTON_Y, button_w, button_h)
        load_button = pygame.Rect(width - step * 3, TOOLBAR_BUTTON_Y, button_w, button_h)

        status_panel = pygame.Rect(
            width - STATUS_WIDTH - STATUS_MARGIN,
            SYMBOL_PANEL_TOP + 10,
            STATUS_WIDTH,
            STATUS_HEIGHT,
        )
        speed_slider = pygame.Rect(
            status_panel.x + SLIDER_OFFSET[0],
            status_panel.y + SLIDER_OFFSET[1],
            SLIDER_SIZE[0],
            SLIDER_SIZE[1],
        )

        execution_panel = pygame.Rect(
            width - EXECUTION_WIDTH - STATUS_MARGIN,
            status_panel.bottom + 10,
            EXECUTION_WIDTH,
            EXECUTION_HEIGHT,
        )

        panel_w, panel_h = INPUT_PANEL_SIZE
        input_panel = pygame.Rect(
            INPUT_PANEL_MARGIN,
            height - panel_h - INPUT_PANEL_MARGIN,
            min(panel_w, width - INPUT_PANEL_MARGIN * 2),
            panel_h,
        )
        input_field = pygame.Rect(input_panel.x + 10, input_panel.y + 40, 200, 30)
        test_button = pygame.Rect(input_field.right + 10, input_field.y, 80, 30)

        help_w, help_h = HELP_PANEL_SIZE
        help_panel = pygame.Rect(
            (width - help_w) // 2,
            max(TOOLBAR_HEIGHT, (height - help_h) // 2),
            help_w,
            min(help_h, height - TOOLBAR_HEIGHT - 20),
        )

        string_strip = pygame.Rect(0, input_panel.y - 52, width, 44)

        return cls(
            width=width,
            height=height,
            toolbar=pygame.Rect(0, 0, width, TOOLBAR_HEIGHT),
            load_button=load_button,
            save_button=save_button,
            help_button=help_button,
            symbol_panel=pygame.Rect(0, SYMBOL_PANEL_TOP, width, SYMBOL_PANEL_HEIGHT),
            symbol_row_origin=SYMBOL_ROW_ORIGIN,
            status_panel=status_panel,
            speed_slider=speed_slider,
            execution_panel=execution_panel,
            input_panel=input_panel,
            input_field=input_field,
            test_button=test_button,
            help_panel=help_panel,
            string_strip=string_strip,
        )

    def help_visible_lines(self) -> int:
        """How many help lines fit in the panel at once."""
        content = self.help_panel.height - HELP_TITLE_HEIGHT - HELP_FOOTER_HEIGHT
        return max(1, content // HELP_LINE_HEIGHT)

    def symbol_button(self, index: int) -> pygame.Rect:
        """Rectangle for the nth button in the symbol palette."""
        x0, y0 = self.symbol_row_origin
        w, h = SYMBOL_BUTTON
        return pygame.Rect(x0 + index * (w + SYMBOL_BUTTON_GAP), y0, w, h)

    def opaque_panels(self, *, execution_active: bool, help_open: bool) -> List[pygame.Rect]:
        """
        The regions currently painted over the canvas.

        A click landing on one of these belongs to the UI and must not also be
        interpreted as a click on the automaton behind it. Panels that are not
        currently drawn are not included, so the canvas is as large as it
        actually looks.
        """
        panels = [self.toolbar, self.symbol_panel, self.status_panel, self.input_panel]
        if execution_active:
            panels.append(self.execution_panel)
        if help_open:
            panels.append(self.help_panel)
        return panels
