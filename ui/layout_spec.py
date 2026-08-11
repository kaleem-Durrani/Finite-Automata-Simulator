"""Window layout.

Every UI region is computed here, once, from the window size. Both the drawing
code and the hit-testing code read the same rectangles, so a panel cannot be
drawn in one place and clicked in another.

Before this existed, hit-testing used hardcoded bands ("anything above y=120 is
the toolbar; anything below height-150 is the input area"). Those bands did not
match what was actually drawn: they covered a third to a half of the window,
which killed right-click over most of the canvas, while missing the status
panel and the execution panel entirely.

Regions whose size depends on content -- how many symbols exist, whether a
panel is collapsed -- are methods rather than fields. They still live here, so
there is still exactly one definition of where anything sits; they just cannot
be computed from the window size alone.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import pygame

# Toolbar
TOOLBAR_HEIGHT = 50
TOOLBAR_BUTTON = (80, 30)
TOOLBAR_BUTTON_Y = 10
TOOLBAR_BUTTON_GAP = 10
#: The pan tool is an icon, not a word, so it gets a square button.
TOOL_BUTTON_WIDTH = 38
#: Space between the tool group and the file/theme group.
TOOLBAR_GROUP_GAP = 18

# Floating symbol palette.
#
# A card on the canvas rather than a full-width band. The band ran the whole
# width of the window, so the right-hand panels were drawn on top of it, and it
# reserved 70px of height to show one 30px row of chips. The card is only as
# wide as the alphabet it holds and sits clear of everything else.
SYMBOL_MARGIN = 12
SYMBOL_CHIP = (34, 30)
SYMBOL_CHIP_GAP = 6
SYMBOL_CARD_PAD = 9
#: Room for the caption printed down the left of the card, so the palette says
#: what it is without a separate heading above it.
SYMBOL_CAPTION_WIDTH = 60
#: Chips wrap past this many, so a large alphabet grows downward in a tidy grid
#: instead of running off the side of the window.
SYMBOLS_PER_ROW = 12

# Right-hand column
PANEL_WIDTH = 280
PANEL_MARGIN = 12
PANEL_GAP = 8
#: Every panel is an accordion: the header is always drawn and is the click
#: target that opens and closes it. Collapsed, the header *is* the panel -- the
#: notch that says what is inside without spending the space to show it.
PANEL_HEADER_HEIGHT = 34

# Transport buttons inside the run panel: back, play/pause, forward, stop.
RUN_BUTTON_SIZE = (38, 28)
RUN_BUTTON_GAP = 8
#: Offsets down from the run panel's header. One stack, one direction.
RUN_BUTTONS_TOP = 76
RUN_SLIDER_TOP = 116
RUN_HINT_TOP = 140
#: Total body height the run panel needs for that stack.
RUN_BODY_HEIGHT = 160

#: Buttons on the modal dialogs. Every dialog offers its actions as buttons as
#: well as keys -- a dialog whose only exit is a keystroke it mentions in small
#: print at the bottom is a dialog most people are stuck in.
DIALOG_BUTTON = (92, 30)

# Speed slider, inside the run panel
SLIDER_SIZE = (150, 16)
SPEED_MIN_MS = 500
SPEED_MAX_MS = 3000

# Test-string panel (bottom left).
#
# One row: the field and the Test button. The caption is gone -- a text field
# with a Test button beside it does not need to be told it tests a string --
# and the verdict band only exists once there is a verdict to put in it.
INPUT_MARGIN = 12
INPUT_WIDTH = 430
INPUT_ROW_HEIGHT = 56
INPUT_VERDICT_HEIGHT = 46
INPUT_COLLAPSED_SIZE = (150, 34)

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
    transition_button: pygame.Rect
    pan_button: pygame.Rect
    theme_button: pygame.Rect
    load_button: pygame.Rect
    save_button: pygame.Rect
    help_button: pygame.Rect

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
        theme_button = pygame.Rect(width - step * 4, TOOLBAR_BUTTON_Y, button_w, button_h)
        pan_button = pygame.Rect(
            theme_button.x - TOOLBAR_GROUP_GAP - TOOL_BUTTON_WIDTH,
            TOOLBAR_BUTTON_Y, TOOL_BUTTON_WIDTH, button_h)
        transition_button = pygame.Rect(
            pan_button.x - TOOLBAR_BUTTON_GAP - TOOL_BUTTON_WIDTH,
            TOOLBAR_BUTTON_Y, TOOL_BUTTON_WIDTH, button_h)

        help_w, help_h = HELP_PANEL_SIZE
        help_panel = pygame.Rect(
            (width - help_w) // 2,
            max(TOOLBAR_HEIGHT, (height - help_h) // 2),
            help_w,
            min(help_h, height - TOOLBAR_HEIGHT - 20),
        )

        # The tape strip sits above the test panel's single row.
        strip_y = height - INPUT_MARGIN - INPUT_ROW_HEIGHT - 8 - 44
        string_strip = pygame.Rect(0, strip_y, width, 44)

        return cls(
            width=width,
            height=height,
            toolbar=pygame.Rect(0, 0, width, TOOLBAR_HEIGHT),
            transition_button=transition_button,
            pan_button=pan_button,
            theme_button=theme_button,
            load_button=load_button,
            save_button=save_button,
            help_button=help_button,
            help_panel=help_panel,
            string_strip=string_strip,
        )

    # ------------------------------------------------------------------
    # Symbol palette
    # ------------------------------------------------------------------

    def symbol_card(self, count: int) -> pygame.Rect:
        """The floating palette card holding ``count`` chips.

        ``count`` includes the trailing add button, because it is laid out in
        the same grid and the card has to be wide enough to contain it.
        """
        count = max(1, count)
        columns = min(count, SYMBOLS_PER_ROW)
        rows = math.ceil(count / SYMBOLS_PER_ROW)
        chip_w, chip_h = SYMBOL_CHIP
        inner_w = columns * chip_w + (columns - 1) * SYMBOL_CHIP_GAP
        inner_h = rows * chip_h + (rows - 1) * SYMBOL_CHIP_GAP
        return pygame.Rect(
            SYMBOL_MARGIN,
            TOOLBAR_HEIGHT + SYMBOL_MARGIN,
            SYMBOL_CAPTION_WIDTH + inner_w + SYMBOL_CARD_PAD * 2,
            inner_h + SYMBOL_CARD_PAD * 2,
        )

    def symbol_chip(self, index: int, count: int) -> pygame.Rect:
        """Rectangle for the nth chip in the palette."""
        card = self.symbol_card(count)
        chip_w, chip_h = SYMBOL_CHIP
        column = index % SYMBOLS_PER_ROW
        row = index // SYMBOLS_PER_ROW
        return pygame.Rect(
            card.x + SYMBOL_CARD_PAD + SYMBOL_CAPTION_WIDTH
            + column * (chip_w + SYMBOL_CHIP_GAP),
            card.y + SYMBOL_CARD_PAD + row * (chip_h + SYMBOL_CHIP_GAP),
            chip_w, chip_h,
        )

    # ------------------------------------------------------------------
    # Test-string panel
    # ------------------------------------------------------------------

    def input_panel(self, *, expanded: bool, has_verdict: bool) -> pygame.Rect:
        """Where the test-string panel sits, in whichever form it is in."""
        if not expanded:
            w, h = INPUT_COLLAPSED_SIZE
            return pygame.Rect(INPUT_MARGIN, self.height - h - INPUT_MARGIN, w, h)
        height = INPUT_ROW_HEIGHT + (INPUT_VERDICT_HEIGHT if has_verdict else 0)
        width = min(INPUT_WIDTH, self.width - INPUT_MARGIN * 2)
        return pygame.Rect(INPUT_MARGIN, self.height - height - INPUT_MARGIN,
                           width, height)

    def input_field(self, panel: pygame.Rect) -> pygame.Rect:
        """The text field inside an expanded test panel."""
        return pygame.Rect(panel.x + 11, panel.y + 11,
                           panel.width - 11 - 82 - 10 - 34, 34)

    def test_button(self, panel: pygame.Rect) -> pygame.Rect:
        field = self.input_field(panel)
        return pygame.Rect(field.right + 10, field.y, 82, 34)

    def input_collapse_button(self, panel: pygame.Rect) -> pygame.Rect:
        """The chevron that folds the test panel away."""
        return pygame.Rect(panel.right - 30, panel.y + 13, 22, 22)

    # ------------------------------------------------------------------
    # Right-hand column
    # ------------------------------------------------------------------

    def column_home_x(self) -> int:
        return self.width - PANEL_WIDTH - PANEL_MARGIN

    def column_top(self) -> int:
        return TOOLBAR_HEIGHT + PANEL_MARGIN

    def column_limit(self) -> int:
        """The lowest a panel may reach before it is dropped.

        Kept clear of the tape strip so the column cannot stack down over it.
        """
        return self.string_strip.top - 8

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    def help_visible_lines(self) -> int:
        """How many help lines fit in the panel at once."""
        content = self.help_panel.height - HELP_TITLE_HEIGHT - HELP_FOOTER_HEIGHT
        return max(1, content // HELP_LINE_HEIGHT)

    def speed_slider(self, panel: pygame.Rect) -> pygame.Rect:
        """The playback-speed slider, inside whichever panel hosts it.

        Measured down from the header rather than up from the bottom edge: the
        run panel's contents are a single stack, and mixing the two directions
        is how the keyboard hint and the playback row ended up drawn on top of
        each other.
        """
        return pygame.Rect(panel.x + 12,
                           panel.y + PANEL_HEADER_HEIGHT + RUN_SLIDER_TOP,
                           SLIDER_SIZE[0], SLIDER_SIZE[1])

    def panel_header(self, panel: pygame.Rect) -> pygame.Rect:
        """The always-drawn header strip that toggles a panel open and shut."""
        return pygame.Rect(panel.x, panel.y, panel.width, PANEL_HEADER_HEIGHT)

    def run_buttons(self, panel: pygame.Rect) -> List[pygame.Rect]:
        """Back, play/pause, forward and stop, inside the run panel.

        Stepping through a run used to be keyboard-only, which meant the panel
        described four commands it gave you no way to issue.
        """
        y = panel.y + PANEL_HEADER_HEIGHT + RUN_BUTTONS_TOP
        x = panel.x + 12
        w, h = RUN_BUTTON_SIZE
        return [pygame.Rect(x + index * (w + RUN_BUTTON_GAP), y, w, h)
                for index in range(4)]

    def confirm_buttons(self, panel: pygame.Rect) -> Tuple[pygame.Rect, pygame.Rect]:
        """Cancel and confirm, in that reading order."""
        w, h = DIALOG_BUTTON
        y = panel.bottom - h - 16
        return (pygame.Rect(panel.right - 16 - w * 2 - 10, y, w, h),
                pygame.Rect(panel.right - 16 - w, y, w, h))


def chip_size() -> Tuple[int, int]:
    return SYMBOL_CHIP
