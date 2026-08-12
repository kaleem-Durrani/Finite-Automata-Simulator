"""The centred, scrollable help overlay.

Plain functions over a :class:`ui.widgets.Chrome`: the panel is handed the
layout and the current scroll offset, and it draws. It owns ``HELP_LINES``
because the drawing and the scrolling have to agree on the line count, and
this is the one module that can be the source of that number.
"""

import pygame

from rendering import primitives
from ui.layout_spec import HELP_LINE_HEIGHT, HELP_TITLE_HEIGHT, LayoutSpec
from ui.widgets import Chrome

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
    "Algorithms:",
    "- Right-click empty canvas for",
    "  Minimise, Trim, and the",
    "  marking table",
    "- The marking table fills one",
    "  round at a time; click a cell",
    "  to read why it holds that",
    "  number. Esc closes it",
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
    "- Ctrl+A: Toggle accepting",
    "- Ctrl+T: Make a trap (loop all",
    "  symbols back to itself)",
    "- Ctrl+0: Fit view to automaton",
    "",
    "Every plain key picks a symbol,",
    "so an alphabet may contain any",
    "letter without a shortcut",
    "stealing it.",
    "",
    "Creating Transitions:",
    "- Pick a symbol from the",
    "  palette, top left",
    "- Switch on the arrow tool,",
    "  click the source state,",
    "  then click its target",
    "- Or hold Shift and click the",
    "  source, if you prefer keys",
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
    "- Right arrow: Next step",
    "- Left arrow: Previous step",
    "- TAB: Toggle animation",
    "- ESC: Stop visualization",
    "",
    "File Operations:",
    "- Save/Load buttons in toolbar",
    "- Filenames are relative to",
    "  the project folder",
]


def draw(chrome: Chrome, *, layout: LayoutSpec, scroll_offset: int) -> None:
    """Draw the help panel with scrollable content.

    Both the geometry and the line count come from the layout and from
    HELP_LINES, which is also what the scroll handler reads. They used to be
    independent constants that disagreed.
    """
    palette = chrome.palette
    screen = chrome.screen
    panel_rect = layout.help_panel
    primitives.elevated_panel(screen, panel_rect, palette.panel_raised,
                              radius=chrome.radius.lg,
                              border=palette.border_strong,
                              shadow=palette.shadow, lift=6,
                              bevel_light=palette.bevel_light,
                              bevel_dark=palette.bevel_dark)

    # Title
    title_text = chrome.fonts.ui("heading").render("Controls", True, palette.text)
    title_rect = title_text.get_rect(centerx=panel_rect.centerx, y=panel_rect.y + 10)
    screen.blit(title_text, title_rect)

    content_y = panel_rect.y + HELP_TITLE_HEIGHT
    visible_lines = layout.help_visible_lines()

    start_line = max(0, min(scroll_offset,
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
        font = chrome.fonts.ui("small_strong" if is_heading else "small")
        colour = palette.text if is_heading else palette.text_muted
        text_surface = font.render(text, True, colour)
        screen.blit(text_surface, (text_x, display_y))

    # Scrollbar
    if len(HELP_LINES) > visible_lines:
        scrollbar_x = panel_rect.right - 15
        scrollbar_height = visible_lines * HELP_LINE_HEIGHT
        scrollbar_rect = pygame.Rect(scrollbar_x, content_y, 6, scrollbar_height)
        primitives.panel(screen, scrollbar_rect, palette.control, radius=3)

        thumb_height = max(20, int(scrollbar_height * visible_lines / len(HELP_LINES)))
        thumb_y = content_y + int((scrollbar_height - thumb_height) * start_line
                                  / (len(HELP_LINES) - visible_lines))
        thumb_rect = pygame.Rect(scrollbar_x, thumb_y, 6, thumb_height)
        primitives.panel(screen, thumb_rect, palette.border_strong, radius=3)

        footer = chrome.fonts.ui("small").render("Scroll for more", True,
                                                 palette.text_faint)
        screen.blit(footer, footer.get_rect(centerx=panel_rect.centerx,
                                            y=panel_rect.bottom - 20))
