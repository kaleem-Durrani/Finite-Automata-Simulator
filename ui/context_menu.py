"""The right-click menu: its rows, its box, and where that box is allowed to sit.

Plain functions over an explicit :class:`~ui.widgets.Chrome` and an explicit
:class:`ContextMenu`, so the menu can be placed, drawn and hit-tested without
the manager that happens to be holding it.

``MenuItem`` and ``SEPARATOR`` keep their names because the application builds
its menus out of them.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame

from rendering import primitives
from ui.events import UiEvent
from ui.widgets import Chrome

#: A menu row that is a rule rather than a command.
SEPARATOR = "---"

CONTEXT_MENU_WIDTH = 168
CONTEXT_MENU_ITEM_HEIGHT = 27
# Left inset for item labels, leaving room for the toggle marker.
CONTEXT_MENU_GUTTER = 28

#: Breathing room between a nudged context menu and the window edge.
CONTEXT_MENU_MARGIN = 8


@dataclass
class MenuItem:
    """One row of a context menu.

    ``event`` is the thing to do, as a value -- not a string with the state id
    packed into it. That packing is what let ``straighten:a>b>c`` split at the
    wrong ``>`` and flatten somebody else's edge.

    ``checked`` carries the current value for a toggle, so the menu can show
    what a state already is rather than making the user pick something and
    then look to see whether it changed anything.
    """
    label: str
    event: Optional[UiEvent] = None
    checked: Optional[bool] = None

    @property
    def is_separator(self) -> bool:
        return self.label == SEPARATOR


@dataclass
class ContextMenu:
    """A context menu: a position and a list of items."""
    position: Tuple[int, int]
    items: List[MenuItem]
    visible: bool = True
    selected_index: int = -1


def place(*, position: Tuple[int, int], items: List[MenuItem],
          screen_width: int, screen_height: int) -> ContextMenu:
    """
    A context menu at the specified position, nudged to fit on screen.

    Drawing and hit-testing both derive their rows from this one stored
    position, so a row pushed past the bottom edge is not merely invisible:
    no mouse position can ever land on it, and there is no keyboard
    fallback. Right-clicking low on the canvas would otherwise lose the
    last items of the menu -- "Delete state" among them.

    Args:
        position: (x, y) position to show the menu
        items: List of (label, action) tuples
        screen_width: Width of the window the menu must stay inside.
        screen_height: Height of the window the menu must stay inside.
    """
    height = len(items) * CONTEXT_MENU_ITEM_HEIGHT
    margin = CONTEXT_MENU_MARGIN
    x, y = position
    # Clamped rather than flipped: the menu stays under the pointer, and a
    # menu taller than the window still starts at the top, showing as much
    # as there is room for instead of hanging off both ends. The margin
    # keeps a nudged menu from sitting flush against the window edge.
    x = max(margin, min(x, screen_width - CONTEXT_MENU_WIDTH - margin))
    y = max(margin, min(y, screen_height - height - margin))
    return ContextMenu((x, y), items)


def bounds(*, menu: Optional[ContextMenu]) -> pygame.Rect:
    """Bounding box of the open context menu."""
    if not menu:
        return pygame.Rect(0, 0, 0, 0)
    menu_x, menu_y = menu.position
    return pygame.Rect(menu_x, menu_y, CONTEXT_MENU_WIDTH,
                       len(menu.items) * CONTEXT_MENU_ITEM_HEIGHT)


def event_at(*, menu: Optional[ContextMenu],
             position: Tuple[int, int]) -> Optional[UiEvent]:
    """The event for the item under the pointer, if any."""
    if not menu:
        return None

    menu_x, menu_y = menu.position

    for index, item in enumerate(menu.items):
        item_y = menu_y + index * CONTEXT_MENU_ITEM_HEIGHT
        item_rect = pygame.Rect(menu_x, item_y, CONTEXT_MENU_WIDTH,
                                CONTEXT_MENU_ITEM_HEIGHT)
        if item_rect.collidepoint(position) and not item.is_separator:
            return item.event

    return None


def draw(chrome: Chrome, *, menu: Optional[ContextMenu],
         mouse_pos: Tuple[int, int]) -> None:
    """Draw the context menu if visible."""
    if not menu or not menu.visible:
        return

    menu_x, menu_y = menu.position
    item_height = CONTEXT_MENU_ITEM_HEIGHT
    menu_width = CONTEXT_MENU_WIDTH

    screen = chrome.screen
    palette = chrome.palette
    menu_rect = bounds(menu=menu)
    # A rectangular shadow under a rectangular menu. The circular
    # soft_shadow used before bulged out below the bottom edge as a dark
    # disc.
    primitives.elevated_panel(screen, menu_rect, palette.panel_raised,
                              radius=chrome.radius.md,
                              border=palette.border_strong,
                              shadow=palette.shadow, lift=5,
                              bevel_light=palette.bevel_light,
                              bevel_dark=palette.bevel_dark)

    # Menu items
    for i, item in enumerate(menu.items):
        label, checked = item.label, item.checked
        item_y = menu_y + i * item_height
        item_rect = pygame.Rect(menu_x, item_y, menu_width, item_height)

        # Highlight hovered item
        if item_rect.collidepoint(mouse_pos) and not item.is_separator:
            primitives.panel(screen, item_rect.inflate(-6, -2),
                             palette.control_hover, radius=chrome.radius.sm)
            menu.selected_index = i

        # Separator line
        if item.is_separator:
            line_y = item_y + item_height // 2
            pygame.draw.line(screen, palette.border,
                           (menu_x + 10, line_y), (menu_x + menu_width - 10, line_y))
        else:
            colour = palette.error if label.startswith("Delete") else palette.text
            text_surface = chrome.fonts.ui("small").render(label, True, colour)
            text_rect = text_surface.get_rect(
                midleft=(menu_x + CONTEXT_MENU_GUTTER, item_y + item_height // 2))
            screen.blit(text_surface, text_rect)

            if checked is not None:
                centre = (menu_x + 15, item_y + item_height // 2)
                if checked:
                    primitives.filled_circle(screen, centre, 5, palette.accent)
                    primitives.filled_circle(screen, centre, 2,
                                             palette.panel_raised)
                else:
                    primitives.ring(screen, centre, 5, 1, palette.border_strong)
