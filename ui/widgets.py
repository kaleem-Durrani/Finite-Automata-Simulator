"""The pieces every panel draws with.

Plain functions over an explicit :class:`Chrome`, not methods on the manager.
That is the whole point of the split: a panel module can be read, tested and
changed without the 2,000-line class it used to live inside, because the only
things it can reach are the surface, the design tokens and the fonts.

Icons are drawn rather than typed. A glyph is only as reliable as the font
behind it, and a missing glyph is a blank button.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import pygame

from rendering import primitives
from rendering.fonts import FontBook
from rendering.theme import Theme

Colour = Tuple[int, ...]


@dataclass(frozen=True)
class Chrome:
    """A surface, the design tokens, and the font faces.

    Passed to every drawing function so none of them has to reach back into
    the manager for state it should not be reading in the first place.
    """

    screen: pygame.Surface
    theme: Theme
    fonts: FontBook

    @property
    def palette(self):
        return self.theme.palette

    @property
    def space(self):
        return self.theme.space

    @property
    def radius(self):
        return self.theme.radius


def button(chrome: Chrome, rect: pygame.Rect, label: str, *,
           active: bool = False, hovered: bool = False, accent: bool = False,
           pressed: bool = False) -> None:
    """A labelled button with real depth.

    Raised at rest, and pressed -- shadow gone, bevels inverted, label nudged
    down a pixel -- while the mouse is held on it. The nudge is what makes a
    click feel like a click.
    """
    palette = chrome.palette
    if accent or active:
        fill, text_colour, border = (palette.accent, palette.text_on_accent,
                                     palette.accent)
    elif hovered:
        fill, text_colour, border = (palette.control_hover, palette.text,
                                     palette.border_strong)
    else:
        fill, text_colour, border = (palette.control, palette.text,
                                     palette.border)

    nudge = primitives.raised_button(
        chrome.screen, rect, fill, radius=chrome.radius.md, border=border,
        bevel_light=palette.bevel_light, bevel_dark=palette.bevel_dark,
        pressed=pressed, shadow=palette.shadow)
    if label:
        surface = chrome.fonts.ui("body_strong").render(label, True, text_colour)
        chrome.screen.blit(surface, surface.get_rect(
            center=(rect.centerx, rect.centery + nudge)))


def section_label(chrome: Chrome, text: str,
                  position: Tuple[int, int]) -> None:
    """A small uppercase caption above a group of controls."""
    surface = chrome.fonts.ui("small_strong").render(
        text.upper(), True, chrome.palette.text_faint)
    chrome.screen.blit(surface, position)


def card(chrome: Chrome, rect: pygame.Rect, *,
         fill: Optional[Colour] = None) -> None:
    """The raised panel every floating surface in the interface is made of."""
    palette = chrome.palette
    primitives.elevated_panel(chrome.screen, rect,
                              palette.panel if fill is None else fill,
                              radius=chrome.radius.lg, border=palette.border,
                              shadow=palette.shadow,
                              bevel_light=palette.bevel_light,
                              bevel_dark=palette.bevel_dark)


def elide(font: pygame.font.Font, text: str, budget: float) -> str:
    """``text``, shortened with an ellipsis until it fits within ``budget``."""
    if budget <= 0 or font.size(text)[0] <= budget:
        return text
    trimmed = text
    while trimmed and font.size(trimmed + "…")[0] > budget:
        trimmed = trimmed[:-1]
    return trimmed + "…" if trimmed else "…"


# ----------------------------------------------------------------------
# Icons
# ----------------------------------------------------------------------

def chevron(chrome: Chrome, rect: pygame.Rect, colour: Colour, *,
            pointing: str) -> None:
    """A caret, drawn as three points so it needs nothing from the font."""
    mid_x, mid_y = rect.centerx, rect.centery
    reach, drop = 5, 3
    if pointing == "up":
        points = [(mid_x - reach, mid_y + drop), (mid_x, mid_y - drop),
                  (mid_x + reach, mid_y + drop)]
    elif pointing == "down":
        points = [(mid_x - reach, mid_y - drop), (mid_x, mid_y + drop),
                  (mid_x + reach, mid_y - drop)]
    elif pointing == "left":
        points = [(mid_x + drop, mid_y - reach), (mid_x - drop, mid_y),
                  (mid_x + drop, mid_y + reach)]
    else:  # "right", for a folded side panel
        points = [(mid_x - drop, mid_y - reach), (mid_x + drop, mid_y),
                  (mid_x - drop, mid_y + reach)]
    pygame.draw.lines(chrome.screen, colour, False, points, 2)


def hand(chrome: Chrome, rect: pygame.Rect, colour: Colour) -> None:
    """The pan tool."""
    cx, cy = rect.centerx, rect.centery
    pygame.draw.rect(chrome.screen, colour,
                     pygame.Rect(cx - 5, cy - 1, 10, 8), border_radius=3)
    for index in range(3):
        pygame.draw.rect(chrome.screen, colour,
                         pygame.Rect(cx - 5 + index * 4, cy - 7, 2, 7),
                         border_radius=1)
    pygame.draw.rect(chrome.screen, colour,
                     pygame.Rect(cx + 4, cy - 3, 3, 5), border_radius=1)


def arrow(chrome: Chrome, rect: pygame.Rect, colour: Colour) -> None:
    """The transition tool."""
    cx, cy = rect.centerx, rect.centery
    pygame.draw.line(chrome.screen, colour, (cx - 8, cy + 3), (cx + 5, cy - 4), 2)
    primitives.polygon(chrome.screen, [(cx + 8, cy - 6), (cx + 3, cy - 5),
                                       (cx + 6, cy - 1)], colour)


def play(chrome: Chrome, rect: pygame.Rect, colour: Colour) -> None:
    cx, cy = rect.centerx, rect.centery
    primitives.polygon(chrome.screen, [(cx - 4, cy - 6), (cx + 6, cy),
                                       (cx - 4, cy + 6)], colour)


def pause(chrome: Chrome, rect: pygame.Rect, colour: Colour) -> None:
    cx, cy = rect.centerx, rect.centery
    for offset in (-5, 1):
        pygame.draw.rect(chrome.screen, colour,
                         pygame.Rect(cx + offset, cy - 6, 4, 12), border_radius=1)


def cross(chrome: Chrome, rect: pygame.Rect, colour: Colour) -> None:
    cx, cy = rect.centerx, rect.centery
    pygame.draw.line(chrome.screen, colour, (cx - 5, cy - 5), (cx + 5, cy + 5), 2)
    pygame.draw.line(chrome.screen, colour, (cx + 5, cy - 5), (cx - 5, cy + 5), 2)
