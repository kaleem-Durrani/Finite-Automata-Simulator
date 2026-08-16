"""The floating symbol palette: what you can draw transitions with.

One card, one chip per symbol in the alphabet, and the add button sharing the
same grid. Every rectangle arrives from the caller -- the palette does not
compute its own geometry, because a chip that is positioned while drawing
cannot be clicked until it has been painted once.
"""

from typing import Mapping, Optional, Tuple

import pygame

from editor import EPSILON_LABEL
from rendering import primitives
from ui import widgets
from ui.widgets import Chrome


def draw(chrome: Chrome, *,
         card_rect: pygame.Rect,
         symbol_buttons: Mapping[str, pygame.Rect],
         add_button_rect: pygame.Rect,
         selected_symbol: Optional[str],
         mouse_pos: Optional[Tuple[int, int]] = None,
         pressed_rect: Optional[pygame.Rect] = None) -> None:
    """Draw the alphabet selector with available symbols.

    Reads the rectangles computed in _recompute_symbol_buttons rather than
    producing them as a side effect of drawing.
    """
    palette = chrome.palette
    widgets.card(chrome, card_rect)

    # The caption rides inside the card, down its left edge, so the palette
    # explains itself without a heading above it claiming another row of
    # height. The chips themselves are the point: what you can draw with is
    # visible at all times, which is why this never collapses.
    caption = chrome.fonts.ui("small_strong").render(
        "SYMBOL", True, palette.text_faint)
    chrome.screen.blit(caption, caption.get_rect(
        midleft=(card_rect.x + chrome.space.sm, card_rect.centery)))

    if mouse_pos is None:
        mouse_pos = pygame.mouse.get_pos()
    mono = chrome.fonts.mono("input")

    for index, (symbol, button_rect) in enumerate(symbol_buttons.items()):
        selected = symbol == selected_symbol
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

        primitives.panel(chrome.screen, button_rect, fill,
                         radius=chrome.radius.md, border=border)
        # `None` is the epsilon move. Shown as the letter, stored as the
        # engine spells it, so a real epsilon in the alphabet stays a
        # different chip from the empty move.
        surface = mono.render(EPSILON_LABEL if symbol is None else symbol,
                              True, text_color)
        chrome.screen.blit(surface, surface.get_rect(center=button_rect.center))

        # A hairline in this symbol's edge colour, tying the palette to the
        # arrows it draws.
        swatch = pygame.Rect(button_rect.x + 7, button_rect.bottom - 5,
                             button_rect.width - 14, 2)
        pygame.draw.rect(chrome.screen, chrome.theme.edge_color(index), swatch,
                         border_radius=1)

    widgets.button(chrome, add_button_rect, "+",
                   hovered=add_button_rect.collidepoint(mouse_pos),
                   pressed=pressed_rect == add_button_rect)
