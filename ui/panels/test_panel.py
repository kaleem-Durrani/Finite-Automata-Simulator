"""The test-string panel in the bottom-left corner, and its verdict badge.

Two forms of the same panel: a pill when folded, and when expanded a sunken
field with a Test button beside it, growing a verdict band once there is a
verdict to put in it.

Plain functions over a :class:`~ui.widgets.Chrome`. Every rectangle and every
scrap of state arrives as a keyword argument, so nothing here reads the
manager, and the caller decides once -- in one place -- where each value comes
from.
"""

from typing import Optional

import pygame

import fsa
from rendering import primitives
from ui import widgets
from ui.widgets import Chrome


def draw(chrome: Chrome, *,
         panel: pygame.Rect,
         field: pygame.Rect,
         test_button: pygame.Rect,
         collapse_button: pygame.Rect,
         input_text: str,
         input_active: bool,
         input_expanded: bool,
         test_result: str,
         test_verdict: str,
         automaton: "fsa.DFA",
         mouse_pos: tuple,
         pressed_rect: Optional[pygame.Rect] = None) -> None:
    """Draw the input area for testing strings.

    ``field``, ``test_button`` and ``collapse_button`` are only read when
    ``input_expanded`` is set; the folded pill is the panel rectangle alone.
    """
    palette = chrome.palette

    if not input_expanded:
        # Folded: a pill that still says what it opens. This panel used to
        # be 600x118 of permanently reserved canvas for one text field.
        hovered = panel.collidepoint(mouse_pos)
        widgets.card(chrome, panel,
                     fill=palette.control_hover if hovered else None)
        label = chrome.fonts.ui("small_strong").render(
            "Test a string", True, palette.text)
        chrome.screen.blit(label, label.get_rect(
            midleft=(panel.x + chrome.space.md, panel.centery)))
        widgets.chevron(
            chrome, pygame.Rect(panel.right - 30, panel.centery - 9, 18, 18),
            palette.text_muted, pointing="up")
        return

    widgets.card(chrome, panel)

    widgets.chevron(chrome, collapse_button,
                    palette.accent if collapse_button.collidepoint(mouse_pos)
                    else palette.text_muted, pointing="down")

    # The field is sunken -- the one recessed surface on a screen of raised
    # ones, which is what makes it read as "type here" without a label.
    primitives.sunken_well(chrome.screen, field, palette.field,
                           radius=chrome.radius.md,
                           border=(palette.accent if input_active
                                   else palette.border),
                           well_shadow=palette.well_shadow)

    font = chrome.fonts.mono("input")
    display_text = input_text if len(input_text) <= 22 else input_text[-22:]
    if display_text:
        # Character by character, so a symbol the alphabet does not contain
        # shows red as it is typed -- the mistake is visible before Test is
        # pressed, at the exact position it was made.
        x = field.left + chrome.space.sm
        for char in display_text:
            valid = char in automaton.alphabet
            glyph = font.render(char, True,
                                palette.text if valid else palette.error)
            chrome.screen.blit(glyph, glyph.get_rect(
                midleft=(x, field.centery)))
            x += glyph.get_width()
        caret_x = min(x + 2, field.right - 5)
    else:
        hint = font.render("epsilon" if input_active else "type here",
                           True, palette.text_faint)
        chrome.screen.blit(hint, hint.get_rect(
            midleft=(field.left + chrome.space.sm, field.centery)))
        caret_x = field.left + chrome.space.sm

    if input_active and pygame.time.get_ticks() % 1100 < 560:
        pygame.draw.line(chrome.screen, palette.accent,
                         (caret_x, field.top + 6), (caret_x, field.bottom - 6), 2)

    widgets.button(chrome, test_button, "Test", accent=True,
                   hovered=test_button.collidepoint(mouse_pos),
                   pressed=pressed_rect == test_button)

    if test_result:
        draw_verdict(chrome, message=test_result, panel=panel, field=field,
                     verdict=test_verdict)


def draw_verdict(chrome: Chrome, *, message: str, panel: pygame.Rect,
                 field: pygame.Rect, verdict: str) -> None:
    """Show the result of the last run, coloured by its verdict.

    The colour comes from the verdict the engine reported, not from
    searching the message for the word "accepted" -- which used to paint a
    rejection green whenever that word appeared in the user's own input.
    """
    palette = chrome.palette
    if verdict == "accept":
        color, mark = palette.success, "ACCEPTED"
    elif verdict == "no_initial_state":
        color, mark = palette.warning, "NO START STATE"
    elif verdict:
        color, mark = palette.error, "REJECTED"
    else:
        color, mark = palette.text_muted, ""

    y = field.bottom + 10
    if mark:
        badge_font = chrome.fonts.ui("small_strong")
        badge_text = badge_font.render(mark, True, palette.text_on_accent)
        badge = pygame.Rect(panel.x + chrome.space.md, y,
                            badge_text.get_width() + 14,
                            badge_text.get_height() + 6)
        primitives.panel(chrome.screen, badge, color, radius=chrome.radius.sm)
        chrome.screen.blit(badge_text, badge_text.get_rect(center=badge.center))
        detail_x = badge.right + chrome.space.sm
    else:
        detail_x = panel.x + chrome.space.md

    # The engine's own sentence, trimmed to the panel.
    detail = message
    surface = chrome.fonts.ui("small").render(detail, True, palette.text_muted)
    available = panel.right - detail_x - chrome.space.md
    while surface.get_width() > available and len(detail) > 12:
        detail = detail[:-4] + "..."
        surface = chrome.fonts.ui("small").render(detail, True, palette.text_muted)
    chrome.screen.blit(surface, (detail_x, y + 2))
