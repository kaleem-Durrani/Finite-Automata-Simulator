"""The right-hand column: the shared panel frame, the status panel, the legend.

Every panel in the column is drawn inside the same card and the same clickable
header, so the frame lives here once and each panel asks it for the body
rectangle that is left over. The two simplest residents of the column -- what
the automaton currently is, and what the state styles mean -- live here beside
it because they are a handful of rows each and nothing else reads them.

Plain functions over a :class:`ui.widgets.Chrome` plus explicit keyword
arguments. Nothing in here may import ``ui.ui_manager``.
"""

from typing import Optional, Tuple

import pygame

import fsa
from rendering import primitives
from ui.layout_spec import LayoutSpec
from ui.widgets import Chrome, card, chevron

#: What each right-hand panel is called. A collapsed panel shows only this, so
#: it doubles as the notch's label: folding a panel away must not cost the user
#: the knowledge of what is inside it.
PANEL_TITLES = {
    "status": "Automaton",
    "run": "Run",
    "diagnostics": "Diagnostics",
    "legend": "Legend",
}


def panel_frame(chrome: Chrome, *, key: str, rect: pygame.Rect,
                collapsed: bool, layout: LayoutSpec,
                mouse_pos: Optional[Tuple[int, int]] = None,
                ) -> Optional[pygame.Rect]:
    """Draw a right-column panel's card and its header.

    Returns the rectangle left over for the body, or ``None`` when the
    panel is folded down to its header. The header is always drawn and is
    always the click target, so a collapsed panel is a labelled notch
    rather than a thing that has vanished.
    """
    palette = chrome.palette
    card(chrome, rect)

    header = layout.panel_header(rect)
    if mouse_pos is None:
        mouse_pos = pygame.mouse.get_pos()
    hovered = header.collidepoint(mouse_pos)
    title = chrome.fonts.ui("small_strong").render(
        PANEL_TITLES.get(key, key).upper(), True,
        palette.text if hovered else palette.text_faint)
    chrome.screen.blit(title, title.get_rect(
        midleft=(header.x + chrome.space.md, header.centery)))

    chevron(chrome, pygame.Rect(header.right - 28, header.y + 8, 18, 18),
            palette.text if hovered else palette.text_muted,
            pointing="right" if collapsed else "down")

    body = pygame.Rect(rect.x, header.bottom, rect.width,
                       rect.bottom - header.bottom)
    return body if body.height > 6 else None


def _deterministic(automaton: "fsa.AnyAutomaton") -> bool:
    """Whether the machine has one target per (state, symbol) and no epsilon.

    A `DFA` value is deterministic by construction; only an `NFA` has to be
    asked.
    """
    checker = getattr(automaton, "is_deterministic", None)
    return True if checker is None else bool(checker())


def draw_status(chrome: Chrome, *, rect: pygame.Rect, automaton: "fsa.AnyAutomaton",
                warn_no_accepting: bool, collapsed: bool, layout: LayoutSpec,
                mouse_pos: Optional[Tuple[int, int]] = None) -> None:
    """Draw status information about the current automaton."""
    palette = chrome.palette
    body = panel_frame(chrome, key="status", rect=rect, collapsed=collapsed,
                       layout=layout, mouse_pos=mouse_pos)
    if body is None:
        return

    info_x = body.x + chrome.space.md
    value_x = info_x + 92

    rows = [
        ("States", str(len(automaton.states)), False),
        ("Alphabet", ", ".join(sorted(automaton.alphabet)) if automaton.alphabet
         else "empty", not automaton.alphabet),
        ("Start", automaton.initial or "none", automaton.initial is None),
        ("Accepting", str(len(automaton.accept)), warn_no_accepting),
        # Stated, not flagged. A nondeterministic machine is a legal thing
        # to have drawn, and this codebase has already learned once what
        # happens when a design choice is labelled a fault with a button
        # beside it -- see docs/LESSONS.md.
        ("Kind", "DFA" if _deterministic(automaton) else "NFA", False),
    ]

    label_font = chrome.fonts.ui("small")
    value_font = chrome.fonts.ui("small_strong")
    row_y = body.y + 4
    for label, value, warn in rows:
        chrome.screen.blit(label_font.render(label, True, palette.text_muted),
                           (info_x, row_y))
        colour = palette.text if value not in ("none", "empty") else palette.text_faint
        if warn:
            colour = palette.warning
        surface = value_font.render(value, True, colour)
        available = body.right - value_x - chrome.space.md
        while surface.get_width() > available and len(value) > 4:
            value = value[:-4] + "..."
            surface = value_font.render(value, True, colour)
        chrome.screen.blit(surface, (value_x, row_y))
        row_y += 19

    if warn_no_accepting:
        chrome.screen.blit(
            chrome.fonts.ui("small").render("No string can be accepted", True,
                                            palette.warning),
            (info_x, row_y))


def draw_legend(chrome: Chrome, *, rect: Optional[pygame.Rect],
                automaton: "fsa.AnyAutomaton", legend_dead: bool,
                legend_unreachable: bool, collapsed: bool, layout: LayoutSpec,
                mouse_pos: Optional[Tuple[int, int]] = None) -> None:
    """Explain the state styles, showing only the kinds actually present.

    A diagram that dims some states and hatches others is only useful if
    the reader knows what those mean. Listing every kind all the time would
    be noise, so entries appear as the automaton acquires them.

    Lives in the sliding right column, so it can never collide with the
    other panels -- each takes the space the one above releases. ``rect`` is
    that slot, or ``None`` on a frame where the column has no room for it.
    """
    if rect is None:
        return
    palette = chrome.palette
    entries = [("normal", palette.state_fill, palette.state_ring, "plain")]

    if automaton.accept:
        entries.append(("accepting", palette.accept_fill,
                        palette.accept_ring, "double"))
    if legend_dead:
        entries.append(("trap", palette.dead_fill, palette.dead_ring, "hatch"))
    if legend_unreachable:
        entries.append(("unreachable", palette.unreachable_fill,
                        palette.unreachable_ring, "dashed"))

    if len(entries) < 2:
        return

    row_h = 22
    body = panel_frame(chrome, key="legend", rect=rect, collapsed=collapsed,
                       layout=layout, mouse_pos=mouse_pos)
    if body is None:
        return

    font = chrome.fonts.ui("small")
    y = body.y + 3
    for label, fill, ring, style in entries:
        if y + row_h > body.bottom:
            break
        centre = (body.x + chrome.space.md + 9, y + 7)
        primitives.filled_circle(chrome.screen, centre, 9, fill)
        if style == "hatch":
            primitives.hatch_circle(chrome.screen, centre, 8,
                                    palette.dead_hatch, spacing=4, width=1)
        if style == "dashed":
            primitives.dashed_ring(chrome.screen, centre, 9, 2, ring, dashes=8)
        else:
            primitives.ring(chrome.screen, centre, 9, 2, ring)
        if style == "double":
            primitives.ring(chrome.screen, centre, 6, 1, ring)

        chrome.screen.blit(font.render(label, True, palette.text_muted),
                           (body.x + chrome.space.md + 26, y))
        y += row_h
