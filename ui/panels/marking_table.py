"""The marking table: Moore's algorithm as the grid a student fills by hand.

The plan ships Moore rather than Hopcroft because "Hopcroft is the better
algorithm and the worse lesson", and this panel is where that choice is
honoured or wasted: a refinement queue cannot be watched, while a triangle of
squares can be copied onto paper and checked against the tool cell by cell. So
the grid is drawn as that artifact -- square cells, ids down the left and
along the bottom, the round that filled a cell inside it, which is also the
length of the shortest word telling those two apart. Empty cells are the
result, not unfinished business: those pairs merge, and they are coloured so.

Plain functions over a :class:`ui.widgets.Chrome`, every input an explicit
keyword argument. Nothing here reads the manager, or imports it.
"""

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Tuple

import pygame

from fsa.minimize import Mark, MarkingTable, Pair
from fsa.symbols import StateId
from rendering import primitives
from ui import widgets
from ui.panels.diagnostics import wrap  # one wrapper for a width, not two
from ui.widgets import Chrome

#: Padding, header and footer in pixels, and the lines the footer has room for
#: above the trap warning. Duplicated from the spacing scale rather than read
#: off the theme because :func:`cell_size` is a pure function of the table and
#: the rectangle -- a caller sizing a window has no Chrome to hand.
PAD, HEAD_H, FOOT_H, MAX_EXPLAIN_LINES = 16, 62, 104, 3
#: Below this a cell has no room for a digit or a label, and the grid
#: becomes a triangle of blank squares -- which shows none of the three
#: things the panel exists to show. Say so instead of drawing confetti.
LEGIBLE_CELL = 18
#: Leading for the explanation. The face is 16px tall, so a 15px step put
#: descenders into the ascenders of the line below.
EXPLAIN_LEADING = 18

#: Cells stop growing at MAX_CELL, past which the grid reads as a few tiles
#: rather than a table; under MIN_CELL there is no grid worth drawing at all.
MAX_CELL, MIN_CELL = 54, 4


@dataclass(frozen=True)
class TableHits:
    """The cells and the close button, returned for the caller to hit-test."""

    cells: Mapping[Pair, pygame.Rect]
    close: Optional[pygame.Rect]


def cell_size(table: MarkingTable, panel: pygame.Rect) -> int:
    """The side of one square cell, so the whole triangle fits ``panel``.

    The row-label gutter and the column-label strip are one cell each, so the
    block is ``len(states)`` cells across and as tall. 0 means no grid at all.
    """
    span = len(table.states)
    if span < 2:
        return 0
    return max(0, min(MAX_CELL, (panel.width - 2 * PAD) // span,
                      (panel.height - HEAD_H - FOOT_H) // span))


def _label(font: pygame.font.Font, state: StateId, budget: int) -> str:
    """A state id shortened to fit its axis, but never to nothing.

    :func:`widgets.elide` returns a bare ellipsis once the budget is smaller
    than one character plus the ellipsis, and a row labelled "..." is a row
    nobody can identify -- worse, the caller was dropping those entirely, so
    half a twenty-state grid had no labels at all while its neighbours did.
    Falling back to the first character keeps every row distinguishable.
    """
    shortened = widgets.elide(font, state, budget)
    if shortened.strip("…"):
        return shortened
    return state[:1]


def _fit_font(chrome: Chrome, cell: int,
              ladder: Tuple[str, ...]) -> Optional[pygame.font.Font]:
    """The largest face in ``ladder`` fitting a cell, or ``None`` for no text.

    ``None`` is a real answer: on a twenty-state machine cells are smaller
    than the smallest legible digit, and confetti reads worse than squares.
    """
    for name in ladder:
        font = chrome.fonts.ui(name)
        if font.get_height() <= cell - 2:
            return font
    return None


def _caption(table: MarkingTable, revealed: float, complete: bool) -> str:
    """One line naming the round now arriving, and what filling it means."""
    arriving = math.ceil(revealed) - 1
    if not table.pairs:
        return "Nothing to fill in: there is no pair of states."
    if complete:
        empty = len(table.equivalent_pairs)
        return (f"{table.rounds} round{'' if table.rounds == 1 else 's'} "
                f"filled the table. " + (
                    f"{empty} cells stayed empty: those pairs merge." if empty
                    else "Every pair differs: this is already minimal."))
    if arriving < 1:
        return "Round 0 marks each pair where just one of the two accepts."
    return (f"Round {arriving}: a symbol leads to a round-{arriving - 1} "
            f"pair; a {arriving}-letter word splits them.")


def _explain(table: MarkingTable, pair: Optional[Pair], revealed: float,
             complete: bool) -> str:
    """Why those two states differ, or do not, or do not yet."""
    if pair is None:
        return "Click a cell to read why those two states are different."
    mark = table.by_pair().get(pair)
    if mark is not None and mark.round < revealed:
        return mark.explain()
    if mark is None and complete:
        return (f"{pair[0]} and {pair[1]} are equivalent: no word tells them "
                f"apart, so they merge into one state.")
    # Deliberately not mark.explain(): this pair is marked in a round nobody
    # has seen yet, and answering early is what an animated table must not do.
    return f"{pair[0]} and {pair[1]} are not marked yet."


def _paint_cell(chrome: Chrome, rect: pygame.Rect, *, mark: Optional[Mark],
                revealed: float, arriving: int, complete: bool, radius: int,
                font: Optional[pygame.font.Font]) -> None:
    """One square, in whichever of the three readings a cell can have."""
    palette = chrome.palette
    if mark is not None and mark.round < revealed:
        newest = mark.round == arriving
        fill = palette.accent if newest else palette.control
        edge = palette.accent if newest else palette.border_strong
        ink = palette.text_on_accent if newest else palette.text
        text = str(mark.round)
    elif mark is None and complete:
        # Never marked, and the run is over. Colouring this like an unreached
        # cell would say "not finished" about the table's one real conclusion.
        fill, edge, ink, text = (palette.accept_fill, palette.accept_ring,
                                 palette.success, "=")
    else:
        fill, edge, ink, text = palette.field, palette.border, palette.text, ""
    primitives.panel(chrome.screen, rect, fill, border=edge, radius=radius)
    if text and font is not None:
        glyph = font.render(text, True, ink)
        if glyph.get_width() <= rect.width - 2:
            chrome.screen.blit(glyph, glyph.get_rect(center=rect.center))


def _draw_grid(chrome: Chrome, *, table: MarkingTable, area: pygame.Rect,
               cell: int, revealed: float, complete: bool,
               selected: Optional[Pair],
               mouse_pos: Tuple[int, int]) -> Dict[Pair, pygame.Rect]:
    """The lower triangle: row ``i`` against column ``j``, for every ``j < i``."""
    palette, screen, states = chrome.palette, chrome.screen, table.states
    marks = table.by_pair()
    digits = _fit_font(chrome, cell, ("body_strong", "small_strong", "tiny"))
    labels = _fit_font(chrome, cell, ("small_strong", "tiny"))
    radius = min(chrome.radius.sm, max(1, cell // 4))
    last, half = len(states) - 1, cell // 2
    # Left-aligned with the title, caption and explanation rather than
    # centred: a centred triangle on a wide panel floats away from the
    # text that explains it.
    left = area.x
    # Top-aligned for the same reason as left-aligned: the caption above
    # the grid describes it, and a small table centred in a large panel
    # drifts away from the sentence that explains it.
    top = area.y

    cells: Dict[Pair, pygame.Rect] = {}
    for row in range(1, len(states)):
        for column in range(row):
            pair = (states[column], states[row])  # sorted axes: canonical
            cells[pair] = rect = pygame.Rect(
                left + (column + 1) * cell, top + (row - 1) * cell, cell, cell)
            _paint_cell(chrome, rect.inflate(-2, -2), mark=marks.get(pair),
                        revealed=revealed, arriving=math.ceil(revealed) - 1,
                        complete=complete, font=digits, radius=radius)

    # Every state but the first has a row and every state but the last has a
    # column, so one loop draws both axes and they cannot drift apart. The
    # invented trap is warned about on the axis as well as in the footer.
    if labels is not None:
        for index, state in enumerate(states):
            # The row gutter is a whole cell wide and the label is centred in
            # it, so it may use the neighbouring margin: budgeting only the
            # cell made two-character ids vanish while their neighbours stayed.
            text = _label(labels, state, cell + PAD - 4)
            glyph = labels.render(text, True, palette.warning
                                  if state == table.invented_trap
                                  else palette.text_muted)
            if index:
                screen.blit(glyph, glyph.get_rect(center=(
                    left + half, top + (index - 1) * cell + half)))
            if index < last:
                screen.blit(glyph, glyph.get_rect(center=(
                    left + (index + 1) * cell + half,
                    top + last * cell + half)))

    chosen = None if selected is None else (min(selected), max(selected))
    for key, rect in cells.items():
        if key == chosen or rect.collidepoint(mouse_pos):
            pygame.draw.rect(screen, palette.selected_ring if key == chosen
                             else palette.hover_ring, rect.inflate(-2, -2),
                             2 if key == chosen else 1, border_radius=radius)
    return cells


def draw(chrome: Chrome, *, table: MarkingTable, panel: pygame.Rect,
         revealed: float, selected: Optional[Pair] = None,
         mouse_pos: Optional[Tuple[int, int]] = None) -> TableHits:
    """Draw the marking table with ``revealed`` rounds of it filled in.

    ``revealed`` is a float so the caller can ease it: a cell fills once
    ``mark.round < revealed``, so 0.0 is an empty grid and ``table.rounds`` or
    more is the finished table. ``selected`` names the pair whose explanation
    is printed, in either order; ``mouse_pos`` defaults to the live cursor.
    """
    palette, screen = chrome.palette, chrome.screen
    if mouse_pos is None:
        mouse_pos = pygame.mouse.get_pos()
    widgets.card(chrome, panel)
    complete = revealed >= table.rounds
    x, budget = panel.x + PAD, panel.width - 2 * PAD

    close = pygame.Rect(panel.right - PAD - 26, panel.y + PAD - 2, 26, 26)
    widgets.button(chrome, close, "", hovered=close.collidepoint(mouse_pos))
    widgets.cross(chrome, close, palette.text_muted)

    title, font = chrome.fonts.ui("title"), chrome.fonts.ui("small")
    head = widgets.elide(title, "Moore's marking table", budget - 34)
    screen.blit(title.render(head, True, palette.text), (x, panel.y + PAD - 2))
    screen.blit(font.render(
        widgets.elide(font, _caption(table, revealed, complete), budget), True,
        palette.text_muted), (x, panel.y + PAD + 26))

    area = pygame.Rect(x, panel.y + HEAD_H, budget,
                       panel.height - HEAD_H - FOOT_H)
    cell = cell_size(table, panel)
    cells: Dict[Pair, pygame.Rect] = {}
    if cell < LEGIBLE_CELL:
        # Fewer than two states means no pairs; a tiny panel cannot show them.
        note = (f"Not enough room to label {len(table.pairs)} cells -- make "
                f"the window taller." if table.pairs
                else "Fewer than two states: already minimal.")
        surface = font.render(widgets.elide(font, note, budget), True,
                              palette.text_faint)
        # Clamped below the header: on a very short panel `area` has a negative
        # height, and centring in it put the note on top of the title.
        centre = max(area.centery, panel.y + HEAD_H + surface.get_height())
        screen.blit(surface, surface.get_rect(center=(area.centerx, centre)))
    else:
        cells = _draw_grid(chrome, table=table, area=area, cell=cell,
                           revealed=revealed, complete=complete,
                           selected=selected, mouse_pos=mouse_pos)

    if complete and table.equivalent_pairs:
        legend = "=  equivalent: no word tells the pair apart, so both merge"
        screen.blit(font.render(widgets.elide(font, legend, budget), True,
                                palette.success), (x, panel.bottom - 100))

    pair = None if selected is None else (min(selected), max(selected))
    message = _explain(table, pair if pair in cells else None, revealed,
                       complete) if cells else ""
    lines = wrap(font, message, budget)
    if len(lines) > MAX_EXPLAIN_LINES:
        # Say it was cut. Without the ellipsis the sentence simply stops, and
        # the amber trap note underneath reads as its next words.
        lines = lines[:MAX_EXPLAIN_LINES]
        lines[-1] = widgets.elide(font, lines[-1] + " ...", budget)
    for index, line in enumerate(lines):
        screen.blit(font.render(line, True, palette.text),
                    (x, panel.bottom - 84 + index * EXPLAIN_LEADING))

    if table.invented_trap is not None:
        note = (f"{table.invented_trap} was added by the tool to complete a "
                f"partial delta -- you did not draw it.")
        screen.blit(font.render(widgets.elide(font, note, budget), True,
                                palette.warning), (x, panel.bottom - 30))
    return TableHits(cells=MappingProxyType(cells), close=close)
