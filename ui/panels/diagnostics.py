"""The diagnostics panel: what is structurally wrong, and what to do about it.

Plain functions over a :class:`ui.widgets.Chrome` plus explicit keyword
arguments; nothing here reads the manager. The rows and the Fix button are
hit-testable, so drawing them has to tell the caller where they landed --
:func:`draw_diagnostics` returns a :class:`DiagnosticHits` instead of writing
the rectangles onto somebody else's attributes. The frame around all of it is
the column's, so the panel is a card and a clickable header like every other
resident of the right-hand column.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pygame

from fsa.analysis import Defect
from rendering import primitives
from ui.layout_spec import LayoutSpec
from ui.panels.column import panel_frame
from ui.widgets import Chrome, button


@dataclass(frozen=True)
class DiagnosticHits:
    """The clickable rectangles this panel drew, for the caller to hit-test.

    ``rows`` pairs each clickable defect row with the payload naming the
    states it would focus; ``fix_button`` is the "incomplete" row's Fix
    button, or ``None`` on a frame that did not draw one.
    """

    rows: List[Tuple[pygame.Rect, Dict[str, Any]]] = field(default_factory=list)
    fix_button: Optional[pygame.Rect] = None


#: A row is at most this many lines. Rows are sized from the wrapped text, so
#: a generous cap costs nothing when the message is short -- and the messages
#: that need the room are the ones carrying the explanation.
MAX_LINES = 6
MAX_ROWS = 4
LINE_HEIGHT = 14
ROW_GAP = 6
#: Padding above and below the text inside a row.
ROW_PADDING = 8


def _has_fix(defect: Defect) -> bool:
    return bool(defect.kind == "incomplete")


def _text_budget(row_width: int, defect: Defect) -> int:
    """Width left for the message after the dot and any button."""
    return row_width - 28 - (72 if _has_fix(defect) else 0)


def row_height(lines: int) -> int:
    """How tall a row with this many wrapped lines has to be."""
    return max(1, lines) * LINE_HEIGHT + ROW_PADDING


def wrap(font: pygame.font.Font, message: str, budget: int,
         *, limit: int = MAX_LINES) -> List[str]:
    """Break a defect message into lines that fit.

    Wrapping rather than truncating, because the tail of these messages is the
    part that names the states and symbols -- and, for the incomplete row, the
    sentence explaining that a missing arrow rejects a string for a reason that
    has nothing to do with the language. Cutting that off leaves a warning with
    the lesson removed.

    ``limit`` caps the lines produced. It is a parameter rather than the
    constant it used to be because the other readers of this function are not
    rows in this panel: the marking table's explanation and the exercise
    panel's prompt have their own idea of how much room a thing deserves, and
    a second wrapper for a width is exactly the kind of duplicate that drifts.
    """
    limit = max(1, limit)
    lines: List[str] = []
    current = ""
    for word in message.split():
        trial = (current + " " + word).strip()
        if font.size(trial)[0] <= budget or not current:
            current = trial
            continue
        lines.append(current)
        current = word
        if len(lines) == limit:
            break
    if current and len(lines) < limit:
        lines.append(current)

    # Only the very last line may be elided, and only when the message is
    # genuinely longer than the row can ever show.
    if len(lines) == limit and font.size(lines[-1])[0] > budget:
        while lines[-1] and font.size(lines[-1] + "...")[0] > budget:
            lines[-1] = lines[-1][:-2]
        lines[-1] += "..."
    return lines


def body_height(chrome: Chrome, diagnostics: Sequence[Defect],
                width: int) -> int:
    """How tall the panel body must be to show these rows in full.

    Measured with the same font and the same wrap the drawing uses, so the
    panel cannot be laid out one size and painted another.
    """
    font = chrome.fonts.ui("tiny")
    total = 6
    for defect in diagnostics[:MAX_ROWS]:
        lines = wrap(font, defect.message, _text_budget(width - 12, defect))
        total += row_height(len(lines)) + ROW_GAP
    return total


def draw_diagnostics(chrome: Chrome, *,
                     rect: pygame.Rect,
                     diagnostics: Sequence[Defect],
                     collapsed: bool,
                     layout: LayoutSpec,
                     pressed_rect: Optional[pygame.Rect] = None,
                     mouse_pos: Optional[Tuple[int, int]] = None,
                     ) -> DiagnosticHits:
    """Structural problems with the automaton, each one actionable.

    Rows with named states can be clicked to jump the camera to them; the
    "incomplete" row carries a Fix button that adds a trap state and routes
    every missing transition to it in one click. The visual feedback *is*
    the lesson: a rejection for want of an arrow is a different problem
    from a wrong language, and this panel is where that becomes concrete.

    Args:
        chrome: Surface, theme and fonts.
        rect: The panel's slot in the right-hand column.
        diagnostics: The defects to list; at most the first four are drawn.
        collapsed: Whether the panel is folded down to its header.
        layout: The window layout, which supplies the header strip.
        pressed_rect: The rect the mouse is currently held down on, or None.
        mouse_pos: The cursor position, for hover. Defaults to the live
            position when the caller has none to hand.

    Returns:
        The rows and the Fix button that were drawn, so the caller can
        hit-test exactly what is on screen this frame.
    """
    palette = chrome.palette
    screen = chrome.screen
    hit_rows: List[Tuple[pygame.Rect, Dict[str, Any]]] = []
    fix_button: Optional[pygame.Rect] = None

    body = panel_frame(chrome, key="diagnostics", rect=rect,
                       collapsed=collapsed, layout=layout, mouse_pos=mouse_pos)
    if body is None:
        return DiagnosticHits(rows=hit_rows, fix_button=fix_button)

    if mouse_pos is None:
        mouse_pos = pygame.mouse.get_pos()

    font = chrome.fonts.ui("tiny")
    row_y = body.y + 2
    for defect in diagnostics[:MAX_ROWS]:
        text_lines = wrap(font, defect.message,
                          _text_budget(body.width - 12, defect))
        height = row_height(len(text_lines))
        if row_y + height > body.bottom:
            break
        row = pygame.Rect(body.x + 6, row_y, body.width - 12, height)

        colour = palette.error if defect.is_blocking else palette.warning
        if defect.kind == "unreachable_states":
            colour = palette.unreachable_ring
        primitives.filled_circle(screen, (row.x + 10, row.y + 10), 4, colour)

        for index, text in enumerate(text_lines):
            screen.blit(font.render(text, True, palette.text_muted),
                        (row.x + 22, row.y + 3 + index * LINE_HEIGHT))

        if _has_fix(defect):
            fix = pygame.Rect(row.right - 68, row.y + 6, 64, 24)
            button(chrome, fix, "Complete", accent=True,
                   hovered=fix.collidepoint(mouse_pos),
                   pressed=pressed_rect == fix)
            fix_button = fix
        elif defect.states:
            hit_rows.append((row, {"focus_states": list(defect.states)}))

        row_y += height + ROW_GAP


    return DiagnosticHits(rows=hit_rows, fix_button=fix_button)
