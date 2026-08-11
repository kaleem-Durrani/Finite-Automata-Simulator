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
    for defect in diagnostics[:4]:
        if row_y + 36 > body.bottom:
            break
        row = pygame.Rect(body.x + 6, row_y, body.width - 12, 36)

        colour = palette.error if defect.is_blocking else palette.warning
        if defect.kind == "unreachable_states":
            colour = palette.unreachable_ring
        primitives.filled_circle(screen, (row.x + 10, row.y + 10), 4, colour)

        has_fix = defect.kind == "incomplete"
        budget = row.width - 28 - (44 if has_fix else 0)

        # Two wrapped lines rather than one truncated one: a single line
        # always cut exactly the part that named the states and symbols,
        # which is the panel's entire teaching content.
        words = defect.message.split()
        lines: List[str] = []
        current = ""
        for word in words:
            trial = (current + " " + word).strip()
            if font.size(trial)[0] <= budget or not current:
                current = trial
            else:
                lines.append(current)
                current = word
                if len(lines) == 2:
                    break
        if current and len(lines) < 2:
            lines.append(current)
        if len(lines) == 2 and font.size(lines[1])[0] > budget:
            while lines[1] and font.size(lines[1] + "...")[0] > budget:
                lines[1] = lines[1][:-2]
            lines[1] += "..."

        for j, text in enumerate(lines):
            screen.blit(font.render(text, True, palette.text_muted),
                        (row.x + 22, row.y + 3 + j * 14))

        if has_fix:
            fix = pygame.Rect(row.right - 40, row.y + 6, 36, 24)
            button(chrome, fix, "Fix", accent=True,
                   hovered=fix.collidepoint(mouse_pos),
                   pressed=pressed_rect == fix)
            fix_button = fix
        elif defect.states:
            hit_rows.append((row, {"focus_states": list(defect.states)}))

        row_y += 40

    return DiagnosticHits(rows=hit_rows, fix_button=fix_button)
