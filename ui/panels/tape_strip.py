"""The input tape strip along the bottom of the window.

One cell per symbol of the string under test, the read head above the cell
being read, and the motion that carries them: the strip slides up from the
bottom edge when a run starts, the scroll between cells eases, and the cell
under the head pops as the position changes.

Plain functions over a :class:`ui.widgets.Chrome` plus explicit keyword
arguments. Nothing in here may import ``ui.ui_manager``.
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import pygame

from rendering import primitives
from rendering.animation import Animated, Timer
from ui.layout_spec import LayoutSpec
from ui.widgets import Chrome

#: The right-hand column's per-panel entries: key, rectangle, slide progress.
Column = Sequence[Tuple[str, pygame.Rect, float]]


@dataclass
class TapeState:
    """What the strip carries from one frame to the next.

    Data only -- every decision made from it is made by :func:`draw`.

    ``cache`` is the content last drawn while the run was live, which the exit
    animation still needs after the application has moved on. ``last_step`` is
    the read-head position the pop was last started for. ``bounds`` is the
    region the last drawn strip occupied.
    """

    last_step: int = -1
    cache: Optional[Tuple[str, int, Any]] = None
    bounds: Optional[pygame.Rect] = None


def draw(chrome: Chrome, *, test_string: str, current_step: int,
         visible: bool, slide: float, was_hidden: bool, layout: LayoutSpec,
         column: Column, scroll: Animated, pop: Timer, state: TapeState,
         run: Optional[Any] = None) -> Optional[pygame.Rect]:
    """Draw the input tape: sliding in and out, scrolling, popping.

    The strip glides up from the bottom edge when a run starts and back
    down when it stops -- which needs the *previous* run's content for the
    exit animation, so the last drawn state is cached. The scroll between
    cells eases rather than jumping, and the cell under the read head pops
    briefly when the position changes.

    ``slide`` is the strip's own 0..1 slide progress and ``was_hidden`` says
    whether it was off screen before this frame advanced that progress.
    Returns the bounds actually drawn, for hit-testing.
    """
    if visible:
        state.cache = (test_string, current_step, run)
    elif state.cache is not None:
        test_string, current_step, run = state.cache
    if slide <= 0.01:
        state.cache = None
        state.bounds = None
        return None

    palette = chrome.palette
    strip = layout.string_strip
    cell_w, cell_h = 34, 40
    gap = 6
    step = cell_w + gap
    count = max(1, len(test_string))
    total = count * step - gap

    # Slide offset: fully below the bottom edge at t=0, in place at t=1.
    # The travel spans the real distance to the window edge; a fixed 74px
    # meant the exit animation stopped mid-screen and the strip blinked out.
    rise = (1.0 - slide) * (layout.height - strip.y + 8)
    top = strip.y + int(rise)

    # The pop restarts whenever the read head moves.
    if current_step != state.last_step:
        state.last_step = current_step
        pop.start()

    # The drawable span stops where the right column begins, so long
    # strings scroll instead of painting cells across the diagnostics
    # panel.
    left_bound = 40
    right_bound = layout.width - 40
    if column:
        right_bound = min(right_bound,
                          min(rect.x for _k, rect, _t in column) - 12)

    centre_x = (left_bound + right_bound) // 2
    span = right_bound - left_bound
    if total <= span:
        target_x = centre_x - total // 2
    else:
        wanted = centre_x - int((current_step + 0.5) * step)
        target_x = max(right_bound - total, min(left_bound, wanted))

    # Ease toward the target, but jump on the frame the strip first
    # appears. The old guard tested t < 0.05, which one 16ms update has
    # already passed, so the strip visibly travelled in from x=0.
    if was_hidden:
        scroll.jump_to(float(target_x))
    else:
        scroll.set(float(target_x))
    start_x = int(scroll.value)

    stopped_at = getattr(run, "stopped_at", len(test_string))

    if not test_string:
        rect = pygame.Rect(centre_x - 30, top, 60, cell_h)
        primitives.panel(chrome.screen, rect, palette.strip_cell,
                         radius=chrome.radius.md, border=palette.border)
        glyph = chrome.fonts.ui("body").render("ε", True, palette.text_muted)
        chrome.screen.blit(glyph, glyph.get_rect(center=rect.center))
        # The ε cell sets no bounds of its own.
        return state.bounds

    state.bounds = pygame.Rect(
        max(left_bound - 8, min(start_x, right_bound) - 8), top - 12,
        min(total, span) + 16, cell_h + 24)

    mono = chrome.fonts.mono("strip")
    pop_amount = 1.0 - pop.progress
    for i, char in enumerate(test_string):
        x = start_x + i * step
        if x < -step or x + cell_w > right_bound + cell_w // 2:
            continue

        consumed = i < current_step
        unreached = i >= stopped_at and stopped_at < len(test_string)
        is_current = i == current_step

        rect = pygame.Rect(x, top, cell_w, cell_h)
        if is_current and pop_amount > 0.01:
            grow = int(4 * pop_amount)
            rect = rect.inflate(grow, grow)

        if is_current:
            fill, text_color = palette.strip_cell_current, palette.text_on_accent
        elif consumed:
            fill, text_color = palette.strip_cell_done, palette.strip_text_done
        else:
            fill, text_color = palette.strip_cell, palette.strip_text
        if unreached and not is_current:
            text_color = palette.text_faint

        primitives.elevated_panel(
            chrome.screen, rect, fill, radius=chrome.radius.md,
            border=palette.accent if is_current else palette.border,
            shadow=palette.shadow, lift=3 if is_current else 2,
            bevel_light=palette.bevel_light, bevel_dark=palette.bevel_dark)
        glyph = mono.render(char, True, text_color)
        chrome.screen.blit(glyph, glyph.get_rect(center=rect.center))

        if is_current:
            # The read head: a marker above the cell, pointing at it.
            primitives.pointer(chrome.screen, (rect.centerx, rect.top - 4),
                               8, palette.accent, direction="down")

        if consumed:
            pygame.draw.line(chrome.screen, palette.success,
                             (rect.x + 8, rect.bottom + 3),
                             (rect.right - 8, rect.bottom + 3), 2)

    if stopped_at < len(test_string):
        marker_x = start_x + stopped_at * step - gap // 2
        pygame.draw.line(chrome.screen, palette.error,
                         (marker_x, top - 4), (marker_x, top + cell_h + 4), 2)

    return state.bounds
