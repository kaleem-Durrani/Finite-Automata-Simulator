"""Where the right-hand panels sit this frame.

Pure geometry over a small bag of animation state. The manager owns the state
and hands it in; nothing here reaches back for it, so the stacking rules can be
read and reasoned about without the rest of the interface.

Two motions compose here and they are deliberately separate. A panel *slides*
horizontally when it appears or leaves, and it *opens* vertically when it is
unfolded. The space a panel occupies scales by its slide, so the panels below
glide up into the gap a departing one releases instead of jumping.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pygame

from rendering.animation import Animated, ease_out
from rendering.theme import Theme
from ui.layout_spec import (
    PANEL_GAP,
    PANEL_HEADER_HEIGHT,
    PANEL_MARGIN,
    PANEL_WIDTH,
    RUN_BODY_HEIGHT,
    LayoutSpec,
)

#: One entry per panel on screen: its key, its rectangle, its slide progress.
Column = List[Tuple[str, pygame.Rect, float]]


@dataclass
class ColumnState:
    """The motion and fold state the column carries between frames."""

    #: 0..1 horizontal slide, per panel key. 1 is fully on screen.
    slides: Dict[str, Animated] = field(default_factory=dict)
    #: 0..1 vertical openness, per panel key. 0 is folded to its header.
    opens: Dict[str, Animated] = field(default_factory=dict)
    #: Which panels the user has folded away.
    collapsed: Dict[str, bool] = field(default_factory=dict)

    def toggle(self, key: str) -> None:
        """Fold a panel away, or bring it back."""
        self.collapsed[key] = not self.collapsed.get(key, False)

    def is_collapsed(self, key: str) -> bool:
        return bool(self.collapsed.get(key))

    def advance(self, dt: float) -> None:
        """Close every animation a little toward its target."""
        for entry in self.slides.values():
            entry.update(dt)
        for entry in self.opens.values():
            entry.update(dt)

    def slide(self, key: str, visible: bool, theme: Theme) -> float:
        entry = self.slides.get(key)
        if entry is None:
            entry = Animated(value=0.0, target=0.0,
                             duration=theme.motion.normal, easing=ease_out)
            self.slides[key] = entry
        entry.set(1.0 if visible else 0.0)
        return entry.value

    def openness(self, key: str, theme: Theme) -> float:
        entry = self.opens.get(key)
        if entry is None:
            entry = Animated(value=1.0, target=1.0,
                             duration=theme.motion.quick, easing=ease_out)
            self.opens[key] = entry
        entry.set(0.0 if self.is_collapsed(key) else 1.0)
        return entry.value


#: States, Alphabet, Start, Accepting, Kind, Denotes. Counted rather than
#: written as a literal, because adding the fifth row and leaving this at four
#: clipped it off the bottom of a panel that was still laid out for four.
STATUS_ROWS = 6


def status_body_height(*, warn_no_accepting: bool) -> int:
    """One line per row, plus the warning line only when there is one."""
    return STATUS_ROWS * 19 + (18 if warn_no_accepting else 0) + 12


def compute(state: ColumnState, *, layout: LayoutSpec, theme: Theme,
            execution_active: bool, legend_rows: int,
            diagnostics_height: int, warn_no_accepting: bool) -> Column:
    """Lay out the right-hand panels for this frame.

    Deterministic from current state -- panel heights depend only on content
    counts known before drawing. Each panel slides horizontally by its own
    progress, and the space it occupies scales with that progress, so panels
    below glide up as one above departs instead of jumping.

    A panel's height is its header plus however much body it currently wants,
    eased between the two so collapsing glides. Bodies are sized from their
    content rather than from a fixed constant, which is what left every panel
    looking half empty.
    """
    width, margin, gap = PANEL_WIDTH, PANEL_MARGIN, PANEL_GAP
    home_x = layout.column_home_x()
    y = float(layout.column_top())
    limit = layout.column_limit()

    wanted = [
        ("status", True, status_body_height(warn_no_accepting=warn_no_accepting)),
        ("run", execution_active, RUN_BODY_HEIGHT),
        # The diagnostics panel measures its own wrapped text and passes the
        # answer in, so the column cannot lay it out one size and the panel
        # then paint another.
        ("diagnostics", diagnostics_height > 6, diagnostics_height),
        ("legend", legend_rows >= 2, legend_rows * 22 + 6),
    ]

    column: Column = []
    for key, visible, body in wanted:
        height = PANEL_HEADER_HEIGHT + int(body * state.openness(key, theme))
        # A collapsed panel is a header, and a header always fits, so only the
        # expanded height can push a panel off the bottom.
        fits = y + PANEL_HEADER_HEIGHT <= limit
        t = state.slide(key, visible and fits, theme)
        if t <= 0.01:
            continue
        height = min(height, max(PANEL_HEADER_HEIGHT, int(limit - y)))
        offset = (1.0 - t) * (width + margin * 2)
        rect = pygame.Rect(int(home_x + offset), int(y), width, height)
        column.append((key, rect, t))
        y += (height + gap) * t
    return column
