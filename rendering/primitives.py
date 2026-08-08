"""Anti-aliased drawing primitives.

``pygame.draw`` produces hard, stair-stepped edges. Its text, meanwhile, is
anti-aliased -- and that mismatch, crisp type against jagged shapes, is most of
why the old interface looked unfinished.

``pygame.gfxdraw`` has anti-aliased primitives, with one serious catch: it
passes coordinates as signed 16-bit integers and raises ``OverflowError`` above
32767 rather than clipping. The camera's zoom is unbounded at 5x, so world
coordinates can exceed that easily. **Every function here clamps before
drawing**, and callers should cull offscreen geometry first.

Thick strokes are built as polygons rather than drawn as lines, because
``pygame.draw.aaline`` has no width parameter (its fifth argument is ``blend``,
which is a common mistake).
"""

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pygame
import pygame.gfxdraw

Point = Tuple[float, float]
Color = Tuple[int, ...]

# gfxdraw passes signed shorts. Stay well inside the limit.
COORD_LIMIT = 30000


def _safe(point: Point) -> Tuple[int, int]:
    """Clamp a point into the range gfxdraw can accept."""
    return (
        int(max(-COORD_LIMIT, min(COORD_LIMIT, point[0]))),
        int(max(-COORD_LIMIT, min(COORD_LIMIT, point[1]))),
    )


def _safe_all(points: Sequence[Point]) -> List[Tuple[int, int]]:
    return [_safe(point) for point in points]


def on_screen(point: Point, size: Tuple[int, int], margin: float = 0.0) -> bool:
    """Whether a point is within the surface, allowing a margin."""
    return (-margin <= point[0] <= size[0] + margin
            and -margin <= point[1] <= size[1] + margin)


# ----------------------------------------------------------------------
# Circles
# ----------------------------------------------------------------------

def filled_circle(surface: pygame.Surface, centre: Point, radius: float,
                  color: Color) -> None:
    """A solid circle with a smooth edge."""
    radius = int(radius)
    if radius < 1:
        return
    x, y = _safe(centre)
    pygame.gfxdraw.filled_circle(surface, x, y, radius, color)
    pygame.gfxdraw.aacircle(surface, x, y, radius, color)


def ring(surface: pygame.Surface, centre: Point, radius: float, width: float,
         color: Color) -> None:
    """An anti-aliased circular outline of a given thickness.

    Drawn as a stack of concentric anti-aliased circles: gfxdraw has no
    stroke-width primitive, and this reads better at small sizes than a
    polygon-based annulus.
    """
    radius = int(radius)
    width = max(1, int(round(width)))
    if radius < 1:
        return

    x, y = _safe(centre)
    for i in range(width):
        r = radius - i
        if r >= 1:
            pygame.gfxdraw.aacircle(surface, x, y, r, color)
    # A second pass on the outer edge firms the line up without thickening it.
    pygame.gfxdraw.aacircle(surface, x, y, radius, color)


# ----------------------------------------------------------------------
# Stamps
# ----------------------------------------------------------------------
#
# Some node chrome is expensive to draw and identical for every node of the
# same kind, size and colour: a dashed ring is dozens of small polygons, and a
# hatch needs two scratch surfaces. Drawing those per node per frame was the
# single largest cost in the renderer -- 50 unreachable states cost twice what
# 50 ordinary ones did, and grew faster.
#
# So each is rendered once into a small surface and blitted thereafter. The key
# includes everything that affects the pixels, so a theme change or a zoom
# simply produces a new entry rather than a stale one.

_STAMPS: Dict[Tuple[Any, ...], pygame.Surface] = {}
_STAMP_LIMIT = 192


def clear_stamp_cache() -> None:
    """Drop every cached stamp. Only tests should need this."""
    _STAMPS.clear()


def _stamp(key: Tuple[Any, ...], size: int,
           draw: Callable[[pygame.Surface], None]) -> pygame.Surface:
    """Fetch a cached stamp, drawing it on first use."""
    cached = _STAMPS.get(key)
    if cached is not None:
        return cached

    layer = pygame.Surface((size, size), pygame.SRCALPHA)
    draw(layer)
    if len(_STAMPS) >= _STAMP_LIMIT:
        # A crude reset rather than an LRU: entries are cheap to rebuild, and
        # the working set is a handful of radii per theme.
        _STAMPS.clear()
    _STAMPS[key] = layer
    return layer


def _blit_centred(surface: pygame.Surface, stamp: pygame.Surface,
                  centre: Point) -> None:
    """Blit a stamp centred on a point.

    The destination is clamped like every other coordinate here. blit rejects
    positions outside the int range rather than clipping, so a node far off
    screen at high zoom would raise -- the same trap gfxdraw sets, reintroduced
    when this drawing moved behind a cache.
    """
    x, y = _safe(centre)
    surface.blit(stamp, (x - stamp.get_width() // 2,
                         y - stamp.get_height() // 2))


def dashed_ring(surface: pygame.Surface, centre: Point, radius: float,
                width: float, color: Color, dashes: int = 16,
                duty: float = 0.55) -> None:
    """A circular outline broken into dashes.

    Used for states no word can reach. A dash pattern is a *shape* difference,
    so it still reads when the colour does not -- in greyscale, on a projector,
    or to a colour-blind reader.

    Cached: this is the same picture for every unreachable state at a given
    size, and drawing it fresh each time made those states twice as expensive
    to render as ordinary ones.
    """
    radius = int(radius)
    width = max(1, int(round(width)))
    if radius < 2 or dashes < 1:
        return

    size = radius * 2 + width * 2 + 4
    key = ("dashes", radius, width, tuple(color), dashes, round(duty, 2))

    def render(layer: pygame.Surface) -> None:
        middle = size / 2
        span = 2 * math.pi / dashes
        for i in range(dashes):
            start = i * span
            end = start + span * duty
            steps = max(2, int(radius * span / 3))
            arc = [
                (middle + math.cos(start + (end - start) * s / steps) * radius,
                 middle + math.sin(start + (end - start) * s / steps) * radius)
                for s in range(steps + 1)
            ]
            stroke_path(layer, arc, width, color)

    _blit_centred(surface, _stamp(key, size, render), centre)


def hatch_circle(surface: pygame.Surface, centre: Point, radius: float,
                 color: Color, spacing: float = 7.0, width: int = 1,
                 angle: float = math.pi / 4) -> None:
    """Fill a circle with diagonal lines.

    The second signal marking a trap state, alongside its colour. Built by
    drawing lines across a square and masking them to the circle -- clipping
    each line analytically is more code for the same pixels.

    Cached, because it allocated two surfaces every time it was called, which
    for a canvas full of traps meant two allocations per node per frame.
    """
    radius = int(radius)
    if radius < 4:
        return

    size = radius * 2 + 2
    key = ("hatch", radius, tuple(color), round(spacing, 1), width,
           round(angle, 3))

    def render(layer: pygame.Surface) -> None:
        direction = (math.cos(angle), math.sin(angle))
        normal = (-direction[1], direction[0])
        reach = size * 1.5
        steps = int(reach / spacing)
        for i in range(-steps, steps + 1):
            offset_x = normal[0] * i * spacing + size / 2
            offset_y = normal[1] * i * spacing + size / 2
            pygame.draw.line(
                layer, color,
                (offset_x - direction[0] * reach, offset_y - direction[1] * reach),
                (offset_x + direction[0] * reach, offset_y + direction[1] * reach),
                width)

        mask = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(mask, size // 2, size // 2, radius,
                                     (255, 255, 255, 255))
        layer.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)

    _blit_centred(surface, _stamp(key, size, render), centre)


def soft_shadow(surface: pygame.Surface, centre: Point, radius: float,
                color: Color, layers: int = 5, spread: float = 5.0) -> None:
    """A blurred disc under a node, to lift it off the canvas.

    Approximated with a few translucent circles of decreasing size rather than a
    real blur, which would cost a surface allocation per frame.
    """
    if radius < 2:
        return
    base_alpha = color[3] if len(color) > 3 else 60
    for i in range(layers, 0, -1):
        t = i / layers
        alpha = int(base_alpha * (1.0 - t) ** 1.5)
        if alpha <= 1:
            continue
        r = int(radius + spread * t)
        x, y = _safe((centre[0], centre[1] + spread * 0.45))
        pygame.gfxdraw.filled_circle(surface, x, y, r, (*color[:3], alpha))


def glow(surface: pygame.Surface, centre: Point, radius: float, color: Color,
         layers: int = 6, spread: float = 14.0) -> None:
    """A halo around a node, used for the active state during execution."""
    if radius < 1:
        return
    base_alpha = color[3] if len(color) > 3 else 50
    x, y = _safe(centre)
    for i in range(layers, 0, -1):
        t = i / layers
        alpha = int(base_alpha * (1.0 - t) ** 2)
        if alpha <= 1:
            continue
        pygame.gfxdraw.filled_circle(surface, x, y, int(radius + spread * t),
                                     (*color[:3], alpha))


# ----------------------------------------------------------------------
# Strokes
# ----------------------------------------------------------------------

def _quad(a: Point, b: Point, half: float) -> List[Point]:
    """A rectangle covering the segment a-b with the given half-width."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return []
    nx, ny = -dy / length * half, dx / length * half
    return [(a[0] + nx, a[1] + ny), (b[0] + nx, b[1] + ny),
            (b[0] - nx, b[1] - ny), (a[0] - nx, a[1] - ny)]


def stroke_path(surface: pygame.Surface, path: Sequence[Point], width: float,
                color: Color) -> None:
    """
    Draw a polyline with a given thickness, anti-aliased.

    Each segment becomes a quad, with a small disc at the joins so corners do
    not show notches. ``pygame.draw.aaline`` cannot do width, and
    ``pygame.draw.lines`` with width > 1 is not anti-aliased.
    """
    if len(path) < 2:
        return

    half = max(0.5, width / 2.0)
    for i in range(len(path) - 1):
        quad = _quad(path[i], path[i + 1], half)
        if not quad:
            continue
        points = _safe_all(quad)
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)

    if width > 2.2:
        joint = int(half)
        for point in path[1:-1]:
            x, y = _safe(point)
            pygame.gfxdraw.filled_circle(surface, x, y, joint, color)
            pygame.gfxdraw.aacircle(surface, x, y, joint, color)


def polygon(surface: pygame.Surface, points: Sequence[Point], color: Color) -> None:
    """A filled, anti-aliased polygon."""
    if len(points) < 3:
        return
    safe = _safe_all(points)
    pygame.gfxdraw.filled_polygon(surface, safe, color)
    pygame.gfxdraw.aapolygon(surface, safe, color)


def line(surface: pygame.Surface, start: Point, end: Point, width: float,
         color: Color) -> None:
    """A single anti-aliased segment of a given thickness."""
    stroke_path(surface, [start, end], width, color)


def dashed_path(surface: pygame.Surface, path: Sequence[Point], width: float,
                color: Color, dash: float = 9.0, gap: float = 7.0) -> None:
    """A dashed polyline, used for the edge being drawn by the user."""
    if len(path) < 2:
        return

    carry = 0.0
    drawing = True
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg < 1e-9:
            continue
        travelled = 0.0
        while travelled < seg:
            wanted = (dash if drawing else gap) - carry
            take = min(wanted, seg - travelled)
            if drawing:
                start = (a[0] + (b[0] - a[0]) * (travelled / seg),
                         a[1] + (b[1] - a[1]) * (travelled / seg))
                finish = (a[0] + (b[0] - a[0]) * ((travelled + take) / seg),
                          a[1] + (b[1] - a[1]) * ((travelled + take) / seg))
                line(surface, start, finish, width, color)
            travelled += take
            carry += take
            if carry >= (dash if drawing else gap) - 1e-9:
                drawing = not drawing
                carry = 0.0


# ----------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------

def panel(surface: pygame.Surface, rect: pygame.Rect, color: Color,
          radius: int = 8, border: Optional[Color] = None,
          border_width: int = 1) -> None:
    """A rounded rectangle, optionally outlined.

    ``pygame.draw.rect`` gained ``border_radius`` in pygame 2 and antialiases
    the corners itself, so this needs no gfxdraw.
    """
    if rect.width <= 0 or rect.height <= 0:
        return
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border is not None and border_width > 0:
        pygame.draw.rect(surface, border, rect, width=border_width,
                         border_radius=radius)


def translucent_panel(surface: pygame.Surface, rect: pygame.Rect, color: Color,
                      radius: int = 8, border: Optional[Color] = None) -> None:
    """A panel with real alpha.

    Passing an RGBA colour straight to ``pygame.draw`` silently discards the
    alpha channel on an opaque surface, which is why the old "translucent"
    overlays were never translucent. Drawing to an SRCALPHA scratch surface and
    blitting it is what actually blends.
    """
    if rect.width <= 0 or rect.height <= 0:
        return
    scratch = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(scratch, color, scratch.get_rect(), border_radius=radius)
    if border is not None:
        pygame.draw.rect(scratch, border, scratch.get_rect(), width=1,
                         border_radius=radius)
    surface.blit(scratch, rect.topleft)


def dim(surface: pygame.Surface, color: Color) -> None:
    """Wash the whole surface, for the backdrop behind a modal dialog."""
    veil = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    veil.fill(color)
    surface.blit(veil, (0, 0))


def dot_grid(surface: pygame.Surface, color: Color, spacing: float,
             origin: Point, size: Tuple[int, int]) -> None:
    """A faint dot grid, to give the canvas a sense of place while panning.

    Skipped when the spacing gets tight, both because it stops reading as a
    grid and because the dot count would grow without bound.
    """
    if spacing < 14:
        return

    start_x = origin[0] % spacing
    start_y = origin[1] % spacing
    x = start_x
    while x < size[0]:
        y = start_y
        while y < size[1]:
            surface.set_at((int(x), int(y)), color)
            y += spacing
        x += spacing
