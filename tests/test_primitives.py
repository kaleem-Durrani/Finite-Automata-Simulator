"""Drawing primitives.

Mostly about the stamp cache: node chrome that is expensive to draw and
identical for every node of the same kind, size and colour. Getting the cache
key wrong would show as stale colours after a theme switch, which is exactly the
kind of bug a test should catch rather than an eye.
"""

import pygame
import pytest

from rendering import primitives


@pytest.fixture(autouse=True)
def _display():
    pygame.init()
    surface = pygame.display.set_mode((240, 240))
    primitives.clear_stamp_cache()
    yield surface
    pygame.quit()


def blank(size=(240, 240)):
    surface = pygame.Surface(size, pygame.SRCALPHA)
    surface.fill((0, 0, 0, 0))
    return surface


def painted(surface) -> int:
    """How many pixels are not fully transparent."""
    count = 0
    for x in range(0, surface.get_width(), 2):
        for y in range(0, surface.get_height(), 2):
            if surface.get_at((x, y)).a > 0:
                count += 1
    return count


# ---------------------------------------------------------------------------
# The stamp cache
# ---------------------------------------------------------------------------


def test_repeated_draws_reuse_one_stamp():
    """Drawing this fresh each time made unreachable states twice as expensive
    to render as ordinary ones, and it grew faster than linearly."""
    surface = blank()
    for i in range(20):
        primitives.dashed_ring(surface, (60 + i, 60), 24, 2, (255, 0, 0, 255))
    assert len(primitives._STAMPS) == 1


def test_a_different_colour_is_a_different_stamp():
    """Otherwise switching theme would leave the old colour on screen."""
    surface = blank()
    primitives.dashed_ring(surface, (60, 60), 24, 2, (255, 0, 0, 255))
    primitives.dashed_ring(surface, (60, 60), 24, 2, (0, 255, 0, 255))
    assert len(primitives._STAMPS) == 2


def test_a_different_radius_is_a_different_stamp():
    """Radius follows the zoom, so this must not go stale either."""
    surface = blank()
    primitives.dashed_ring(surface, (60, 60), 24, 2, (255, 0, 0, 255))
    primitives.dashed_ring(surface, (60, 60), 30, 2, (255, 0, 0, 255))
    assert len(primitives._STAMPS) == 2


def test_the_cache_does_not_grow_without_bound():
    surface = blank()
    for radius in range(5, 5 + primitives._STAMP_LIMIT + 40):
        primitives.dashed_ring(surface, (60, 60), radius % 90 + 5, 2,
                               (radius % 255, 0, 0, 255))
    assert len(primitives._STAMPS) <= primitives._STAMP_LIMIT


def test_a_cached_stamp_draws_the_same_pixels():
    """The cache must be invisible. Second draw, same picture."""
    first = blank()
    primitives.dashed_ring(first, (120, 120), 40, 3, (255, 128, 0, 255))

    second = blank()
    primitives.dashed_ring(second, (120, 120), 40, 3, (255, 128, 0, 255))

    assert pygame.image.tostring(first, "RGBA") == pygame.image.tostring(second, "RGBA")


# ---------------------------------------------------------------------------
# The shapes themselves
# ---------------------------------------------------------------------------


def test_a_dashed_ring_is_broken():
    """It has to read as dashes, not as a solid outline."""
    dashed = blank()
    primitives.dashed_ring(dashed, (120, 120), 50, 2, (255, 255, 255, 255))

    solid = blank()
    primitives.ring(solid, (120, 120), 50, 2, (255, 255, 255, 255))

    assert 0 < painted(dashed) < painted(solid) * 1.6
    assert painted(dashed) > 0


def test_a_hatch_stays_inside_its_circle():
    surface = blank()
    centre, radius = (120, 120), 50
    primitives.hatch_circle(surface, centre, radius, (255, 0, 0, 200))

    for x in range(0, 240, 3):
        for y in range(0, 240, 3):
            if surface.get_at((x, y)).a > 0:
                distance = ((x - centre[0]) ** 2 + (y - centre[1]) ** 2) ** 0.5
                assert distance <= radius + 2, f"({x},{y}) escaped the circle"


def test_a_hatch_actually_marks_the_circle():
    surface = blank()
    primitives.hatch_circle(surface, (120, 120), 50, (255, 0, 0, 200))
    assert painted(surface) > 50


def test_tiny_shapes_are_skipped_rather_than_drawn_wrong():
    surface = blank()
    primitives.hatch_circle(surface, (120, 120), 2, (255, 0, 0, 255))
    primitives.dashed_ring(surface, (120, 120), 1, 1, (255, 0, 0, 255))
    primitives.filled_circle(surface, (120, 120), 0, (255, 0, 0, 255))
    assert painted(surface) == 0


# ---------------------------------------------------------------------------
# Coordinate safety
# ---------------------------------------------------------------------------


def test_extreme_coordinates_do_not_overflow():
    """gfxdraw takes signed shorts and raises above 32767 rather than clipping.

    The camera's zoom is unbounded, so world coordinates reach that easily.
    """
    surface = blank()
    for point in [(10 ** 9, 10 ** 9), (-10 ** 9, 5), (0, -10 ** 12)]:
        primitives.filled_circle(surface, point, 20, (255, 0, 0, 255))
        primitives.ring(surface, point, 20, 2, (255, 0, 0, 255))
        primitives.dashed_ring(surface, point, 20, 2, (255, 0, 0, 255))
        primitives.hatch_circle(surface, point, 20, (255, 0, 0, 255))
        primitives.stroke_path(surface, [point, (5, 5)], 2, (255, 0, 0, 255))
        primitives.polygon(surface, [point, (5, 5), (9, 9)], (255, 0, 0, 255))


def test_translucent_panels_keep_their_alpha():
    """Passing RGBA to pygame.draw discards alpha on an opaque surface, which
    is why the old 'translucent' overlays never were."""
    surface = pygame.Surface((100, 100))
    surface.fill((0, 0, 0))
    primitives.translucent_panel(surface, pygame.Rect(10, 10, 80, 80),
                                 (255, 255, 255, 128), radius=4)
    pixel = surface.get_at((50, 50))
    assert 100 < pixel.r < 200, f"expected a blend, got {pixel}"
