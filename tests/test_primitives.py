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


# ---------------------------------------------------------------------------
# Elevation
# ---------------------------------------------------------------------------


def test_a_shaded_circle_reuses_one_stamp():
    """The gradient is a scanline loop plus a mask -- the most expensive fill
    in the module. Redrawing it per node per frame would undo the cache win."""
    surface = blank()
    for i in range(12):
        primitives.shaded_circle(surface, (60 + i, 60), 24, (100, 120, 150))
    assert len(primitives._STAMPS) == 1


def test_a_shaded_circle_restamps_for_radius_and_colour():
    """Radius follows the zoom and colour follows the theme; a stale entry
    for either would be visible immediately."""
    surface = blank()
    primitives.shaded_circle(surface, (60, 60), 24, (100, 120, 150))
    primitives.shaded_circle(surface, (60, 60), 30, (100, 120, 150))
    primitives.shaded_circle(surface, (60, 60), 24, (150, 120, 100))
    assert len(primitives._STAMPS) == 3


def test_a_shaded_circle_is_lit_from_above():
    """Top lighter, bottom darker: otherwise it is just a flat disc and the
    function has no reason to exist."""
    surface = blank()
    primitives.shaded_circle(surface, (120, 120), 50, (100, 100, 100))
    top = surface.get_at((120, 80))
    bottom = surface.get_at((120, 160))
    assert top.r > bottom.r + 10, f"top {top} vs bottom {bottom}"


def test_a_shaded_circle_stays_inside_its_radius():
    surface = blank()
    centre, radius = (120, 120), 40
    primitives.shaded_circle(surface, centre, radius, (200, 60, 60, 255))
    for x in range(0, 240, 2):
        for y in range(0, 240, 2):
            if surface.get_at((x, y)).a > 0:
                distance = ((x - centre[0]) ** 2 + (y - centre[1]) ** 2) ** 0.5
                # +1.5: the anti-aliased rim owns the pixel ring at radius+1.
                assert distance <= radius + 1.5, f"({x},{y}) escaped the circle"


def test_an_elevated_panel_casts_a_shadow_below_itself():
    surface = blank()
    rect = pygame.Rect(60, 60, 100, 50)
    primitives.elevated_panel(surface, rect, (60, 60, 70), lift=4)
    below = 0
    for x in range(rect.left + 10, rect.right - 10, 2):
        for y in range(rect.bottom, rect.bottom + 4):
            if surface.get_at((x, y)).a > 0:
                below += 1
    assert below > 0, "no shadow pixels under the panel"


def test_a_bevelled_panel_is_lighter_at_the_top_than_the_bottom():
    surface = blank()
    rect = pygame.Rect(60, 60, 100, 50)
    primitives.elevated_panel(surface, rect, (100, 100, 100), radius=6,
                              bevel_light=(255, 255, 255, 90),
                              bevel_dark=(0, 0, 0, 90))
    top = surface.get_at((110, rect.top + 1))
    bottom = surface.get_at((110, rect.bottom - 2))
    assert top.r > bottom.r, f"top {top} vs bottom {bottom}"


def test_a_raised_button_reports_its_depth():
    """The return value is the label's downward nudge. A cap that sinks while
    its label stays put reads as two broken parts, not one pressed button."""
    surface = blank()
    rect = pygame.Rect(60, 60, 100, 40)
    assert primitives.raised_button(surface, rect, (120, 120, 130)) == 0
    assert primitives.raised_button(surface, rect, (120, 120, 130),
                                    pressed=True) == 1


def test_a_pressed_button_casts_no_shadow():
    """Pressed means flush with the surface, so nothing may leak below."""
    surface = blank()
    rect = pygame.Rect(60, 60, 100, 40)
    primitives.raised_button(surface, rect, (120, 120, 130), pressed=True)
    for x in range(rect.left, rect.right, 2):
        for y in range(rect.bottom, rect.bottom + 5):
            assert surface.get_at((x, y)).a == 0, f"shadow at ({x},{y})"


def test_a_pressed_button_is_darker_than_a_raised_one():
    rect = pygame.Rect(60, 60, 100, 40)
    raised = blank()
    primitives.raised_button(raised, rect, (120, 120, 130))
    pressed = blank()
    primitives.raised_button(pressed, rect, (120, 120, 130), pressed=True)
    assert pressed.get_at(rect.center).r < raised.get_at(rect.center).r


def test_a_pressed_button_wears_its_bevel_upside_down():
    """Concave surfaces put the dark edge on top -- the swap is most of what
    makes a press look like a press rather than a colour change."""
    rect = pygame.Rect(60, 60, 100, 40)
    kwargs = dict(bevel_light=(255, 255, 255, 90), bevel_dark=(0, 0, 0, 90))
    raised = blank()
    primitives.raised_button(raised, rect, (120, 120, 130), **kwargs)
    pressed = blank()
    primitives.raised_button(pressed, rect, (120, 120, 130), pressed=True,
                             **kwargs)
    assert raised.get_at((110, 61)).r > raised.get_at((110, 98)).r
    assert pressed.get_at((110, 61)).r < pressed.get_at((110, 98)).r


def test_a_sunken_well_is_darker_near_its_top_edge():
    surface = blank()
    rect = pygame.Rect(60, 60, 120, 40)
    primitives.sunken_well(surface, rect, (200, 200, 200), radius=6,
                           well_shadow=(0, 0, 0, 90))
    top = surface.get_at((120, rect.top + 1))
    bottom = surface.get_at((120, rect.bottom - 5))
    assert top.r < bottom.r, f"top {top} vs bottom {bottom}"


def test_a_pointer_stays_by_its_tip():
    for direction in ("up", "down", "left", "right"):
        surface = blank()
        primitives.pointer(surface, (120, 120), 10, (255, 0, 0, 255), direction)
        assert painted(surface) > 0
        for x in range(0, 240, 2):
            for y in range(0, 240, 2):
                if surface.get_at((x, y)).a > 0:
                    assert abs(x - 120) <= 11 and abs(y - 120) <= 11, \
                        f"{direction}: ({x},{y}) strayed from the tip"


def test_a_pointer_points_the_way_it_is_told():
    """The body must sit behind the tip: the half-plane beyond it stays empty,
    give or take one anti-aliased pixel."""
    # For each direction: which axis runs past the tip, and which way.
    beyond = {
        "down": ("y", 1),
        "up": ("y", -1),
        "right": ("x", 1),
        "left": ("x", -1),
    }
    for direction, (axis, sign) in beyond.items():
        surface = blank()
        primitives.pointer(surface, (120, 120), 10, (255, 255, 255, 255),
                           direction)
        assert painted(surface) > 0
        for x in range(0, 240, 2):
            for y in range(0, 240, 2):
                if surface.get_at((x, y)).a > 0:
                    along = x if axis == "x" else y
                    assert (along - 120) * sign <= 1, f"{direction}: ({x},{y})"


def test_elevation_primitives_survive_extreme_coordinates():
    """blit refuses destinations outside the int range rather than clipping --
    the gfxdraw trap again, one layer up."""
    surface = blank()
    for point in [(10 ** 9, 10 ** 9), (-10 ** 9, 5), (0, -10 ** 12)]:
        primitives.shaded_circle(surface, point, 20, (255, 0, 0, 255))
        for direction in ("up", "down", "left", "right"):
            primitives.pointer(surface, point, 8, (255, 0, 0, 255), direction)
    # Rect coordinates themselves are limited to the C int range, so this is
    # as far offscreen as a caller can even ask for.
    far = pygame.Rect(10 ** 9, -(10 ** 9), 60, 30)
    primitives.elevated_panel(surface, far, (40, 40, 50))
    primitives.raised_button(surface, far, (40, 40, 50))
    primitives.raised_button(surface, far, (40, 40, 50), pressed=True)
    primitives.sunken_well(surface, far, (40, 40, 50),
                           well_shadow=(0, 0, 0, 90))
