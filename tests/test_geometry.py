"""Tests for edge and arrow geometry.

Pure maths, no display. These pin the three defects the previous renderer had
-- edges that missed their nodes, a start arrow drawn backwards, and self-loops
that ignored their orientation -- plus the one I introduced while fixing them,
where straight edges collapsed to a single point and vanished.
"""

import math

import pytest

from rendering import geometry as g

RADIUS = 30.0
SOURCE = (200.0, 200.0)
TARGET = (400.0, 200.0)


def close(a, b, tol=1.0):
    return g.distance(a, b) <= tol


# ---------------------------------------------------------------------------
# Edges meet their nodes
# ---------------------------------------------------------------------------


def test_straight_edge_is_not_degenerate():
    """A straight edge used to trim down to a single point and disappear.

    Trimming walks the polyline for the crossing; a bare pair of endpoints
    gives it nothing to walk.
    """
    path = g.edge_path(SOURCE, TARGET, RADIUS, RADIUS, 0.0)
    assert len(path) >= 2
    assert g.distance(path[0], path[-1]) > 100


def test_straight_edge_starts_and_ends_on_the_boundaries():
    path = g.edge_path(SOURCE, TARGET, RADIUS, RADIUS, 0.0)
    assert g.distance(path[0], SOURCE) == pytest.approx(RADIUS, abs=1.0)
    assert g.distance(path[-1], TARGET) == pytest.approx(RADIUS, abs=1.0)


@pytest.mark.parametrize("arc", [-60.0, -34.0, -12.0, 12.0, 34.0, 60.0])
def test_curved_edges_meet_the_boundaries(arc):
    """Curves used to be trimmed along the chord, so they missed the node.

    The endpoints were pulled back by one radius in the straight-line
    direction, then the curve was drawn between those points -- arriving near,
    but not on, the circle.
    """
    path = g.edge_path(SOURCE, TARGET, RADIUS, RADIUS, arc)
    assert g.distance(path[0], SOURCE) == pytest.approx(RADIUS, abs=1.0)
    assert g.distance(path[-1], TARGET) == pytest.approx(RADIUS, abs=1.0)


@pytest.mark.parametrize("arc", [0.0, 34.0])
def test_edges_never_enter_either_node(arc):
    """No sampled point may lie inside a state circle."""
    path = g.edge_path(SOURCE, TARGET, RADIUS, RADIUS, arc)
    for point in path:
        assert g.distance(point, SOURCE) >= RADIUS - 1.0
        assert g.distance(point, TARGET) >= RADIUS - 1.0


def test_edges_work_at_any_angle():
    for degrees in range(0, 360, 15):
        angle = math.radians(degrees)
        target = (SOURCE[0] + math.cos(angle) * 240,
                  SOURCE[1] + math.sin(angle) * 240)
        for arc in (0.0, 30.0, -30.0):
            path = g.edge_path(SOURCE, target, RADIUS, RADIUS, arc)
            assert g.distance(path[0], SOURCE) == pytest.approx(RADIUS, abs=1.2)
            assert g.distance(path[-1], target) == pytest.approx(RADIUS, abs=1.2)


def test_coincident_states_do_not_crash():
    path = g.edge_path(SOURCE, SOURCE, RADIUS, RADIUS, 0.0)
    assert len(path) >= 2


# ---------------------------------------------------------------------------
# Self loops
# ---------------------------------------------------------------------------


def test_self_loop_begins_and_ends_on_the_node():
    path = g.self_loop_path(SOURCE, RADIUS)
    assert g.distance(path[0], SOURCE) == pytest.approx(RADIUS, abs=0.5)
    assert g.distance(path[-1], SOURCE) == pytest.approx(RADIUS, abs=0.5)


def test_self_loop_leaves_the_node():
    """It must actually bulge out, not hug the boundary."""
    path = g.self_loop_path(SOURCE, RADIUS)
    furthest = max(g.distance(point, SOURCE) for point in path)
    assert furthest > RADIUS * 1.5


def test_self_loop_honours_its_angle():
    """The old loop was a tangent circle in the same place above every node."""
    up = g.self_loop_path(SOURCE, RADIUS, angle=-math.pi / 2)
    down = g.self_loop_path(SOURCE, RADIUS, angle=math.pi / 2)

    up_extreme = min(point[1] for point in up)
    down_extreme = max(point[1] for point in down)
    assert up_extreme < SOURCE[1] - RADIUS
    assert down_extreme > SOURCE[1] + RADIUS


def test_self_loop_arrowhead_points_back_at_the_node():
    """It used to be hardcoded horizontal, regardless of the curve."""
    path = g.self_loop_path(SOURCE, RADIUS)
    head = g.arrowhead(path, 10.0)
    assert len(head) == 3
    tip = head[0]
    assert g.distance(tip, SOURCE) == pytest.approx(RADIUS, abs=0.6)


# ---------------------------------------------------------------------------
# The start marker
# ---------------------------------------------------------------------------


def test_start_marker_points_into_the_state():
    """It was drawn backwards, detached, on every automaton ever displayed.

    A 25-unit stub had 30 units trimmed off each end, so at zoom 1 it ran from
    x-30 to x-65 -- pointing away.
    """
    path = g.start_marker_path(SOURCE, RADIUS)
    tail, tip = path[0], path[-1]

    assert g.distance(tip, SOURCE) < g.distance(tail, SOURCE), "must point inwards"
    assert g.distance(tip, SOURCE) == pytest.approx(RADIUS, abs=0.5), "must touch the boundary"
    assert g.distance(tail, tip) > 20, "must be long enough to see"


def test_start_marker_arrowhead_is_at_the_state():
    path = g.start_marker_path(SOURCE, RADIUS)
    head = g.arrowhead(path, 10.0)
    assert g.distance(head[0], SOURCE) == pytest.approx(RADIUS, abs=0.5)


# ---------------------------------------------------------------------------
# Arrowheads and travel
# ---------------------------------------------------------------------------


def test_arrowhead_follows_a_curve():
    """Taken from the last two tessellated points, so curves are handled."""
    path = g.edge_path(SOURCE, TARGET, RADIUS, RADIUS, 60.0)
    head = g.arrowhead(path, 12.0)
    assert close(head[0], path[-1], 0.01)
    # The barbs sit behind the tip, along the direction of travel.
    heading = g.tangent_at_end(path)
    for barb in head[1:]:
        back = (barb[0] - head[0][0], barb[1] - head[0][1])
        assert back[0] * heading[0] + back[1] * heading[1] < 0


def test_point_at_moves_at_constant_speed():
    """The token must not hurry through the middle of a curve.

    Interpolating the bezier parameter directly would; interpolating by arc
    length does not.
    """
    path = g.edge_path(SOURCE, TARGET, RADIUS, RADIUS, 70.0)
    samples = [g.point_at(path, i / 40) for i in range(41)]
    gaps = [g.distance(samples[i], samples[i + 1]) for i in range(40)]
    assert max(gaps) < min(gaps) * 1.8, "spacing should be roughly even"


def test_point_at_endpoints():
    path = g.edge_path(SOURCE, TARGET, RADIUS, RADIUS, 0.0)
    assert close(g.point_at(path, 0.0), path[0])
    assert close(g.point_at(path, 1.0), path[-1])
    assert close(g.point_at(path, -5.0), path[0])
    assert close(g.point_at(path, 5.0), path[-1])


def test_segment_count_is_bounded():
    """Unbounded tessellation meant zooming multiplied the work without limit."""
    assert g.segment_count(1) >= g.MIN_SEGMENTS
    assert g.segment_count(10_000_000) <= g.MAX_SEGMENTS


# ---------------------------------------------------------------------------
# Bidirectional pairs
# ---------------------------------------------------------------------------


def test_opposite_edges_get_the_same_arc_value():
    """The separation comes from the direction flip, not from the sign.

    Negating the arc for the reverse edge cancels the perpendicular's own flip
    and puts both curves back on top of each other -- which is what the
    previous renderer did.
    """
    forward = g.auto_arc("q0", "q1", bidirectional=True)
    backward = g.auto_arc("q1", "q0", bidirectional=True)
    assert forward == backward
    assert forward != 0


def test_a_lone_edge_stays_straight():
    assert g.auto_arc("q0", "q1", bidirectional=False) == 0.0


def test_opposite_edges_do_not_overlap():
    forward = g.edge_path(SOURCE, TARGET, RADIUS, RADIUS,
                          g.auto_arc("q0", "q1", True))
    backward = g.edge_path(TARGET, SOURCE, RADIUS, RADIUS,
                           g.auto_arc("q1", "q0", True))
    mid_f = g.point_at(forward, 0.5)
    mid_b = g.point_at(backward, 0.5)
    assert g.distance(mid_f, mid_b) > 40

    # And they bow to opposite sides of the line joining the two states.
    line_y = SOURCE[1]
    assert (mid_f[1] - line_y) * (mid_b[1] - line_y) < 0


# ---------------------------------------------------------------------------
# Self-loop placement
# ---------------------------------------------------------------------------


def test_a_lone_state_loops_upwards():
    assert g.quietest_direction(SOURCE, []) == pytest.approx(-math.pi / 2)


def test_a_loop_points_away_from_the_other_edges():
    """A fixed loop direction sits on top of whatever else meets the node."""
    # One neighbour directly above: the loop should point down.
    angle = g.quietest_direction(SOURCE, [(SOURCE[0], SOURCE[1] - 200)])
    assert math.sin(angle) > 0.9

    # One neighbour to the right: the loop should point left.
    angle = g.quietest_direction(SOURCE, [(SOURCE[0] + 200, SOURCE[1])])
    assert math.cos(angle) < -0.9


def test_a_loop_splits_the_difference_between_neighbours():
    """Two neighbours left and right, so the loop goes up or down, not sideways."""
    angle = g.quietest_direction(SOURCE, [(SOURCE[0] - 200, SOURCE[1]),
                                          (SOURCE[0] + 200, SOURCE[1])])
    assert abs(math.cos(angle)) < 0.2


def test_symmetric_neighbours_fall_back_to_the_default():
    """Nothing is quieter than anything else, so pick the default."""
    angle = g.quietest_direction(SOURCE, [(SOURCE[0] - 100, SOURCE[1]),
                                          (SOURCE[0] + 100, SOURCE[1]),
                                          (SOURCE[0], SOURCE[1] - 100),
                                          (SOURCE[0], SOURCE[1] + 100)])
    assert angle == pytest.approx(-math.pi / 2)


def test_a_loop_label_sits_outside_the_loop():
    """Placed at the path midpoint it lands inside the loop, on the node."""
    angle = -math.pi / 2
    anchor = g.self_loop_label_anchor(SOURCE, RADIUS, angle)
    path = g.self_loop_path(SOURCE, RADIUS, angle)

    assert g.distance(anchor, SOURCE) > RADIUS * 1.5
    furthest = max(g.distance(p, SOURCE) for p in path)
    assert g.distance(anchor, SOURCE) >= furthest * 0.85


def test_a_loop_label_follows_the_loop_direction():
    up = g.self_loop_label_anchor(SOURCE, RADIUS, -math.pi / 2)
    down = g.self_loop_label_anchor(SOURCE, RADIUS, math.pi / 2)
    assert up[1] < SOURCE[1]
    assert down[1] > SOURCE[1]


def test_label_anchor_sits_off_the_line():
    path = g.edge_path(SOURCE, TARGET, RADIUS, RADIUS, 0.0)
    on_line = g.label_anchor(path, 0.0)
    offset = g.label_anchor(path, 14.0)
    assert g.distance(on_line, offset) == pytest.approx(14.0, abs=0.5)
