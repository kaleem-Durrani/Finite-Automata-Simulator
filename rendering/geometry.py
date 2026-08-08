"""Path geometry for automaton diagrams.

Pure functions over plain tuples. No pygame, no state, no drawing -- so this is
unit-testable, and the SVG and TikZ exporters can share it with the on-screen
renderer rather than reimplementing the same curves and getting different
pictures.

Three things here were wrong in the previous renderer and are fixed:

* **Curved edges were trimmed along the chord.** The straight-line direction was
  used to pull the endpoints back by one radius, then the curve was drawn
  between those points -- so a curved edge left the node boundary and arrived
  somewhere near, but not on, the target. Trimming now happens *in curve space*:
  the tessellated path is clipped where it actually crosses each circle.

* **Self-loops were a tangent circle** with an arrowhead hardcoded at a 45
  degree point, in the same place above every node, and the arc offset was
  silently discarded. A loop is now a cubic bezier leaving and re-entering the
  node boundary at a settable angle, with its arrowhead oriented from the curve.

* **Segment count was unbounded.** ``int(length / 10)`` with no ceiling meant
  zooming in multiplied pure-Python tessellation without limit. It is clamped.
"""

import math
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]
Path = List[Point]

MIN_SEGMENTS = 10
MAX_SEGMENTS = 72

# How far a self-loop reaches from the node, as a multiple of the node radius.
SELF_LOOP_REACH = 2.6
# Half-angle between the two points where a self-loop meets the node.
SELF_LOOP_SPREAD = math.radians(32)


# ----------------------------------------------------------------------
# Vector helpers
# ----------------------------------------------------------------------

def distance(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def direction(a: Point, b: Point) -> Point:
    """Unit vector from ``a`` to ``b``; (1, 0) if they coincide."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return (1.0, 0.0)
    return (dx / length, dy / length)


def perpendicular(vector: Point) -> Point:
    """The vector rotated a quarter turn."""
    return (-vector[1], vector[0])


def lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def offset(point: Point, vector: Point, amount: float) -> Point:
    return (point[0] + vector[0] * amount, point[1] + vector[1] * amount)


# ----------------------------------------------------------------------
# Curves
# ----------------------------------------------------------------------

def segment_count(length: float) -> int:
    """How finely to tessellate a curve of the given on-screen length."""
    return max(MIN_SEGMENTS, min(MAX_SEGMENTS, int(length / 9)))


def quadratic(p0: Point, control: Point, p1: Point, segments: int) -> Path:
    """Sample a quadratic bezier."""
    points: Path = []
    for i in range(segments + 1):
        t = i / segments
        u = 1.0 - t
        points.append((
            u * u * p0[0] + 2 * u * t * control[0] + t * t * p1[0],
            u * u * p0[1] + 2 * u * t * control[1] + t * t * p1[1],
        ))
    return points


def cubic(p0: Point, c0: Point, c1: Point, p1: Point, segments: int) -> Path:
    """Sample a cubic bezier."""
    points: Path = []
    for i in range(segments + 1):
        t = i / segments
        u = 1.0 - t
        points.append((
            u ** 3 * p0[0] + 3 * u * u * t * c0[0] + 3 * u * t * t * c1[0] + t ** 3 * p1[0],
            u ** 3 * p0[1] + 3 * u * u * t * c0[1] + 3 * u * t * t * c1[1] + t ** 3 * p1[1],
        ))
    return points


def path_length(path: Sequence[Point]) -> float:
    """Total length along a polyline."""
    return sum(distance(path[i], path[i + 1]) for i in range(len(path) - 1))


def point_at(path: Sequence[Point], fraction: float) -> Point:
    """The point a given fraction of the way along a polyline.

    This is what makes a token travel at a believable speed: interpolating the
    bezier parameter directly would make it hurry through the curved middle and
    dawdle at the ends.
    """
    if not path:
        return (0.0, 0.0)
    if len(path) == 1:
        return path[0]

    fraction = max(0.0, min(1.0, fraction))
    total = path_length(path)
    if total < 1e-9:
        return path[0]

    target = total * fraction
    walked = 0.0
    for i in range(len(path) - 1):
        span = distance(path[i], path[i + 1])
        if walked + span >= target:
            local = (target - walked) / span if span > 1e-9 else 0.0
            return lerp(path[i], path[i + 1], local)
        walked += span
    return path[-1]


def tangent_at_end(path: Sequence[Point]) -> Point:
    """Direction of travel at the end of a polyline."""
    if len(path) < 2:
        return (1.0, 0.0)
    for i in range(len(path) - 1, 0, -1):
        if distance(path[i - 1], path[i]) > 1e-6:
            return direction(path[i - 1], path[i])
    return (1.0, 0.0)


# ----------------------------------------------------------------------
# Trimming
# ----------------------------------------------------------------------

def trim_path(path: Sequence[Point], start_centre: Optional[Point],
              start_radius: float, end_centre: Optional[Point],
              end_radius: float) -> Path:
    """
    Clip a path where it crosses the source and target circles.

    Works on the tessellated curve rather than on the straight line between the
    endpoints, so a curved edge meets the node boundary exactly where it looks
    like it should.
    """
    points = list(path)
    if len(points) < 2:
        return points

    if start_centre is not None and start_radius > 0:
        first = 0
        while first < len(points) - 1 and distance(points[first], start_centre) < start_radius:
            first += 1
        if first > 0:
            # points[first - 1] is inside the circle, points[first] is outside.
            crossing = _circle_crossing(points[first], points[first - 1],
                                        start_centre, start_radius)
            points = ([crossing] if crossing else []) + points[first:]

    if end_centre is not None and end_radius > 0:
        last = len(points) - 1
        while last > 0 and distance(points[last], end_centre) < end_radius:
            last -= 1
        if last < len(points) - 1:
            # points[last] is outside, points[last + 1] is inside.
            crossing = _circle_crossing(points[last], points[last + 1],
                                        end_centre, end_radius)
            points = points[:last + 1] + ([crossing] if crossing else [])

    return points if len(points) >= 2 else list(path)


def _circle_crossing(outside: Point, inside: Point, centre: Point,
                     radius: float) -> Optional[Point]:
    """Where the segment from ``outside`` to ``inside`` meets the circle.

    The endpoints straddle the boundary, so bisection is exact enough for a
    pixel and easier to reason about than solving the quadratic. The order of
    the arguments matters: passing them the wrong way round converges on the
    wrong end of the segment, which silently pulled every edge off its node.
    """
    lo, hi = 0.0, 1.0
    for _ in range(20):
        mid = (lo + hi) / 2
        if distance(lerp(outside, inside, mid), centre) > radius:
            lo = mid
        else:
            hi = mid
    return lerp(outside, inside, (lo + hi) / 2)


# ----------------------------------------------------------------------
# Edges
# ----------------------------------------------------------------------

def edge_path(source: Point, target: Point, source_radius: float,
              target_radius: float, arc: float = 0.0) -> Path:
    """
    The drawn path of a transition from one state to another.

    Args:
        source: Centre of the source state.
        target: Centre of the target state.
        source_radius: Radius of the source state.
        target_radius: Radius of the target state.
        arc: Perpendicular bow, in the same units as the coordinates. Zero
            draws a straight edge.

    Returns:
        A polyline that starts on the source circle and ends on the target one.
    """
    span = distance(source, target)
    if span < 1e-6:
        return [source, target]

    if abs(arc) < 0.01:
        # Sampled rather than a bare pair of endpoints. Trimming walks the
        # polyline looking for where it leaves each circle, and a two-point
        # line gives it nothing to walk -- which collapsed every straight edge
        # to a single point, so straight transitions did not render at all.
        steps = segment_count(span)
        raw = [lerp(source, target, i / steps) for i in range(steps + 1)]
    else:
        mid = lerp(source, target, 0.5)
        bow = perpendicular(direction(source, target))
        control = offset(mid, bow, arc * 2.0)
        raw = quadratic(source, control, target, segment_count(span + abs(arc)))

    return trim_path(raw, source, source_radius, target, target_radius)


def self_loop_path(centre: Point, radius: float, angle: float = -math.pi / 2,
                   reach: float = SELF_LOOP_REACH) -> Path:
    """
    The drawn path of a transition from a state to itself.

    Leaves the node boundary on one side of ``angle`` and returns on the other,
    so both ends sit on the circle and the arrowhead can be oriented from the
    curve like any other edge.

    Args:
        centre: Centre of the state.
        radius: Radius of the state.
        angle: Direction the loop points, in radians. Defaults to straight up.
        reach: How far the loop extends, as a multiple of the radius.
    """
    leave = angle - SELF_LOOP_SPREAD
    arrive = angle + SELF_LOOP_SPREAD

    start = (centre[0] + math.cos(leave) * radius,
             centre[1] + math.sin(leave) * radius)
    end = (centre[0] + math.cos(arrive) * radius,
           centre[1] + math.sin(arrive) * radius)

    extent = radius * reach
    spread = radius * 1.25
    axis = (math.cos(angle), math.sin(angle))
    side = perpendicular(axis)

    control_a = (centre[0] + axis[0] * extent - side[0] * spread,
                 centre[1] + axis[1] * extent - side[1] * spread)
    control_b = (centre[0] + axis[0] * extent + side[0] * spread,
                 centre[1] + axis[1] * extent + side[1] * spread)

    return cubic(start, control_a, control_b, end, segment_count(extent * 3))


def start_marker_path(centre: Point, radius: float,
                      angle: float = math.pi, length: float = 46.0) -> Path:
    """
    The little arrow that marks the initial state.

    It runs *towards* the state and stops on its boundary. The old one was built
    from a 25-unit stub that then had 30 units trimmed off each end, so it was
    drawn backwards, detached, on every automaton ever displayed.
    """
    outward = (math.cos(angle), math.sin(angle))
    tip = offset(centre, outward, radius)
    tail = offset(centre, outward, radius + length)
    return [tail, tip]


def arrowhead(path: Sequence[Point], size: float,
              spread: float = 0.42) -> List[Point]:
    """
    A triangle at the end of a path, pointing the way the path travels.

    Taken from the last two tessellated points, so it is correct for curves and
    self-loops as well as straight lines.
    """
    if len(path) < 2:
        return []

    tip = path[-1]
    heading = tangent_at_end(path)
    angle = math.atan2(heading[1], heading[0])

    return [
        tip,
        (tip[0] - size * math.cos(angle - spread),
         tip[1] - size * math.sin(angle - spread)),
        (tip[0] - size * math.cos(angle + spread),
         tip[1] - size * math.sin(angle + spread)),
    ]


def label_anchor(path: Sequence[Point], distance_from_path: float = 0.0) -> Point:
    """Where an edge's label should sit: the midpoint, nudged off the line."""
    if not path:
        return (0.0, 0.0)
    mid = point_at(path, 0.5)
    if abs(distance_from_path) < 1e-6 or len(path) < 2:
        return mid

    before = point_at(path, 0.45)
    after = point_at(path, 0.55)
    return offset(mid, perpendicular(direction(before, after)), distance_from_path)


def auto_arc(source_id: str, target_id: str, bidirectional: bool,
             magnitude: float = 34.0) -> float:
    """
    Default bow for an edge the user has not curved by hand.

    A pair of states with transitions both ways gets one edge bowed to each
    side, so they do not lie on top of one another.

    Both directions get the *same* sign, which looks wrong and is not. The bow
    is applied along the perpendicular of the source-to-target direction, and
    that direction is already reversed for the opposite edge -- so the
    perpendicular is reversed too, and the two curves separate. Negating the
    arc as well, which is what the previous renderer did, cancels the flip and
    puts both edges back on exactly the same curve.
    """
    del source_id, target_id  # kept for call-site clarity
    return magnitude if bidirectional else 0.0
