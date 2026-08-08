"""SVG.

The one exporter that reproduces what was on screen, because it uses the same
:mod:`fsa.geometry` the renderer does. Where DOT re-lays-out the graph and TikZ
restates the curves in its own idiom, this draws the identical paths -- so a
figure taken from the tool looks like the tool.

Self-contained, no external stylesheet, deterministic byte-for-byte. Colours are
inlined rather than themed: an exported figure has to survive being dropped into
someone else's document.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from fsa import geometry
from fsa.analysis import dead_states, unreachable_states
from fsa.automaton import DFA
from fsa.layout import Layout
from fsa.symbols import StateId

Point = Tuple[float, float]

STATE_RADIUS = 30.0
MARGIN = 46.0

#: A light, print-friendly palette. Matches the application's light theme, so a
#: screenshot and an export sit together without clashing.
INK = "#1c1c1e"
MUTED = "#6b6b6b"
ACCEPT = "#059669"
ACCEPT_FILL = "#e9faf2"
TRAP = "#be3c3c"
TRAP_FILL = "#fdeded"
UNREACHABLE = "#8c7caa"
STATE_FILL = "#ffffff"
EDGE = "#4a4a4e"
PLATE = "#fafaf7"


def _f(value: float) -> str:
    """Format a number for SVG: fixed precision, no trailing noise.

    Determinism matters -- a golden-file test compares bytes, and repr() of a
    float is not stable enough to rely on.
    """
    text = f"{value:.2f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))


def _path_data(points: Sequence[Point]) -> str:
    """A polyline as an SVG path."""
    if not points:
        return ""
    head = f"M {_f(points[0][0])} {_f(points[0][1])}"
    rest = " ".join(f"L {_f(x)} {_f(y)}" for x, y in points[1:])
    return f"{head} {rest}".strip()


def _polygon_points(points: Sequence[Point]) -> str:
    return " ".join(f"{_f(x)},{_f(y)}" for x, y in points)


def to_svg(automaton: DFA, layout: Optional[Layout] = None,
           annotate: bool = True, title: Optional[str] = None) -> str:
    """Render an automaton as a standalone SVG document.

    Args:
        automaton: What to draw.
        layout: Where the states go. Without one they are placed on a grid.
        annotate: Style trap and unreachable states the way the application
            does. Turn it off for a plain figure.
        title: An accessible title for the document.
    """
    layout = layout or Layout.grid(sorted(automaton.states))
    positions: Dict[StateId, Point] = {
        state: layout.position_of(state) for state in sorted(automaton.states)
    }

    dead = dead_states(automaton) if (annotate and automaton.accept) else frozenset()
    unreachable = unreachable_states(automaton) if annotate else frozenset()

    grouped = automaton.grouped_transitions()
    loop_angles = _loop_angles(positions, grouped)
    paths = _edge_paths(positions, grouped, layout, loop_angles)

    view = _viewbox(positions, paths)
    body: List[str] = []

    if automaton.initial is not None and automaton.initial in positions:
        marker = geometry.start_marker_path(positions[automaton.initial], STATE_RADIUS)
        body.append(f'  <path d="{_path_data(marker)}" fill="none" '
                    f'stroke="{INK}" stroke-width="2"/>')
        body.append(f'  <polygon points="'
                    f'{_polygon_points(geometry.arrowhead(marker, 10.0))}" '
                    f'fill="{INK}"/>')

    # Edges, then nodes, then labels -- so nothing is drawn over a label.
    for edge in sorted(paths):
        path = paths[edge]
        body.append(f'  <path d="{_path_data(path)}" fill="none" '
                    f'stroke="{EDGE}" stroke-width="1.6"/>')
        body.append(f'  <polygon points="'
                    f'{_polygon_points(geometry.arrowhead(path, 9.0))}" '
                    f'fill="{EDGE}"/>')

    for state in sorted(positions):
        x, y = positions[state]
        if state in unreachable:
            fill, stroke, dash = STATE_FILL, UNREACHABLE, ' stroke-dasharray="5 4"'
        elif state in dead:
            fill, stroke, dash = TRAP_FILL, TRAP, ""
        elif state in automaton.accept:
            fill, stroke, dash = ACCEPT_FILL, ACCEPT, ""
        else:
            fill, stroke, dash = STATE_FILL, INK, ""

        body.append(f'  <circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(STATE_RADIUS)}" '
                    f'fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>')
        if state in automaton.accept:
            body.append(f'  <circle cx="{_f(x)}" cy="{_f(y)}" '
                        f'r="{_f(STATE_RADIUS * 0.76)}" fill="none" '
                        f'stroke="{ACCEPT}" stroke-width="1.6"/>')

    for edge in sorted(paths):
        source, target = edge
        label = ", ".join(sorted(grouped[edge]))
        if source == target:
            anchor = geometry.self_loop_label_anchor(
                positions[source], STATE_RADIUS, loop_angles[source])
        else:
            anchor = geometry.label_anchor(paths[edge], -12.0)
        body.append(_text_with_plate(anchor, label))

    for state in sorted(positions):
        x, y = positions[state]
        colour = TRAP if state in dead else (
            UNREACHABLE if state in unreachable else INK)
        body.append(
            f'  <text x="{_f(x)}" y="{_f(y)}" fill="{colour}" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="14" '
            f'font-weight="600" text-anchor="middle" '
            f'dominant-baseline="central">{_escape(automaton.label_of(state))}</text>')

    header = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{_f(view[0])} {_f(view[1])} {_f(view[2])} {_f(view[3])}" '
        f'width="{_f(view[2])}" height="{_f(view[3])}">',
    ]
    if title:
        header.append(f"  <title>{_escape(title)}</title>")
    header.append(f'  <rect x="{_f(view[0])}" y="{_f(view[1])}" '
                  f'width="{_f(view[2])}" height="{_f(view[3])}" fill="#ffffff"/>')

    return "\n".join(header + body + ["</svg>"]) + "\n"


def _text_with_plate(anchor: Point, label: str) -> str:
    """A label on a pale plate, so it stays readable where edges cross."""
    width = max(14.0, 7.6 * len(label) + 8)
    x, y = anchor
    return (f'  <g><rect x="{_f(x - width / 2)}" y="{_f(y - 9)}" '
            f'width="{_f(width)}" height="18" rx="4" fill="{PLATE}" '
            f'fill-opacity="0.92"/>'
            f'<text x="{_f(x)}" y="{_f(y)}" fill="{MUTED}" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="12" '
            f'text-anchor="middle" dominant-baseline="central">'
            f'{_escape(label)}</text></g>')


def _loop_angles(positions: Dict[StateId, Point],
                 grouped: Mapping[Tuple[StateId, StateId], Any]) -> Dict[StateId, float]:
    neighbours: Dict[StateId, List[Point]] = {}
    for (source, target) in grouped:
        if source == target:
            continue
        for a, b in ((source, target), (target, source)):
            if a in positions and b in positions:
                neighbours.setdefault(a, []).append(positions[b])
    return {state: geometry.quietest_direction(point, neighbours.get(state, []))
            for state, point in positions.items()}


def _edge_paths(positions: Dict[StateId, Point],
                grouped: Mapping[Tuple[StateId, StateId], Any],
                layout: Layout,
                loop_angles: Dict[StateId, float],
                ) -> Dict[Tuple[StateId, StateId], List[Point]]:
    paths: Dict[Tuple[StateId, StateId], List[Point]] = {}
    for edge in grouped:
        source, target = edge
        if source not in positions or target not in positions:
            continue
        if source == target:
            paths[edge] = geometry.self_loop_path(
                positions[source], STATE_RADIUS, angle=loop_angles[source])
            continue
        arc = layout.arc_of(source, target)
        if not arc:
            arc = geometry.auto_arc(source, target, (target, source) in grouped)
        paths[edge] = geometry.edge_path(positions[source], positions[target],
                                         STATE_RADIUS, STATE_RADIUS, arc)
    return paths


def _viewbox(positions: Dict[StateId, Point],
             paths: Mapping[Tuple[StateId, StateId], Sequence[Point]],
             ) -> Tuple[float, float, float, float]:
    """A box containing every node and every curve, with a margin."""
    xs: List[float] = []
    ys: List[float] = []
    for x, y in positions.values():
        xs += [x - STATE_RADIUS, x + STATE_RADIUS]
        ys += [y - STATE_RADIUS, y + STATE_RADIUS]
    for path in paths.values():
        for x, y in path:
            xs.append(x)
            ys.append(y)
    if not xs:
        return (0.0, 0.0, 200.0, 120.0)

    # The start marker reaches left of its state, and labels sit above edges.
    min_x, max_x = min(xs) - MARGIN - 46, max(xs) + MARGIN
    min_y, max_y = min(ys) - MARGIN, max(ys) + MARGIN
    return (min_x, min_y, max_x - min_x, max_y - min_y)


def write_svg(automaton: DFA, path: str, layout: Optional[Layout] = None,
              annotate: bool = True, title: Optional[str] = None) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(to_svg(automaton, layout, annotate, title))
