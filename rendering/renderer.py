"""Paints a Scene.

Knows about pixels, the camera and the theme. Knows nothing about automata:
everything it draws arrives as a :class:`~rendering.scene.Scene` of plain
geometry, so the model underneath can change without touching this file.

Draw order is three passes -- edges, then nodes, then every label. The previous
renderer drew each edge's label during the edge pass, so the next node painted
over it; transition labels were routinely buried under the states they
connected.
"""

from typing import List, Optional, Sequence, Tuple

import pygame

from fsa import geometry
from rendering import primitives
from rendering.camera import Camera
from rendering.fonts import FontBook
from rendering.scene import EdgeVisual, NodeKind, NodeVisual, Scene, TokenVisual
from rendering.theme import Theme

Point = Tuple[float, float]

# World-space size of a state, before zoom.
STATE_RADIUS = 30.0
# How far outside the window a node may be before it is skipped.
CULL_MARGIN = 140.0


def _elide(font: pygame.font.Font, text: str, budget: float) -> str:
    """``text``, shortened with an ellipsis until it fits within ``budget``.

    A state's label is free text, so it has no natural relationship to the size
    of the circle it sits in. Left alone it is blitted at full width over
    whatever is behind it -- a long name painted a band clean across the
    diagram, on top of the very transitions it was meant to describe.
    """
    if budget <= 0 or font.size(text)[0] <= budget:
        return text
    ellipsis = "…"
    trimmed = text
    while trimmed and font.size(trimmed + ellipsis)[0] > budget:
        trimmed = trimmed[:-1]
    return trimmed + ellipsis if trimmed else ellipsis


def _mix(a: Sequence[int], b: Sequence[int], t: float) -> Tuple[int, int, int]:
    """Blend two colours. Used to interpolate along an animated value."""
    t = max(0.0, min(1.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


class Renderer:
    """Draws the workspace."""

    def __init__(self, screen: pygame.Surface, theme: Optional[Theme] = None,
                 fonts: Optional[FontBook] = None):
        self.screen = screen
        self.camera = Camera(screen.get_width(), screen.get_height())
        self.theme = theme or Theme("dark")
        self.fonts = fonts or FontBook()

    # ------------------------------------------------------------------
    # Frame
    # ------------------------------------------------------------------

    def update_screen_size(self, width: int, height: int) -> None:
        """Track a resized window.

        Also rebinds the surface. The old version updated only the camera and
        left ``self.screen`` pointing at the pre-resize surface.
        """
        self.screen = pygame.display.get_surface()
        self.camera.screen_width = width
        self.camera.screen_height = height

    def clear(self) -> None:
        """Fill the canvas and lay down the dot grid."""
        palette = self.theme.palette
        self.screen.fill(palette.canvas)

        spacing = 48 * self.camera.zoom
        origin = self.camera.world_to_screen((0.0, 0.0))
        primitives.dot_grid(self.screen, palette.canvas_dot, spacing, origin,
                            self.screen.get_size())

    def draw_scene(self, scene: Scene) -> None:
        """Paint a whole frame of the workspace."""
        visible_edges = [edge for edge in scene.edges if self._edge_visible(edge)]
        visible_nodes = [node for node in scene.nodes if self._node_visible(node)]

        if scene.start_marker is not None:
            self._draw_start_marker(scene.start_marker.path)

        for edge in visible_edges:
            self._draw_edge(edge)

        if scene.ghost_edge is not None:
            self._draw_ghost_edge(scene.ghost_edge)

        for node in visible_nodes:
            self._draw_node(node)

        # Labels last, so nothing can be painted over them.
        for edge in visible_edges:
            self._draw_edge_label(edge)
        for node in visible_nodes:
            self._draw_node_label(node)

        # Tokens last and all of them: a nondeterministic move puts one on
        # every branch, and none of them may be hidden behind a node -- the
        # whole point of drawing several is that the viewer counts them.
        for token in scene.tokens:
            self._draw_token(token)

    # ------------------------------------------------------------------
    # Culling
    # ------------------------------------------------------------------

    def _node_visible(self, node: NodeVisual) -> bool:
        screen_pos = self.camera.world_to_screen(node.position)
        margin = CULL_MARGIN + node.radius * self.camera.zoom
        return primitives.on_screen(screen_pos, self.screen.get_size(), margin)

    def _edge_visible(self, edge: EdgeVisual) -> bool:
        """Keep an edge if any sampled point is near the window.

        Culling is not only a performance measure: gfxdraw raises OverflowError
        above a signed short, and at maximum zoom a far-off coordinate exceeds
        that easily. Primitives clamp as well, so this is belt and braces.
        """
        if not edge.path:
            return False
        size = self.screen.get_size()
        step = max(1, len(edge.path) // 6)
        for point in list(edge.path)[::step] + [edge.path[-1]]:
            if primitives.on_screen(self.camera.world_to_screen(point), size,
                                    CULL_MARGIN):
                return True
        return False

    def _to_screen(self, path: Sequence[Point]) -> List[Point]:
        return [self.camera.world_to_screen(point) for point in path]

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _draw_node(self, node: NodeVisual) -> None:
        palette = self.theme.palette
        centre = self.camera.world_to_screen(node.position)
        zoom = self.camera.zoom

        # A brief swell as the state is entered, and a steady lift while active.
        scale = 1.0 + 0.09 * node.settle + 0.04 * node.active
        radius = node.radius * zoom * scale
        if radius < 2:
            return

        if node.active > 0.01:
            primitives.glow(self.screen, centre, radius,
                            (*palette.active_glow[:3],
                             int(palette.active_glow[3] * node.active)),
                            spread=16 * zoom * node.active)

        primitives.soft_shadow(self.screen, centre, radius, palette.shadow,
                               spread=5 * max(0.6, zoom))

        # Each kind gets its own fill and ring colour, and its own shape signal:
        # accepting states carry a second ring, traps a diagonal hatch,
        # unreachable states a dashed outline. Colour alone was not enough --
        # normal and trap differed only by a slightly darker grey, which is no
        # difference at all on a projector or to a colour-blind reader.
        if node.kind is NodeKind.DEAD:
            fill = palette.dead_fill
            ring_color = palette.dead_ring
        elif node.kind is NodeKind.UNREACHABLE:
            fill = palette.unreachable_fill
            ring_color = palette.unreachable_ring
        elif node.is_accept:
            fill = palette.accept_fill
            ring_color = palette.accept_ring
        else:
            fill = palette.state_fill
            ring_color = palette.state_ring

        # Interaction states blend on top, strongest last.
        if node.hover > 0.01:
            ring_color = _mix(ring_color, palette.hover_ring, node.hover * 0.7)
        if node.selected > 0.01:
            ring_color = _mix(ring_color, palette.selected_ring, node.selected)
        if node.active > 0.01:
            ring_color = _mix(ring_color, palette.active_ring, node.active)
            fill = _mix(fill, palette.active_glow[:3], 0.22 * node.active)

        # A vertical gradient makes the state read as a softly lit sphere
        # rather than a flat disc; the interaction states above still blend
        # into the base colour before the gradient is applied.
        primitives.shaded_circle(self.screen, centre, radius, fill)

        if node.kind is NodeKind.DEAD and radius > 8:
            primitives.hatch_circle(self.screen, centre, radius - 1,
                                    palette.dead_hatch,
                                    spacing=max(5.0, 7.0 * zoom))

        weight = max(1.4, (2.0 + 1.4 * max(node.selected, node.active)) * min(2.0, zoom))
        if node.kind is NodeKind.UNREACHABLE:
            primitives.dashed_ring(self.screen, centre, radius, weight, ring_color)
        else:
            primitives.ring(self.screen, centre, radius, weight, ring_color)

        if node.is_accept:
            inner = max(3.0, radius * 0.76)
            accept_color = (palette.dead_ring if node.kind is NodeKind.DEAD
                            else palette.accept_ring)
            primitives.ring(self.screen, centre, inner,
                            max(1.2, 1.8 * min(2.0, zoom)), accept_color)

    def _draw_node_label(self, node: NodeVisual) -> None:
        zoom = self.camera.zoom
        if node.radius * zoom < 11:
            return

        palette = self.theme.palette
        if node.kind is NodeKind.DEAD:
            colour = palette.dead_text
        elif node.kind is NodeKind.UNREACHABLE:
            colour = palette.unreachable_text
        else:
            colour = palette.state_text
        font = self.fonts.scaled("state", zoom)
        # Kept inside the disc: the label belongs to this state, so it must not
        # be readable as belonging to whatever it would otherwise overlap.
        surface = font.render(
            _elide(font, node.label, node.radius * zoom * 1.7), True, colour)
        centre = self.camera.world_to_screen(node.position)
        self.screen.blit(surface, surface.get_rect(
            center=(int(centre[0]), int(centre[1]))))

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def _edge_color(self, edge: EdgeVisual) -> Tuple[int, int, int]:
        palette = self.theme.palette
        base = self.theme.edge_color(edge.color_index)
        if edge.muted > 0.01:
            base = _mix(base, palette.edge_muted, edge.muted)
        if edge.active > 0.01:
            base = _mix(base, palette.edge_active, edge.active)
        return base

    def _draw_edge(self, edge: EdgeVisual) -> None:
        if len(edge.path) < 2:
            return

        path = self._to_screen(edge.path)
        colour = self._edge_color(edge)
        zoom = self.camera.zoom
        width = max(1.2, (1.9 + 1.5 * edge.active) * min(2.2, zoom))

        primitives.stroke_path(self.screen, path, width, colour)

        if edge.show_arrowhead:
            head = geometry.arrowhead(path, max(6.0, 11.0 * min(2.0, zoom)))
            primitives.polygon(self.screen, head, colour)

    def _draw_edge_label(self, edge: EdgeVisual) -> None:
        zoom = self.camera.zoom
        if not edge.label or zoom < 0.42:
            return

        palette = self.theme.palette
        path = self._to_screen(edge.path)
        if edge.label_at is not None:
            anchor = self.camera.world_to_screen(edge.label_at)
        else:
            anchor = geometry.label_anchor(path, -13.0 * min(1.6, zoom))

        font = self.fonts.scaled("edge", zoom)
        colour = palette.label_text
        if edge.active > 0.01:
            colour = _mix(colour, palette.edge_active, edge.active)
        surface = font.render(edge.label, True, colour)
        rect = surface.get_rect(center=(int(anchor[0]), int(anchor[1])))

        # A plate behind the text so it stays readable where edges cross.
        plate = rect.inflate(9, 5)
        primitives.panel(self.screen, plate, palette.label_plate,
                         radius=self.theme.radius.sm)
        self.screen.blit(surface, rect)

    def _draw_ghost_edge(self, ghost) -> None:
        palette = self.theme.palette
        path = self._to_screen(ghost.path)
        colour = palette.accent if ghost.valid else palette.error
        primitives.dashed_path(self.screen, path,
                               max(1.5, 2.0 * min(2.0, self.camera.zoom)), colour)
        head = geometry.arrowhead(path, max(6.0, 10.0 * min(2.0, self.camera.zoom)))
        primitives.polygon(self.screen, head, colour)

        if ghost.label:
            font = self.fonts.scaled("edge", self.camera.zoom)
            surface = font.render(ghost.label, True, colour)
            anchor = geometry.label_anchor(path, -14.0)
            rect = surface.get_rect(center=(int(anchor[0]), int(anchor[1])))
            primitives.panel(self.screen, rect.inflate(9, 5), palette.label_plate,
                             radius=self.theme.radius.sm)
            self.screen.blit(surface, rect)

    def _draw_start_marker(self, path: Sequence[Point]) -> None:
        palette = self.theme.palette
        screen_path = self._to_screen(path)
        zoom = self.camera.zoom
        primitives.stroke_path(self.screen, screen_path,
                               max(1.8, 2.6 * min(2.0, zoom)),
                               palette.initial_marker)
        head = geometry.arrowhead(screen_path, max(7.0, 12.0 * min(2.0, zoom)))
        primitives.polygon(self.screen, head, palette.initial_marker)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _draw_token(self, token: TokenVisual) -> None:
        """One read head: a bright dot with a fading tail behind it.

        Drawn identically however many there are. Tinting or numbering the
        branches was considered and dropped: the branches of a nondeterministic
        move are not ranked, and colouring them differently would suggest an
        order the machine does not have.
        """
        palette = self.theme.palette
        zoom = self.camera.zoom
        radius = max(3.0, token.radius * zoom)

        trail = self._to_screen(token.trail)
        if len(trail) >= 2:
            for i in range(len(trail) - 1):
                t = (i + 1) / len(trail)
                alpha = int(palette.token_trail[3] * t * token.intensity)
                if alpha <= 2:
                    continue
                primitives.stroke_path(
                    self.screen, [trail[i], trail[i + 1]],
                    max(1.0, radius * 0.75 * t),
                    (*palette.token_trail[:3], alpha))

        centre = self.camera.world_to_screen(token.position)
        primitives.glow(self.screen, centre, radius,
                        (*palette.token[:3], int(90 * token.intensity)),
                        layers=4, spread=radius * 1.6)
        primitives.filled_circle(self.screen, centre, radius, palette.token)

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    def fit_to_bounds(self, bounds: Tuple[float, float, float, float],
                      padding: float = 90.0) -> Tuple[float, Point]:
        """Camera zoom and offset that frame the given world rectangle.

        Returned rather than applied, so the caller can ease towards it instead
        of snapping.
        """
        min_x, min_y, max_x, max_y = bounds
        width = max(1.0, max_x - min_x)
        height = max(1.0, max_y - min_y)

        available_w = max(1.0, self.screen.get_width() - padding * 2)
        available_h = max(1.0, self.screen.get_height() - padding * 2)
        zoom = min(available_w / width, available_h / height)
        zoom = max(self.camera.min_zoom, min(1.35, zoom))

        centre_x = (min_x + max_x) / 2
        centre_y = (min_y + max_y) / 2
        offset_x = self.screen.get_width() / (2 * zoom) - centre_x
        offset_y = self.screen.get_height() / (2 * zoom) - centre_y
        return zoom, (offset_x, offset_y)


def default_state_radius() -> float:
    """The world radius of a state. One definition, used by app and renderer."""
    return STATE_RADIUS
