"""Design tokens.

Every colour, radius, spacing step and stroke weight in the application comes
from here. Nothing else may name a colour.

That rule is the point. Before this module there were two palettes -- one on the
renderer, one on the UI manager -- that gave the same concept different values,
plus twenty-three RGB literals inlined at their point of use. Changing "the
colour of an accepting state" meant finding every place that had an opinion
about it.

Two palettes are defined, sharing one set of token names. Adding a third means
filling in the same names again; it cannot mean inventing new ones, because the
drawing code only ever asks for tokens.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

RGB = Tuple[int, int, int]
RGBA = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Palette:
    """One complete set of colour values."""

    name: str
    is_dark: bool

    # Surfaces
    canvas: RGB
    canvas_dot: RGB          # faint dot grid on the workspace
    panel: RGB
    panel_raised: RGB
    border: RGB
    border_strong: RGB
    shadow: RGBA

    # Elevation. Painted by the primitives as 1px inner lines over a surface's
    # own fill, so they carry alpha and one pair serves every control colour.
    bevel_light: RGBA        # inner top-edge highlight on raised surfaces
    bevel_dark: RGBA         # inner bottom-edge lowlight on raised surfaces
    well_shadow: RGBA        # inner top shadow inside sunken fields

    # Type
    text: RGB
    text_muted: RGB
    text_faint: RGB
    text_on_accent: RGB

    # Interaction
    accent: RGB
    accent_soft: RGBA
    control: RGB
    control_hover: RGB
    control_active: RGB
    field: RGB

    # States. Each kind differs in fill, in ring colour, *and* in a shape
    # signal -- a double ring, a hatch, a dashed outline -- so the four are
    # still distinguishable in greyscale or to a colour-blind reader. Colour
    # here is reinforcement, never the only channel.
    state_fill: RGB
    state_ring: RGB
    state_text: RGB
    accept_fill: RGB
    accept_ring: RGB
    active_ring: RGB
    active_glow: RGBA
    dead_fill: RGB
    dead_ring: RGB
    dead_text: RGB
    dead_hatch: RGBA
    unreachable_fill: RGB
    unreachable_ring: RGB
    unreachable_text: RGB
    initial_marker: RGB
    selected_ring: RGB
    hover_ring: RGB

    # Edges
    edge: RGB
    edge_muted: RGB
    edge_active: RGB
    label_plate: RGB
    label_text: RGB

    # Execution
    token: RGB
    token_trail: RGBA
    strip_cell: RGB
    strip_cell_done: RGB
    strip_cell_current: RGB
    strip_text: RGB
    strip_text_done: RGB

    # Semantics
    success: RGB
    error: RGB
    warning: RGB

    # A hue per alphabet symbol, indexed by position in sorted(alphabet).
    # Chosen to stay distinguishable in both themes and under the common forms
    # of colour blindness; the label is always drawn as well, so colour is
    # reinforcement rather than the only channel.
    edge_cycle: Tuple[RGB, ...]


DARK = Palette(
    name="dark",
    is_dark=True,

    canvas=(19, 21, 26),
    canvas_dot=(33, 37, 46),
    panel=(26, 29, 36),
    panel_raised=(33, 37, 46),
    border=(45, 50, 61),
    border_strong=(65, 72, 86),
    shadow=(0, 0, 0, 110),

    bevel_light=(255, 255, 255, 26),
    bevel_dark=(0, 0, 0, 96),
    well_shadow=(0, 0, 0, 110),

    text=(228, 232, 240),
    text_muted=(140, 150, 168),
    text_faint=(96, 105, 122),
    text_on_accent=(12, 14, 18),

    accent=(56, 189, 248),
    accent_soft=(56, 189, 248, 40),
    control=(38, 43, 53),
    control_hover=(48, 54, 66),
    control_active=(56, 189, 248),
    field=(15, 17, 21),

    state_fill=(46, 53, 66),
    state_ring=(128, 141, 163),
    state_text=(232, 236, 244),
    accept_fill=(30, 60, 54),
    accept_ring=(52, 211, 153),
    active_ring=(56, 189, 248),
    active_glow=(56, 189, 248, 46),
    dead_fill=(58, 34, 38),
    dead_ring=(220, 110, 110),
    dead_text=(214, 168, 168),
    dead_hatch=(220, 110, 110, 70),
    unreachable_fill=(34, 36, 43),
    unreachable_ring=(120, 110, 145),
    unreachable_text=(140, 133, 160),
    initial_marker=(148, 160, 180),
    selected_ring=(251, 191, 36),
    hover_ring=(176, 186, 204),

    edge=(112, 122, 140),
    edge_muted=(66, 72, 85),
    edge_active=(56, 189, 248),
    # Deliberately several steps off `canvas`: the plate's whole job is to lift
    # the symbol off the line it sits on. Matching the canvas makes it vanish.
    label_plate=(40, 45, 56),
    label_text=(198, 206, 220),

    token=(125, 211, 252),
    token_trail=(56, 189, 248, 70),
    strip_cell=(33, 37, 46),
    strip_cell_done=(26, 29, 36),
    strip_cell_current=(56, 189, 248),
    strip_text=(228, 232, 240),
    strip_text_done=(105, 113, 130),

    success=(52, 211, 153),
    error=(248, 113, 113),
    warning=(251, 191, 36),

    edge_cycle=(
        (129, 178, 245),   # blue
        (244, 155, 108),   # orange
        (156, 204, 132),   # green
        (206, 148, 235),   # violet
        (232, 176, 92),    # amber
        (120, 205, 208),   # teal
    ),
)


LIGHT = Palette(
    name="light",
    is_dark=False,

    canvas=(250, 249, 247),
    canvas_dot=(226, 223, 217),
    panel=(255, 255, 255),
    panel_raised=(248, 247, 244),
    border=(226, 223, 217),
    border_strong=(196, 191, 183),
    shadow=(60, 55, 48, 34),

    bevel_light=(255, 253, 247, 170),
    bevel_dark=(60, 55, 48, 42),
    well_shadow=(60, 55, 48, 46),

    text=(28, 28, 30),
    text_muted=(104, 102, 98),
    text_faint=(148, 145, 140),
    text_on_accent=(255, 255, 255),

    accent=(37, 99, 235),
    accent_soft=(37, 99, 235, 28),
    control=(244, 243, 240),
    control_hover=(234, 232, 228),
    control_active=(37, 99, 235),
    field=(255, 255, 255),

    state_fill=(255, 255, 255),
    state_ring=(52, 52, 56),
    state_text=(24, 24, 26),
    accept_fill=(233, 250, 242),
    accept_ring=(5, 150, 105),
    active_ring=(37, 99, 235),
    active_glow=(37, 99, 235, 34),
    dead_fill=(253, 237, 237),
    dead_ring=(190, 60, 60),
    dead_text=(146, 52, 52),
    dead_hatch=(190, 60, 60, 60),
    unreachable_fill=(247, 246, 249),
    unreachable_ring=(140, 124, 170),
    unreachable_text=(120, 110, 140),
    initial_marker=(74, 74, 78),
    selected_ring=(217, 119, 6),
    hover_ring=(96, 94, 90),

    edge=(74, 74, 78),
    edge_muted=(176, 173, 168),
    edge_active=(37, 99, 235),
    # Was exactly `canvas`, which made the plate invisible: the symbol sat
    # directly on its own transition line with nothing separating them.
    label_plate=(231, 228, 220),
    label_text=(48, 48, 52),

    token=(37, 99, 235),
    token_trail=(37, 99, 235, 60),
    strip_cell=(255, 255, 255),
    strip_cell_done=(240, 238, 234),
    strip_cell_current=(37, 99, 235),
    strip_text=(28, 28, 30),
    strip_text_done=(150, 147, 142),

    success=(5, 150, 105),
    error=(200, 42, 42),
    warning=(180, 83, 9),

    edge_cycle=(
        (37, 99, 235),     # blue
        (194, 96, 20),     # orange
        (22, 128, 62),     # green
        (124, 58, 190),    # violet
        (161, 98, 7),      # amber
        (14, 116, 128),    # teal
    ),
)


PALETTES: Dict[str, Palette] = {"dark": DARK, "light": LIGHT}


# ----------------------------------------------------------------------
# Non-colour tokens
# ----------------------------------------------------------------------

class Space:
    """A spacing scale. Layout uses these, never arbitrary pixel counts."""

    xs = 4
    sm = 8
    md = 12
    lg = 16
    xl = 24
    xxl = 32


class Radius:
    """Corner radii."""

    sm = 4
    md = 8
    lg = 12
    pill = 999


class Stroke:
    """Line weights, in screen pixels at zoom 1."""

    hairline = 1
    thin = 2
    medium = 3
    thick = 4


class Motion:
    """Animation durations in milliseconds, and what they are for.

    Kept together so the interface has a consistent rhythm rather than each
    animation inventing its own timing.
    """

    instant = 90       # hover, focus rings
    quick = 160        # selection, small state changes
    normal = 260       # panels, camera easing
    step = 380         # one transition of the automaton -- the token's travel
    slow = 600         # fit-to-content, large camera moves


class Theme:
    """The active palette, plus the tokens that do not vary between palettes."""

    def __init__(self, palette_name: str = "dark"):
        self.palette = PALETTES[palette_name]
        self.space = Space
        self.radius = Radius
        self.stroke = Stroke
        self.motion = Motion

    @property
    def name(self) -> str:
        return self.palette.name

    @property
    def is_dark(self) -> bool:
        return self.palette.is_dark

    def use(self, palette_name: str) -> None:
        """Switch palettes in place, so every holder sees the change."""
        self.palette = PALETTES[palette_name]

    def toggle(self) -> str:
        """Flip between dark and light. Returns the new palette's name."""
        self.use("light" if self.palette.is_dark else "dark")
        return self.palette.name

    def edge_color(self, index: int) -> RGB:
        """The colour for the nth symbol in the sorted alphabet.

        Indexed by position rather than keyed on the literal characters 'a' and
        'b', which left the shipped {0,1} example rendering entirely in one
        colour.
        """
        cycle = self.palette.edge_cycle
        return cycle[index % len(cycle)]

    def __getattr__(self, item: str) -> object:
        """Read palette colours directly off the theme: ``theme.canvas``."""
        try:
            return getattr(self.palette, item)
        except AttributeError as exc:
            raise AttributeError(f"no such design token: {item}") from exc
