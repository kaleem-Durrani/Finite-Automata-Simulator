"""Fonts.

``pygame.font.Font(None, size)`` loads pygame's bundled fallback face, which is
what made every label look like a debug overlay. A real UI font is almost always
present on the system; this picks the best one available and caches it.

Faces are loaded once. The old code constructed a ``Font`` object inside a draw
call that ran every frame while a message was on screen.
"""

from typing import Dict, List, Optional, Tuple

import pygame

# In preference order. The first that the system actually has wins.
UI_FACES: List[str] = [
    "inter",
    "segoeuivariable",
    "segoeui",        # Windows
    "sfnstext",       # macOS
    "helveticaneue",
    "roboto",
    "notosans",
    "dejavusans",     # common on Linux
    "calibri",
    "arial",
    "verdana",
]

MONO_FACES: List[str] = [
    "cascadiamono",
    "cascadiacode",
    "jetbrainsmono",
    "firacode",
    "consolas",       # Windows
    "sfmono",
    "menlo",
    "dejavusansmono",
    "couriernew",
]


class FontBook:
    """Loads and caches the faces the interface uses.

    Sizes are named rather than numeric so that type stays on a scale instead of
    each call site picking a number.
    """

    #: name -> (point size, bold)
    SIZES: Dict[str, Tuple[int, bool]] = {
        "title": (19, True),
        "heading": (15, True),
        "body": (14, False),
        "body_strong": (14, True),
        "small": (12, False),
        "small_strong": (12, True),
        "tiny": (11, False),
        "state": (15, True),      # the label inside a state circle
        "edge": (12, True),       # transition labels
    }

    MONO_SIZES: Dict[str, int] = {
        "input": 16,
        "strip": 20,
        "small": 12,
    }

    def __init__(self) -> None:
        self._ui_face = self._first_available(UI_FACES)
        self._mono_face = self._first_available(MONO_FACES)
        self._cache: Dict[Tuple[str, int, bool], pygame.font.Font] = {}

    @staticmethod
    def _first_available(candidates: List[str]) -> Optional[str]:
        installed = set(pygame.font.get_fonts())
        for name in candidates:
            if name in installed:
                return name
        return None

    @property
    def ui_face_name(self) -> str:
        return self._ui_face or "pygame default"

    @property
    def mono_face_name(self) -> str:
        return self._mono_face or "pygame default"

    def _load(self, face: Optional[str], size: int, bold: bool) -> pygame.font.Font:
        key = (face or "", size, bold)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if face is None:
            font = pygame.font.Font(None, int(size * 1.45))
            font.set_bold(bold)
        else:
            font = pygame.font.SysFont(face, size, bold=bold)
        self._cache[key] = font
        return font

    def ui(self, name: str = "body") -> pygame.font.Font:
        """A UI face at a named size."""
        size, bold = self.SIZES[name]
        return self._load(self._ui_face, size, bold)

    def mono(self, name: str = "input") -> pygame.font.Font:
        """A monospaced face, for input and for the tape strip."""
        return self._load(self._mono_face, self.MONO_SIZES[name], False)

    def scaled(self, name: str, zoom: float) -> pygame.font.Font:
        """A UI face scaled by the camera zoom, for text drawn in the world.

        Clamped so labels stay legible when zoomed out and do not consume the
        whole screen when zoomed in.
        """
        size, bold = self.SIZES[name]
        scaled = max(9, min(40, round(size * zoom)))
        return self._load(self._ui_face, scaled, bold)
