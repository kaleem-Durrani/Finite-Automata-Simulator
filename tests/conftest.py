"""Shared test configuration.

Pygame requires a video driver to be selected before ``pygame.display`` is
used. Setting the dummy driver here — before any test module imports pygame —
lets the whole suite run headlessly, locally and in CI.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# The GUI modules still live at the repository root; the engine lives under
# src/. Both are put on the path so tests can run without an editable install.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
