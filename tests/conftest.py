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

# The project is not installed as a package yet, so tests import the modules
# directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
