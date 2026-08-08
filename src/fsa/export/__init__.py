"""Getting an automaton out of the tool.

Three formats, three jobs:

* :mod:`~fsa.export.dot` for Graphviz, which re-lays-out the graph with a real
  layout engine. Best when the arrangement does not matter.
* :mod:`~fsa.export.tikz` for LaTeX, restating the diagram in TikZ's own idiom
  so it looks like the rest of the document.
* :mod:`~fsa.export.svg` for anything else, reproducing the on-screen diagram
  exactly -- it shares :mod:`fsa.geometry` with the renderer, so the curves are
  literally the same curves.

All three are pure string builders over ``(automaton, layout)``, with
deterministic output, so they can be tested against golden files.
"""

from typing import Optional

from fsa.automaton import DFA
from fsa.export.dot import to_dot, write_dot
from fsa.export.svg import to_svg, write_svg
from fsa.export.tikz import to_tikz, write_tikz
from fsa.layout import Layout

#: Format name -> file extension.
FORMATS = {"dot": ".dot", "tikz": ".tex", "svg": ".svg"}


def render(automaton: DFA, layout: Optional[Layout] = None,
           format: str = "svg") -> str:
    """Render an automaton in the named format."""
    if format == "dot":
        return to_dot(automaton)
    if format == "tikz":
        return to_tikz(automaton, layout)
    if format == "svg":
        return to_svg(automaton, layout)
    raise ValueError(f"unknown format {format!r}; expected one of "
                     f"{', '.join(sorted(FORMATS))}")


__all__ = ["FORMATS", "render",
           "to_dot", "write_dot", "to_tikz", "write_tikz", "to_svg", "write_svg"]
