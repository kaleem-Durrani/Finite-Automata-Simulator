"""TikZ, for LaTeX.

Emits a ``tikzpicture`` using the ``automata`` library, which is what a
theory-of-computation write-up expects. Positions come from the document's
layout, converted to centimetres and y-flipped, so the figure matches what was
on screen -- the arrangement is usually the part the author cared about.

Self-loops and bowed edges are expressed as TikZ options (``loop above``,
``bend left``) rather than as explicit coordinates, so TikZ draws them in its
own house style instead of a transcription of our pixels.
"""

import math
from typing import Any, List, Mapping, Optional, Tuple

from fsa.automaton import DFA
from fsa.layout import Layout

#: World pixels per centimetre. A state is 60px across, so this puts a state a
#: little over a centimetre wide, which reads well at figure size.
PIXELS_PER_CM = 52.0


def _escape(text: str) -> str:
    """Escape the characters LaTeX treats specially."""
    for char, replacement in (("\\", r"\textbackslash{}"), ("&", r"\&"),
                              ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                              ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                              ("~", r"\textasciitilde{}"),
                              ("^", r"\textasciicircum{}")):
        text = text.replace(char, replacement)
    return text


def _node_name(state: str) -> str:
    """A TikZ-safe node name."""
    return "s" + "".join(char if char.isalnum() else "_" for char in state)


def to_tikz(automaton: DFA, layout: Optional[Layout] = None,
            caption: Optional[str] = None,
            standalone: bool = False) -> str:
    """Render an automaton as a TikZ picture.

    Args:
        automaton: What to draw.
        layout: Where the states go. Without one they are placed in a row.
        caption: Wrap the picture in a figure with this caption.
        standalone: Emit a complete compilable document rather than a fragment.
    """
    layout = layout or Layout.grid(sorted(automaton.states))
    centre = layout.centre()

    lines: List[str] = []
    if standalone:
        lines += [
            r"\documentclass[tikz,border=8pt]{standalone}",
            r"\usepackage{tikz}",
            r"\usetikzlibrary{automata, positioning, arrows.meta}",
            r"\begin{document}",
        ]
    if caption:
        lines.append(r"\begin{figure}[htbp]")
        lines.append(r"\centering")

    lines.append(r"\begin{tikzpicture}[shorten >=1pt, node distance=2.4cm, "
                 r"on grid, auto, every state/.style={minimum size=1cm}]")

    for state in sorted(automaton.states):
        options = ["state"]
        if state == automaton.initial:
            options.append("initial")
        if state in automaton.accept:
            options.append("accepting")

        x, y = layout.position_of(state)
        # TikZ's y axis points up; the canvas's points down.
        tx = (x - centre[0]) / PIXELS_PER_CM
        ty = -(y - centre[1]) / PIXELS_PER_CM
        lines.append(f"  \\node[{', '.join(options)}] ({_node_name(state)}) "
                     f"at ({tx:.2f},{ty:.2f}) {{${_escape(automaton.label_of(state))}$}};")

    lines.append("")
    lines.append(r"  \path[->]")

    grouped = automaton.grouped_transitions()
    for (source, target), symbols in sorted(grouped.items()):
        label = _escape(", ".join(sorted(symbols)))
        if source == target:
            edge = f"[loop {_loop_side(layout, source, grouped)}]"
        elif (target, source) in grouped:
            # A pair in both directions: bend them apart, as on the canvas.
            edge = "[bend left=22]"
        elif layout.arc_of(source, target):
            bend = "left" if layout.arc_of(source, target) > 0 else "right"
            edge = f"[bend {bend}=18]"
        else:
            edge = ""
        lines.append(f"    ({_node_name(source)}) edge {edge} "
                     f"node {{${label}$}} ({_node_name(target)})")

    lines.append("  ;")
    lines.append(r"\end{tikzpicture}")

    if caption:
        lines.append(f"\\caption{{{_escape(caption)}}}")
        lines.append(r"\end{figure}")
    if standalone:
        lines.append(r"\end{document}")

    return "\n".join(lines) + "\n"


def _loop_side(layout: Layout, state: str,
               grouped: Mapping[Tuple[str, str], Any]) -> str:
    """Which side a self-loop should sit on.

    Points away from the average direction of the state's other edges, the same
    rule the canvas uses, so the figure and the screen agree.
    """
    here = layout.position_of(state)
    dx = dy = 0.0
    for (source, target) in grouped:
        if source == target:
            continue
        other = target if source == state else (source if target == state else None)
        if other is None:
            continue
        there = layout.position_of(other)
        length = math.hypot(there[0] - here[0], there[1] - here[1]) or 1.0
        dx += (there[0] - here[0]) / length
        dy += (there[1] - here[1]) / length

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return "above"
    if abs(dy) >= abs(dx):
        return "above" if dy > 0 else "below"
    return "left" if dx > 0 else "right"


def write_tikz(automaton: DFA, path: str, layout: Optional[Layout] = None,
               caption: Optional[str] = None, standalone: bool = False) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(to_tikz(automaton, layout, caption, standalone))
