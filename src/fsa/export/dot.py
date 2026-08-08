"""Graphviz DOT.

Deliberately does *not* emit coordinates. Graphviz has its own layout engine
and is better at it than a hand-dragged canvas; passing positions through would
fight it. Use SVG when the on-screen arrangement is the point.

Output is deterministic: states sorted, edges sorted, symbols sorted.
"""

from typing import List, Optional

from fsa.analysis import dead_states, unreachable_states
from fsa.automaton import DFA
from fsa.symbols import StateId


def _quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _label_for(automaton: DFA, state: StateId) -> str:
    return automaton.label_of(state)


def to_dot(automaton: DFA, name: str = "automaton",
           annotate: bool = True) -> str:
    """Render an automaton as a DOT digraph.

    Args:
        automaton: What to draw.
        name: The digraph's name.
        annotate: Style trap and unreachable states differently, the way the
            application does. Turn it off for a plain diagram.
    """
    dead = dead_states(automaton) if (annotate and automaton.accept) else frozenset()
    unreachable = unreachable_states(automaton) if annotate else frozenset()

    lines: List[str] = [
        f"digraph {_quote(name)} {{",
        "  rankdir=LR;",
        '  node [shape=circle, fontname="Helvetica", fontsize=12];',
        '  edge [fontname="Helvetica", fontsize=11];',
        "",
    ]

    if automaton.initial is not None:
        lines.append('  __start [shape=none, label="", width=0, height=0];')

    for state in sorted(automaton.states):
        attributes = [f"label={_quote(_label_for(automaton, state))}"]
        if state in automaton.accept:
            attributes.append("shape=doublecircle")
        if state in unreachable:
            attributes.append("style=dashed")
        elif state in dead:
            attributes.append("style=filled")
            attributes.append('fillcolor="#fdeded"')
            attributes.append('color="#be3c3c"')
        lines.append(f"  {_quote(state)} [{', '.join(attributes)}];")

    lines.append("")
    if automaton.initial is not None:
        lines.append(f"  __start -> {_quote(automaton.initial)};")

    for (source, target), symbols in sorted(automaton.grouped_transitions().items()):
        label = ", ".join(sorted(symbols))
        lines.append(f"  {_quote(source)} -> {_quote(target)} "
                     f"[label={_quote(label)}];")

    lines.append("}")
    return "\n".join(lines) + "\n"


def write_dot(automaton: DFA, path: str, name: Optional[str] = None) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(to_dot(automaton, name or "automaton"))
