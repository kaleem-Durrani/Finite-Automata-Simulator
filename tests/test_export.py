"""Exporters.

All three are pure string builders, so they can be checked exactly. The
important properties are that output is deterministic (a golden file is
worthless otherwise), that the SVG really does reproduce the on-screen
geometry, and that text going into DOT or LaTeX is escaped.
"""

import xml.etree.ElementTree as ET

import pytest

import fsa
from fsa import Document, Layout, geometry
from fsa.export import FORMATS, render, to_dot, to_svg, to_tikz

SVG_NS = "{http://www.w3.org/2000/svg}"


@pytest.fixture
def demo():
    """a*b+ over {a, b}, with a genuine trap."""
    document = Document().add_symbol("a").add_symbol("b")
    document, q0 = document.add_state((220.0, 220.0))
    document, q1 = document.add_state((470.0, 220.0))
    document, q2 = document.add_state((345.0, 400.0))
    document = (document.add_transition(q0, "a", q0)
                        .add_transition(q0, "b", q1)
                        .add_transition(q1, "a", q2)
                        .add_transition(q1, "b", q1)
                        .add_transition(q2, "a", q2)
                        .add_transition(q2, "b", q2))
    return document.toggle_accept(q1).set_initial(q0)


# ---------------------------------------------------------------------------
# Every format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_output_is_deterministic(demo, fmt):
    """A golden file is worthless if the bytes move between runs."""
    assert render(demo.automaton, demo.layout, fmt) == \
           render(demo.automaton, demo.layout, fmt)


@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_every_state_and_symbol_appears(demo, fmt):
    text = render(demo.automaton, demo.layout, fmt)
    for state in demo.automaton.states:
        assert state in text.replace("s" + state, state), f"{fmt}: {state} missing"
    for symbol in demo.automaton.alphabet:
        assert symbol in text, f"{fmt}: {symbol} missing"


@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_an_empty_automaton_does_not_crash(fmt):
    assert render(fsa.DFA(), Layout(), fmt)


def test_an_unknown_format_is_refused(demo):
    with pytest.raises(ValueError, match="unknown format"):
        render(demo.automaton, demo.layout, "png")


# ---------------------------------------------------------------------------
# DOT
# ---------------------------------------------------------------------------


def test_dot_marks_the_initial_and_accepting_states(demo):
    text = to_dot(demo.automaton)
    assert "__start -> \"q0\"" in text
    assert 'shape=doublecircle' in text


def test_dot_styles_a_trap(demo):
    text = to_dot(demo.automaton)
    trap_line = next(line for line in text.splitlines() if line.startswith('  "q2" ['))
    assert "fillcolor" in trap_line


def test_dot_can_omit_annotation(demo):
    text = to_dot(demo.automaton, annotate=False)
    assert "fillcolor" not in text
    assert "shape=doublecircle" in text, "accepting is structure, not annotation"


def test_dot_groups_symbols_on_one_edge(demo):
    text = to_dot(demo.automaton)
    assert '"q2" -> "q2" [label="0, 1"]' in text or '"q2" -> "q2" [label="a, b"]' in text


def test_dot_escapes_quotes():
    automaton = fsa.DFA().with_state('he said "hi"')
    assert r'\"hi\"' in to_dot(automaton)


# ---------------------------------------------------------------------------
# TikZ
# ---------------------------------------------------------------------------


def test_tikz_uses_the_automata_library_idiom(demo):
    text = to_tikz(demo.automaton, demo.layout)
    assert r"\begin{tikzpicture}" in text
    assert "state, initial" in text
    assert "accepting" in text
    assert "loop" in text, "self-loops become a TikZ option"


def test_tikz_flips_the_y_axis(demo):
    """The canvas's y grows downward; TikZ's grows up."""
    text = to_tikz(demo.automaton, demo.layout)
    rows = {}
    for line in text.splitlines():
        if line.strip().startswith(r"\node"):
            name = line.split("(")[1].split(")")[0]
            coords = line.split(" at (")[1].split(")")[0]
            rows[name] = float(coords.split(",")[1])
    # q2 is lowest on the canvas, so it must be lowest in TikZ too.
    assert rows["sq2"] < rows["sq0"]


def test_tikz_escapes_latex_specials():
    automaton = fsa.DFA().with_state("q_1").with_label("q_1", "a&b%c")
    text = to_tikz(automaton)
    assert r"\&" in text and r"\%" in text
    assert "sq_1" in text, "the node name stays safe"


def test_tikz_standalone_compiles_as_a_document(demo):
    text = to_tikz(demo.automaton, demo.layout, standalone=True)
    assert text.startswith(r"\documentclass")
    assert r"\end{document}" in text.rstrip().splitlines()[-1]


def test_tikz_caption_wraps_a_figure(demo):
    text = to_tikz(demo.automaton, demo.layout, caption="Recognises a*b+")
    assert r"\begin{figure}" in text
    assert "Recognises a*b+" in text


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------


def test_svg_is_valid_xml(demo):
    root = ET.fromstring(to_svg(demo.automaton, demo.layout))
    assert root.tag == f"{SVG_NS}svg"


def test_svg_draws_one_circle_per_state_plus_accept_rings(demo):
    root = ET.fromstring(to_svg(demo.automaton, demo.layout))
    circles = root.findall(f"{SVG_NS}circle")
    assert len(circles) == len(demo.automaton.states) + len(demo.automaton.accept)


def test_svg_places_states_where_the_layout_says(demo):
    """This is the exporter that must match the screen."""
    root = ET.fromstring(to_svg(demo.automaton, demo.layout))
    drawn = {(round(float(c.get("cx"))), round(float(c.get("cy"))))
             for c in root.findall(f"{SVG_NS}circle")}
    for state in demo.automaton.states:
        x, y = demo.layout.position_of(state)
        assert (round(x), round(y)) in drawn, f"{state} not drawn at its position"


def test_svg_edges_use_the_same_geometry_as_the_renderer(demo):
    """Not similar geometry -- the same function, so they cannot drift."""
    root = ET.fromstring(to_svg(demo.automaton, demo.layout))
    paths = [p.get("d") for p in root.findall(f"{SVG_NS}path")]

    expected = geometry.edge_path(demo.layout.position_of("q0"),
                                  demo.layout.position_of("q1"),
                                  30.0, 30.0, 0.0)
    start = f"M {expected[0][0]:.2f}".rstrip("0").rstrip(".")
    assert any(path.startswith(start[:8]) for path in paths if path)


def test_svg_viewbox_contains_everything(demo):
    root = ET.fromstring(to_svg(demo.automaton, demo.layout))
    min_x, min_y, width, height = (float(v) for v in root.get("viewBox").split())

    for state in demo.automaton.states:
        x, y = demo.layout.position_of(state)
        assert min_x <= x - 30 and x + 30 <= min_x + width
        assert min_y <= y - 30 and y + 30 <= min_y + height


def test_svg_marks_the_state_kinds_differently(demo):
    root = ET.fromstring(to_svg(demo.automaton, demo.layout))
    strokes = {c.get("stroke") for c in root.findall(f"{SVG_NS}circle")}
    assert len(strokes) >= 3, "normal, accepting and trap must differ"
    assert any(c.get("stroke-dasharray")
               for c in ET.fromstring(
                   to_svg(demo.automaton.with_state("island"), demo.layout)
               ).findall(f"{SVG_NS}circle")), "unreachable is dashed"


def test_svg_escapes_markup():
    automaton = fsa.DFA().with_state("x").with_label("x", "<script>&")
    text = to_svg(automaton)
    assert "&lt;script&gt;&amp;" in text
    ET.fromstring(text)


def test_svg_title_is_included(demo):
    root = ET.fromstring(to_svg(demo.automaton, demo.layout, title="Demo"))
    assert root.find(f"{SVG_NS}title").text == "Demo"


def test_svg_without_a_layout_still_places_states():
    automaton = (fsa.DFA().with_states(["a", "b", "c"])
                 .with_transition("a", "x", "b"))
    root = ET.fromstring(to_svg(automaton))
    centres = {(c.get("cx"), c.get("cy")) for c in root.findall(f"{SVG_NS}circle")}
    assert len(centres) == 3, "laid out on a grid, not stacked at the origin"
