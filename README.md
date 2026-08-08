# Finite Automata Simulator

A visual DFA editor and simulator built with Python and Pygame. Draw states and transitions,
then step through how a string is processed, one symbol at a time.

> **Status: early.** This is a working rewrite of an older prototype, and it is being actively
> improved. Please read [Known limitations](#known-limitations) before relying on it.

## Features

- **Visual DFA design** — create and edit deterministic finite automata on an infinite canvas
- **Explained results** — a rejection says *why*: no transition defined, a symbol outside the
  alphabet, or the run halting in a non-accepting state
- **Animated execution** — a token travels along the edge it is taking, the state it enters lights
  up, and the input tape advances under a moving read head
- **Step-by-step, in both directions** — walk forwards and backwards through the run, or play it
  back at an adjustable speed
- **Four visually distinct state kinds** — normal, accepting (double ring), trap (hatched), and
  unreachable (dashed). Each differs in fill, ring colour *and* shape, so they stay
  distinguishable in greyscale and to colour-blind readers. Traps and unreachable states are
  derived from the transition function, not declared, and a legend appears listing whichever
  kinds are actually on screen
- **Dark and light themes** — toggle from the toolbar
- **Pan and zoom** — with eased camera movement and fit-to-content
- **Transition rendering** — anti-aliased curves, self-loops, bidirectional pairs separated
  automatically, and labels on plates so they stay legible where edges cross
- **Configurable symbol palette** — `a`, `b`, `0`, `1` by default; more can be added at runtime
- **Context menus** — right-click a state to change its type or delete it
- **JSON file format** — human-readable and hand-editable

## Requirements

- Python 3.12+
- Pygame 2.6+

## Installation

```bash
pip install pygame
python main.py
```

## Controls

### Mouse

| Action | Effect |
|---|---|
| Left click | Select a state, or drag it |
| Shift + left click | Start creating a transition (then click the target state) |
| Right click | Open a context menu |
| Middle click + drag | Pan the view |
| Scroll wheel | Zoom in/out |

### Keyboard

| Key | Effect |
|---|---|
| `Space` | Add a new state at the centre of the view |
| `Delete` | Remove the selected state |
| `Q` | Toggle the selected state as accepting |
| `W` | Make the selected state a trap (loop every symbol back to it) |
| `R` | Fit the view to the automaton |
| `N` / `P` | Next / previous step during execution |
| `Tab` | Toggle automatic playback during execution |
| `Esc` | Stop the execution trace, or close a dialog |

Typing a symbol that is in the palette selects it for the next transition.

### Context menu

Right-click a state to toggle whether it accepts, make it the initial state, turn it into a trap,
or delete it. The toggles show what the state already is. Right-click empty canvas to add a state
there or fit the view.

"Make a trap" is an operation, not a label: it removes the state's accepting status and points
every symbol back at itself, which genuinely traps it. It then renders as a trap because it *is*
one. There is no separate "dead end" flag to set — a state that looks like a trap in the diagram
is one, and vice versa.

## Getting started

1. Press `Space` a few times to add states.
2. Pick a symbol from the palette at the top left.
3. Shift+click a state to begin a transition, then click the target state.
4. Right-click a state and choose Accepting.
5. Type a string into the input field at the bottom and press `Enter`.
6. Press `N` and `P` to step through the run.

## The bundled demo

The simulator opens with a three-state demo over the alphabet `{a, b}`. It recognises the
language **a\*b⁺** — any number of `a`s followed by at least one `b`:

| Input | Result |
|---|---|
| `b` | accepted |
| `ab` | accepted |
| `aab` | accepted |
| `abb` | accepted |
| `a` | rejected |
| `aa` | rejected |
| `ba` | rejected |
| `aba` | rejected |
| *(empty)* | rejected |

`examples/simple_binary.json` contains a second automaton over `{0, 1}`.

## File format

Automata are stored as JSON:

```json
{
  "states": { "q0": { "position": [200, 200], "state_type": "normal" } },
  "transitions": { "q0": { "0": "q0", "1": "q1" } },
  "alphabet": ["0", "1"],
  "initial_state": "q0",
  "accept_states": ["q1"],
  "dead_end_states": ["q2"],
  "next_state_id": 3
}
```

## Project layout

```
main.py                    application class, event loop, scene building
src/fsa/                   the automata engine -- no pygame, no dependencies
  automaton.py               immutable DFA; flat transition function
  simulate.py                Run, Verdict, and explain()
  analysis.py                reachability, dead states, defects
core/                      legacy editor model, being retired
rendering/
  theme.py                   design tokens: two palettes, one set of names
  fonts.py                   system font selection and caching
  geometry.py                path maths -- edges, self-loops, arrowheads
  primitives.py              anti-aliased drawing over gfxdraw
  animation.py               easing curves and interpolated values
  scene.py                   what to draw, described without pixels
  renderer.py                paints a scene
ui/
  layout_spec.py             every UI rectangle, computed once from the window size
  ui_manager.py              panels, dialogs, palette, help
tests/                     conformance spec, engine, geometry, app regressions
examples/                  sample automata
```

## Saving and loading

Save and Load prompt for a filename. Names are resolved against the project folder, and `.json`
is appended if you leave the extension off, so a file saved in one session is findable in the
next regardless of where you launched Python from.

The window title shows the current file, with a `*` when there are unsaved changes. Quitting or
loading with unsaved changes asks for confirmation first.

## Known limitations

These are known and tracked. Listing them here rather than letting you discover them the hard way:

- **The editor still uses a separate legacy model.** Simulation and analysis go through the
  engine, so verdicts and trap states are correct, but editing operations do not yet. Deleting the
  start state, for instance, still promotes another state rather than leaving none.
- **Transitions cannot be deleted or edited individually.** Drawing a new transition on the same
  symbol replaces the old one; otherwise the only way to remove one is to delete a state.
- **There is no undo.**
- **`q`, `w`, `r`, `n`, and `p` cannot be used as alphabet symbols**, because those keys are bound
  to editor shortcuts.
- **The symbol palette and the automaton's alphabet are separate.** Loading a file adds its
  symbols to the palette, but the two are not otherwise kept in step.
- No NFA, ε-transitions, minimisation, equivalence checking, or regular-expression conversion yet.

## Development

```bash
pip install -e ".[dev]"
ruff check .      # lint
mypy              # type check
pytest            # tests, headless
```

`src/fsa/` is the automata engine: immutable, dependency-free, and importable without a display.
`rendering/` is the view layer — `theme.py` holds every colour and spacing token, `geometry.py` is
pure path maths shared with the future SVG exporter, and `scene.py` is the boundary that keeps the
renderer ignorant of automata.

The test suite includes `tests/conformance/cases.json`: a specification of DFA semantics where
every expected verdict and run was computed by hand from the transition function rather than
recorded from the implementation. Cases the simulator currently gets wrong are marked and run as
strict expected-failures, so fixing one turns it green and CI notices if a defect disappears
without the marker being removed.

Neither `src/fsa/` nor `core/` may import pygame — the automaton has to stay usable without a
display. CI enforces this by parsing the import statements.

## Roadmap

In rough order: move the editor itself onto the engine and retire `core/`; add a CLI and
DOT/TikZ/SVG export; undo; edit and delete individual transitions; then minimisation, equivalence
checking with counterexamples, and NFA support.

## License

MIT — see [LICENSE](LICENSE).
