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
| Arrow tool + click | Click the source state, then its target, to draw a transition |
| Shift + left click | The same thing without leaving the pointer tool |
| Right click | Open a context menu for a state, a transition arrow, or the canvas |
| Space + drag | Pan the view — or switch on the hand tool in the toolbar |
| Right or middle drag | Pan the view |
| Scroll wheel | Zoom in/out |

A right *drag* pans and a right *click* opens a menu; the menu waits for the button to come up and
only appears if the pointer never travelled. Panning with the right button matters on a trackpad,
where there is no middle button to hold.

`Space` is the same story in one key: held it pans, tapped it adds a state. Which one it was is
only knowable when the key comes up, so the state appears on release.

### Tools and dialogs

The toolbar carries two tools beside the file buttons: an arrow that draws
transitions and a hand that pans. They are exclusive, and clicking the active one
returns you to the pointer. Every keyboard command has a button too — the run panel
has back, play/pause, forward and stop, and every dialog has real Cancel and confirm
buttons. The confirm button names what it will do ("Quit", "Discard", "Save") rather
than saying "Yes" to a question you would have to read twice.

### Panels

The window opens maximised, with the canvas full-bleed underneath everything else. The symbol
palette is a small card in the top-left corner and is always visible — what you can draw with
should never be hidden. Everything else folds away: click any right-hand panel's title to collapse
it to a labelled notch, and the panels below glide up into the space. The test-string box folds
down to a pill in the bottom corner and opens on a click, or on its own whenever a run produces a
verdict.

### Keyboard

| Key | Effect |
|---|---|
| `Ctrl+Z` / `Ctrl+Y` | Undo / redo the last edit (`Ctrl+Shift+Z` also redoes) |
| `Space` | Tapped, add a new state at the centre; held, pan the view |
| `Delete` | Remove the selected state |
| `Ctrl+A` | Toggle the selected state as accepting |
| `Ctrl+T` | Make the selected state a trap (loop every symbol back to it) |
| `Ctrl+0` | Fit the view to the automaton |
| `→` / `←` | Next / previous step during execution |
| `Tab` | Toggle automatic playback during execution |
| `Esc` | Stop the execution trace, or close a dialog |

Typing a symbol that is in the palette selects it for the next transition. Every editing
shortcut is a chord or a non-letter key precisely so that this is safe: while `Q`, `W`, `R`, `N`
and `P` were bare shortcuts, an automaton over an alphabet containing them could be drawn with the
mouse but never typed at — pressing `q` toggled accepting instead of choosing `q`.

### Context menu

Right-click a state to toggle whether it accepts, make it the initial state, rename it, turn it
into a trap, or delete it. The toggles show what the state already is. Right-click empty canvas to
add a state there or fit the view.

Right-click a transition arrow to remove any one of the symbols travelling along it, or to
straighten a curved edge. A rename sets a *display label* only: the state keeps its id, so
transitions and saved files are unaffected, and clearing the field puts the id back on screen.

"Make a trap" is an operation, not a label: it removes the state's accepting status and points
every symbol back at itself, which genuinely traps it. It then renders as a trap because it *is*
one. There is no separate "dead end" flag to set — a state that looks like a trap in the diagram
is one, and vice versa.

## Getting started

1. Press `Space` a few times to add states.
2. Pick a symbol from the palette at the top left.
3. Switch on the arrow tool, click a state, then click the target.
4. Right-click a state and choose Accepting.
5. Type a string into the input field at the bottom and press `Enter`.
6. Press the arrow keys to step through the run.

## The bundled demo

The simulator opens with a three-state demo over `{a, b}`. It recognises **a\*b⁺** — any number of
`a`s followed by at least one `b`.

<!-- generated: fsa sample -->
| Accepted | Rejected |
|---|---|
| `b` | *(empty)* |
| `ab` | `a` |
| `bb` | `aa` |
| `aab` | `ba` |
| `abb` | `aaa` |
| `bbb` | `aba` |
<!-- /generated -->

That table is produced by `fsa sample`, not written by hand — a test regenerates it and fails if
the README drifts. The previous README listed three examples and had the verdict **inverted on all
three**, which is exactly the failure mode generating it prevents.

`examples/simple_binary.json` holds a second automaton over `{0, 1}`, recognising `0*1+`.

## Command line

The engine ships with a CLI that needs no display and no pygame:

```
fsa test    machine.json 0110     # run a word; exit 0 accepted, 1 rejected
fsa run     machine.json 0110     # ...and show every step
fsa check   machine.json          # structural problems; exit 1 if any
fsa show    machine.json          # the transition table
fsa sample  machine.json -n 10    # words it accepts, shortest first
fsa export  machine.json -f svg   # dot | tikz | svg
fsa new     machine.json -a 0 1   # an empty automaton
fsa gui                           # the editor, from a checkout
```

Exit codes are the point: **0** means yes, **1** means no, **2** means it could not run. So
`fsa test m.json 0110 && echo in-the-language` works, and a marking script can loop over
submissions and branch on the status without parsing anything.

```console
$ fsa run examples/simple_binary.json 011
      state  read  next
      -----  ----  ----
   0  q0      0    q0
   1  q0      1    q1
   2  q1      1    q1
      q1           (accepting)

'011' was accepted in q1
```

## Exporting

Three formats, three jobs:

| Format | Use it when |
|---|---|
| **SVG** | You want the diagram you drew. Shares its geometry with the on-screen renderer, so the curves are literally the same curves. |
| **DOT** | You want Graphviz to lay it out properly. Positions are deliberately not passed through. |
| **TikZ** | It is going into LaTeX. Positions are carried over; loops and bends become TikZ options so it looks native. |

```bash
fsa export machine.json -f svg  -o figure.svg
fsa export machine.json -f dot  | dot -Tpng > figure.png
fsa export machine.json -f tikz -o figure.tex
```

## File format

A versioned JSON envelope. The automaton and its layout are separate, so moving a state cannot
change the language. Every collection is sorted and each transition is one line, so saving twice
produces identical bytes and a file diffs like source.

```json
{
  "version": 2,
  "automaton": {
    "states": ["q0", "q1"],
    "alphabet": ["0", "1"],
    "initial": "q0",
    "accept": ["q1"],
    "transitions": [["q0", "0", "q0"], ["q0", "1", "q1"]],
    "labels": {}
  },
  "layout": {
    "positions": { "q0": [220.0, 220.0], "q1": [470.0, 220.0] },
    "arcs": [["q0", "q1", 34.0]]
  },
  "next_id": 2
}
```

Files in the pre-versioning format still open. Their `dead_end_states` list is deliberately
dropped on read: it was a flag that made the simulator reject early without any transition saying
so, and honouring it would reintroduce the defect that removing it fixed. A state that is
genuinely a trap still reads as one, because that is derived from the edges now.

## Project layout

```
main.py                    application shell: window, input routing, scene building
editor.py                  editing state: selection, hover, drag, dirty
src/fsa/                   the automata engine -- no pygame, no dependencies
  automaton.py               immutable DFA; flat transition function
  simulate.py                Run, Verdict, and explain()
  analysis.py                reachability, dead states, defects
  language.py                enumerating accepted words
  layout.py                  positions and curve offsets
  document.py                an automaton together with its layout
  serialize.py               versioned, byte-stable JSON
  geometry.py                path maths, shared by the renderer and the exporters
  cli.py                     the command line
  export/                    dot, tikz, svg
rendering/
  camera.py                  viewport pan and zoom
  theme.py                   design tokens: two palettes, one set of names
  fonts.py                   system font selection and caching
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

- **Transitions cannot be deleted or edited from the canvas.** Drawing a new transition on the
  same symbol replaces the old one; otherwise the only way to remove one is to delete a state.
- **There is no undo.** The document is immutable, so this is a small change now rather than a
  structural one, but it is not done.
- **`q`, `w`, `r`, `n` and `p` can be alphabet symbols but cannot be *typed* to select one**,
  because those keys are still bound to editor shortcuts. Click them in the palette instead.
- No NFA, ε-transitions, minimisation, equivalence checking, or regular-expression conversion yet.

## Development

```bash
pip install -e ".[dev]"
ruff check .          # lint
mypy                  # type check
mypy --strict src/fsa # the engine is held to a higher standard
pytest                # tests, headless
```

`pip install .` installs the engine and the `fsa` command with **no** dependencies at all —
pygame is an optional `[gui]` extra. CI installs into a clean environment with pygame absent and
drives the CLI there, so "works without a display" is checked rather than claimed.

`src/fsa/` is the automata engine: immutable, dependency-free, and importable without a display.
Everything is a value, so a snapshot cannot change underneath the code holding it.

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

In rough order: explain rejections on the canvas as well as in the status line, with a
diagnostics panel; undo; edit and delete individual transitions; then minimisation, equivalence
checking with counterexamples, and NFA support.

## License

MIT — see [LICENSE](LICENSE).
