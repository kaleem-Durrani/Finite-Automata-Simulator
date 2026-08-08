# Finite Automata Simulator

A visual DFA editor and simulator built with Python and Pygame. Draw states and transitions,
then step through how a string is processed, one symbol at a time.

> **Status: early.** This is a working rewrite of an older prototype, and it is being actively
> improved. Please read [Known limitations](#known-limitations) before relying on it.

## Features

- **Visual DFA design** — create and edit deterministic finite automata on an infinite canvas
- **Configurable symbol palette** — `a`, `b`, `0`, `1` by default; more can be added at runtime
- **String testing** — test a string against the automaton and see the result
- **Step-by-step execution** — walk forwards and backwards through the run, or play it back
  automatically at an adjustable speed
- **Pan and zoom** — navigate larger automata
- **State types** — normal, accepting, and trap ("dead end") states, each visually distinct
- **Transition rendering** — curved arrows with labels, self-loops, and grouped multi-symbol edges
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
| `W` | Toggle the selected state as a trap state |
| `R` | Reset the camera |
| `N` / `P` | Next / previous step during execution |
| `Tab` | Toggle automatic playback during execution |
| `Esc` | Stop the execution trace, or close a dialog |

Typing a symbol that is in the palette selects it for the next transition.

### Context menu

Right-click a state to set it as accepting, as a trap state, as normal, or as the initial state,
or to delete it. Right-click empty canvas to add a state there or reset the view.

## Getting started

1. Press `Space` a few times to add states.
2. Pick a symbol from the palette at the top left.
3. Shift+click a state to begin a transition, then click the target state.
4. Right-click a state to mark it as accepting.
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
main.py                 application class, event loop, frame composition
core/state.py           State object: position, type, hit-testing
core/dfa.py             the automaton: states, transitions, alphabet, simulation, JSON I/O
core/camera.py          viewport pan/zoom, world <-> screen conversion
rendering/renderer.py   drawing states, arrows, labels, self-loops
ui/ui_manager.py        toolbar, input field, symbol palette, context menus, help, overlays
examples/               sample automata
```

## Saving and loading

Save and Load prompt for a filename. Names are resolved against the project folder, and `.json`
is appended if you leave the extension off, so a file saved in one session is findable in the
next regardless of where you launched Python from.

The window title shows the current file, with a `*` when there are unsaved changes. Quitting or
loading with unsaved changes asks for confirmation first.

## Known limitations

These are known and tracked. Listing them here rather than letting you discover them the hard way:

- **Marking a state as a trap state changes how strings are evaluated** rather than the trap being
  derived from the transition function. If a "dead end" state has an outgoing path to an accepting
  state, the simulator rejects anyway — so the verdict can disagree with the diagram. This is the
  most significant known defect; it is pinned by four failing cases in the conformance suite.
- **A rejected string does not say why it was rejected.** No transition defined, symbol outside
  the alphabet, and halting in a non-accepting state all report the same bare "REJECTED".
- **The empty string cannot be tested** from the input field.
- **Shift+click drags the state** as well as starting a transition, because the click is handled
  twice. The transition still gets made; the state also moves.
- **Right-click does nothing near the top and bottom of the window** — those bands are reserved
  for the toolbar and input area regardless of what is actually drawn there.
- **The help panel does not scroll**, so its last few lines are unreachable.
- **The animation speed slider cannot be dragged.**
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

The test suite includes `tests/conformance/cases.json`: a specification of DFA semantics where
every expected verdict and run was computed by hand from the transition function rather than
recorded from the implementation. Cases the simulator currently gets wrong are marked and run as
strict expected-failures, so fixing one turns it green and CI notices if a defect disappears
without the marker being removed.

`core/` must not import pygame — the automaton model has to stay usable without a display. CI
enforces this.

## Roadmap

In rough order: give every input event a single owner (fixing shift+click, right-click coverage,
and help scrolling); extract a tested, Pygame-free automata engine and derive trap states from the
transition function instead of a flag; add a CLI and DOT/TikZ/SVG export; explain *why* a string
was rejected; undo; then minimisation, equivalence checking with counterexamples, and NFA support.

## License

MIT — see [LICENSE](LICENSE).
