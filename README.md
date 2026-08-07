# Finite Automata Simulator

A visual DFA editor and simulator built with Python and Pygame. Draw states and transitions,
then step through how a string is processed, one symbol at a time.

> **Status: early.** This is a working rewrite of an older prototype, and it is being actively
> improved. Please read [Known limitations](#known-limitations) before relying on it — in
> particular, **saving and loading do not round-trip yet**.

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
| `Esc` | Stop the execution trace |

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

## Known limitations

These are known and tracked. Listing them here rather than discovering them the hard way:

- **Save and Load do not round-trip.** Save writes a timestamped file into the current working
  directory; Load always opens the bundled example. There is no file picker yet.
- **Loading does not fully refresh the canvas** — transition arrows from the previous automaton
  can persist after a load.
- **Deleting a state while dragging it, or while a transition is pending, can crash the app.**
  Press `Esc` to cancel a pending transition before deleting.
- **Transitions cannot be deleted or edited individually.** Drawing a new transition on the same
  symbol replaces the old one; otherwise the only way to remove one is to delete a state.
- **There is no undo.**
- **`q`, `w`, `r`, `n`, and `p` cannot be used as alphabet symbols**, because those keys are bound
  to editor shortcuts.
- **A rejected string does not say why it was rejected** — no transition, symbol outside the
  alphabet, and halting in a non-accepting state all report the same result.
- **Marking a state as a trap state changes how strings are evaluated** rather than being derived
  from the transition function, so it can disagree with the diagram.
- **The help panel does not scroll**, so its last few lines are not reachable.
- No NFA, ε-transitions, minimisation, equivalence checking, or regular-expression conversion yet.

## Roadmap

In rough order: fix the crashes and make save/load round-trip; extract a tested, Pygame-free
automata engine; add a CLI and DOT/TikZ/SVG export; explain *why* a string was rejected; undo;
then minimisation, equivalence checking with counterexamples, and NFA support.

## License

MIT — see [LICENSE](LICENSE).
