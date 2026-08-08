# Lessons

Mistakes made on this project that are likely to recur. Each one cost real time
or shipped a real defect. Kept short on purpose — if an entry stops earning its
place, delete it.

---

## Testing

**A test double that is more permissive than reality proves nothing.**
Test helpers built mouse events with a `mod=` field. Pygame never sets `.mod` on
mouse events — only key events have it. `pygame.event.post` preserves arbitrary
attributes, so 100 tests stayed green while `python main.py` died on the first
click. *Build synthetic inputs with exactly the fields the real thing carries,
and no more.*

**A guard that has never been shown to fail is not a guard.**
The "engine must not import pygame" check grepped for the word `pygame`, so it
flagged the docstring explaining the rule, and would equally have missed a real
import hidden in a comment-free line it did not scan. *Write the test that
proves the guard fires, feeding it a genuine violation.*

**Headless passes are not proof for anything input-related.**
Every crash the user hit had a green suite behind it. Fuzzing 3–4k real-shaped
events catches a lot; it still did not catch the ones a real window did.
*Render to PNG and look at it. Run the app.*

**Tests that call handlers directly prove nothing about routing.**
Calling `_handle_mouse_down` bypasses the UI-consume gate — the exact thing that
was broken. *Push events through the real queue and the real loop.*

---

## Model and state

**Two representations of the same fact will drift apart.**
`transitions` and `transition_groups` held the same edges. One was updated on
load and the other was not, so the renderer drew a different automaton than the
simulator ran. *Derive the second view; never store it.*

**A snapshot that shares references is not a snapshot.**
`to_dict()` returned the live `transitions` dict. Taking a snapshot and then
loading a file emptied the snapshot, because `from_dict` cleared the very object
it pointed at. *Copy on the way out, or return immutable values.*

**App-level pointers into a model outlive the thing they point at.**
Three separate crashes, all "state deleted, something still references it".
*One `forget(id)` called from every removal path, not a guard per call site.*

**Set iteration order is not stable across processes.**
Python randomises string hashing, so `list(some_set)` serialised differently on
different runs. Saved files could not be diffed and a comparison test was flaky
by luck. *Sort anything set-derived before it leaves the process.*

---

## Geometry and rendering

**Check the degenerate case before the general one.**
Edge trimming walks the polyline for the boundary crossing. A straight edge was
two points with nothing between them, so every straight edge collapsed to a
single point and vanished from the canvas. *Densify, or handle the two-point
case explicitly.*

**Two sign flips cancel.**
Bidirectional edges bowed by `+arc` and `−arc`, but the perpendicular is already
reversed for the opposite direction — so both curves landed in exactly the same
place. The previous renderer had this bug too and nobody noticed for years.
*When separation depends on a sign, assert the two results actually differ.*

**Colour alone is not a signal.**
Normal and trap states differed by a slightly darker grey. That is no difference
on a projector, in greyscale, or to a colour-blind reader. *Give every category
a shape difference too — a second ring, a hatch, a dash — and treat colour as
reinforcement.*

**Passing RGBA to `pygame.draw` silently discards alpha** on an opaque surface.
Every "translucent" panel was fully opaque for months. *Draw to an `SRCALPHA`
scratch surface and blit it.*

**`pygame.gfxdraw` takes signed 16-bit coordinates** and raises `OverflowError`
above 32767 rather than clipping. Unbounded zoom reaches that easily. *Cull
first, then clamp, then draw.*

---

## Design

**Correct is not the same as useful.**
With no accepting state, every state is genuinely a trap — so the whole canvas
greyed out while the user was still drawing. The maths was right and the display
was useless. *Ask what the user should conclude, not only what is true.*

**Deriving something means removing the control that used to set it.**
Trap-ness became computed from the transition function, but the menu items that
set the old flag stayed — writing a value nothing read, with no visible effect.
*When a fact becomes derived, follow it all the way to the UI in the same
change.*

**A control that changes nothing visible is worse than no control.**
Same root cause, worth stating separately: if an action cannot be observed, it
will be pressed repeatedly and reported as broken. *Every action reports, or
does not exist.*

---

## Tooling

**`strict = true` is not a valid per-module mypy option.**
Setting it under `[[tool.mypy.overrides]]` applies it to the whole project. It
turned 336 test-annotation complaints into an apparent failure. *List the
individual flags per module; run `mypy --strict <pkg>` as its own step.*

**`pip install` without `--user` half-succeeds on this machine** and leaves a
broken package that later installs consider satisfied. See the note in the
project memory. *Use `--user` on the first attempt.*
