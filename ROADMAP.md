# Roadmap — Phases 11 onward

`IMPROVEMENT_PLAN.md` took this project from a prototype to a tested engine, a
CLI, three exporters, a GUI that behaves, and an algorithms layer. Phases 0–9
are done. This document plans what turns it into a *complete* tool.

Read that plan's Phase 5 gate first. It asks what this project is for, and the
answer decides whether Phase 18 (release) is worth an hour. Nothing here
substitutes for that decision.

---

## Two rules that govern every phase

**1. The engine keeps zero required runtime dependencies.**

`pyproject.toml` says `dependencies = []`, and that is not an accident — Phase 3
bought a pygame-free, importable engine and CI enforces it. Every library named
below lands in exactly one of three places:

| Where | For | Examples |
|---|---|---|
| `dev` extra | Testing only. Never imported by `src/fsa`. | `hypothesis`, `automata-lib`, `greenery`, `pytest-benchmark` |
| optional extra | A better experience when present, with a working fallback when absent. | `pydot` + Graphviz, `rich`, `pyperclip` |
| nowhere | Anything that would become a hard runtime dependency of the engine. | — |

The `dot` binary is **not** on this machine's PATH, so Graphviz layout is
optional by necessity as well as by principle.

**2. Libraries replace toil, not the subject matter.**

This is a teaching tool. Implementing the subset construction *is* the point;
so is Moore's algorithm, which is why the plan chose it over Hopcroft. Handing
those to a library would leave a GUI wrapped around somebody else's engine and
delete the reason the project exists.

So `automata-lib` and `greenery` are used as **differential oracles** — we
implement the algorithm, then check our answer against a mature independent
implementation over thousands of random machines. That is strictly stronger
verification than either alone, and it is the honest way to let a battle-tested
library carry weight here.

Where a library removes genuine toil with no lesson attached — random data
generation with shrinking, graph layout, XML parsing, terminal tables — we use
it without hesitation.

---

## The phases

| Phase | Hours | Buys |
|---|---:|---|
| 11 · Verification infrastructure | 6 | Every later phase is checked against an independent implementation |
| 12a · NFA in the engine | 8 | **The biggest capability gap in the tool** |
| 12b · NFA in the editor | 8 | Drawing and running a nondeterministic machine |
| 13 · Regular expressions, both directions | 12 | Closes Kleene's theorem: the spine of the course |
| 14 · Exercises and self-grading | 8 | **Highest value per hour.** Turns an editor into a course instrument |
| 15 · Layout and editing affordances | 9 | The tool stops fighting the person using it |
| 16 · Interop: `.jff`, DOT, PNG | 8 | An adoption path into courses that already run JFLAP |
| 17 · Transducers, grammars, pumping lemma | 12 | Covers the rest of a regular-languages syllabus |
| 18 · Release | 7 | Someone other than you can install and use it |
| 19 · CFG and pushdown automata *(optional)* | 20+ | Beyond "finite automata"; only if the scope decision asks for it |

Order matters: 11 first because everything after it changes the engine, and 12
before 13 because regular expressions are built out of NFAs.

---

### Phase 11 — Verification infrastructure · 6h

Sharpen the saw before touching the engine. The current property tests
hand-roll random automata with `random.Random(seed)`; they work, but a failure
gives you a seed and a 6-state machine rather than the two-state machine that
actually reproduces it.

- **`hypothesis`** — strategies for `DFA`, `NFA`, words and `Document`, in
  `tests/strategies.py`. Replaces the hand-rolled generators. The win is
  shrinking: a failure arrives already minimised.
- **`automata-lib` as an oracle** — `tests/oracle.py` converts our `DFA` to
  theirs and back, so `minimize`, `equivalent`, the boolean operations and
  (later) `determinize` can be checked against a mature independent
  implementation. Test-only; `src/fsa` never imports it. CI already greps for
  forbidden imports; extend that guard to cover the oracle libraries.
- **`pytest-benchmark`** — pin the plan's performance criteria (100 states under
  8ms) as a test that fails on regression rather than a number in a document.

**Exit criteria**
1. `tests/strategies.py` exports `dfas()`, `nfas()`, `words()`, `documents()`;
   the existing hand-rolled generators are gone.
2. A differential test runs ≥1000 hypothesis-generated machines through
   `minimize` and `equivalent` against `automata-lib` and finds no disagreement.
3. `! grep -rq "hypothesis\|automata\b\|greenery" src/` in CI.
4. A benchmark test fails if a 100-state frame exceeds 8ms.

---

### Phase 12a — NFA in the engine · 8h

The engine models δ as `{(state, symbol): state}` — one target, always. Every
automata course spends as much time on nondeterminism as on determinism, so
this is the single change that most enlarges what the tool can teach.

- **`src/fsa/nfa.py`** — an `NFA` value type with
  `transitions: Mapping[Tuple[StateId, Optional[Symbol]], FrozenSet[StateId]]`,
  where a `None` symbol is an ε-move. Frozen and structurally equal, exactly
  like `DFA`.
- `epsilon_closure`, `run`, `accepts` — a run of an NFA is a *set* of
  configurations, and the GUI will want to show that set travelling.
- **`determinize()`** — the subset construction, reachable subsets only. State
  ids name their subset (`{q0,q2}`), because the whole lesson is seeing where a
  DFA state came from. Deterministic naming, sorted, so two runs agree.
- `DFA.as_nfa()` for the trivial direction, so the boolean operations and
  equivalence can accept either.
- **Serialization**: version 3 for a document holding an NFA. Version 2 keeps
  emitting exactly the bytes it does today, because example files, an existing
  round-trip test and the generated README table all depend on them.
- **CLI**: `fsa determinize machine.json`.

**Exit criteria**
1. Property: `accepts(nfa, w) == accepts(determinize(nfa), w)` over
   hypothesis-generated NFAs and words.
2. Property: `determinize` output is complete and deterministic by construction.
3. Differential: our `determinize` agrees with `automata-lib`, comparing
   languages rather than state names.
4. ε-closure is tested against a machine with an ε-*cycle*, which is the case
   that breaks naive implementations.
5. Every existing file still loads, and a v2 document still serialises byte-for-
   byte as it did before.

---

### Phase 12b — NFA in the editor · 8h

Split from 12a deliberately. `Document.automaton` is typed `DFA`, and widening
it reaches the editor, `analysis`, `ops`, `product`, `minimize`, all three
exporters and most of the GUI — every one of which assumes one target per
`(state, symbol)`. That is the expensive half, and doing it in the same change
as the engine work would make both hard to review.

- **`Document` holds either.** Decide the shape: a union, or store an `NFA`
  always and treat a `DFA` as the deterministic case. The second is tempting and
  probably wrong — most of the algorithms layer is genuinely DFA-only, and
  making them all re-check determinism per call trades a type error for a
  runtime one.
- **The canvas accepts two edges on one symbol.** `editor.add_transition`
  overwrites today.
- **ε-edges** are drawable and drawn distinctly.
- **The diagnostics panel reports nondeterminism as a *fact*, not a defect** —
  the same lesson as partial δ, and the same trap as the complete/trim cycle: a
  legal design choice must not be labelled a fault with a Fix button.
- **The run animation shows a set of states**, not a token on one.
- **Menu**: Determinize, alongside Minimise and Trim.

**Exit criteria**
1. Event replay: an NFA with two `a`-edges from one state is drawn on the
   canvas, saved, reloaded and determinized from the menu.
2. Running a word on an NFA lights every state the machine is in, not one.
3. Every DFA-only algorithm refuses a nondeterministic document with an error
   naming `determinize`, and a test proves each one does.

---

### Phase 13 — Regular expressions, both directions · 12h

Closes Kleene's theorem. With Phase 12 in place this is mostly construction.

- **`src/fsa/regex.py`** — parse a regular expression to an AST. Hand-written
  recursive descent (the grammar is four productions and the parser is a
  teaching artifact in its own right); **`lark`** is the fallback if the syntax
  grows beyond that.
- **Thompson's construction** → `NFA`, one ε-machine per operator.
- **State elimination** → regex, via a GNFA. The output wants simplifying, and
  simplification is where these implementations usually get sloppy: test that
  the *language* is preserved, never that the string matches an expected one.
- **`greenery` as the oracle** — it converts regex ↔ FSM and is well-tested.
  Differential-test both directions against it.
- **CLI**: `fsa from-regex "a*b+" -o machine.json`, `fsa to-regex machine.json`.
- **GUI**: "From regular expression…" in the canvas menu; the status panel shows
  the derived expression for the current machine.

**Exit criteria**
1. Round trip: `equivalent(from_regex(r), from_regex(to_regex(from_regex(r))))`
   for hypothesis-generated expressions.
2. Differential vs `greenery` on ≥500 expressions, comparing languages.
3. Precedence and associativity tested explicitly: `ab|c`, `a|bc`, `(ab)*`,
   `a**`, and the empty expression.
4. Event replay: typing an expression produces a drawable, laid-out machine.

---

### Phase 14 — Exercises and self-grading · 8h

The highest value per hour in this document, because the hard parts are already
written. `IMPROVEMENT_PLAN.md` cites the DAVID study: the weaker cohort beat the
stronger one on DFA, RegEx and PDA purely from getting counterexample strings
back. You have `counterexample()`. What is missing is the *task*.

- **An exercise format** (`.fsx`, JSON): a prompt in prose, a reference machine
  or regular expression, an alphabet, and optional visible examples. The
  reference may be stored as a regex so the answer is not readable at a glance.
- **`fsa check machine.json --against exercise.fsx`** — exit 0 correct, 1 wrong,
  and on wrong: *"your machine accepts `bb`; the reference rejects it"*, the
  shortest such word.
- **`fsa mark exercises/ submissions/ -o results.csv`** — batch marking, one row
  per submission, with the counterexample. **`rich`** for the terminal table
  when present; plain text when not.
- **GUI**: open an exercise, see the prompt, press Test → the verdict names the
  distinguishing word and offers to run it on the canvas so the student watches
  their machine take the wrong path.

**Exit criteria**
1. `check --against` is correct on a corpus of right and wrong submissions, and
   its exit codes let a shell script branch without parsing.
2. The reported word is always shortest, and always actually distinguishes.
3. `mark` over a directory of 20 submissions produces a CSV a marker can read.
4. Event replay: failing an exercise in the GUI puts the counterexample in the
   test field and runs it.

---

### Phase 15 — Layout, rendering performance, and editing affordances · 13h

The tool currently fights anyone editing a machine of more than six states.

- **Tessellation must follow *screen* length, not world length.**
  `geometry.segments()` picks `int(length / 9)` from the world length, so a
  hundred-state machine fitted to the screen still builds every edge from
  40--72 anti-aliased quads while each is a few pixels long. Phase 8 warned
  about exactly this and its exit criterion (100 states under 8ms) was never
  measured; the Phase 11 benchmark measures it at **~97ms a frame, twelve
  times over**. The fix is to pass a scale into the path builders. It is not
  a one-liner: `_edge_paths` feeds both the renderer *and* `nearest_edge`,
  so hit-testing precision changes with it -- arguably correctly, since a
  click happens in screen space, but it needs its own tests.
- **Graphviz layout, optional.** `Layout.auto` places by BFS layers, which is
  honest but plain. `dot` produces genuinely better layered drawings. Via
  **`pydot`**, behind a capability check, falling back to BFS layers when the
  binary is absent — which it is on this machine, so the fallback path is the
  one that gets tested by default.
- **"Tidy up" on the canvas menu.** `Layout.auto` has 29 tests and is reachable
  only as a side effect of minimise and trim. A machine drawn by hand cannot be
  arranged at all. One menu item; it should have existed already.
- **Multi-select** by box drag, **copy/paste** and **duplicate**. Cross-process
  paste via **`pyperclip`** when present, serialising the selection as JSON.
- **Recent files**, stored under **`platformdirs`**' user data directory, and a
  file *picker* rather than a bare text prompt — today Load requires typing a
  filename from memory.
- **Keyboard-only editing**: transitions currently need the mouse. Tab to move
  between states, Enter to start an edge.

**Exit criteria**
1. The 100-state frame benchmark passes at under 8ms, and its `xfail` mark
   is removed rather than loosened.
2. Layout falls back cleanly with no `dot` on PATH, and a test forces that path.
2. Event replay for tidy-up, box-select, copy/paste and duplicate.
3. A complete three-state machine can be built without touching the mouse.
4. Recent files survive a restart and drop entries whose file has gone.

---

### Phase 16 — Interop: `.jff`, DOT import, PNG · 8h

`IMPROVEMENT_PLAN.md`'s Phase 10 first half, plus the two smaller formats.

- **`src/fsa/export/jff.py`** — lossless JFLAP `.jff` import and export via
  `xml.etree` (stdlib; the FA schema is genuinely simple). **Budget the hidden
  dependency honestly:** "round-trips through real JFLAP" needs a fixture corpus
  *and* a JRE plus JFLAP 7.1 under its non-commercial licence. If fixtures
  cannot be got, scope down to "our own export re-imports identically" and *say
  so in the README* rather than overclaiming.
- **DOT import** via `pydot`, so a machine written for Graphviz can be opened.
- **PNG export** of the canvas — pygame already saves surfaces; this is the
  format people actually paste into an assignment.

**Exit criteria**
1. `from_jff(to_jff(d)) == d` for hypothesis-generated documents.
2. ≥5 real JFLAP fixtures import with hand-verified languages — or the README
   states plainly that this was not achieved and why.
3. PNG export writes a file whose pixel dimensions match the requested region.

---

### Phase 17 — Transducers, grammars, and the pumping lemma · 12h

The rest of a regular-languages syllabus.

- **Moore and Mealy machines** — output on the state and on the transition.
  Shares the layout, the renderer and the run animation; the tape strip grows an
  output row. Moore ↔ Mealy conversion is a standard exercise and a natural
  `fsa convert`.
- **Regular grammars ↔ finite automata** — right-linear grammar to NFA and back.
  A text panel for the productions beside the diagram.
- **A pumping-lemma assistant** — the tool plays the adversary: the student
  picks *p*, the tool picks a string, the student decomposes it, the tool finds
  the *i* that breaks it. This is the one part of a regular-languages course
  with no good interactive tool anywhere, and it is a genuine differentiator.

**Exit criteria**
1. Property: a Mealy machine and its Moore conversion produce the same output
   for every input word (modulo the leading symbol Moore emits).
2. Property: `equivalent(nfa, from_grammar(to_grammar(nfa)))`.
3. The pumping assistant is correct on a known non-regular language (`aⁿbⁿ`)
   and correctly *fails to break* a regular one.

---

### Phase 18 — Release · 7h

Only worth doing once the scope decision says someone other than you will run
this.

- `CHANGELOG.md`, `CONTRIBUTING.md`, a documented release process, `v0.1.0`.
- **Check the PyPI name is free before promising it.**
- CI matrix across supported Pythons; a coverage gate.
- Docs: the README is generated and test-locked already; a small `mkdocs` site
  if the CLI surface justifies it.

**Exit criteria**
1. `pip install` from a built wheel in a clean venv on a second machine.
2. Every README code block executed in CI.
3. `fsa --version` matches the tag.

---

### Phase 19 — CFG and pushdown automata *(optional)* · 20h+

Explicitly beyond "finite automata", and a different data model: a stack, and
nondeterminism that cannot be determinized away. Worth it only if the scope
decision says this is a course tool that has to cover the whole course. Plan it
properly rather than bolting it on — the `DFA`/`NFA` split above is the right
precedent.

---

## What this deliberately does not include

- **Turing machines.** A different tool.
- **A web port.** `IMPROVEMENT_PLAN.md` is explicit: do not build web work
  before a real number from real users exists.
- **Replacing the engine with `automata-lib`.** It would delete the reason the
  project exists. It is an oracle, not an implementation.
