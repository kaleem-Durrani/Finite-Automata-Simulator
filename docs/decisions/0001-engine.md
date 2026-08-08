# 0001 — Write the automata engine rather than wrapping automata-lib

**Status:** accepted
**Date:** 2026-08-08

## Context

The simulator needs a tested, display-free automata engine. `automata-lib`
(MIT, JOSS-published, ~405 stars, ~56k downloads/month) already exists and is
the obvious thing to reach for. Writing another automata library is the most
easily wasted effort available on this project, so the burden of proof is on
building rather than on adopting.

Evaluated `automata-lib` 9.2.0 against five criteria. Own the engine only if at
least two are unacceptable.

## Findings

**1. Partial transition functions — acceptable.**
Rejected by default (`MissingSymbolError`), accepted with `allow_partial=True`.
A half-drawn automaton in an editor is almost always partial, so the flag would
be permanently on, but it works.

**2. "No initial state yet" — unacceptable.**
```
initial_state=None -> InvalidStateError: None is not a valid initial state
initial_state=''   -> InvalidStateError:  is not a valid initial state
```
An editor must be able to represent an automaton whose start state has not been
chosen. It is the state the document is in between the user's first click and
their decision, and it is what the document falls back to when the start state
is deleted. The library treats it as a construction error, so this legal
editing state cannot be modelled at all.

**3. State identity across operations — unacceptable.**
```
before minify: ['q0', 'q1', 'q2']
after  minify: ['0', '1', '2']
```
Minimisation renames every state. The editor keys canvas positions off state
ids, so the entire layout is lost the moment the user minimises — which is the
single most important algorithm the tool will ship. Recovering the mapping means
reimplementing the correspondence the library discarded.

**4. Rejection reasons — unacceptable, and this one is the product.**
`accepts_input` returns a bare `bool`. `read_input_stepwise` raises
`RejectionException: the DFA stopped on a non-final state (None)`. There is no
verdict, no offending symbol, no position.

Telling a student *why* their string was rejected — no transition defined vs.
symbol outside the alphabet vs. halted in a non-accepting state — is the one
thing this tool is being built to do that the alternatives do not. It cannot be
built on a boolean.

**5. Dependencies — a mark against.**
`networkx`, `frozendict`, `typing-extensions`, `cached-method`. The engine is
meant to be stdlib-only so it stays trivially installable, importable without a
display, and cheap to guard in CI.

## Decision

Write the engine. Three of five criteria are unacceptable, and criterion 4 is
the project's reason for existing.

Scope of what we own: the immutable model, simulation with verdicts, and
structural analysis. These are a few hundred lines of textbook material, and
they are the parts where our requirements and the library's design genuinely
diverge.

## What we are *not* claiming

`automata-lib` is good software and is better than what we will write at the
algorithm layer. Two consequences:

- When Phase 9 implements minimisation, product constructions and equivalence,
  property-test our results against `automata-lib` as an independent oracle.
  Agreeing with a mature published library on 300 random automata is far
  stronger evidence than our own unit tests.
- If the engine ever needs an algorithm that is hard to get right and not
  pedagogically interesting to implement, take theirs.

Recorded so this is a considered boundary rather than reflex.

## Consequences

- `src/fsa/` has zero third-party imports, enforced in CI.
- State ids are stable by construction; every operation returning a new
  automaton preserves the identity of states it did not change.
- `initial: StateId | None` is a legal value throughout.
- `Run` carries a five-way verdict, the offending symbol, and the position it
  stopped at, and can explain itself in one sentence.
- We are responsible for our own correctness, which is what the conformance
  suite written from the theory in Phase 1 is for.
