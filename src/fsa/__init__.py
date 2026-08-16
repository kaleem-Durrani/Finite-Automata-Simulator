"""A finite automata engine.

Pure Python, standard library only, and no display dependency: nothing here
imports pygame, and CI enforces that. The GUI is one front end among several.

    >>> from fsa import DFA, run
    >>> a = (DFA()
    ...      .with_states(["q0", "q1"])
    ...      .with_transition("q0", "a", "q1")
    ...      .with_accept("q1"))
    >>> run(a, "a").explain()
    "'a' was accepted in q1"
    >>> run(a, "aa").explain()
    "'aa' was rejected: no transition from q1 on 'a' at position 1 -- this automaton is incomplete"

There are two machine types and one document type. :class:`DFA` promises
determinism in its type and :class:`NFA` does not; a :class:`Document` holds an
NFA always, and hands out the deterministic reading through
:meth:`Document.as_dfa` for the algorithms that need one. :data:`AnyAutomaton`
names the pair, for the few functions that read only what both expose.
"""

from fsa import (
    equivalence,
    exercise,
    export,
    geometry,
    language,
    nfa,
    ops,
    product,
    regex,
    subset,
)
from fsa.analysis import (
    Defect,
    co_reachable,
    dead_states,
    defects,
    is_complete,
    is_trap,
    missing_transitions,
    reachable,
    unreachable_states,
)
from fsa.automaton import DFA
from fsa.document import Document
from fsa.equivalence import counterexample, equivalent
from fsa.errors import (
    AutomatonError,
    DuplicateStateError,
    IllegalSymbolError,
    IncompleteAutomatonError,
    NondeterministicError,
    UnknownStateError,
)

# `exercise.check`, `exercise.loads` and `exercise.dumps` are deliberately NOT
# re-exported: `fsa.loads` and `fsa.dumps` already mean documents, and a second
# pair under those names would make which format was meant a question about
# import order. The three values below have names of their own.
from fsa.exercise import Exercise, ExerciseFormatError, Result
from fsa.layout import AnyAutomaton, Layout, Point
from fsa.minimize import MarkingTable, marking_table, minimize

# `nfa.accepts` and `nfa.run` are deliberately NOT re-exported: the names
# already belong to the DFA simulator, and a nondeterministic run returns a
# different value. Reach them through the module, so which one you meant is
# written at the call site.
from fsa.nfa import EPSILON, NFA
from fsa.ops import complete, trim
from fsa.product import (
    complement,
    difference,
    intersection,
    symmetric_difference,
    union,
)
from fsa.serialize import (
    DocumentFormatError,
    dumps,
    load_or_error,
    loads,
    save_or_error,
)
from fsa.simulate import Run, Step, Verdict, accepts, run
from fsa.subset import determinize
from fsa.symbols import StateId, Symbol, is_legal_symbol, normalize_alphabet

__all__ = [
    # model
    "DFA",
    "Document",
    "Layout",
    "Point",
    "geometry",
    "language",
    "export",
    "ops",
    "product",
    "equivalence",
    "nfa",
    "regex",
    "subset",
    "NFA",
    "AnyAutomaton",
    "EPSILON",
    "determinize",
    "NondeterministicError",
    "StateId",
    "Symbol",
    "is_legal_symbol",
    "normalize_alphabet",
    # simulation
    "run",
    "accepts",
    "Run",
    "Step",
    "Verdict",
    # analysis
    "defects",
    "Defect",
    "missing_transitions",
    "is_complete",
    "reachable",
    "co_reachable",
    "dead_states",
    "unreachable_states",
    "is_trap",
    # operations
    "complete",
    "trim",
    "minimize",
    "marking_table",
    "MarkingTable",
    "equivalent",
    "counterexample",
    "union",
    "intersection",
    "difference",
    "symmetric_difference",
    "complement",
    # documents
    "dumps",
    "loads",
    "load_or_error",
    "save_or_error",
    "DocumentFormatError",
    # exercises
    "exercise",
    "Exercise",
    "Result",
    "ExerciseFormatError",
    # errors
    "AutomatonError",
    "IllegalSymbolError",
    "UnknownStateError",
    "DuplicateStateError",
    "IncompleteAutomatonError",
]
