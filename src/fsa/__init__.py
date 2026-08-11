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
"""

from fsa import equivalence, export, geometry, language, ops, product
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
    UnknownStateError,
)
from fsa.layout import Layout, Point
from fsa.minimize import MarkingTable, marking_table, minimize
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
    # errors
    "AutomatonError",
    "IllegalSymbolError",
    "UnknownStateError",
    "DuplicateStateError",
    "IncompleteAutomatonError",
]
