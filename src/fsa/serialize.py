"""Reading and writing documents.

The format is a versioned envelope. Output is deterministic -- every collection
is sorted and the transition function is a list of ``[state, symbol, target]``
triples -- so two saves of the same automaton are byte-identical and a saved
file can be diffed like source.

There is no migration framework. Save had never once round-tripped before this
was written, so no file in the old format exists anywhere except the one example
checked into the repository. :func:`read_legacy` handles that one, is used once,
and can be deleted the moment the example is re-saved.

**Version 2 holds a DFA, version 3 an NFA.** The version records which value
type wrote the file, not what the machine in it happens to look like: a
deterministic NFA is still written as 3. That keeps writing a function of the
type rather than of the contents, so no edit to a machine can silently change
the shape of the file it lives in, and version 2 goes on emitting exactly the
bytes it emits today -- which the checked-in example, a byte-for-byte
round-trip test and the generated README table all depend on. Reading is the
tolerant direction: :func:`loads` opens a version 3 file whenever the machine
in it is deterministic, and :func:`loads_nfa` opens every version, because
every DFA is an NFA.

An NFA differs from a DFA in exactly two ways a file has to carry -- a move may
have several targets, and a move may read nothing -- and both are written into
the same triple list version 2 uses, because this format is advertised as
hand-editable and one rule is easier to hand-edit than two:

* **Several targets are several entries.** ``["q0", "a", "q1"]`` and
  ``["q0", "a", "q2"]`` both appear, each its own triple. Grouping them into
  ``["q0", "a", ["q1", "q2"]]`` would be shorter and worse: adding a branch
  would then edit an entry instead of adding one, so a diff would stop reading
  like the edit that produced it, and version 2's one rule -- an entry is an
  arrow -- would become two. A repeated ``(source, symbol)`` *is* the
  nondeterminism, which is the thing the file is trying to show.
* **Epsilon is JSON ``null``.** ``["q0", null, "q1"]`` is a move that reads
  nothing. Not ``""``, ``"e"`` or ``"ε"``: ``ε`` is one printable non-space
  character and therefore a perfectly legal alphabet symbol here, so any of
  those spellings would give one line two meanings and cost the alphabet a
  letter. ``null`` maps exactly onto :data:`fsa.nfa.EPSILON` and can collide
  with nothing. Epsilon moves sort first within a state, matching
  :meth:`fsa.nfa.NFA.sorted_transitions` and how they read on screen: the move
  that costs nothing comes first.

:class:`~fsa.document.Document` still holds a ``DFA`` -- widening it is Phase
12b -- so version 3 is written and read as a bare ``(NFA, Layout, next_id)``
triple by :func:`dumps_nfa` and :func:`loads_nfa` rather than as a document.
"""

import json
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from fsa.automaton import DFA
from fsa.document import Document
from fsa.errors import AutomatonError, NondeterministicError
from fsa.layout import Layout
from fsa.nfa import NFA, from_dfa, to_dfa
from fsa.symbols import StateId, Symbol

#: The version a document holding a DFA is written as. Frozen, and not merely
#: by convention: the example file, the README table generated from it and a
#: byte-for-byte test all pin the exact output of version 2.
VERSION = 2

#: The version a document holding an NFA is written as.
NFA_VERSION = 3

#: Every version this build understands, in the order the error message lists
#: them. Anything else is refused rather than guessed at -- a file from a newer
#: build may well parse, and the machine it produced would not be the one that
#: was saved.
READABLE_VERSIONS = (VERSION, NFA_VERSION)


class DocumentFormatError(AutomatonError):
    """The bytes are not a document this version understands."""


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------

def _layout_to_dict(layout: Layout) -> Dict[str, Any]:
    """The drawing half of an envelope.

    Identical in both versions, and shared rather than copied: where a state is
    drawn has nothing to do with how many targets its moves have.
    """
    return {
        # Layout normalises coordinates to storage precision on the way
        # in, so there is nothing left to round here.
        "positions": {
            state: [point[0], point[1]]
            for state, point in sorted(layout.positions.items())
        },
        "arcs": [
            [source, target, offset]
            for (source, target), offset in sorted(layout.arc_offsets.items())
        ],
    }


def to_dict(document: Document) -> Dict[str, Any]:
    """A plain, sorted, JSON-ready snapshot."""
    automaton = document.automaton
    return {
        "version": VERSION,
        "automaton": {
            "states": sorted(automaton.states),
            "alphabet": sorted(automaton.alphabet),
            "initial": automaton.initial,
            "accept": sorted(automaton.accept),
            "transitions": [
                [source, symbol, target]
                for (source, symbol), target in sorted(automaton.transitions.items())
            ],
            "labels": {k: v for k, v in sorted(automaton.labels.items())},
        },
        "layout": _layout_to_dict(document.layout),
        "next_id": document.next_id,
    }


def to_nfa_dict(automaton: NFA, layout: Layout, next_id: int) -> Dict[str, Any]:
    """A version 3 envelope: the same keys as :func:`to_dict`, one version up.

    The triple list is :meth:`fsa.nfa.NFA.sorted_transitions` flattened, one
    line per arrow. The order has to come from there rather than from a
    ``sorted`` call here, and not for tidiness: the keys hold
    ``Optional[Symbol]`` and ``sorted`` raises on them, because ``None`` does
    not compare with ``str``. Its targets arrive sorted too, which is what stops
    the file's lines shuffling between processes (see docs/LESSONS.md).
    """
    return {
        "version": NFA_VERSION,
        "automaton": {
            "states": sorted(automaton.states),
            "alphabet": sorted(automaton.alphabet),
            "initial": automaton.initial,
            "accept": sorted(automaton.accept),
            "transitions": [
                [source, symbol, target]
                for source, symbol, targets in automaton.sorted_transitions()
                for target in targets
            ],
            "labels": {k: v for k, v in sorted(automaton.labels.items())},
        },
        "layout": _layout_to_dict(layout),
        "next_id": next_id,
    }


def dumps(document: Document) -> str:
    """Serialise a document to text."""
    return json.dumps(to_dict(document), indent=2) + "\n"


def dumps_nfa(automaton: NFA, layout: Layout, next_id: int) -> str:
    """Serialise a nondeterministic machine and its drawing to text.

    Takes the three pieces separately because there is no document type that
    holds an NFA yet; :func:`loads_nfa` gives the same three back.
    """
    return json.dumps(to_nfa_dict(automaton, layout, next_id), indent=2) + "\n"


def save(document: Document, path: str) -> None:
    """Write a document to a file."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(dumps(document))


def save_nfa(automaton: NFA, layout: Layout, next_id: int, path: str) -> None:
    """Write a nondeterministic machine and its drawing to a file."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(dumps_nfa(automaton, layout, next_id))


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------

def _layout_from_dict(data: Dict[str, Any], states: FrozenSet[StateId]) -> Layout:
    """The drawing half of an envelope, as a :class:`~fsa.layout.Layout`.

    Anything naming a state the file does not define is dropped rather than
    raising. A position for a state that no longer exists is junk left behind by
    an edit, not a corrupt file, and refusing to open the file over it would
    lose the machine as well.
    """
    body = data.get("layout") or {}
    positions = {
        str(state): (float(point[0]), float(point[1]))
        for state, point in dict(body.get("positions", {})).items()
        if state in states
    }
    arcs = {
        (str(entry[0]), str(entry[1])): float(entry[2])
        for entry in body.get("arcs", [])
        if len(entry) == 3 and entry[0] in states and entry[1] in states
    }
    return Layout(positions, arcs)


def _next_id_from(data: Dict[str, Any], states: FrozenSet[StateId]) -> int:
    """The id counter to start from: whatever the file declared, raised to
    something safe.

    A hand-written file will not have thought about an id counter, and a file
    that declares one may still name it below a state it contains -- either way
    the next state added would collide with an existing one. Both are fixed by
    the rule :meth:`fsa.document.Document.of` applies to a DFA: at least one
    past the highest ``qN`` in the file. The rule is written twice today,
    here and there, because a ``Document`` cannot hold an NFA yet; Phase 12b
    should leave one copy.
    """
    highest = -1
    for state in states:
        if state.startswith("q") and state[1:].isdigit():
            highest = max(highest, int(state[1:]))
    declared = data.get("next_id")
    return max(highest + 1, declared if isinstance(declared, int) else 0)


def from_dict(data: Dict[str, Any]) -> Document:
    """Rebuild a document from a parsed envelope, of any known version."""
    if not isinstance(data, dict):
        raise DocumentFormatError("not an object")

    version = data.get("version")
    if version is None:
        return read_legacy(data)
    if version == NFA_VERSION:
        return _document_from_nfa(*from_nfa_dict(data))
    if version != VERSION:
        raise DocumentFormatError(
            f"unsupported version {version!r}; this build reads "
            f"{' and '.join(str(known) for known in READABLE_VERSIONS)}")

    body = data.get("automaton")
    if not isinstance(body, dict):
        raise DocumentFormatError("missing 'automaton'")

    states = frozenset(str(s) for s in body.get("states", []))
    alphabet = frozenset(str(s) for s in body.get("alphabet", []))

    transitions: Dict[Tuple[StateId, str], StateId] = {}
    for entry in body.get("transitions", []):
        if len(entry) != 3:
            raise DocumentFormatError(f"malformed transition: {entry!r}")
        source, symbol, target = (str(part) for part in entry)
        if source in states and target in states and symbol in alphabet:
            transitions[(source, symbol)] = target

    initial = body.get("initial")
    automaton = DFA(
        states=states,
        alphabet=alphabet,
        transitions=transitions,
        initial=initial if initial in states else None,
        accept=frozenset(s for s in body.get("accept", []) if s in states),
        labels={k: str(v) for k, v in dict(body.get("labels", {})).items()
                if k in states},
    )

    document = Document.of(automaton, _layout_from_dict(data, states))
    declared = data.get("next_id")
    if isinstance(declared, int):
        document = Document(document.automaton, document.layout,
                            max(document.next_id, declared))
    return document


def from_nfa_dict(data: Dict[str, Any]) -> Tuple[NFA, Layout, int]:
    """Rebuild a machine, its drawing and its id counter from an envelope.

    Reads version 3, and also everything :func:`from_dict` reads, lifted with
    :func:`fsa.nfa.from_dfa`. Every DFA is an NFA, so refusing an older file
    here would only mean a caller had to know which kind of file it was about
    to open before it opened it.

    Unlike :func:`from_dict`, this does *not* place states that arrive without
    coordinates. It returns the file's three values as they were written, so
    that ``loads_nfa(dumps_nfa(x)) == x`` exactly rather than nearly; inventing
    a position would make that false for every machine whose layout is partial.
    Placement belongs to whatever holds the machine and its drawing together --
    ``Document.of`` for a DFA, and its Phase 12b counterpart for this.
    """
    if not isinstance(data, dict):
        raise DocumentFormatError("not an object")

    if data.get("version") != NFA_VERSION:
        document = from_dict(data)
        return from_dfa(document.automaton), document.layout, document.next_id

    body = data.get("automaton")
    if not isinstance(body, dict):
        raise DocumentFormatError("missing 'automaton'")

    states = frozenset(str(s) for s in body.get("states", []))
    alphabet = frozenset(str(s) for s in body.get("alphabet", []))

    collected: Dict[Tuple[StateId, Optional[Symbol]], Set[StateId]] = {}
    for entry in body.get("transitions", []):
        if len(entry) != 3:
            raise DocumentFormatError(f"malformed transition: {entry!r}")
        raw_source, raw_symbol, raw_target = entry
        source, target = str(raw_source), str(raw_target)

        symbol: Optional[Symbol] = None
        if raw_symbol is not None:
            symbol = str(raw_symbol)
            if symbol not in alphabet:
                continue
        # An epsilon move is deliberately *not* checked against the alphabet.
        # Epsilon was never in it, so that check would drop every epsilon move
        # in the file -- which is the whole of what version 3 exists to carry.

        if source in states and target in states:
            # setdefault, not assignment: a second line on the same
            # ``(source, symbol)`` is a second branch, not a correction of the
            # first. Overwriting here is exactly the bug that made
            # nondeterminism unsaveable.
            collected.setdefault((source, symbol), set()).add(target)

    initial = body.get("initial")
    automaton = NFA(
        states=states,
        alphabet=alphabet,
        transitions={key: frozenset(targets) for key, targets in collected.items()},
        initial=initial if initial in states else None,
        accept=frozenset(s for s in body.get("accept", []) if s in states),
        labels={k: str(v) for k, v in dict(body.get("labels", {})).items()
                if k in states},
    )
    return automaton, _layout_from_dict(data, states), _next_id_from(data, states)


def _document_from_nfa(automaton: NFA, layout: Layout, next_id: int) -> Document:
    """A version 3 file as a document, when it can be one.

    ``Document.automaton`` is a ``DFA`` until Phase 12b, and a version 3 file
    holding a machine that happens to be deterministic is an ordinary thing to
    write -- so it opens. Anything else is refused with the state and symbol
    that made it nondeterministic, which is the only honest answer available:
    determinising here would open a file and show a machine with different
    states from the one that was saved. :func:`fsa.nfa.to_dfa` refuses for the
    same reason, and its message is worth repeating rather than replacing.
    """
    try:
        deterministic = to_dfa(automaton)
    except NondeterministicError as exc:
        raise DocumentFormatError(
            f"this file holds a nondeterministic machine, and this build can "
            f"only open deterministic ones as a document ({exc})") from exc

    document = Document.of(deterministic, layout)
    return Document(document.automaton, document.layout,
                    max(document.next_id, next_id))


def _parse(text: str) -> Any:
    """Text to a parsed envelope, with a failure a person can act on."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise DocumentFormatError(f"not valid JSON (line {exc.lineno})") from exc


def loads(text: str) -> Document:
    """Parse a document from text."""
    return from_dict(_parse(text))


def loads_nfa(text: str) -> Tuple[NFA, Layout, int]:
    """Parse a machine, its drawing and its id counter from text."""
    return from_nfa_dict(_parse(text))


def load(path: str) -> Document:
    """Read a document from a file."""
    with open(path, "r", encoding="utf-8") as handle:
        return loads(handle.read())


def load_nfa(path: str) -> Tuple[NFA, Layout, int]:
    """Read a nondeterministic machine and its drawing from a file."""
    with open(path, "r", encoding="utf-8") as handle:
        return loads_nfa(handle.read())


# ----------------------------------------------------------------------
# The one old format
# ----------------------------------------------------------------------

def read_legacy(data: Dict[str, Any]) -> Document:
    """Read the pre-versioning format.

    Positions lived on each state and the transition function was nested. The
    ``dead_end_states`` list is deliberately dropped: it was a flag that made
    the simulator reject early without any transition saying so, and honouring
    it would reintroduce exactly the defect that removing it fixed. A state that
    was genuinely a trap still reads as one, because that is now derived from
    the edges.
    """
    raw_states = dict(data.get("states", {}))
    states = frozenset(str(s) for s in raw_states)
    alphabet = frozenset(str(s) for s in data.get("alphabet", []))

    transitions: Dict[Tuple[StateId, str], StateId] = {}
    for source, symbol_map in dict(data.get("transitions", {})).items():
        if source not in states:
            continue
        for symbol, target in dict(symbol_map).items():
            if target in states and symbol in alphabet:
                transitions[(str(source), str(symbol))] = str(target)

    initial = data.get("initial_state")
    automaton = DFA(
        states=states,
        alphabet=alphabet,
        transitions=transitions,
        initial=initial if initial in states else None,
        accept=frozenset(s for s in data.get("accept_states", []) if s in states),
    )

    positions = {}
    for state, body in raw_states.items():
        point = dict(body).get("position")
        if isinstance(point, (list, tuple)) and len(point) == 2:
            positions[str(state)] = (float(point[0]), float(point[1]))

    arcs = {}
    for key, offset in dict(data.get("arc_offsets", {})).items():
        source, _, target = str(key).partition("|")
        if source in states and target in states:
            arcs[(source, target)] = float(offset)

    document = Document.of(automaton, Layout(positions, arcs))
    declared = data.get("next_state_id")
    if isinstance(declared, int):
        document = Document(document.automaton, document.layout,
                            max(document.next_id, declared))
    return document


def describe_error(exc: Exception) -> str:
    """A short sentence about a failure, for showing to a user."""
    if isinstance(exc, DocumentFormatError):
        return str(exc)
    if isinstance(exc, OSError):
        return exc.strerror or str(exc)
    if isinstance(exc, AutomatonError):
        return f"malformed automaton ({exc})"
    return str(exc)


def load_or_error(path: str) -> Tuple[Optional[Document], str]:
    """Read a file, returning either the document or a reason.

    Failures are returned rather than printed. Nobody running a windowed
    application reads stdout.
    """
    try:
        return load(path), ""
    except (OSError, AutomatonError, ValueError, TypeError, KeyError) as exc:
        return None, describe_error(exc)


def load_nfa_or_error(path: str) -> Tuple[Optional[Tuple[NFA, Layout, int]], str]:
    """Read a file as a nondeterministic machine, returning it or a reason.

    The three values come back as one tuple so that failure has somewhere to
    live: ``None`` is the whole result being absent, which two of three
    ``Optional`` return values could not say.
    """
    try:
        return load_nfa(path), ""
    except (OSError, AutomatonError, ValueError, TypeError, KeyError) as exc:
        return None, describe_error(exc)


def save_or_error(document: Document, path: str) -> Tuple[bool, str]:
    """Write a file, returning success and a reason on failure."""
    try:
        save(document, path)
        return True, ""
    except OSError as exc:
        return False, describe_error(exc)


def save_nfa_or_error(automaton: NFA, layout: Layout, next_id: int,
                      path: str) -> Tuple[bool, str]:
    """Write a nondeterministic machine, returning success and a reason."""
    try:
        save_nfa(automaton, layout, next_id, path)
        return True, ""
    except OSError as exc:
        return False, describe_error(exc)


__all__: List[str] = [
    "VERSION", "NFA_VERSION", "READABLE_VERSIONS", "DocumentFormatError",
    "to_dict", "dumps", "save", "save_or_error",
    "from_dict", "loads", "load", "load_or_error", "read_legacy",
    "to_nfa_dict", "dumps_nfa", "save_nfa", "save_nfa_or_error",
    "from_nfa_dict", "loads_nfa", "load_nfa", "load_nfa_or_error",
]
