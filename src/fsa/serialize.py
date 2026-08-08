"""Reading and writing documents.

The format is a versioned envelope. Output is deterministic -- every collection
is sorted and the transition function is a list of ``[state, symbol, target]``
triples -- so two saves of the same automaton are byte-identical and a saved
file can be diffed like source.

There is no migration framework. Save had never once round-tripped before this
was written, so no file in the old format exists anywhere except the one example
checked into the repository. :func:`read_legacy` handles that one, is used once,
and can be deleted the moment the example is re-saved.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from fsa.automaton import DFA
from fsa.document import Document
from fsa.errors import AutomatonError
from fsa.layout import Layout
from fsa.symbols import StateId

VERSION = 2


class DocumentFormatError(AutomatonError):
    """The bytes are not a document this version understands."""


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------

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
        "layout": {
            # Layout normalises coordinates to storage precision on the way
            # in, so there is nothing left to round here.
            "positions": {
                state: [point[0], point[1]]
                for state, point in sorted(document.layout.positions.items())
            },
            "arcs": [
                [source, target, offset]
                for (source, target), offset in sorted(document.layout.arc_offsets.items())
            ],
        },
        "next_id": document.next_id,
    }


def dumps(document: Document) -> str:
    """Serialise a document to text."""
    return json.dumps(to_dict(document), indent=2) + "\n"


def save(document: Document, path: str) -> None:
    """Write a document to a file."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(dumps(document))


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------

def from_dict(data: Dict[str, Any]) -> Document:
    """Rebuild a document from a parsed envelope, of any known version."""
    if not isinstance(data, dict):
        raise DocumentFormatError("not an object")

    version = data.get("version")
    if version is None:
        return read_legacy(data)
    if version != VERSION:
        raise DocumentFormatError(
            f"unsupported version {version!r}; this build reads {VERSION}")

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

    layout_body = data.get("layout") or {}
    positions = {
        str(state): (float(point[0]), float(point[1]))
        for state, point in dict(layout_body.get("positions", {})).items()
        if state in states
    }
    arcs = {
        (str(entry[0]), str(entry[1])): float(entry[2])
        for entry in layout_body.get("arcs", [])
        if len(entry) == 3 and entry[0] in states and entry[1] in states
    }

    document = Document.of(automaton, Layout(positions, arcs))
    declared = data.get("next_id")
    if isinstance(declared, int):
        document = Document(document.automaton, document.layout,
                            max(document.next_id, declared))
    return document


def loads(text: str) -> Document:
    """Parse a document from text."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DocumentFormatError(f"not valid JSON (line {exc.lineno})") from exc
    return from_dict(data)


def load(path: str) -> Document:
    """Read a document from a file."""
    with open(path, "r", encoding="utf-8") as handle:
        return loads(handle.read())


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


def save_or_error(document: Document, path: str) -> Tuple[bool, str]:
    """Write a file, returning success and a reason on failure."""
    try:
        save(document, path)
        return True, ""
    except OSError as exc:
        return False, describe_error(exc)


__all__: List[str] = [
    "VERSION", "DocumentFormatError",
    "to_dict", "dumps", "save", "save_or_error",
    "from_dict", "loads", "load", "load_or_error", "read_legacy",
]
