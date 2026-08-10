"""Undo/redo on the editor model.

The document is an immutable value, so history is a list of previous values.
These tests pin the properties that make that scheme trustworthy: a long random
edit session round-trips exactly, the depth cap drops the oldest entries and
nothing else, a drag is one entry no matter how many motion events fed it, and
restoring a document does not resurrect editor pointers that were dropped when
their state disappeared.

No pygame. The model is testable without a display.
"""

import random

import fsa
from editor import UNDO_DEPTH, EditorModel


def make_document(states: int = 3, symbols=("a", "b")) -> fsa.Document:
    """A row of states over an alphabet, built directly on the document so the
    editor under test starts with empty history."""
    document = fsa.Document()
    for symbol in symbols:
        document = document.add_symbol(symbol)
    for i in range(states):
        document, _ = document.add_state((float(i) * 120.0, 0.0))
    return document


def random_edit(editor: EditorModel, rng: random.Random) -> None:
    """One editor-level edit, chosen at random. May be a no-op."""
    states = sorted(editor.automaton.states)
    symbols = sorted(editor.automaton.alphabet)
    operations = ["add"]
    if states:
        operations += ["remove", "accept", "initial", "drag", "rename"]
        if symbols:
            operations += ["wire", "unwire", "trap"]
    operation = rng.choice(operations)
    if operation == "add":
        editor.add_state((rng.uniform(0.0, 900.0), rng.uniform(0.0, 900.0)))
    elif operation == "remove":
        editor.remove_state(rng.choice(states))
    elif operation == "accept":
        editor.toggle_accept(rng.choice(states))
    elif operation == "initial":
        editor.set_initial(rng.choice(states))
    elif operation == "drag":
        state = rng.choice(states)
        editor.begin_drag(state, editor.position_of(state))
        for _ in range(rng.randrange(1, 6)):
            editor.update_drag((rng.uniform(0.0, 900.0), rng.uniform(0.0, 900.0)))
        editor.end_drag()
    elif operation == "rename":
        editor.rename(rng.choice(states), rng.choice(["start", "loop", "end", ""]))
    elif operation == "wire":
        editor.add_transition(rng.choice(states), rng.choice(symbols),
                              rng.choice(states), arc=rng.choice([0.0, 40.0, -40.0]))
    elif operation == "unwire":
        editor.remove_transition(rng.choice(states), rng.choice(symbols))
    elif operation == "trap":
        editor.make_trap(rng.choice(states))


# ---------------------------------------------------------------------------
# The basics
# ---------------------------------------------------------------------------


def test_undo_with_no_history_returns_none():
    editor = EditorModel(make_document())
    assert not editor.can_undo
    assert not editor.can_redo
    assert editor.undo() is None
    assert editor.redo() is None


def test_undo_restores_the_previous_value():
    editor = EditorModel(make_document())
    before = editor.document
    state = editor.add_state((400.0, 400.0))
    assert editor.can_undo

    assert editor.undo() == f"add {state}"
    assert editor.document == before
    assert editor.dirty
    assert editor.can_redo


def test_plain_apply_defaults_to_edit():
    editor = EditorModel(make_document())
    editor.apply(editor.document.toggle_accept("q0"))
    assert editor.undo() == "edit"


def test_random_session_round_trips():
    """THE property: undo-all is the original document, redo-all is the final
    one, for an arbitrary mix of edits. Value equality, so this covers the
    automaton, the layout (positions and arcs), labels, and the id counter."""
    rng = random.Random(20260810)
    editor = EditorModel(make_document())
    original = editor.document
    for _ in range(40):
        random_edit(editor, rng)
    final = editor.document

    while editor.can_undo:
        assert editor.undo() is not None
    assert editor.document == original

    while editor.can_redo:
        assert editor.redo() is not None
    assert editor.document == final


def test_fresh_edit_after_undo_clears_redo():
    editor = EditorModel(make_document())
    editor.toggle_accept("q0")
    assert editor.undo() == "accept q0"
    assert editor.can_redo

    editor.add_transition("q0", "a", "q1")
    assert not editor.can_redo
    assert editor.redo() is None


def test_cap_drops_the_oldest_entries():
    editor = EditorModel(fsa.Document())
    snapshots = []  # snapshots[i] is the document after i edits
    for i in range(220):
        snapshots.append(editor.document)
        editor.add_state((float(i % 20) * 100.0, float(i // 20) * 100.0))

    undone = 0
    while editor.undo() is not None:
        undone += 1
    assert undone == UNDO_DEPTH == 200
    assert editor.document == snapshots[20]


# ---------------------------------------------------------------------------
# Edits that must not multiply
# ---------------------------------------------------------------------------


def test_drag_records_exactly_one_entry():
    editor = EditorModel(make_document())
    origin = editor.position_of("q0")
    editor.begin_drag("q0", origin)
    for step in range(1, 30):
        editor.update_drag((origin[0] + step * 4.0, origin[1] + step * 2.0))
    assert editor.end_drag()

    assert editor.undo() == "move q0"
    assert editor.position_of("q0") == origin
    assert not editor.can_undo


def test_drag_back_to_start_records_nothing():
    editor = EditorModel(make_document())
    origin = editor.position_of("q0")
    editor.begin_drag("q0", origin)
    editor.update_drag((origin[0] + 80.0, origin[1] + 80.0))
    editor.update_drag(origin)
    assert not editor.end_drag()
    assert not editor.can_undo


def test_noop_apply_records_nothing():
    editor = EditorModel(make_document())
    same = fsa.Document(editor.automaton, editor.layout, editor.document.next_id)
    assert same is not editor.document and same == editor.document

    editor.apply(same, action="pointless")
    assert not editor.can_undo
    assert not editor.dirty


# ---------------------------------------------------------------------------
# History and the rest of the editor's state
# ---------------------------------------------------------------------------


def test_replace_clears_both_stacks():
    """History must not cross files: undoing past a load would silently turn
    'the previous document' into 'a different file's document'."""
    editor = EditorModel(make_document())
    editor.toggle_accept("q0")
    editor.toggle_accept("q1")
    editor.undo()
    assert editor.can_undo and editor.can_redo

    editor.replace(make_document(states=1), "elsewhere.json")
    assert not editor.can_undo
    assert not editor.can_redo
    assert editor.undo() is None
    assert editor.redo() is None


def test_undo_of_delete_leaves_selection_cleared():
    """Undo brings the deleted state back; it must not bring back the pointer
    that forget_missing dropped when the state died."""
    editor = EditorModel(make_document())
    editor.select("q1")
    assert editor.remove_state("q1")
    assert editor.selection is None

    assert editor.undo() == "delete q1"
    assert "q1" in editor.automaton.states
    assert editor.selection is None
    # The restored state is fully live: it has a position and is hit-testable.
    assert editor.state_at(editor.position_of("q1"), 30.0) == "q1"


# ---------------------------------------------------------------------------
# Rename
# ---------------------------------------------------------------------------


def test_rename_sets_label_and_is_undoable():
    editor = EditorModel(make_document())
    assert editor.rename("q0", "start")
    assert editor.automaton.label_of("q0") == "start"
    assert editor.dirty

    assert editor.undo() == "rename q0"
    assert editor.automaton.label_of("q0") == "q0"
    assert editor.redo() == "rename q0"
    assert editor.automaton.label_of("q0") == "start"


def test_rename_blank_resets_to_the_id():
    editor = EditorModel(make_document())
    editor.rename("q0", "start")
    assert editor.rename("q0", "   ")
    assert editor.automaton.label_of("q0") == "q0"
    # Cleared, not overwritten with the id. label_of cannot tell the two apart
    # but equality can, and equality is what decides whether this was an edit.
    assert dict(editor.automaton.labels) == {}


def test_clearing_an_absent_label_is_not_an_edit():
    """The blank-rename path must reach the same value the document already
    has, or a rename that changes nothing still dirties the file."""
    editor = EditorModel(make_document())
    before = editor.document

    assert editor.rename("q0", "")
    assert editor.document == before
    assert not editor.dirty
    assert not editor.can_undo


def test_renaming_to_the_id_itself_is_not_an_edit():
    editor = EditorModel(make_document())
    before = editor.document

    assert editor.rename("q0", "q0")
    assert editor.document == before
    assert not editor.can_undo


def test_a_label_is_stored_stripped():
    """So it cannot differ from the name reported back to the user by padding
    that neither of them can see."""
    editor = EditorModel(make_document())
    editor.rename("q0", "  start  ")
    assert editor.automaton.label_of("q0") == "start"


def test_rename_unknown_state_is_refused():
    editor = EditorModel(make_document())
    assert not editor.rename("q9", "ghost")
    assert not editor.can_undo
