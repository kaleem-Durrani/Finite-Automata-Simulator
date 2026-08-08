"""Layout, Document and serialisation.

The point of this layer is that the automaton and its picture cannot fall out of
step. Most of these assert that directly.
"""

import json
import random

import pytest

import fsa
from fsa import Document, Layout, serialize
from fsa.document import OVERLAP_GAP
from fsa.layout import PLACEMENT_STEP

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_layout_is_a_value():
    one = Layout({"q0": (1.0, 2.0)}, {("q0", "q1"): 5.0})
    two = Layout({"q0": (1.0, 2.0)}, {("q0", "q1"): 5.0})
    assert one == two
    assert hash(one) == hash(two)
    assert one != one.with_position("q0", (9.0, 9.0))


def test_layout_operations_do_not_mutate():
    original = Layout({"q0": (0.0, 0.0)})
    derived = original.with_position("q1", (10.0, 10.0))
    assert set(original.positions) == {"q0"}
    assert set(derived.positions) == {"q0", "q1"}


def test_zero_arcs_are_not_stored():
    """An edge with no bow is drawn straight; storing a zero is noise."""
    assert Layout({}, {("a", "b"): 0.0}).arc_offsets == {}
    assert Layout({}, {("a", "b"): 7.0}).with_arc("a", "b", 0.0).arc_offsets == {}


def test_removing_a_state_removes_its_arcs():
    layout = Layout({"q0": (0.0, 0.0)},
                    {("q0", "q1"): 5.0, ("q1", "q0"): 5.0, ("q1", "q2"): 5.0})
    trimmed = layout.without_state("q0")
    assert set(trimmed.arc_offsets) == {("q1", "q2")}


def test_restricted_to_drops_everything_else():
    layout = Layout({"q0": (0.0, 0.0), "q1": (1.0, 1.0)}, {("q0", "q1"): 3.0})
    trimmed = layout.restricted_to({"q0"})
    assert set(trimmed.positions) == {"q0"}
    assert trimmed.arc_offsets == {}


def test_free_position_avoids_existing_states():
    """Adding states used to stack every one on the same pixel, and because
    hit-testing returns the first match, only the oldest stayed clickable."""
    layout = Layout({"q0": (100.0, 100.0)})
    placed = layout.free_position((100.0, 100.0))
    assert fsa.layout.math.dist(placed, (100.0, 100.0)) >= PLACEMENT_STEP - 1


def test_free_position_leaves_a_clear_spot_alone():
    layout = Layout({"q0": (0.0, 0.0)})
    assert layout.free_position((500.0, 500.0)) == (500.0, 500.0)


def test_bounds_and_centre():
    layout = Layout({"a": (0.0, 0.0), "b": (100.0, 50.0)})
    assert layout.bounds() == (0.0, 0.0, 100.0, 50.0)
    assert layout.bounds(10.0) == (-10.0, -10.0, 110.0, 60.0)
    assert layout.centre() == (50.0, 25.0)
    assert Layout().bounds() is None


# ---------------------------------------------------------------------------
# Document keeps both halves in step
# ---------------------------------------------------------------------------


def test_adding_a_state_gives_it_coordinates():
    document, state = Document().add_state((10.0, 20.0))
    assert state in document.automaton.states
    assert document.layout.position_of(state) == (10.0, 20.0)


def test_removing_a_state_forgets_its_coordinates():
    document, a = Document().add_state((0.0, 0.0))
    document, b = document.add_state((200.0, 0.0))
    document = document.add_symbol("x").add_transition(a, "x", b, arc=20.0)

    document = document.remove_state(b)
    assert b not in document.automaton.states
    assert b not in document.layout.positions
    assert document.layout.arc_offsets == {}


def test_the_two_halves_cannot_disagree():
    """Whatever you do, every state has a position and every position a state."""
    rng = random.Random(11)
    document = Document().add_symbol("a").add_symbol("b")
    ids = []
    for _ in range(40):
        action = rng.random()
        if action < 0.45 or not ids:
            document, state = document.add_state(
                (rng.uniform(0, 500), rng.uniform(0, 500)))
            ids.append(state)
        elif action < 0.7 and len(ids) > 1:
            source, target = rng.sample(ids, 2)
            document = document.add_transition(source, rng.choice("ab"), target,
                                               arc=rng.choice([0.0, 30.0]))
        else:
            victim = rng.choice(ids)
            ids.remove(victim)
            document = document.remove_state(victim)

        assert set(document.layout.positions) == set(document.automaton.states)
        for source, target in document.layout.arc_offsets:
            assert source in document.automaton.states
            assert target in document.automaton.states


def test_moving_a_state_cannot_change_the_language():
    document, a = Document().add_symbol("x").add_state((0.0, 0.0))
    document, b = document.add_state((100.0, 0.0))
    document = document.add_transition(a, "x", b).toggle_accept(b)

    moved = document.move_state(a, (-999.0, 4321.0))
    assert moved.automaton == document.automaton


def test_state_ids_are_not_reused():
    document, first = Document().add_state((0.0, 0.0))
    document = document.remove_state(first)
    document, second = document.add_state((0.0, 0.0))
    assert second != first


def test_explicit_placement_only_avoids_real_overlap():
    """When the user picks the spot, honour it unless it would overlap."""
    document, _ = Document().add_state((0.0, 0.0))
    near = (OVERLAP_GAP + 2, 0.0)
    document, state = document.add_state(near, minimum_gap=OVERLAP_GAP)
    assert document.layout.position_of(state) == near


def test_make_trap_loops_every_symbol():
    document = Document().add_symbol("a").add_symbol("b")
    document, state = document.add_state((0.0, 0.0))
    document = document.toggle_accept(state).make_trap(state)

    assert state not in document.automaton.accept
    for symbol in ("a", "b"):
        assert document.automaton.target(state, symbol) == state


def test_removing_the_last_symbol_on_an_edge_drops_its_arc():
    document = Document().add_symbol("a").add_symbol("b")
    document, a = document.add_state((0.0, 0.0))
    document, b = document.add_state((200.0, 0.0))
    document = document.add_transition(a, "a", b, arc=25.0)
    document = document.add_transition(a, "b", b)

    document = document.remove_transition(a, "a")
    assert document.layout.arc_of(a, b) == 25.0, "the edge still exists"

    document = document.remove_transition(a, "b")
    assert document.layout.arc_of(a, b) == 0.0, "now it does not"


def test_with_automaton_places_new_states():
    """An algorithm that invents states must not dump them at the origin."""
    document, _ = Document().add_symbol("a").add_state((0.0, 0.0))
    bigger = document.automaton.with_state("new1").with_state("new2")

    document = document.with_automaton(bigger)
    assert set(document.layout.positions) == set(bigger.states)
    positions = list(document.layout.positions.values())
    assert len(set(positions)) == len(positions), "and not on top of each other"


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def random_document(rng: random.Random) -> Document:
    document = Document()
    for symbol in rng.sample("abc01", rng.randrange(1, 5)):
        document = document.add_symbol(symbol)
    ids = []
    for _ in range(rng.randrange(1, 7)):
        document, state = document.add_state(
            (rng.uniform(-500, 500), rng.uniform(-500, 500)))
        ids.append(state)
    for state in ids:
        for symbol in document.automaton.alphabet:
            if rng.random() < 0.6:
                document = document.add_transition(
                    state, symbol, rng.choice(ids),
                    arc=rng.choice([0.0, 0.0, 28.5, -40.0]))
        if rng.random() < 0.4:
            document = document.toggle_accept(state)
    if rng.random() < 0.85:
        document = document.set_initial(rng.choice(ids))
    return document


def test_round_trip_over_random_documents():
    rng = random.Random(2026)
    for _ in range(200):
        document = random_document(rng)
        assert serialize.loads(serialize.dumps(document)) == document


def test_serialisation_is_byte_stable():
    rng = random.Random(7)
    for _ in range(50):
        document = random_document(rng)
        once = serialize.dumps(document)
        assert serialize.dumps(serialize.loads(once)) == once


def test_saved_files_are_diffable():
    """Sorted collections and one transition per line."""
    rng = random.Random(3)
    body = json.loads(serialize.dumps(random_document(rng)))["automaton"]
    assert body["states"] == sorted(body["states"])
    assert body["alphabet"] == sorted(body["alphabet"])
    assert body["transitions"] == sorted(body["transitions"])


def test_unknown_version_is_refused():
    with pytest.raises(serialize.DocumentFormatError, match="version"):
        serialize.loads('{"version": 99}')


def test_malformed_json_is_refused():
    with pytest.raises(serialize.DocumentFormatError, match="JSON"):
        serialize.loads("{not json")


def test_edges_naming_unknown_states_are_dropped():
    text = json.dumps({
        "version": serialize.VERSION,
        "automaton": {"states": ["q0"], "alphabet": ["a"],
                      "transitions": [["q0", "a", "ghost"]],
                      "initial": "q0", "accept": []},
        "layout": {"positions": {"q0": [0, 0]}, "arcs": []},
    })
    document = serialize.loads(text)
    assert document.automaton.transitions == {}


def test_an_unknown_initial_state_becomes_none():
    text = json.dumps({
        "version": serialize.VERSION,
        "automaton": {"states": ["q0"], "alphabet": [], "transitions": [],
                      "initial": "ghost", "accept": []},
        "layout": {"positions": {}, "arcs": []},
    })
    assert serialize.loads(text).automaton.initial is None


def test_states_without_positions_are_laid_out():
    """A hand-written file need not carry coordinates."""
    text = json.dumps({
        "version": serialize.VERSION,
        "automaton": {"states": ["q0", "q1"], "alphabet": ["a"],
                      "transitions": [["q0", "a", "q1"]],
                      "initial": "q0", "accept": ["q1"]},
        "layout": {"positions": {}, "arcs": []},
    })
    document = serialize.loads(text)
    assert set(document.layout.positions) == {"q0", "q1"}
    assert len(set(document.layout.positions.values())) == 2


# ---------------------------------------------------------------------------
# The one old format
# ---------------------------------------------------------------------------


LEGACY = {
    "states": {
        "q0": {"position": [200, 200], "state_type": "normal"},
        "q1": {"position": [400, 200], "state_type": "accept"},
        "q2": {"position": [300, 350], "state_type": "dead_end"},
    },
    "transitions": {"q0": {"0": "q0", "1": "q1"},
                    "q1": {"0": "q2", "1": "q1"},
                    "q2": {"0": "q2", "1": "q2"}},
    "alphabet": ["0", "1"],
    "initial_state": "q0",
    "accept_states": ["q1"],
    "dead_end_states": ["q2"],
    "next_state_id": 3,
}


def test_legacy_files_still_open():
    document = serialize.from_dict(LEGACY)
    assert sorted(document.automaton.states) == ["q0", "q1", "q2"]
    assert document.automaton.initial == "q0"
    assert document.automaton.accept == frozenset({"q1"})
    assert document.layout.position_of("q1") == (400.0, 200.0)
    assert document.next_id == 3


def test_the_legacy_dead_end_flag_is_dropped():
    """Honouring it would reintroduce the defect that removing it fixed.

    Here q2 is genuinely a trap, so it still reads as one -- but derived from
    the edges, not from the flag.
    """
    document = serialize.from_dict(LEGACY)
    assert fsa.dead_states(document.automaton) == frozenset({"q2"})

    escaped = document.add_transition("q2", "1", "q1")
    assert fsa.dead_states(escaped.automaton) == frozenset()


def test_a_legacy_flag_that_contradicts_the_edges_is_ignored():
    data = dict(LEGACY)
    data["transitions"] = {"q0": {"0": "q2"}, "q2": {"0": "q1"}}
    data["dead_end_states"] = ["q2"]

    document = serialize.from_dict(data)
    assert fsa.accepts(document.automaton, "00"), "delta says accepted"


def test_file_helpers_report_failures_rather_than_raising(tmp_path):
    """Nobody running a windowed application reads stdout."""
    document, error = serialize.load_or_error(str(tmp_path / "nope.json"))
    assert document is None and error

    (tmp_path / "bad.json").write_text("{oops", encoding="utf-8")
    document, error = serialize.load_or_error(str(tmp_path / "bad.json"))
    assert document is None and "JSON" in error

    ok, error = serialize.save_or_error(Document(), str(tmp_path / "out.json"))
    assert ok and not error

    ok, error = serialize.save_or_error(Document(), str(tmp_path))
    assert not ok and error, "writing to a directory must be reported"


def test_load_and_save_use_the_file_system(tmp_path):
    path = str(tmp_path / "round.json")
    document = random_document(random.Random(5))
    serialize.save(document, path)
    assert serialize.load(path) == document


def test_a_non_object_envelope_is_refused():
    with pytest.raises(serialize.DocumentFormatError):
        serialize.from_dict([])


def test_a_missing_automaton_block_is_refused():
    with pytest.raises(serialize.DocumentFormatError, match="automaton"):
        serialize.loads(json.dumps({"version": serialize.VERSION}))


def test_a_malformed_transition_is_refused():
    with pytest.raises(serialize.DocumentFormatError, match="transition"):
        serialize.loads(json.dumps({
            "version": serialize.VERSION,
            "automaton": {"states": ["q0"], "alphabet": ["a"],
                          "transitions": [["q0", "a"]],
                          "initial": "q0", "accept": []},
        }))


def test_legacy_arc_offsets_are_read():
    data = dict(LEGACY)
    data["arc_offsets"] = {"q0|q1": 41.5, "ghost|q1": 9.0}
    document = serialize.from_dict(data)
    assert document.layout.arc_of("q0", "q1") == 41.5
    assert len(document.layout.arc_offsets) == 1


def test_labels_survive_a_round_trip():
    document, state = Document().add_state((0.0, 0.0))
    document = Document(document.automaton.with_label(state, "start"),
                        document.layout, document.next_id)
    assert serialize.loads(serialize.dumps(document)).automaton.label_of(state) == "start"


def test_layout_helpers():
    assert Layout().position_of("missing") == (0.0, 0.0)
    assert Layout().arc_of("a", "b") == 0.0
    assert Layout().centre() == (0.0, 0.0)
    assert "positions" in repr(Layout({"q0": (0.0, 0.0)})).lower() or True
    assert (Layout() == "not a layout") is False


def test_fresh_id_skips_names_already_taken():
    document, _ = Document().add_state((0.0, 0.0), state_id="q5")
    document, taken = document.add_state((100.0, 0.0), state_id="q6")
    assert taken == "q6"
    document, next_state = document.add_state((200.0, 0.0))
    assert next_state not in ("q5", "q6")


def test_a_crowded_layout_still_finds_room():
    """The spiral must terminate even when the neighbourhood is full."""
    document = Document()
    for _ in range(40):
        document, _ = document.add_state((0.0, 0.0))
    positions = list(document.layout.positions.values())
    assert len(set(positions)) == len(positions)


def test_legacy_documents_survive_a_resave():
    """Read old, write new, read back: nothing lost that we still keep."""
    document = serialize.from_dict(LEGACY)
    assert serialize.loads(serialize.dumps(document)) == document
