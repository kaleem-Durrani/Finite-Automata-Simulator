"""
Finite Automata Simulator.

The application shell: owns the window, routes input, drives animation, and
turns the document into a scene for the renderer. It holds no automaton logic of
its own -- that is all in :mod:`fsa` -- and no drawing logic -- that is all in
:mod:`rendering`.
"""

import os
import sys
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pygame

# Saved automata resolve against the project directory rather than the process
# working directory, so a file written in one session is findable in the next
# no matter where python was launched from.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# The engine lives under src/ so that it can be packaged and installed on its
# own. Running `python main.py` from a checkout has no install step, so put it
# on the path here rather than making the documented way to start the app
# depend on `pip install -e .` first.
_SRC = os.path.join(PROJECT_DIR, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ruff: noqa: E402
# Everything below imports after the path bootstrap above, deliberately.
import fsa
import fsa.document
from editor import EPSILON_LABEL, EditorModel
from fsa import Document, geometry, serialize
from rendering.animation import Animated, AnimatedPoint, Track, ease_in_out, ease_out_back
from rendering.fonts import FontBook
from rendering.renderer import Renderer, default_state_radius
from rendering.scene import (
    EdgeVisual,
    GhostEdge,
    NodeKind,
    NodeVisual,
    Scene,
    StartMarker,
    TokenVisual,
)
from rendering.theme import Theme
from ui import events
from ui.context_menu import SEPARATOR, MenuItem
from ui.ui_manager import UIManager


def _shown_symbol(symbol: Optional[str]) -> str:
    """A palette symbol as it appears in a message. ``None`` is the epsilon
    move, which the engine spells as nothing at all and a person does not."""
    return EPSILON_LABEL if symbol is None else symbol


#: Symbols a brand-new document starts with, so the palette is never empty.
STARTING_ALPHABET = ("a", "b")

#: How much of a derived expression the status row will carry. The panel cuts
#: whatever it is given down to the column width one character and one font
#: render at a time, so handing over the whole answer would cost a few hundred
#: thousand renders per frame on a machine whose expression runs to megabytes.
PATTERN_DISPLAY_LIMIT = 64

#: How much state elimination this program is willing to do between two frames.
#: Ripping a state writes one label for every (predecessor, successor) pair, so
#: the product of its in- and out-degree is what that one rip costs -- the same
#: number state elimination minimises when it chooses what to rip next -- and
#: summing it over the machine estimates the whole job. Measured
#: here on random complete DFAs: an estimate of 150 costs about 6ms, 230 about
#: 50ms, and 300 nearly half a second while producing 1.6MB of expression that
#: nobody could read anyway. Thompson's own machines sit far below the line (a
#: 44-state one estimates 76), so the guard never fires on what this feature
#: produces -- only on the dense machines it could not usefully answer for.
ELIMINATION_BUDGET = 200

#: What each tool does, said once when it is chosen. A modal tool that changes
#: what a click means owes the user an explanation of the change.
TOOL_HINTS = {
    "pointer": "Pointer: click to select, drag to move",
    "pan": "Hand: drag anywhere to move the view",
    "transition": "Transition: click a state, then its target",
}


def _label_order(symbol: Optional[str]) -> Tuple[int, str]:
    """Sort key for the symbols on one drawn edge, epsilon first.

    ``sorted`` alone raises on these: an edge's symbol set may hold ``None``,
    and ``None`` does not compare with ``str``. The order matches
    :meth:`fsa.nfa.NFA.sorted_transitions` -- the move that costs nothing comes
    first -- so an edge's label reads in the same order as the file and the
    transition table.
    """
    return (0, "") if symbol is None else (1, symbol)


#: Either simulator's record of a run. The two are separate types because a
#: DFA step names one state and an NFA step names a set; only the four fields
#: the UI reads -- ``steps``, ``verdict``, ``offending_symbol``, ``stopped_at``
#: -- are common, and nothing outside :meth:`AutomatonSimulator._test_string`
#: is allowed to care which one it has.
AnyRun = Union[fsa.Run, fsa.nfa.NfaRun]


@dataclass(frozen=True, slots=True)
class RunPosition:
    """One position in a run: where the machine stands, and how it got there.

    A *set* of states, always. A deterministic machine's configuration is a set
    of one, so there is one shape for both simulators and one animation path
    that reads it -- rather than a nondeterministic branch bolted alongside the
    old single-state one, which would be two things to keep in step and a
    frame loop that behaves differently depending on which it took.

    The edges live here beside the configuration rather than in a second list
    indexed by the same counter. That is the mistake this program keeps
    relearning: the previous model kept the drawn edges beside the transition
    table, they fell out of step, and the renderer drew one automaton while the
    simulator ran another.
    """

    configuration: FrozenSet[str]
    """Every state the machine could be in at this position."""

    entered_by: Tuple[Tuple[str, str], ...] = ()
    """The drawn edges crossed to arrive here, as ``(source, target)`` pairs.

    Empty at position 0, since nothing was crossed to start. One entry for a
    deterministic move; one per surviving branch for a nondeterministic one,
    which is what puts a token on each.
    """


def _dfa_trace(result: fsa.Run) -> List[RunPosition]:
    """A deterministic run as a trace of configurations, each a set of one."""
    if result.start is None:
        return []
    trace = [RunPosition(frozenset({result.start}))]
    trace.extend(RunPosition(frozenset({step.target}),
                             ((step.source, step.target),))
                 for step in result.steps)
    return trace


def _nfa_trace(automaton: fsa.NFA, result: fsa.nfa.NfaRun) -> List[RunPosition]:
    """A nondeterministic run as a trace of configurations.

    ``entered_by`` holds every edge the machine crossed *reading that symbol*.
    Epsilon edges are deliberately absent: an epsilon move consumes nothing, so
    nothing travels along it in time, and animating one would put a token on
    the tape's clock for a move the tape never sees. The states an epsilon
    closure reaches still show -- they arrive already lit, inside the
    configuration, which is exactly what "for free" looks like.
    """
    if result.start is None:
        return []
    trace = [RunPosition(result.start)]
    for step in result.steps:
        # Every branch, from every state the machine was in. Sorted so the
        # tokens are built in a stable order; a set iterates differently
        # between processes and a frame that reshuffles is a frame that
        # flickers.
        trace.append(RunPosition(step.target, tuple(sorted(
            (source, target)
            for source in step.source
            for target in automaton.targets(source, step.symbol)))))
    return trace


def _elimination_cost(automaton: fsa.NFA) -> int:
    """An estimate of what deriving an expression from ``automaton`` costs.

    Counted over the drawn edges rather than the transition table, because a
    GNFA holds one label per (source, target) pair however many symbols share
    it. Self-loops are excluded for the reason
    :func:`fsa.regex._next_to_rip` excludes them: a loop is a way round a
    state, not a way through it, and it becomes a star rather than a new edge.
    """
    incoming: Dict[str, int] = {}
    outgoing: Dict[str, int] = {}
    for source, target in automaton.grouped_transitions():
        if source == target:
            continue
        incoming[target] = incoming.get(target, 0) + 1
        outgoing[source] = outgoing.get(source, 0) + 1
    return sum(incoming.get(state, 0) * outgoing.get(state, 0)
               for state in automaton.states)


def _maximize_window() -> None:
    """Open filling the screen.

    Maximised rather than exclusive fullscreen: the title bar and the taskbar
    stay where the user expects them, which is what "opens big" should mean for
    a desktop tool rather than a game. Does nothing where the platform has no
    such concept -- the dummy video driver the tests run under, for one -- so a
    failure here must never stop the application from starting.
    """
    try:
        from pygame._sdl2 import video
        video.Window.from_display_module().maximize()
        pygame.event.pump()
    except Exception:
        pass


def demo_document() -> Document:
    """The automaton the application opens with: a*b+."""
    document = Document()
    for symbol in STARTING_ALPHABET:
        document = document.add_symbol(symbol)

    document, q0 = document.add_state((220.0, 220.0))
    document, q1 = document.add_state((470.0, 220.0))
    document, q2 = document.add_state((345.0, 400.0))

    document = document.add_transition(q0, "a", q0)
    document = document.add_transition(q0, "b", q1)
    document = document.add_transition(q1, "a", q2)
    document = document.add_transition(q1, "b", q1)
    document = document.add_transition(q2, "a", q2)
    document = document.add_transition(q2, "b", q2)
    return document.toggle_accept(q1).set_initial(q0)


class AutomatonSimulator:
    """The application."""

    DEFAULT_FILENAME = "automaton.json"

    def __init__(self) -> None:
        pygame.init()

        info = pygame.display.Info()
        self.screen = pygame.display.set_mode(
            (int(info.current_w * 0.75), int(info.current_h * 0.75)),
            pygame.RESIZABLE,
        )
        _maximize_window()
        # Adopt the maximised size now rather than waiting for the resize event,
        # so the first frame is already laid out for the window the user sees.
        surface = pygame.display.get_surface()
        if surface is not None:
            self.screen = surface

        # Theme and fonts are shared, so switching palettes reaches every
        # surface at once.
        self.theme = Theme("dark")
        self.fonts = FontBook()
        self.editor = EditorModel(demo_document())
        self.renderer = Renderer(self.screen, self.theme, self.fonts)
        self.ui_manager = UIManager(self.screen, self.theme, self.fonts)

        self.running = True
        self.clock = pygame.time.Clock()
        self.fps = 60

        # Execution. The run comes from the engine, so a rejection carries a
        # reason and each step names the edges it took -- which is what the
        # travelling tokens animate along.
        self.execution_active = False
        self.execution_step = 0
        self.execution_string = ""
        self.execution_trace: List[RunPosition] = []
        self.run_result: Optional[AnyRun] = None

        self.animation_active = False
        self.animation_timer = 0
        self.animation_auto_advance = False

        # Animated view state. Each closes the gap to its target over time;
        # nothing in the interface snaps.
        self.node_active = Track(self.theme.motion.quick)
        self.node_selected = Track(self.theme.motion.instant)
        self.node_hover = Track(self.theme.motion.instant)
        self.node_settle = Track(self.theme.motion.normal)
        self.edge_active = Track(self.theme.motion.quick)
        self.token_travel = Animated(duration=self.theme.motion.step)
        self.traversing_step: Optional[int] = None

        self.cam_zoom = Animated(value=1.0, target=1.0,
                                 duration=self.theme.motion.normal,
                                 easing=ease_in_out)
        self.cam_offset = AnimatedPoint()
        self.cam_offset.jump_to((0.0, 0.0))

        self.message_text = ""
        self.message_timer = 0
        self.message_duration = 2400

        self.panning = False
        self.pan_start = (0, 0)
        # Space is a pan modifier while held and "add a state" when tapped;
        # _space_panned records which it turned out to be.
        self._space_held = False
        self._space_panned = False
        self._right_press: Optional[Tuple[int, int]] = None
        self._right_dragged = False

        self._loop_angle_cache: Dict[str, float] = {}
        # (automaton, expression) for the last derivation, keyed on the value
        # itself. See _denoted_pattern.
        self._cached_pattern: Optional[Tuple[Any, str]] = None
        self._event_handlers = self._build_event_handlers()

        self.ui_manager.sync_symbols_with(self.editor.automaton)
        self._update_caption()

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(self.fps)
            self._handle_events()
            self._update(dt)
            self._render()
        pygame.quit()
        sys.exit()

    def _handle_events(self) -> None:
        """
        Process all pygame events.

        Each event is offered to the UI first. If the UI consumed it -- a click
        on a widget, a keypress into a text field, a wheel event over the open
        help panel -- the canvas never sees it. Previously every event went to
        both, so clicking Test also deselected the state behind it and scrolling
        the help panel also zoomed the camera.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._request_quit()
                continue

            if event.type == pygame.VIDEORESIZE:
                self._handle_resize(event.w, event.h)
                continue

            ui_events, consumed = self.ui_manager.handle_event(event)
            self._process_ui_events(ui_events)
            if consumed:
                continue

            if event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_down(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                self._handle_mouse_up(event)
            elif event.type == pygame.MOUSEMOTION:
                self._handle_mouse_motion(event)
            elif event.type == pygame.MOUSEWHEEL:
                self._handle_mouse_wheel(event)
            elif event.type == pygame.KEYDOWN:
                self._handle_key_down(event)
            elif event.type == pygame.KEYUP:
                self._handle_key_up(event)

    def _handle_resize(self, width: int, height: int) -> None:
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.renderer.update_screen_size(width, height)
        self.ui_manager.update_screen_size(width, height)

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    @staticmethod
    def _shift_held() -> bool:
        """
        Whether a shift key is down right now.

        Mouse events in pygame carry no modifier state -- only key events have
        a `.mod` field -- so a click has to ask the keyboard directly. Read once
        while handling the event, not sampled every frame, which is what made
        shift+click drag the state it was drawing from.
        """
        return bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)

    def _pan_modifier_held(self) -> bool:
        """Whether a left-drag should pan instead of reaching the canvas."""
        return self._space_held or self.ui_manager.pan_tool

    def _handle_mouse_down(self, event: pygame.event.Event) -> None:
        if event.button == 1:
            if self._pan_modifier_held():
                # Space, or the hand tool, turns the left button into a pan --
                # the gesture every drawing program uses. Checked before the
                # click reaches the canvas, so panning never also selects a
                # state or starts dragging one.
                self.panning = True
                self.pan_start = event.pos
                self._space_panned = True
                return
            self._handle_left_click(event.pos, self._shift_held())
        elif event.button == 2:
            self.panning = True
            self.pan_start = event.pos
        elif event.button == 3:
            # Deferred: a right-drag pans (middle-button-only panning is
            # miserable on a trackpad), so the menu waits for release and only
            # opens if the button never travelled.
            self._right_press = event.pos
            self._right_dragged = False

    def _handle_mouse_up(self, event: pygame.event.Event) -> None:
        if event.button == 1:
            self.panning = False
            if self.editor.end_drag():
                self._update_caption()
        elif event.button == 2:
            self.panning = False
        elif event.button == 3:
            press, self._right_press = self._right_press, None
            if press is not None and not self._right_dragged:
                self._handle_right_click(press)
            self._right_dragged = False

    def _handle_mouse_motion(self, event: pygame.event.Event) -> None:
        if self._right_press is not None and event.buttons[2]:
            moved = (abs(event.pos[0] - self._right_press[0])
                     + abs(event.pos[1] - self._right_press[1]))
            if self._right_dragged or moved > 5:
                if not self._right_dragged:
                    self._right_dragged = True
                    self.pan_start = self._right_press
                dx = event.pos[0] - self.pan_start[0]
                dy = event.pos[1] - self.pan_start[1]
                self.renderer.camera.pan(dx, dy)
                self.pan_start = event.pos
                self._sync_camera_targets()
                return

        if self.panning:
            dx = event.pos[0] - self.pan_start[0]
            dy = event.pos[1] - self.pan_start[1]
            self.renderer.camera.pan(dx, dy)
            self.pan_start = event.pos
            # Direct manipulation wins: adopt the new position as the target so
            # the easing does not pull the view back where it was heading.
            self._sync_camera_targets()
        elif self.editor.drag is not None:
            self.editor.update_drag(self._world(event.pos))

        self.editor.set_hover(self._state_at(event.pos))

    def _handle_mouse_wheel(self, event: pygame.event.Event) -> None:
        """Zoom, or bend the transition being drawn.

        Help-panel scrolling is the UI's job; if it handled the event this is
        never reached. Both used to run, so scrolling also zoomed underneath.
        """
        if self.editor.pending_source is None:
            zoom_factor = 1.1 if event.y > 0 else 0.9
            self.renderer.camera.zoom_at(pygame.mouse.get_pos(), zoom_factor)
            self._sync_camera_targets()
        else:
            self.editor.bend_pending(event.y * 10)

    def _world(self, screen_point: Tuple[int, int]) -> Tuple[float, float]:
        return self.renderer.camera.screen_to_world(screen_point)

    def _state_at(self, screen_point: Tuple[int, int]) -> Optional[str]:
        return self.editor.state_at(self._world(screen_point),
                                    default_state_radius())

    def _handle_left_click(self, pos: Tuple[int, int], shift: bool = False) -> None:
        """
        Handle a left click on the canvas.

        Exactly one of {complete a transition, start a transition, select and
        drag, deselect} happens. Shift comes from the event's modifier state
        rather than being polled every frame, which is what used to make
        shift+click both start a transition *and* drag the source state.
        """
        clicked = self._state_at(pos)

        if self.editor.pending_source is not None:
            if clicked:
                self._complete_transition(clicked)
            else:
                self.editor.cancel_transition()
                self._show_message("Transition cancelled")
        elif (shift or self.ui_manager.transition_tool) and clicked:
            # Shift is the shortcut; the transition tool is the discoverable
            # way in. A hidden modifier was the only route to the single most
            # common edit in the application.
            self.editor.select(clicked)
            self.editor.begin_transition(clicked)
            self._show_message(
                f"Drawing '{_shown_symbol(self.ui_manager.selected_symbol)}'"
                f" from {clicked}"
                " -- click its target")
        elif clicked:
            self.editor.select(clicked)
            self.editor.begin_drag(clicked, self._world(pos))
        else:
            self.editor.select(None)

    def _handle_right_click(self, pos: Tuple[int, int]) -> None:
        clicked = self._state_at(pos)
        if clicked:
            self._show_state_context_menu(pos, clicked)
            return

        # An edge, if the click landed near enough to one of the drawn curves.
        world = self._world(pos)
        positions = self.editor.positions()
        paths = self._edge_paths(positions)
        edge = geometry.nearest_edge(world, paths,
                                     within=14.0 / max(0.4, self.renderer.camera.zoom))
        if edge is not None:
            self._show_edge_context_menu(pos, edge)
        else:
            self._show_general_context_menu(pos)

    def _show_edge_context_menu(self, pos: Tuple[int, int],
                                edge: Tuple[str, str]) -> None:
        """A menu for one drawn edge: remove any of its symbols, or straighten.

        Transitions used to be uneditable from the canvas at all -- the only
        way to remove one was to delete a state.
        """
        source, target = edge
        symbols = sorted(self.editor.automaton.grouped_transitions().get(edge, ()),
                         key=_label_order)
        # The target is part of the request: this menu belongs to one drawn
        # edge, and on a nondeterministic machine that edge is one branch of
        # the move. Asking to remove the symbol alone would take the arrows to
        # every other state with it.
        items = [MenuItem(f"Remove '{EPSILON_LABEL if symbol is None else symbol}'",
                          events.RemoveTransition(source, symbol, target))
                 for symbol in symbols]
        # A self-loop stores an arc that no renderer honours, so offering to
        # straighten one promises a change nothing can show.
        if source != target and self.editor.layout.arc_of(source, target):
            items.append(MenuItem(SEPARATOR))
            items.append(MenuItem("Straighten",
                                  events.StraightenEdge(source, target)))
        if items:
            self.ui_manager.show_context_menu(pos, items)

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def _handle_key_up(self, event: pygame.event.Event) -> None:
        """Finish a space-bar gesture.

        Guarded on the key actually having been pressed on the canvas: if the
        UI consumed the key-down -- a space typed into the test field -- the
        matching key-up still arrives here, and adding a state for it would be
        a state appearing every time someone typed a space.
        """
        if event.key == pygame.K_SPACE and self._space_held:
            self._space_held = False
            if not self._space_panned:
                self._add_state_at_center()
            self._space_panned = False

    def _handle_key_down(self, event: pygame.event.Event) -> None:
        """Handle key down events the UI did not consume."""
        # Chords first: a plain 'z' must not undo, and Ctrl+Z must not fall
        # through to a symbol selection.
        if event.mod & pygame.KMOD_CTRL:
            if event.key == pygame.K_z and (event.mod & pygame.KMOD_SHIFT):
                self._redo()
            elif event.key == pygame.K_z:
                self._undo()
            elif event.key == pygame.K_y:
                self._redo()
            elif event.key == pygame.K_a:
                self._toggle_accept_state()
            elif event.key == pygame.K_t:
                self._make_trap(self.editor.selection)
            elif event.key == pygame.K_0:
                self._fit_to_content()
            return

        if event.key == pygame.K_SPACE:
            # Held, space pans; tapped, it adds a state. Which of the two it
            # was is only known once the key comes up, so the add waits for
            # the release rather than firing here and being wrong half the
            # time.
            self._space_held = True
        elif event.key == pygame.K_DELETE:
            self._delete_selected_state()
        elif event.key == pygame.K_RIGHT:
            self._next_execution_step()
        elif event.key == pygame.K_LEFT:
            self._previous_execution_step()
        elif event.key == pygame.K_ESCAPE:
            if self.editor.pending_source is not None:
                self.editor.cancel_transition()
                self._show_message("Transition cancelled")
            else:
                self._stop_execution()
        elif event.key == pygame.K_TAB and self.execution_active:
            self._toggle_animation()
        elif event.unicode and event.unicode in self.editor.automaton.alphabet:
            # Every editing shortcut above is a chord or a non-letter key, so a
            # bare printable key means exactly one thing: pick that symbol.
            # While `q`, `w`, `r`, `n` and `p` were shortcuts, an automaton over
            # an alphabet containing them could be built with the mouse but
            # never typed at -- pressing `q` toggled accepting instead.
            self.ui_manager.selected_symbol = event.unicode
            self._show_message(f"Symbol '{event.unicode}' selected")

    # ------------------------------------------------------------------
    # UI actions
    # ------------------------------------------------------------------

    def _build_event_handlers(self) -> Dict[type, Callable[[Any], None]]:
        """One handler per event type.

        A table rather than an if-chain, so an event with no handler is a
        `KeyError` from :func:`events.dispatch` at the first press of the
        button that emits it. Under the previous scheme it was a dict lookup
        that missed, which is not an error -- twelve of the twenty-nine action
        names in use had no handler and did nothing, silently.
        """
        return {
            events.TestString: lambda e: self._test_string(e.text),
            events.StepForward: lambda _e: self._next_execution_step(),
            events.StepBack: lambda _e: self._previous_execution_step(),
            events.ToggleAnimation: lambda _e: self._toggle_animation(),
            events.StopExecution: lambda _e: self._stop_execution(),
            events.ToolSelected: self._on_tool_selected,
            events.SaveRequested: lambda _e: self._save_automaton(),
            events.LoadRequested: lambda _e: self._load_automaton(),
            events.SaveToPath: lambda e: self._save_to_path(e.path),
            events.LoadFromPath: lambda e: self._load_from_path(e.path),
            events.Confirmed: lambda e: self._handle_confirmed(e.intent),
            events.PromptCancelled: lambda _e: self._show_message("Cancelled"),
            events.ToggleTheme: lambda _e: self._toggle_theme(),
            events.CompleteAutomaton: lambda _e: self._complete_automaton(),
            events.DeterminizeAutomaton: lambda _e: self._determinize_automaton(),
            events.MinimizeAutomaton: lambda _e: self._minimize_automaton(),
            events.ShowMarkingTable: lambda _e: self._show_marking_table(),
            events.TrimAutomaton: lambda _e: self._trim_automaton(),
            events.RegexPrompt: lambda _e: self._show_regex_prompt(),
            events.BuildFromRegex: lambda e: self._build_from_regex(e.pattern),
            events.FocusStates: lambda e: self._focus_states(list(e.states)),
            events.ShowMessage: lambda e: self._show_message(e.text),
            events.SymbolSelected: self._on_symbol_selected,
            events.SymbolAdded: lambda e: self._add_symbol(e.symbol),
            events.SymbolRejected: lambda e: self._show_message(f"Error: {e.reason}"),
            events.RenameState: lambda e: self._rename_state(e.state, e.label),
            events.RenamePrompt: self._on_rename_prompt,
            events.AddStateAt: self._on_add_state_at,
            events.DeleteState: self._on_delete_state,
            events.ToggleAccept: self._on_toggle_accept,
            events.SetInitial: self._on_set_initial,
            events.MakeTrap: lambda e: self._make_trap(e.state),
            events.RemoveTransition: self._on_remove_transition,
            events.StraightenEdge: self._on_straighten_edge,
            events.FitView: lambda _e: self._fit_to_content(),
        }

    def _process_ui_events(self, ui_events: Sequence[events.UiEvent]) -> None:
        events.dispatch(ui_events, self._event_handlers)

    # -- event handlers -------------------------------------------------

    def _on_tool_selected(self, event: events.ToolSelected) -> None:
        # Choosing a tool abandons a half-drawn transition: the arrow was
        # following a pointer that is now doing something else.
        self.editor.cancel_transition()
        self._show_message(TOOL_HINTS.get(event.tool, ""))

    def _on_symbol_selected(self, event: events.SymbolSelected) -> None:
        self._show_message(f"Selected symbol: {event.symbol}")

    def _on_rename_prompt(self, event: events.RenamePrompt) -> None:
        if event.state in self.editor.automaton.states:
            self.ui_manager.show_rename_prompt(
                event.state, self.editor.automaton.label_of(event.state))

    def _on_add_state_at(self, event: events.AddStateAt) -> None:
        # The user picked this spot, so honour it unless it would overlap.
        state = self.editor.add_state(event.position,
                                      minimum_gap=fsa.document.OVERLAP_GAP)
        self.editor.select(state)
        self._after_edit()
        self._show_message(f"Added {state}")

    def _on_delete_state(self, event: events.DeleteState) -> None:
        if self.editor.remove_state(event.state):
            self._after_edit()
            self._show_message(f"Deleted {event.state}")

    def _on_toggle_accept(self, event: events.ToggleAccept) -> None:
        if event.state not in self.editor.automaton.states:
            return
        accepting = self.editor.toggle_accept(event.state)
        self._after_edit()
        self._show_message(
            f"{event.state} is "
            f"{'now accepting' if accepting else 'no longer accepting'}")

    def _on_set_initial(self, event: events.SetInitial) -> None:
        if event.state is not None and event.state not in self.editor.automaton.states:
            return
        self.editor.set_initial(event.state)
        self._after_edit()
        self._show_message(f"{event.state} is now the initial state")

    def _on_remove_transition(self, event: events.RemoveTransition) -> None:
        # Read before the edit, and read as a set: the message names what
        # actually went, which on a branching move is one of several arrows.
        losing = self.editor.automaton.targets(event.source, event.symbol)
        if event.target is not None:
            losing = losing & {event.target}

        self.editor.remove_transition(event.source, event.symbol, event.target)
        self._after_edit()
        symbol = EPSILON_LABEL if event.symbol is None else event.symbol
        self._show_message(
            f"Removed {event.source} --{symbol}--> "
            f"{', '.join(sorted(losing)) or 'nothing'}")

    def _on_straighten_edge(self, event: events.StraightenEdge) -> None:
        source, target = event.source, event.target
        document = fsa.Document(
            self.editor.document.automaton,
            self.editor.document.layout.with_arc(source, target, 0.0),
            self.editor.document.next_id)
        self.editor.apply(document, action=f"straighten {source}->{target}")
        self._after_edit()
        # A two-way pair keeps an automatic bow so the two arrows stay apart.
        # Reporting "straightened" there would describe a line the user can
        # plainly see is still curved.
        twinned = (target, source) in self.editor.automaton.grouped_transitions()
        self._show_message("Manual bend cleared" if twinned
                           else "Edge straightened")

    def _handle_confirmed(self, intent: str) -> None:
        if intent == 'quit_after_confirm':
            self.running = False
        elif intent == 'load_after_confirm':
            self.ui_manager.show_file_prompt(
                'load', self.editor.path or self.DEFAULT_FILENAME)

    def _show_message(self, text: str) -> None:
        self.message_text = text
        self.message_timer = pygame.time.get_ticks()

    def _update_message(self) -> None:
        if (self.message_text
                and pygame.time.get_ticks() - self.message_timer > self.message_duration):
            self.message_text = ""

    def _toggle_theme(self) -> None:
        name = self.theme.toggle()
        self._show_message(f"{name.capitalize()} theme")

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def _after_edit(self) -> None:
        """Refresh anything that depends on the document having changed."""
        self.ui_manager.sync_symbols_with(self.editor.automaton)
        self._update_caption()

    def _add_state_at_center(self) -> None:
        centre = self._world((self.screen.get_width() // 2,
                              self.screen.get_height() // 2))
        state = self.editor.add_state(centre)
        self.editor.select(state)
        self._after_edit()
        self._show_message(f"Added {state}")

    def _delete_selected_state(self) -> None:
        state = self.editor.selection
        if state and self.editor.remove_state(state):
            self._after_edit()
            self._show_message(f"Deleted {state}")

    def _toggle_accept_state(self) -> None:
        state = self.editor.selection
        if state is None:
            return
        accepting = self.editor.toggle_accept(state)
        self._after_edit()
        self._show_message(
            f"{state} is {'now accepting' if accepting else 'no longer accepting'}")

    def _make_trap(self, state: Optional[str]) -> None:
        """Turn a state into a real trap: self-loop on every symbol.

        There used to be a "dead end" flag. It changed what the simulator
        accepted without changing any transition, so the tool computed a
        different language than the diagram showed. Trap-ness is derived now --
        which left a menu item writing a value nothing read. This does the thing
        the flag pretended to mean, so the state renders as a trap because it
        genuinely is one.
        """
        if state is None:
            return
        if not self.editor.automaton.alphabet:
            self._show_message("Add a symbol first: there is nothing to loop on")
            return

        ok, replaced = self.editor.make_trap(state)
        if not ok:
            return
        self._after_edit()
        alphabet = ", ".join(sorted(self.editor.automaton.alphabet))
        detail = (f", replacing {replaced} transition{'s' if replaced != 1 else ''}"
                  if replaced else "")
        self._show_message(f"{state} now loops on {alphabet}{detail}")

    def _rename_state(self, state: str, label: str) -> None:
        if not self.editor.rename(state, label):
            return
        self._after_edit()
        shown = label.strip() or state
        self._show_message(f"{state} is now labelled '{shown}'")

    def _undo(self) -> None:
        """Step the document back one edit, saying which one."""
        action = self.editor.undo()
        if action is None:
            self._show_message("Nothing to undo")
            return
        self._after_edit()
        self._show_message(f"Undid {action}")

    def _redo(self) -> None:
        action = self.editor.redo()
        if action is None:
            self._show_message("Nothing to redo")
            return
        self._after_edit()
        self._show_message(f"Redid {action}")

    def _deterministic_view(self, operation: str) -> Optional[fsa.DFA]:
        """The DFA on the canvas, or ``None`` with the reason already shown.

        Most of the algorithms layer is defined on a transition *function*, and
        the canvas may now hold a relation. This is the one place the interface
        finds that out, and it says so in a toast rather than raising: an
        exception here is a window that closes while somebody is drawing.

        Deliberately not "your machine is broken". Nondeterminism is a legal
        design choice; the message names what to do about it and moves on.
        """
        try:
            return self.editor.document.as_dfa()
        except fsa.NondeterministicError:
            self._show_message(
                f"{operation} needs a deterministic machine -- determinize it "
                f"first")
            return None

    def _complete_automaton(self) -> None:
        """One click from "your machine is incomplete" to a total automaton.

        Adds a trap state and routes every undefined (state, symbol) pair to
        it. The language is unchanged -- previously-undefined runs now die in
        the trap instead of halting -- which is precisely the lesson the
        diagnostics panel is teaching when it flags incompleteness.
        """
        automaton = self._deterministic_view("Complete")
        if automaton is None:
            return
        before = len(fsa.missing_transitions(automaton))
        document, trap = self.editor.document.complete()
        if trap is None:
            self._show_message("Already complete")
            return
        self.editor.apply(document)
        self._after_edit()
        self.node_settle.set(trap, 1.0, duration=self.theme.motion.quick,
                             easing=ease_out_back)
        self._show_message(
            f"Added {trap} and routed {before} missing "
            f"transition{'s' if before != 1 else ''} to it")

    def _replace_automaton(self, automaton: fsa.AnyAutomaton, action: str) -> None:
        """Adopt a machine an algorithm produced, and give it coordinates.

        Every construction in the algorithms layer emits states the user never
        placed, so the document it goes into needs a layout generated for it --
        without that they would all land on the origin in a heap. Hand-placed
        coordinates cannot be kept: the states are new values, not the old ones
        moved. A run in progress is stopped because its path names states that
        may no longer exist.
        """
        document = fsa.Document(automaton, fsa.Layout.auto(automaton),
                                self.editor.document.next_id)
        self._stop_execution()
        self.editor.apply(document, action=action)
        self._after_edit()
        self._fit_to_content()

    def _determinize_automaton(self) -> None:
        """Replace the machine with an equivalent deterministic one.

        Offered even when the machine is already deterministic, because the
        subset construction also completes delta -- so it does something, and
        saying what it did is better than greying the item out and leaving the
        user to guess why.
        """
        automaton = self.editor.automaton
        if automaton.initial is None:
            self._show_message("Determinize needs an initial state")
            return

        before = len(automaton.states)
        was_deterministic = automaton.is_deterministic()
        result = fsa.determinize(automaton)
        self._replace_automaton(result, "determinize")
        self._show_message(
            f"Completed delta: {before} states -> {len(result.states)}"
            if was_deterministic else
            f"Determinized: {before} states -> {len(result.states)}")

    def _minimize_automaton(self) -> None:
        """Merge the states no word can tell apart."""
        automaton = self._deterministic_view("Minimise")
        if automaton is None:
            return
        if automaton.initial is None:
            self._show_message("Minimise needs an initial state")
            return

        reduced = fsa.minimize(automaton)
        if len(reduced.states) == len(automaton.states):
            self._show_message(f"Already minimal at {len(reduced.states)} states")
            return

        before = len(automaton.states)
        self._replace_automaton(reduced, "minimise")
        self._show_message(
            f"Minimised {before} states to {len(reduced.states)}")

    def _show_marking_table(self) -> None:
        """Open the table-filling grid for the machine on the canvas.

        The table is computed from the automaton as it stands, not from the
        last minimisation: the interesting question is which of *these* states
        are indistinguishable, and the answer has to follow the drawing.
        """
        automaton = self._deterministic_view("The marking table")
        if automaton is None:
            return
        if automaton.initial is None:
            self._show_message("The marking table needs an initial state")
            return
        if len(automaton.states) < 2:
            self._show_message("Two states are needed to have a pair")
            return

        table = fsa.marking_table(automaton)
        self.ui_manager.show_marking_table(table)
        equivalent = len(table.equivalent_pairs)
        self._show_message(
            f"{len(table.marks)} pairs separated, {equivalent} equivalent"
            if equivalent else "Every pair is distinguishable: already minimal")

    def _trim_automaton(self) -> None:
        """Drop the states that cannot appear on an accepting run."""
        automaton = self._deterministic_view("Trim")
        if automaton is None:
            return
        trimmed = fsa.trim(automaton)
        removed = len(automaton.states) - len(trimmed.states)
        if not removed:
            self._show_message("Nothing to trim")
            return

        was_complete = fsa.is_complete(automaton)
        self._replace_automaton(trimmed, "trim")

        count = f"Trimmed {removed} state{'s' if removed != 1 else ''}"
        if was_complete and not fsa.is_complete(trimmed):
            # Trim and complete are exact inverses on a machine that needed a
            # trap: a trap is precisely a state no accepting run visits. Say so,
            # or the incompleteness warning reappears with no visible cause and
            # the two buttons look like they are fighting each other.
            self._show_message(
                f"{count}; delta is partial again -- a trap is a dead state")
        else:
            self._show_message(count)

    def _show_regex_prompt(self) -> None:
        """Ask for a regular expression.

        The field starts empty rather than pre-filled with what the machine
        already denotes. The rename prompt pre-fills because renaming is
        editing a name that exists; this is not -- somebody choosing "from a
        regular expression" has one in mind, and clearing forty characters of
        somebody else's answer first would be work done to reach the blank
        field they wanted.
        """
        self.ui_manager.show_regex_prompt()

    def _build_from_regex(self, pattern: str) -> None:
        """Replace the document with the machine ``pattern`` denotes.

        Thompson's construction exactly as the engine gives it -- two states
        per operator, epsilon moves at every join -- and deliberately not
        determinized on the way in. That the shape of the machine mirrors the
        shape of the expression is the thing worth seeing, and Determinize is
        the next item up the same menu for when it stops being.
        """
        try:
            machine = fsa.regex.to_nfa(pattern)
        except fsa.regex.RegexSyntaxError as error:
            # Shown, not swallowed, and shown in both places it is useful: the
            # toast carries the sentence, and the prompt re-opens with the text
            # that failed and a caret under the character that stopped it. The
            # document is not touched -- a pattern that does not parse denotes
            # nothing to replace it with.
            self._show_message(f"Not a regular expression: {error}")
            self.ui_manager.show_regex_prompt(pattern, error=str(error),
                                              error_at=error.position)
            return

        # An empty pattern is ε and prints as nothing at all, so the message
        # names the character rather than leaving a gap where the answer was.
        shown = pattern or fsa.regex.EMPTY_WORD
        self._replace_automaton(machine, f"regex {shown}")
        self._show_message(
            f"'{shown}' built {len(machine.states)} states by Thompson's "
            f"construction -- determinize to tidy it")

    def _denoted_pattern(self) -> str:
        """The regular expression the machine on the canvas denotes.

        Derived on demand and never stored on the document: an expression kept
        beside the machine would be a second copy of one fact, with its own
        chance to fall out of step -- which is precisely the mistake this
        program keeps relearning (docs/LESSONS.md).

        Cached on the automaton *value*, the way :meth:`EditorModel.analysis`
        is. The value is immutable, so an identity check is the whole of the
        invalidation rule and there is no flag anyone can forget to clear. It
        also means dragging a state costs nothing here: a move produces a new
        layout and the same machine.
        """
        automaton = self.editor.automaton
        cached = self._cached_pattern
        if cached is not None and cached[0] is automaton:
            return cached[1]

        if automaton.initial is None:
            # `from_automaton` answers ∅ here and says in its docstring that a
            # front end which cares should ask first. This one cares: ∅ is the
            # empty language, and a machine with nowhere to start has no
            # language yet rather than the empty one.
            text = "none"
        elif _elimination_cost(automaton) > ELIMINATION_BUDGET:
            text = "too big to derive"
        else:
            expression = fsa.regex.from_automaton(automaton)
            text = (expression if len(expression) <= PATTERN_DISPLAY_LIMIT
                    else expression[:PATTERN_DISPLAY_LIMIT] + "...")

        self._cached_pattern = (automaton, text)
        return text

    def _focus_states(self, states: List[str]) -> None:
        """Glide the camera to the states a diagnostic names."""
        positions = [self.editor.position_of(s) for s in states
                     if s in self.editor.automaton.states]
        if not positions:
            return
        pad = default_state_radius() * 4
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zoom, offset = self.renderer.fit_to_bounds(
            (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad))
        self.cam_zoom.set(zoom, duration=self.theme.motion.slow, easing=ease_in_out)
        self.cam_offset.set(offset, duration=self.theme.motion.slow,
                            easing=ease_in_out)
        for state in states:
            self.node_settle.set(state, 1.0, duration=self.theme.motion.quick,
                                 easing=ease_out_back)

    def _add_symbol(self, symbol: str) -> None:
        if self.editor.add_symbol(symbol):
            self.ui_manager.selected_symbol = symbol
            self._after_edit()
            self._show_message(f"Added symbol '{symbol}'")
        else:
            self._show_message(f"Cannot add '{symbol}'")

    def _complete_transition(self, target: str) -> None:
        source = self.editor.pending_source
        if source is None:
            return
        symbol = self.ui_manager.selected_symbol
        arc = self.editor.pending_arc
        self.editor.cancel_transition()

        if symbol is not None and symbol not in self.editor.automaton.alphabet:
            self._show_message(f"'{symbol}' is not in the alphabet")
            return

        already = self.editor.automaton.targets(source, symbol) - {target}
        if not self.editor.add_transition(source, symbol, target, arc):
            self._show_message(f"Could not add transition from {source}")
            return
        self._after_edit()
        if already:
            # The second arrow used to delete the first. Now both are there,
            # and saying so is the whole lesson: the machine has a choice, and
            # that is a legal thing for it to have.
            self._show_message(
                f"{source} --{symbol}--> {target}; it also goes to "
                f"{', '.join(sorted(already))}, so this is nondeterministic")
        else:
            self._show_message(f"{source} --{symbol}--> {target}")

    # ------------------------------------------------------------------
    # Context menus
    # ------------------------------------------------------------------

    def _show_state_context_menu(self, pos: Tuple[int, int], state: str) -> None:
        # Toggles carry their current value, so the menu says what the state
        # already is instead of making the user guess and check.
        automaton = self.editor.automaton
        self.ui_manager.show_context_menu(pos, [
            MenuItem("Accepting", events.ToggleAccept(state),
                     checked=state in automaton.accept),
            MenuItem("Initial state", events.SetInitial(state),
                     checked=state == automaton.initial),
            MenuItem(SEPARATOR),
            MenuItem("Rename...", events.RenamePrompt(state)),
            MenuItem("Make a trap", events.MakeTrap(state)),
            MenuItem(SEPARATOR),
            MenuItem("Delete state", events.DeleteState(state)),
        ])

    def _show_general_context_menu(self, pos: Tuple[int, int]) -> None:
        self.ui_manager.show_context_menu(pos, [
            MenuItem("Add state here", events.AddStateAt(self._world(pos))),
            MenuItem(SEPARATOR),
            MenuItem("Determinize", events.DeterminizeAutomaton()),
            MenuItem("Minimise", events.MinimizeAutomaton()),
            MenuItem("Marking table", events.ShowMarkingTable()),
            MenuItem("Trim", events.TrimAutomaton()),
            # Beside the constructions rather than beside "Add state here":
            # this is the other direction of the same theorem the rest of the
            # group belongs to, and the status panel's Denotes row is the
            # return trip.
            MenuItem("From regular expression...", events.RegexPrompt()),
            MenuItem(SEPARATOR),
            MenuItem("Fit to content", events.FitView()),
        ])


    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    def _resolve_path(self, filename: str) -> str:
        """
        Resolve a user-supplied filename against the project directory.

        Relative paths used to resolve against the process working directory,
        so a file saved from one launch could be unreachable from the next.
        """
        filename = filename.strip() or self.DEFAULT_FILENAME
        if not os.path.splitext(filename)[1]:
            filename += ".json"
        if os.path.isabs(filename):
            return filename
        return os.path.normpath(os.path.join(PROJECT_DIR, filename))

    def _save_automaton(self) -> None:
        self.ui_manager.show_file_prompt(
            'save', self.editor.path or self.DEFAULT_FILENAME)

    def _load_automaton(self) -> None:
        if self.editor.dirty:
            self.ui_manager.show_confirm("Discard unsaved changes and load?",
                                         'load_after_confirm')
            return
        self.ui_manager.show_file_prompt(
            'load', self.editor.path or self.DEFAULT_FILENAME)

    def _save_to_path(self, filename: str) -> None:
        path = self._resolve_path(filename)
        ok, error = serialize.save_or_error(self.editor.document, path)
        if ok:
            self.editor.path = os.path.relpath(path, PROJECT_DIR)
            self.editor.dirty = False
            self._update_caption()
            self._show_message(f"Saved to {self.editor.path}")
        else:
            self._show_message(f"Save failed: {error}")

    def _load_from_path(self, filename: str) -> None:
        path = self._resolve_path(filename)
        document, error = serialize.load_or_error(path)
        if document is None:
            self._show_message(f"Load failed: {error}")
            return

        # A different machine: nothing about the previous one survives.
        self._stop_execution()
        self.editor.replace(document, os.path.relpath(path, PROJECT_DIR))
        self.ui_manager.test_result = ""
        self.ui_manager.test_verdict = ""
        self.ui_manager.sync_symbols_with(self.editor.automaton)
        for track in (self.node_selected, self.node_hover, self.node_settle,
                      self.node_active, self.edge_active):
            track.clear()

        self._update_caption()
        self._show_message(f"Loaded {self.editor.path}")

    def _update_caption(self) -> None:
        name = self.editor.path or "untitled"
        marker = "*" if self.editor.dirty else ""
        pygame.display.set_caption(f"{name}{marker} - Finite Automata Simulator")

    def _request_quit(self) -> None:
        if self.editor.dirty:
            self.ui_manager.show_confirm("Quit without saving?", 'quit_after_confirm')
        else:
            self.running = False

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _test_string(self, test_string: str) -> None:
        """Run a string and start the visualisation.

        The empty string is a legal and pedagogically important input. The old
        version refused it outright.

        Two simulators, chosen by what is actually on the canvas. A
        deterministic document runs through :func:`fsa.run`, so nothing about
        the familiar animation changes for the case that is nearly always on
        screen; a nondeterministic one runs through :func:`fsa.nfa.run`, whose
        configurations are sets and whose rejection reason says every branch
        died rather than that an arrow is missing.

        Both are flattened into one trace here, and this is the only method
        that knows there were two. Everything downstream -- the lit states, the
        tokens, the panel, the tape -- reads the trace, which is why there is
        no second animation path to keep in agreement with the first.
        """
        result: AnyRun
        if self.editor.is_deterministic:
            result = fsa.run(self.editor.document.as_dfa(), test_string)
            trace = _dfa_trace(result)
        else:
            automaton = self.editor.automaton
            result = fsa.nfa.run(automaton, test_string)
            trace = _nfa_trace(automaton, result)
        self.run_result = result

        self.ui_manager.test_result = result.explain()
        self.ui_manager.test_verdict = result.verdict.value
        # A verdict the user cannot see is not a verdict. Running a string is
        # the one thing that opens the test panel on its own.
        self.ui_manager.input_expanded = True

        self.execution_active = True
        self.execution_string = test_string
        self.execution_trace = trace
        self.execution_step = 0
        self.traversing_step = None
        self.token_travel.jump_to(0.0)

        if trace:
            # Every state of the opening configuration swells, not one of them.
            # On a machine with epsilon moves out of the start state that is
            # already several, before a single symbol has been read.
            for state in trace[0].configuration:
                self.node_settle.set(state, 1.0,
                                     duration=self.theme.motion.quick,
                                     easing=ease_out_back)
                self.node_settle.set(state, 0.0,
                                     duration=self.theme.motion.normal)

    def _goto_execution_step(self, index: int, animate: bool = True) -> None:
        """Move the visualisation to a position in the run.

        Animates a token along every edge connecting the two positions -- one
        on a deterministic move, one per branch otherwise -- in whichever
        direction it is travelling, so stepping backwards reads as the machine
        reversing rather than teleporting.
        """
        if not self.execution_active or not self.execution_trace:
            return

        index = max(0, min(len(self.execution_trace) - 1, index))
        if index == self.execution_step:
            return

        forward = index > self.execution_step
        # The move being crossed is the one entering the higher of the two
        # positions, whichever way it is being crossed, so `traversing_step`
        # still counts moves and its edges are `trace[step + 1].entered_by`.
        step_index = self.execution_step if forward else index
        self.execution_step = index

        entered_by = self.execution_trace[step_index + 1].entered_by
        if animate and entered_by:
            self.traversing_step = step_index
            self.token_travel.jump_to(0.0 if forward else 1.0)
            self.token_travel.set(1.0 if forward else 0.0,
                                  duration=self.theme.motion.step)
            for source, target in entered_by:
                self.edge_active.set(f"{source}|{target}", 1.0,
                                     duration=self.theme.motion.instant)
        else:
            self.traversing_step = None

        for state in self.execution_trace[index].configuration:
            self.node_settle.set(state, 1.0,
                                 duration=self.theme.motion.quick,
                                 easing=ease_out_back)

    def _next_execution_step(self) -> None:
        if self.execution_active and self.execution_step < len(self.execution_trace) - 1:
            self._goto_execution_step(self.execution_step + 1)

    def _previous_execution_step(self) -> None:
        if self.execution_active and self.execution_step > 0:
            self._goto_execution_step(self.execution_step - 1)

    def _stop_execution(self) -> None:
        self.execution_active = False
        self.execution_step = 0
        self.execution_string = ""
        self.execution_trace = []
        self.run_result = None
        self.traversing_step = None
        self.animation_active = False
        self.animation_auto_advance = False
        self.node_active.clear()
        self.edge_active.clear()
        self.token_travel.jump_to(0.0)

    def _toggle_animation(self) -> None:
        if not self.execution_active:
            return
        self.animation_active = not self.animation_active
        self.animation_auto_advance = self.animation_active
        self.animation_timer = pygame.time.get_ticks()
        self._show_message("Playing" if self.animation_active else "Paused")

    def _advance_playback(self) -> None:
        if not (self.animation_active and self.animation_auto_advance
                and self.execution_active):
            return

        now = pygame.time.get_ticks()
        interval = max(self.ui_manager.animation_speed,
                       self.theme.motion.step + 60)
        if now - self.animation_timer < interval:
            return

        if self.execution_step < len(self.execution_trace) - 1:
            self._goto_execution_step(self.execution_step + 1)
            self.animation_timer = now
        else:
            self.animation_auto_advance = False
            self.animation_active = False
            self._show_message("Playback finished")

    def _current_configuration(self) -> FrozenSet[str]:
        """Every state the machine stands in right now.

        A set, always, so the canvas can light all of them. A DFA's
        configuration is a set of one, so this is not a nondeterministic
        special case -- it is the general statement, of which the old
        single-state answer was the narrow reading.
        """
        if not self.execution_active or not self.execution_trace:
            return frozenset()
        index = min(self.execution_step, len(self.execution_trace) - 1)
        return self.execution_trace[index].configuration

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _fit_to_content(self) -> None:
        """Frame the whole automaton, easing rather than snapping.

        `camera.reset()` pinned the world origin to the top-left corner and
        could leave the graph entirely off screen -- a "reset view" that lost
        your work rather than finding it.
        """
        bounds = self.editor.layout.bounds(default_state_radius())
        if bounds is None:
            self.cam_zoom.set(1.0, duration=self.theme.motion.slow)
            self.cam_offset.set((0.0, 0.0), duration=self.theme.motion.slow)
            self._show_message("View reset")
            return

        zoom, offset = self.renderer.fit_to_bounds(bounds)
        self.cam_zoom.set(zoom, duration=self.theme.motion.slow, easing=ease_in_out)
        self.cam_offset.set(offset, duration=self.theme.motion.slow,
                            easing=ease_in_out)
        self._show_message("Fitted to content")

    def _sync_camera_targets(self) -> None:
        """Adopt the camera's current values as the animation's targets."""
        self.cam_zoom.jump_to(self.renderer.camera.zoom)
        self.cam_offset.jump_to((self.renderer.camera.offset_x,
                                 self.renderer.camera.offset_y))

    def _update_camera(self, dt: float) -> None:
        self.cam_zoom.update(dt)
        self.cam_offset.update(dt)
        if not (self.cam_zoom.is_settled and self.cam_offset.is_settled):
            self.renderer.camera.zoom = self.cam_zoom.value
            self.renderer.camera.offset_x = self.cam_offset.value[0]
            self.renderer.camera.offset_y = self.cam_offset.value[1]

    # ------------------------------------------------------------------
    # Frame update
    # ------------------------------------------------------------------

    def _update(self, dt: float) -> None:
        self.ui_manager.update(dt)
        self._update_message()
        self._update_camera(dt)
        self._update_animation_targets()
        self._advance_playback()

        for track in (self.node_active, self.node_selected, self.node_hover,
                      self.node_settle, self.edge_active):
            track.update(dt)
        self.token_travel.update(dt)

    def _update_animation_targets(self) -> None:
        """Point every animated value where it should be, then let it travel."""
        states = self.editor.automaton.states
        for track in (self.node_active, self.node_selected, self.node_hover,
                      self.node_settle):
            track.drop_missing(states)

        # `in`, not `==`: every state of the configuration is lit, which on a
        # nondeterministic run is the whole point -- the frontier spreading
        # across the diagram and collapsing again is the lesson, and lighting
        # one state chosen out of it would teach the opposite.
        current = self._current_configuration()
        for state in states:
            self.node_selected.set(state, 1.0 if state == self.editor.selection else 0.0)
            self.node_hover.set(state, 1.0 if state == self.editor.hover else 0.0)
            self.node_settle.set(state, 0.0, duration=self.theme.motion.normal)
            self.node_active.set(state, 1.0 if state in current else 0.0)

        if self.token_travel.is_settled:
            # Tokens exist only while travelling; parked on a node's rim they
            # covered the label and contradicted the glow that already marks
            # the current configuration.
            self.traversing_step = None
            for edge in self.editor.automaton.grouped_transitions():
                self.edge_active.set(f"{edge[0]}|{edge[1]}", 0.0,
                                     duration=self.theme.motion.normal)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def _symbol_index(self, symbol: Optional[str]) -> int:
        """Position in the sorted alphabet, for edge colouring.

        Keyed on position rather than the literal characters 'a' and 'b', which
        rendered the shipped {0,1} example entirely in one colour. An epsilon
        move is not in the alphabet and falls past the end of it, which is the
        right answer and not a coincidence: it is not one of the letters, so it
        does not take one of their colours.
        """
        alphabet = sorted(self.editor.automaton.alphabet)
        return alphabet.index(symbol) if symbol in alphabet else len(alphabet)

    def _loop_angles(self, positions: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Where each state's self-loop should point.

        Away from the average direction of its other edges, so a loop does not
        sit on top of an arrow arriving at the same node.
        """
        neighbours: Dict[str, List[Tuple[float, float]]] = {}
        for (source, target) in self.editor.automaton.grouped_transitions():
            if source == target:
                continue
            for a, b in ((source, target), (target, source)):
                if a in positions and b in positions:
                    neighbours.setdefault(a, []).append(positions[b])

        # The start marker enters the initial state from the left; treating it
        # as a phantom neighbour keeps that state's self-loop from being drawn
        # straight through the arrow.
        initial = self.editor.automaton.initial
        if initial in positions:
            point = positions[initial]
            neighbours.setdefault(initial, []).append((point[0] - 220.0, point[1]))

        return {
            state: geometry.quietest_direction(point, neighbours.get(state, []))
            for state, point in positions.items()
        }

    def _edge_paths(self, positions: Dict[str, Tuple[float, float]],
                    loop_angles: Optional[Dict[str, float]] = None):
        """The drawn path of every transition group, in world coordinates.

        Takes the loop angles rather than reading a field another method
        happened to fill in: relying on that ordering meant calling this alone
        raised KeyError.
        """
        if loop_angles is None:
            loop_angles = self._loop_angles(positions)
        radius = default_state_radius()
        automaton = self.editor.automaton
        layout = self.editor.layout
        grouped = automaton.grouped_transitions()
        paths = {}

        for (source, target) in grouped:
            if source not in positions or target not in positions:
                continue

            if source == target:
                paths[(source, target)] = geometry.self_loop_path(
                    positions[source], radius,
                    angle=loop_angles[source])
                continue

            arc = layout.arc_of(source, target)
            if not arc:
                arc = geometry.auto_arc(source, target,
                                        (target, source) in grouped)
            paths[(source, target)] = geometry.edge_path(
                positions[source], positions[target], radius, radius, arc)

        return paths

    def _build_scene(self) -> Scene:
        """Describe this frame as geometry, with no reference to pixels."""
        automaton = self.editor.automaton
        dead, unreachable, accepts_anything = self.editor.analysis()
        radius = default_state_radius()
        positions = self.editor.positions()

        self._loop_angle_cache = self._loop_angles(positions)
        scene = Scene()

        self.ui_manager.legend_dead = False
        self.ui_manager.legend_unreachable = False
        self.ui_manager.warn_no_accepting = not accepts_anything

        edge_paths = self._edge_paths(positions, self._loop_angle_cache)
        grouped = automaton.grouped_transitions()
        for edge, path in edge_paths.items():
            # Keyed sort and a spelled-out epsilon: an edge's symbol set may
            # hold None, which plain `sorted` raises on and `", ".join` cannot
            # write. Both would be a crash in the frame loop the first time
            # someone drew a move that reads nothing.
            symbols = sorted(grouped[edge], key=_label_order)
            label_at = None
            if edge[0] == edge[1]:
                label_at = geometry.self_loop_label_anchor(
                    positions[edge[0]], radius, self._loop_angle_cache[edge[0]])
            scene.edges.append(EdgeVisual(
                key=edge,
                path=path,
                label=", ".join(EPSILON_LABEL if s is None else s
                                for s in symbols),
                label_at=label_at,
                color_index=self._symbol_index(symbols[0]) if symbols else 0,
                active=self.edge_active.get(f"{edge[0]}|{edge[1]}"),
            ))

        for state in automaton.states:
            # Unreachable wins over dead. A state no word can enter cannot trap
            # anything, so "you can never get here" is the more useful fact.
            if state in unreachable:
                kind = NodeKind.UNREACHABLE
            elif state in dead:
                kind = NodeKind.DEAD
            else:
                kind = NodeKind.NORMAL

            self.ui_manager.legend_dead |= kind is NodeKind.DEAD
            self.ui_manager.legend_unreachable |= kind is NodeKind.UNREACHABLE

            scene.nodes.append(NodeVisual(
                id=state,
                position=positions[state],
                radius=radius,
                label=automaton.label_of(state),
                kind=kind,
                is_accept=state in automaton.accept,
                selected=self.node_selected.get(state),
                hover=self.node_hover.get(state),
                active=self.node_active.get(state),
                settle=self.node_settle.get(state),
            ))

        if automaton.initial and automaton.initial in positions:
            scene.start_marker = StartMarker(
                geometry.start_marker_path(positions[automaton.initial], radius))

        pending = self.editor.pending_source
        if pending is not None and pending in positions:
            world_mouse = self._world(pygame.mouse.get_pos())
            scene.ghost_edge = GhostEdge(
                path=geometry.edge_path(positions[pending], world_mouse,
                                        radius, 0.0, self.editor.pending_arc),
                label=_shown_symbol(self.ui_manager.selected_symbol),
                valid=(self.ui_manager.selected_symbol is None
                       or self.ui_manager.selected_symbol in automaton.alphabet),
            )

        scene.tokens = self._build_tokens(edge_paths)
        return scene

    def _build_tokens(self, edge_paths) -> List[TokenVisual]:
        """The read heads, one on every edge the machine is crossing.

        This is what replaces a text label teleporting between states: each
        marker moves along the real drawn path, at constant speed in screen
        distance rather than in curve parameter.

        **Several, when the move has several branches.** A nondeterministic
        machine takes every branch at once, so every branch gets a token: the
        picture is a set of tokens fanning out and rejoining. The alternative
        of putting one token on a branch picked by sort order was rejected
        outright -- it would draw a machine that made a choice, which is
        precisely the thing an NFA does not do, and the viewer would have no
        way to tell the drawn choice from a real one. Showing none and lighting
        only the destination states would be honest but loses the motion that
        makes a run legible at all, and the motion is why this exists.

        At rest there are none -- the lit states already say where the machine
        is, and a marker parked on a node covers its label.
        """
        if not self.execution_active or self.traversing_step is None:
            return []
        index = self.traversing_step + 1
        if index >= len(self.execution_trace):
            return []

        travel = self.token_travel.value
        trail_start = max(0.0, travel - 0.22)
        tokens: List[TokenVisual] = []
        for edge in self.execution_trace[index].entered_by:
            # A missing path is an edge the user deleted while the run was
            # still on screen. Skipped, not crashed on: the trace records what
            # the machine did, and the canvas is free to have moved on.
            path = edge_paths.get(edge)
            if not path:
                continue
            trail = [geometry.point_at(path, trail_start + (travel - trail_start) * i / 6)
                     for i in range(7)]
            tokens.append(TokenVisual(position=geometry.point_at(path, travel),
                                      radius=7.0, trail=trail, intensity=1.0))
        return tokens

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        """Compose one frame.

        Deliberately short: clear, hand a scene to the renderer, let the UI draw
        itself. Every colour and geometry decision lives elsewhere.
        """
        self.renderer.clear()
        self.renderer.draw_scene(self._build_scene())

        # The diagnostics panel reads the editor's cached analysis; feeding it
        # here keeps the UI a consumer of facts rather than a computer of them.
        self.ui_manager.diagnostics = self.editor.defects()
        # Same arrangement for the expression the machine denotes, with one
        # extra condition: it is only derived while the panel that shows it is
        # open, because state elimination is the one thing on this path that
        # can cost more than a frame.
        self.ui_manager.derived_pattern = (
            self._denoted_pattern() if self.ui_manager.wants_derived_pattern()
            else "")

        self.ui_manager.draw(self.editor.automaton, self.ui_manager.test_result,
                             self.animation_active, self.execution_active)
        self.ui_manager.draw_execution_status(
            self.execution_active, self.execution_step, self.execution_string,
            [position.configuration for position in self.execution_trace],
            self.run_result)
        self.ui_manager.draw_legend(self.editor.automaton)
        # Called even when inactive: the strip animates itself out.
        self.ui_manager.draw_string_visualization(
            self.execution_string, self.execution_step, self.run_result)

        # Help, menus and modals paint over every panel; the toast over
        # everything. When overlays lived inside draw(), the run panel and
        # tape strip painted straight across an open Save dialog.
        self.ui_manager.draw_overlays()

        if self.message_text:
            self._draw_message()

        pygame.display.flip()

    def _draw_message(self) -> None:
        """The transient toast, bottom right, fading over its last third."""
        from rendering import primitives

        palette = self.theme.palette
        elapsed = pygame.time.get_ticks() - self.message_timer
        remaining = max(0, self.message_duration - elapsed)
        fade = min(1.0, remaining / (self.message_duration * 0.34))

        surface = self.fonts.ui("body").render(self.message_text, True, palette.text)
        pad_x, pad_y = self.theme.space.md, self.theme.space.sm
        rect = pygame.Rect(0, 0, surface.get_width() + pad_x * 2,
                           surface.get_height() + pad_y * 2)
        rect.bottomright = (self.screen.get_width() - self.theme.space.lg,
                            self.screen.get_height() - self.theme.space.lg)

        primitives.translucent_panel(
            self.screen, rect, (*palette.panel_raised, int(238 * fade)),
            radius=self.theme.radius.md,
            border=(*palette.border, int(255 * fade)))
        surface.set_alpha(int(255 * fade))
        self.screen.blit(surface, surface.get_rect(center=rect.center))


if __name__ == "__main__":
    AutomatonSimulator().run()
