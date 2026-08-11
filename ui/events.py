"""What the interface asks the application to do.

One frozen dataclass per request, replacing a ``Dict[str, Any]`` bag whose keys
were strings and whose payloads were strings with separators packed into them.
State ids and coordinates were encoded as ``"straighten:2:q0q2"`` and
``"add_state:120.0,45.0"`` and taken apart again at the far end. Two whole
classes of defect came out of that:

* A separator that occurred inside a state id addressed the wrong thing. Ids
  are opaque and the file format is hand-editable, so ``a>b`` is a legal id and
  splitting on ``>`` silently straightened somebody else's edge. Here an id is
  a field, not a substring, and cannot be mistaken for punctuation.
* An action nobody handled did nothing at all, silently, because a dict lookup
  that misses is not an error. Twelve of the twenty-nine action names in the
  previous scheme had no handler. :func:`dispatch` raises instead, so an
  unhandled request fails in a test rather than in front of a user.

Purely internal interface state -- which field has focus, whether a panel is
folded, when a key began repeating -- is deliberately absent. It was never the
application's business, but it was being announced anyway and nothing was
listening.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Tuple, Type

from fsa.symbols import StateId, Symbol


class UiEvent:
    """Base class. Every event is a frozen dataclass carrying only data."""


class UnknownEventError(TypeError):
    """Raised when an event reaches a dispatcher with no handler for it."""


# ----------------------------------------------------------------------
# Files and chrome
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class SaveRequested(UiEvent):
    """The Save button. Opens the prompt; does not itself write anything."""


@dataclass(frozen=True)
class LoadRequested(UiEvent):
    pass


@dataclass(frozen=True)
class SaveToPath(UiEvent):
    path: str


@dataclass(frozen=True)
class LoadFromPath(UiEvent):
    path: str


@dataclass(frozen=True)
class ToggleTheme(UiEvent):
    pass


@dataclass(frozen=True)
class ToolSelected(UiEvent):
    tool: str


# ----------------------------------------------------------------------
# Alphabet
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class SymbolSelected(UiEvent):
    symbol: Symbol


@dataclass(frozen=True)
class SymbolAdded(UiEvent):
    """A symbol the dialog has already checked against the engine's rules."""
    symbol: Symbol


@dataclass(frozen=True)
class SymbolRejected(UiEvent):
    reason: str


# ----------------------------------------------------------------------
# Running a string
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class TestString(UiEvent):
    text: str


@dataclass(frozen=True)
class StepForward(UiEvent):
    pass


@dataclass(frozen=True)
class StepBack(UiEvent):
    pass


@dataclass(frozen=True)
class ToggleAnimation(UiEvent):
    pass


@dataclass(frozen=True)
class StopExecution(UiEvent):
    pass


# ----------------------------------------------------------------------
# Dialogs
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Confirmed(UiEvent):
    """A guarded action the user agreed to. ``intent`` names which one."""
    intent: str


@dataclass(frozen=True)
class PromptCancelled(UiEvent):
    pass


# ----------------------------------------------------------------------
# Editing the automaton
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class AddStateAt(UiEvent):
    """``position`` is in world coordinates, as a pair of floats -- not two
    numbers rendered into a string and parsed back."""
    position: Tuple[float, float]


@dataclass(frozen=True)
class DeleteState(UiEvent):
    state: StateId


@dataclass(frozen=True)
class ToggleAccept(UiEvent):
    state: StateId


@dataclass(frozen=True)
class SetInitial(UiEvent):
    state: Optional[StateId]


@dataclass(frozen=True)
class MakeTrap(UiEvent):
    state: StateId


@dataclass(frozen=True)
class RenamePrompt(UiEvent):
    state: StateId


@dataclass(frozen=True)
class RenameState(UiEvent):
    state: StateId
    label: str


@dataclass(frozen=True)
class RemoveTransition(UiEvent):
    source: StateId
    symbol: Symbol


@dataclass(frozen=True)
class StraightenEdge(UiEvent):
    source: StateId
    target: StateId


@dataclass(frozen=True)
class CompleteAutomaton(UiEvent):
    pass


@dataclass(frozen=True)
class MinimizeAutomaton(UiEvent):
    """Merge the states no word can tell apart, and re-lay the result out."""


@dataclass(frozen=True)
class TrimAutomaton(UiEvent):
    """Drop every state that cannot appear on an accepting run."""


# ----------------------------------------------------------------------
# Camera and messages
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class FitView(UiEvent):
    pass


@dataclass(frozen=True)
class FocusStates(UiEvent):
    states: Tuple[StateId, ...]


@dataclass(frozen=True)
class ShowMessage(UiEvent):
    text: str


Handler = Callable[[UiEvent], None]


def dispatch(events: Iterable[UiEvent],
             handlers: Mapping[Type[UiEvent], Handler]) -> None:
    """Give every event to its handler, or refuse to continue.

    Exact type lookup rather than isinstance: an event is handled by the
    handler registered for precisely its class, so adding a new event without
    a handler is a loud failure at the first press of the button that emits
    it, not a button that quietly does nothing.
    """
    for event in events:
        handler = handlers.get(type(event))
        if handler is None:
            raise UnknownEventError(
                f"no handler for {type(event).__name__}; "
                f"known: {sorted(h.__name__ for h in handlers)}")
        handler(event)


__all__ = [name for name in dir() if not name.startswith("_")]
