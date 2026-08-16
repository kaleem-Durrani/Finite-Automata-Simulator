"""The run panel: where a run is in the input, and the controls that move it.

Position counter, progress bar, current configuration, what the next step will
do, the four transport buttons, the keyboard hint and the step-speed slider -- one
stack, measured down from the panel header, because mixing the two directions
is how the hint and the playback row ended up drawn on top of each other.

Plain functions over a :class:`ui.widgets.Chrome` plus explicit keyword
arguments. Nothing here reads the manager. The rectangles come from the
:class:`LayoutSpec` the caller passes in, which is the same object hit-testing
reads, so a transport button cannot be drawn in one place and clicked in
another.

A position in a run is a *set* of states. That is not a concession to
nondeterminism bolted on afterwards: a deterministic run's configuration is a
set of one, so there is a single shape here and a single code path, and the
panel cannot report a machine standing somewhere it is not.
"""

from typing import Any, FrozenSet, Iterable, Optional, Sequence, Tuple

import pygame

from rendering import primitives
from ui.layout_spec import RUN_HINT_TOP, SPEED_MAX_MS, SPEED_MIN_MS, LayoutSpec
from ui.panels.column import panel_frame
from ui.widgets import Chrome, button, chevron
from ui.widgets import cross as cross_icon
from ui.widgets import pause as pause_icon
from ui.widgets import play as play_icon

#: What each verdict says on the panel's one detail line.
#:
#: Keyed on the verdict's string value, not the enum, because nothing under
#: ``ui/`` imports the engine. The wording matters more than the lookup: the
#: line used to be the enum name with its underscores stripped, so a run that
#: halted read "reject no transition" -- true of a DFA with a missing arrow,
#: and a lie about an NFA, which had plenty of arrows and simply watched every
#: branch die. "Ran out of moves" is true of both, so there is one sentence
#: rather than two that have to be kept in agreement.
_VERDICT_WORDING = {
    "accept": "accepted",
    "reject_non_accepting": "read to the end, not accepting",
    "reject_no_transition": "ran out of moves",
    "reject_symbol_not_in_alphabet": "symbol not in the alphabet",
    "no_initial_state": "no initial state, so no run",
}


def show_configuration(configuration: Iterable[str]) -> str:
    """A configuration as the panel says it.

    One state reads as itself -- a deterministic run stands *in q1*, and
    wrapping that in braces would make the ordinary case pay for the rare one.
    Two or more read as the set they are, ``{q0, q2}``, because that is what
    the machine is doing and there is no honest way to name one of them.

    Sorted, always: a frozenset iterates in an order that changes between
    processes, and a panel that reads differently on every run is a panel
    nobody can compare against a diagram (docs/LESSONS.md).
    """
    states = sorted(configuration)
    if not states:
        return "-"
    if len(states) == 1:
        return states[0]
    return "{" + ", ".join(states) + "}"


def verdict_line(run: Any) -> str:
    """How the run ended, in the few words the panel has room for.

    Public alongside :func:`show_configuration` for the same reason: these two
    are the panel's wording, the wording is a decision, and a decision that is
    only readable off a rendered surface is a decision nothing can test.
    """
    verdict = getattr(run, "verdict", None)
    value = str(getattr(verdict, "value", verdict or ""))
    text = _VERDICT_WORDING.get(value, value.replace("_", " "))

    # The symbol that stopped it, when one did. "Ran out of moves" without
    # saying on what leaves the user counting characters on the tape.
    symbol = getattr(run, "offending_symbol", None)
    if symbol is not None:
        text = f"{text}: '{symbol}' at {getattr(run, 'stopped_at', '?')}"
    return text


def draw_run_panel(chrome: Chrome, *,
                   panel: Optional[pygame.Rect],
                   layout: LayoutSpec,
                   execution_active: bool,
                   execution_step: int,
                   configurations: Sequence[FrozenSet[str]],
                   run: Optional[Any] = None,
                   animation_active: bool = False,
                   animation_speed: int = 1000,
                   slider: Optional[pygame.Rect] = None,
                   collapsed: bool = False,
                   pressed_rect: Optional[pygame.Rect] = None,
                   mouse_pos: Optional[Tuple[int, int]] = None) -> None:
    """
    Draw the execution trace panel.

    Positions are counted in *transitions taken*, against the length of the
    run. The old panel counted the current index against the length of the
    input string, so it reported "Step 3/5" for a run that had halted after
    two symbols, and never said why it stopped.

    Args:
        chrome: Surface, theme and fonts.
        panel: The run panel's slot in the right-hand column, or None on a
            frame where the column has no run panel in it.
        execution_active: Whether execution visualization is active
        execution_step: Position in the run
        configurations: Where the machine stood at each position -- a set of
            states per position, a set of one for a deterministic run.
        run: The engine's record of the run, if there is one. Either
            simulator's; only ``steps``, ``verdict``, ``offending_symbol`` and
            ``stopped_at`` are read, and both records carry all four.
        animation_active: Whether playback is running, which decides whether
            the middle transport button offers play or pause.
        animation_speed: Milliseconds per step, for the slider.
        slider: The speed slider's rectangle, or None when there is nowhere to
            put it. The playback controls belong to a run, so they exist only
            while one does.
        collapsed: Whether the panel is folded down to its header.
        layout: The window layout; supplies the header and the transport row.
        pressed_rect: The rect the mouse is currently held down on, or None.
            A button matching it draws pressed rather than raised.
        mouse_pos: The cursor position, for hover. Defaults to the live
            position when the caller has none to hand.
    """
    if not execution_active:
        return
    if panel is None:
        return

    if mouse_pos is None:
        mouse_pos = pygame.mouse.get_pos()

    palette = chrome.palette
    body = panel_frame(chrome, key="run", rect=panel, collapsed=collapsed,
                       layout=layout, mouse_pos=mouse_pos)
    if body is None:
        return

    x = body.x + chrome.space.md
    y = body.y + 2

    total_steps = max(0, len(configurations) - 1)
    position = f"{execution_step} / {total_steps}"
    pos_surface = chrome.fonts.ui("small_strong").render(position, True,
                                                         palette.text_muted)
    chrome.screen.blit(pos_surface, (body.right - chrome.space.md
                                     - pos_surface.get_width(), y))

    # Progress bar across the run.
    track = pygame.Rect(x, y + 20, body.width - chrome.space.md * 2, 4)
    primitives.panel(chrome.screen, track, palette.control, radius=2)
    if total_steps:
        done = pygame.Rect(track.x, track.y,
                           int(track.width * execution_step / total_steps), 4)
        primitives.panel(chrome.screen, done, palette.accent, radius=2)

    current = (configurations[execution_step]
               if execution_step < len(configurations) else frozenset())
    state_line = chrome.fonts.ui("body_strong").render(
        f"in {show_configuration(current)}", True, palette.text)
    chrome.screen.blit(state_line, (x, y + 32))

    steps = getattr(run, "steps", ()) or ()
    if execution_step < len(steps):
        # The destination is read out of ``configurations`` rather than off the
        # step, because a DFA step names one state and an NFA step names a set.
        # The caller has already made those the same shape; asking the step
        # would put an isinstance check in a panel.
        detail = (f"next: read '{steps[execution_step].symbol}' to "
                  f"{show_configuration(configurations[execution_step + 1])}")
    elif getattr(run, "verdict", None) is not None:
        detail = verdict_line(run)
    else:
        detail = "run complete"
    chrome.screen.blit(
        chrome.fonts.ui("small").render(detail, True, palette.text_muted),
        (x, y + 54))

    # Transport. Every one of these was keyboard-only, so the panel listed
    # four commands it gave the user no way to issue.
    back, play, forward, stop = layout.run_buttons(panel)
    at_start = execution_step <= 0
    at_end = execution_step >= total_steps

    button(chrome, back, "", hovered=back.collidepoint(mouse_pos),
           pressed=pressed_rect == back)
    chevron(chrome, back, palette.text_faint if at_start else palette.text,
            pointing="left")

    button(chrome, play, "", accent=True, hovered=play.collidepoint(mouse_pos),
           pressed=pressed_rect == play)
    if animation_active:
        pause_icon(chrome, play, palette.text_on_accent)
    else:
        play_icon(chrome, play, palette.text_on_accent)

    button(chrome, forward, "", hovered=forward.collidepoint(mouse_pos),
           pressed=pressed_rect == forward)
    chevron(chrome, forward, palette.text_faint if at_end else palette.text,
            pointing="right")

    button(chrome, stop, "", hovered=stop.collidepoint(mouse_pos),
           pressed=pressed_rect == stop)
    cross_icon(chrome, stop, palette.error)

    draw_playback_controls(chrome, slider=slider,
                           animation_speed=animation_speed)

    hint = "← → step   Tab play   Esc stop"
    chrome.screen.blit(
        chrome.fonts.ui("small").render(hint, True, palette.text_faint),
        (x, body.y + RUN_HINT_TOP))


def draw_playback_controls(chrome: Chrome, *, slider: Optional[pygame.Rect],
                           animation_speed: int) -> None:
    """The step-speed slider, inside the run panel.

    This used to live in the status panel and was drawn whether or not
    anything was running -- a "Paused" dot and a speed slider for an
    animation that did not exist, which is most of what made that panel
    look half empty. The dot is gone as well: the play button shows
    whether playback is running, so a second indicator saying the same
    thing is just something else to keep in step.

    Args:
        chrome: Surface, theme and fonts.
        slider: The slider's rectangle, or None when the run panel is absent
            or collapsed and there is nowhere to draw it.
        animation_speed: Milliseconds per step, between ``SPEED_MIN_MS`` and
            ``SPEED_MAX_MS``.
    """
    palette = chrome.palette
    if slider is None:
        return

    speed_ratio = (animation_speed - SPEED_MIN_MS) / (SPEED_MAX_MS - SPEED_MIN_MS)
    speed_ratio = max(0.0, min(1.0, speed_ratio))
    chrome.screen.blit(
        chrome.fonts.ui("small").render(f"{animation_speed} ms", True,
                                        palette.text_faint),
        (slider.right + 12, slider.y))

    # Track, filled portion, then handle, following the slid panel.
    track = pygame.Rect(slider.x, slider.centery - 2, slider.width, 4)
    primitives.panel(chrome.screen, track, palette.control, radius=2)
    filled = pygame.Rect(track.x, track.y, int(track.width * speed_ratio), 4)
    primitives.panel(chrome.screen, filled, palette.accent, radius=2)

    handle_x = slider.x + speed_ratio * slider.width
    primitives.filled_circle(chrome.screen, (handle_x, slider.centery), 7,
                             palette.panel_raised)
    primitives.ring(chrome.screen, (handle_x, slider.centery), 7, 2,
                    palette.accent)
