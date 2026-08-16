"""The exercise panel: the task, in view while the answer is being drawn.

A prompt read once in a dialog and then dismissed is a prompt that half a class
answers from memory of its first sentence. So an open exercise takes a slot in
the right-hand column beside the other facts about what is on screen, and folds
away like them when the canvas needs the room -- the header stays behind as the
notch saying an exercise is open at all.

Three things share the panel, in the order a student needs them:

* **the verdict**, when there is one, directly under the title. It is put above
  the prompt rather than below it because after a check the sentence naming the
  distinguishing word is the new information; the prompt has already been read
  once, and is still underneath it to be read again.
* **the prompt**, which is the task.
* **the examples**, which are the prompt made concrete.

The Check button is drawn last and anchored to the bottom of the body, so it is
on screen whatever the column does to the panel's height -- a task with no way
to hand it in would be worse than no panel. Everything above it flows top-down
into the space that leaves, and a prompt too long for the room loses its tail
rather than pushing the button off the panel.

Plain functions over a :class:`ui.widgets.Chrome`, every input an explicit
keyword argument. Nothing here reads the manager, or imports it: what the Check
button *does* is the application's business, so this module draws a rectangle
and reports where it landed, exactly as the diagnostics panel does with Fix.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pygame

from fsa.exercise import Exercise
from rendering import primitives
from ui import widgets
from ui.layout_spec import LayoutSpec
from ui.panels.column import panel_frame

# One wrapper for a width, not two -- the same function the diagnostics rows and
# the marking table's explanation are broken with.
from ui.panels.diagnostics import wrap
from ui.widgets import Chrome

#: Padding inside the body, and the step between two wrapped lines.
PAD = 12
LEADING = 15
#: The title row, which is one line of `small_strong` plus air beneath it.
TITLE_HEIGHT = 20
#: The gap between the panel's sections.
GAP = 8
#: The verdict badge: CORRECT or NOT YET, on the same footing as the test
#: panel's ACCEPTED and REJECTED.
BADGE_HEIGHT = 20

#: The Check button, the air beneath it, and therefore the strip at the bottom
#: of the body reserved for it. ``FOOTER_HEIGHT`` is derived rather than written
#: down: it was a round number once, and being two pixels short of the button
#: plus its gap cost the panel its last example row -- the measured height said
#: the line fitted and the drawing loop, testing against the same button from
#: the other end, said it did not.
CHECK_SIZE = (78, 26)
CHECK_PAD = 8
FOOTER_HEIGHT = CHECK_SIZE[1] + CHECK_PAD + GAP

#: How much of each thing the panel will show. The prompt gets the most because
#: it is the task; the examples get one line each and are a reminder, not a
#: list; the feedback gets four because
#: :attr:`fsa.exercise.Result.message` names a word, both verdicts and, when
#: the alphabets disagree, the clause explaining why the word looks foreign.
PROMPT_MAX_LINES = 10
FEEDBACK_MAX_LINES = 4
#: Example words shown per row, before the rest are summarised as "+n more".
EXAMPLES_SHOWN = 4

#: What the two verdicts are called. "NOT YET" rather than "WRONG": the panel is
#: reporting that a word still tells the two machines apart, which is a state of
#: the work and not a mark out of ten -- and the sentence beside it says exactly
#: which word to go and look at. The colour is the warning amber for the same
#: reason it is not the test panel's rejection red: a rejected *string* and a
#: wrong *answer* are different claims, and they must not look alike.
BADGES = {"correct": "CORRECT", "wrong": "NOT YET"}


def spell(word: str) -> str:
    """A word as this panel names it: quoted, or the empty word in words.

    The same spelling :mod:`fsa.exercise` gives its messages, so the examples
    and the verdict sentence beneath them talk about words the same way. It is
    written out again here rather than imported because it is private there,
    and because the reason for it is worth repeating: the empty word is the
    commonest counterexample in this whole feature, and a bare pair of quotes
    reads like a defect in the marker rather than a fact about the answer.
    """
    return "the empty word" if not word else repr(word)


def _listed(words: Sequence[str]) -> str:
    """A row of example words, cut to the ones that fit on one line."""
    shown = [spell(word) for word in words[:EXAMPLES_SHOWN]]
    extra = len(words) - len(shown)
    if extra > 0:
        shown.append(f"+{extra} more")
    return ", ".join(shown)


@dataclass(frozen=True)
class Content:
    """Every line this panel will draw, wrapped and counted before it draws.

    The column asks for the panel's height a frame before the body is painted,
    so measuring and drawing have to be the same walk over the same wrapped
    text. Making that walk a value both of them read is what stops the panel
    being laid out one size and painted another -- the defect the diagnostics
    panel already learned to design out.
    """

    title: str
    prompt: Tuple[str, ...]
    examples: Tuple[Tuple[str, str], ...]
    feedback: Tuple[str, ...]


def measure(chrome: Chrome, *, exercise: Optional[Exercise], message: str,
            width: int) -> Content:
    """Wrap everything the panel shows to a body of ``width`` pixels."""
    if exercise is None:
        return Content(title="", prompt=(), examples=(), feedback=())

    budget = width - PAD * 2
    prompt = wrap(chrome.fonts.ui("small"), exercise.prompt, budget,
                  limit=PROMPT_MAX_LINES)

    tiny = chrome.fonts.ui("tiny")
    examples: List[Tuple[str, str]] = []
    if exercise.accept_examples:
        examples.append(("accepts", _listed(exercise.accept_examples)))
    if exercise.reject_examples:
        examples.append(("rejects", _listed(exercise.reject_examples)))
    # Elided rather than wrapped: an example row is a reminder of what the
    # prompt already said, and a three-line one would crowd out the prompt.
    examples = [(label, widgets.elide(tiny, words, budget - 52))
                for label, words in examples]

    feedback = wrap(chrome.fonts.ui("small"), message, budget,
                    limit=FEEDBACK_MAX_LINES) if message else []

    return Content(title=exercise.name(), prompt=tuple(prompt),
                   examples=tuple(examples), feedback=tuple(feedback))


def content_height(content: Content) -> int:
    """How far below the top of the body the last line of text reaches.

    The one walk both :func:`body_height` and :func:`draw` are counted against,
    which is what makes "the panel asked for enough room" a property a test can
    state rather than something to be discovered by looking at it.
    """
    total = 4 + TITLE_HEIGHT
    if content.feedback:
        total += GAP + BADGE_HEIGHT + len(content.feedback) * LEADING
    total += GAP + len(content.prompt) * LEADING
    if content.examples:
        total += GAP + len(content.examples) * LEADING
    return total


def body_height(chrome: Chrome, *, exercise: Optional[Exercise], message: str,
                width: int) -> int:
    """How tall the body must be to show this exercise in full.

    Zero when no exercise is open, which is how the column knows to leave the
    panel out altogether rather than drawing an empty card.
    """
    if exercise is None:
        return 0
    return content_height(measure(chrome, exercise=exercise, message=message,
                                  width=width)) + FOOTER_HEIGHT


def check_button(body: pygame.Rect) -> pygame.Rect:
    """Where the Check button sits in a body of this size."""
    return pygame.Rect(body.right - PAD - CHECK_SIZE[0],
                       body.bottom - CHECK_SIZE[1] - CHECK_PAD, *CHECK_SIZE)


def ceiling(body: pygame.Rect) -> int:
    """The lowest a line of text may reach before the footer claims the space."""
    return check_button(body).top - GAP


def _badge(chrome: Chrome, position: Tuple[int, int], verdict: str) -> None:
    """The CORRECT / NOT YET pill, drawn in the verdict's own colour."""
    palette = chrome.palette
    label = BADGES.get(verdict, "")
    if not label:
        return
    colour = palette.success if verdict == "correct" else palette.warning
    font = chrome.fonts.ui("small_strong")
    text = font.render(label, True, palette.text_on_accent)
    rect = pygame.Rect(position[0], position[1],
                       text.get_width() + 14, text.get_height() + 4)
    primitives.panel(chrome.screen, rect, colour, radius=chrome.radius.sm)
    chrome.screen.blit(text, text.get_rect(center=rect.center))


def draw(chrome: Chrome, *, rect: pygame.Rect, exercise: Optional[Exercise],
         message: str, verdict: str, collapsed: bool, layout: LayoutSpec,
         pressed_rect: Optional[pygame.Rect] = None,
         mouse_pos: Optional[Tuple[int, int]] = None) -> Optional[pygame.Rect]:
    """Draw the open exercise, and return the Check button's rectangle.

    Args:
        chrome: Surface, theme and fonts.
        rect: The panel's slot in the right-hand column.
        exercise: The task on show. ``None`` draws nothing at all.
        message: The last verdict's sentence, or empty before the first check.
        verdict: ``"correct"``, ``"wrong"``, or empty for not yet checked.
        collapsed: Whether the panel is folded down to its header.
        layout: The window layout, which supplies the header strip.
        pressed_rect: The rect the mouse is held down on, for the pressed look.
        mouse_pos: The cursor, for hover. Defaults to the live position.

    Returns:
        Where the Check button was drawn, so the caller can hit-test exactly
        what is on screen -- or ``None`` on a frame that drew none, which is
        every frame the panel is folded away or absent.
    """
    if exercise is None:
        return None

    body = panel_frame(chrome, key="exercise", rect=rect, collapsed=collapsed,
                       layout=layout, mouse_pos=mouse_pos)
    if body is None:
        return None

    if mouse_pos is None:
        mouse_pos = pygame.mouse.get_pos()

    palette = chrome.palette
    screen = chrome.screen
    content = measure(chrome, exercise=exercise, message=message,
                      width=rect.width)

    # The footer first, so the action is never the thing that gets clipped.
    check = check_button(body)
    widgets.button(chrome, check, "Check", accent=True,
                   hovered=check.collidepoint(mouse_pos),
                   pressed=pressed_rect == check)
    floor = ceiling(body)

    text_x = body.x + PAD
    y = body.y + 4

    title = chrome.fonts.ui("small_strong")
    screen.blit(title.render(
        widgets.elide(title, content.title, body.width - PAD * 2), True,
        palette.text), (text_x, y))
    y += TITLE_HEIGHT

    if content.feedback:
        y += GAP
        _badge(chrome, (text_x, y), verdict)
        y += BADGE_HEIGHT
        # The sentence itself in the full text colour, not the muted one used
        # for the prompt: it is the one line on this panel written about the
        # machine actually on the canvas.
        font = chrome.fonts.ui("small")
        for line in content.feedback:
            if y + LEADING > floor:
                break
            screen.blit(font.render(line, True, palette.text), (text_x, y))
            y += LEADING

    y += GAP
    font = chrome.fonts.ui("small")
    for line in content.prompt:
        if y + LEADING > floor:
            break
        screen.blit(font.render(line, True, palette.text_muted), (text_x, y))
        y += LEADING

    if content.examples:
        y += GAP
        tiny = chrome.fonts.ui("tiny")
        for label, words in content.examples:
            if y + LEADING > floor:
                break
            colour = palette.success if label == "accepts" else palette.error
            screen.blit(tiny.render(label, True, colour), (text_x, y))
            screen.blit(tiny.render(words, True, palette.text_faint),
                        (text_x + 52, y))
            y += LEADING

    return check
