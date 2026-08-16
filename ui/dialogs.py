"""The modal dialogs: the file prompt, the confirmation, and add-symbol.

Every dialog is the same shape -- a dimmed screen, a centred raised box with a
title, and a row of buttons above a footer hint -- so the frame is drawn once
here and each dialog fills it in.

The rect helpers are pure functions of the window size, deliberately. They are
computed rather than recorded while drawing, so a dialog's buttons are live on
the frame it opens rather than the frame after.
"""

from typing import Optional, Tuple

import pygame

from rendering import primitives
from ui import widgets
from ui.layout_spec import LayoutSpec
from ui.widgets import Chrome

#: Dialog sizes, tall enough for a row of buttons above the footer hint.
CONFIRM_DIALOG_SIZE = (420, 172)
FILE_PROMPT_SIZE = (440, 200)

#: What the regular-expression prompt says about the syntax when it has no
#: complaint to make instead. Four operators and the one rule nobody guesses:
#: an empty pattern is the empty word, not a cancelled dialog.
REGEX_HINT = "| choice, * + ? repetition, () grouping. Empty means ε."

#: What the exercise prompt says about where a task comes from. The bundled
#: directory is named because the alternative is a blank field and a filename
#: nobody has been told, which is a dialog most people cancel.
EXERCISE_HINT = "An .fsx exercise. The extension and examples/ are assumed."


# ----------------------------------------------------------------------
# Where a dialog sits
# ----------------------------------------------------------------------

def modal_rect(screen_width: int, screen_height: int,
               width: int, height: int) -> pygame.Rect:
    """Where a centred dialog of this size sits.

    Computed rather than recorded while drawing, so a dialog's buttons are
    live on the frame it opens rather than the frame after -- the same rule
    the add-symbol dialog's buttons already followed.
    """
    return pygame.Rect(
        (screen_width - width) // 2,
        (screen_height - height) // 2,
        width,
        height,
    )


def confirm_rect(screen_width: int, screen_height: int) -> pygame.Rect:
    return modal_rect(screen_width, screen_height, *CONFIRM_DIALOG_SIZE)


def file_prompt_rect(screen_width: int, screen_height: int) -> pygame.Rect:
    return modal_rect(screen_width, screen_height, *FILE_PROMPT_SIZE)


def symbol_dialog_buttons(screen_width: int,
                          screen_height: int) -> Tuple[pygame.Rect, pygame.Rect]:
    """Cancel and Add rectangles for the add-symbol dialog.

    Computed rather than recorded during drawing, so the buttons are live on
    the frame the dialog opens instead of the frame after.
    """
    width, height = 300, 180
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    return (pygame.Rect(x + 50, y + 130, 80, 25),
            pygame.Rect(x + 170, y + 130, 80, 25))


# ----------------------------------------------------------------------
# The shared frame
# ----------------------------------------------------------------------

def draw_modal_frame(chrome: Chrome, width: int, height: int, title: str, *,
                     screen_width: int, screen_height: int) -> pygame.Rect:
    """Dim the screen and draw an empty centred dialog box, returning it."""
    palette = chrome.palette
    primitives.dim(chrome.screen, (0, 0, 0, 150 if palette.is_dark else 90))

    rect = modal_rect(screen_width, screen_height, width, height)
    primitives.elevated_panel(chrome.screen, rect, palette.panel_raised,
                              radius=chrome.radius.lg,
                              border=palette.border_strong,
                              shadow=palette.shadow, lift=7,
                              bevel_light=palette.bevel_light,
                              bevel_dark=palette.bevel_dark)

    title_surface = chrome.fonts.ui("heading").render(title, True, palette.text)
    chrome.screen.blit(title_surface, title_surface.get_rect(
        centerx=rect.centerx, y=rect.y + chrome.space.lg))

    return rect


def _button(chrome: Chrome, rect: pygame.Rect, label: str, *,
            pressed_rect: Optional[pygame.Rect], mouse_pos, accent: bool = False):
    """One dialog button, hovered and pressed against the manager's state."""
    widgets.button(chrome, rect, label, accent=accent,
                   hovered=rect.collidepoint(mouse_pos),
                   pressed=pressed_rect == rect)


# ----------------------------------------------------------------------
# The dialogs
# ----------------------------------------------------------------------

def draw_file_prompt(chrome: Chrome, *, layout: LayoutSpec,
                     screen_width: int, screen_height: int,
                     file_prompt_mode: Optional[str], file_prompt_text: str,
                     rename_target: str = "",
                     regex_error: str = "",
                     regex_error_at: Optional[int] = None,
                     pressed_rect: Optional[pygame.Rect] = None,
                     mouse_pos: Optional[Tuple[int, int]] = None) -> None:
    """Draw the filename prompt, or one of its rename and regex variants."""
    titles = {"save": "Save as", "load": "Load file",
              "exercise": "Open an exercise",
              "rename": f"Rename {rename_target}",
              "regex": "From a regular expression"}
    # Drawn at the size :func:`file_prompt_rect` reports, which is what
    # hit-testing uses. They were 160 and 200, so every button in this dialog
    # was drawn twenty pixels below the rectangle that answered the click and
    # only the bottom third of it could be pressed.
    rect = draw_modal_frame(chrome, *FILE_PROMPT_SIZE,
                            title=titles.get(file_prompt_mode or '', ""),
                            screen_width=screen_width,
                            screen_height=screen_height)

    palette = chrome.palette
    small = chrome.fonts.ui("small")
    budget = rect.width - chrome.space.lg * 2
    hints = {"rename": "A display label. Leave empty to use the state's id.",
             "regex": REGEX_HINT,
             "exercise": EXERCISE_HINT}
    # The hint stays put when there is also a complaint to make, rather than
    # being replaced by it: how to write one of these is exactly what somebody
    # who has just written one wrong needs in front of them.
    hint_text = hints.get(
        file_prompt_mode or "",
        "Relative to the project folder. '.json' is added if omitted.")
    hint = small.render(widgets.elide(small, hint_text, budget), True,
                        palette.text_muted)
    chrome.screen.blit(hint, (rect.x + chrome.space.lg, rect.y + 50))

    field = pygame.Rect(rect.x + chrome.space.lg, rect.y + 74,
                        rect.width - chrome.space.lg * 2, 34)
    primitives.panel(chrome.screen, field, palette.field,
                     radius=chrome.radius.md, border=palette.accent,
                     border_width=2)

    # Show the tail of the text so the caret stays visible on long paths.
    shown = file_prompt_text[-40:]
    mono = chrome.fonts.mono("input")
    text_surface = mono.render(shown, True, palette.text)
    text_rect = text_surface.get_rect(midleft=(field.left + 6, field.centery))
    chrome.screen.blit(text_surface, text_rect)

    if file_prompt_mode == "regex" and regex_error:
        # The complaint goes below the field, under the character it is about,
        # in the space the button row was already leaving empty. The caret is
        # drawn in the field's own face so the columns line up, and only while
        # the whole pattern is on screen -- the field shows the tail of a long
        # one, and a caret under the wrong character is worse than none.
        if regex_error_at is not None and shown == file_prompt_text:
            marker = mono.render("^", True, palette.warning)
            marker_x = text_rect.left + mono.size(file_prompt_text[:regex_error_at])[0]
            chrome.screen.blit(
                marker, (min(marker_x, field.right - marker.get_width()),
                         field.bottom - 3))
        chrome.screen.blit(
            small.render(widgets.elide(small, regex_error, budget), True,
                         palette.warning),
            (rect.x + chrome.space.lg, rect.y + 126))

    if pygame.time.get_ticks() % 1100 < 560:
        caret_x = min(text_rect.right + 2, field.right - 5)
        pygame.draw.line(chrome.screen, palette.accent,
                         (caret_x, field.top + 7), (caret_x, field.bottom - 7), 2)

    verbs = {"save": "Save", "load": "Load", "rename": "Rename",
             "regex": "Build", "exercise": "Open"}
    cancel, confirm = layout.confirm_buttons(rect)
    if mouse_pos is None:
        mouse_pos = pygame.mouse.get_pos()
    _button(chrome, cancel, "Cancel", pressed_rect=pressed_rect,
            mouse_pos=mouse_pos)
    _button(chrome, confirm, verbs.get(file_prompt_mode or '', "OK"), accent=True,
            pressed_rect=pressed_rect, mouse_pos=mouse_pos)

    footer = chrome.fonts.ui("small").render("Enter to confirm, Escape to cancel",
                                             True, palette.text_faint)
    chrome.screen.blit(footer, footer.get_rect(x=rect.x + chrome.space.lg,
                                               centery=cancel.centery))


def draw_confirm_dialog(chrome: Chrome, *, layout: LayoutSpec,
                        screen_width: int, screen_height: int,
                        confirm_intent: Optional[str], confirm_message: str,
                        pressed_rect: Optional[pygame.Rect] = None,
                        mouse_pos: Optional[Tuple[int, int]] = None) -> None:
    """Draw the yes/no confirmation dialog."""
    rect = draw_modal_frame(chrome, *CONFIRM_DIALOG_SIZE,
                            title="Unsaved changes",
                            screen_width=screen_width,
                            screen_height=screen_height)

    palette = chrome.palette
    message = chrome.fonts.ui("body").render(confirm_message, True,
                                             palette.text_muted)
    chrome.screen.blit(message, message.get_rect(centerx=rect.centerx,
                                                 y=rect.y + 56))

    # The verb, not "Yes": a button that names what it will do is one the
    # user can answer without re-reading the question above it.
    verbs = {"quit_after_confirm": "Quit",
             "load_after_confirm": "Discard",
             "new_after_confirm": "Discard"}
    cancel, confirm = layout.confirm_buttons(rect)
    if mouse_pos is None:
        mouse_pos = pygame.mouse.get_pos()
    _button(chrome, cancel, "Cancel", pressed_rect=pressed_rect,
            mouse_pos=mouse_pos)
    _button(chrome, confirm, verbs.get(confirm_intent or "", "Confirm"),
            accent=True, pressed_rect=pressed_rect, mouse_pos=mouse_pos)

    footer = chrome.fonts.ui("small").render(
        "Y confirms, N cancels", True, palette.text_faint)
    chrome.screen.blit(footer, footer.get_rect(x=rect.x + chrome.space.lg,
                                               centery=cancel.centery))


def draw_add_symbol_dialog(chrome: Chrome, *,
                           screen_width: int, screen_height: int,
                           new_symbol_input: str,
                           pressed_rect: Optional[pygame.Rect] = None,
                           mouse_pos: Optional[Tuple[int, int]] = None) -> None:
    """The add-symbol dialog, on the same modal frame as its siblings.

    It predated the elevation pass and looked like a different application
    next to the Save dialog: square corners, hard borders, flat buttons,
    and no dimmed backdrop despite being modal to the keyboard.
    """
    palette = chrome.palette
    rect = draw_modal_frame(chrome, 300, 180, "Add a symbol",
                            screen_width=screen_width,
                            screen_height=screen_height)

    hint = chrome.fonts.ui("small").render(
        "One printable character.", True, palette.text_muted)
    chrome.screen.blit(hint, (rect.x + 20, rect.y + 48))

    field = pygame.Rect(rect.x + 20, rect.y + 68, 260, 32)
    primitives.sunken_well(chrome.screen, field, palette.field,
                           radius=chrome.radius.md,
                           border=palette.accent,
                           well_shadow=palette.well_shadow)

    if new_symbol_input:
        glyph = chrome.fonts.mono("input").render(
            new_symbol_input, True, palette.text)
        chrome.screen.blit(glyph, glyph.get_rect(
            midleft=(field.left + 8, field.centery)))
        caret_x = field.left + 8 + glyph.get_width() + 2
    else:
        caret_x = field.left + 8

    if pygame.time.get_ticks() % 1100 < 560:
        pygame.draw.line(chrome.screen, palette.accent,
                         (caret_x, field.top + 6),
                         (caret_x, field.bottom - 6), 2)

    cancel, add = symbol_dialog_buttons(screen_width, screen_height)
    if mouse_pos is None:
        mouse_pos = pygame.mouse.get_pos()
    _button(chrome, cancel, "Cancel", pressed_rect=pressed_rect,
            mouse_pos=mouse_pos)
    _button(chrome, add, "Add", accent=True, pressed_rect=pressed_rect,
            mouse_pos=mouse_pos)

    footer = chrome.fonts.ui("small").render(
        "Enter to add, Escape to cancel", True, palette.text_faint)
    chrome.screen.blit(footer, footer.get_rect(
        centerx=rect.centerx, y=rect.bottom - 24))
