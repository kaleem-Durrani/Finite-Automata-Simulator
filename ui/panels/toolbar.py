"""The top toolbar: the title, the tool group, and the file/theme group.

Plain functions over a :class:`ui.widgets.Chrome` plus explicit keyword
arguments. Nothing here reads the manager, so the toolbar can be drawn -- and
tested -- against a bare surface, a theme and a layout.

The rectangles are not computed here. They come from the :class:`LayoutSpec`
the caller passes in, which is the same object hit-testing reads, so a button
cannot be drawn in one place and clicked in another.
"""

from typing import Optional, Tuple

import pygame

from ui.layout_spec import LayoutSpec
from ui.widgets import Chrome, arrow, button, hand


def draw_toolbar(chrome: Chrome, *,
                 layout: LayoutSpec,
                 tool: str,
                 show_help: bool,
                 pressed_rect: Optional[pygame.Rect] = None,
                 mouse_pos: Optional[Tuple[int, int]] = None) -> None:
    """Draw the main toolbar at the top of the screen.

    Args:
        chrome: Surface, theme and fonts.
        layout: The window layout; supplies the toolbar band and every button.
        tool: The active tool, ``"pointer"``, ``"pan"`` or ``"transition"``.
            Exclusive by construction: there is one name, so turning a tool on
            cannot leave another one also on.
        show_help: Whether the help panel is open, which lights the Help
            button the way an active tool lights its own.
        pressed_rect: The rect the mouse is currently held down on, or None.
            A button matching it draws pressed rather than raised.
        mouse_pos: The cursor position, for hover. Defaults to the live
            position when the caller has none to hand.
    """
    if mouse_pos is None:
        mouse_pos = pygame.mouse.get_pos()

    palette = chrome.palette
    toolbar = layout.toolbar
    pygame.draw.rect(chrome.screen, palette.panel, toolbar)
    pygame.draw.line(chrome.screen, palette.border,
                     (0, toolbar.bottom - 1), (toolbar.right, toolbar.bottom - 1))

    title = chrome.fonts.ui("title").render("Finite Automata", True, palette.text)
    chrome.screen.blit(title, title.get_rect(
        midleft=(chrome.space.lg, toolbar.centery)))

    subtitle = chrome.fonts.ui("small").render(
        "simulator", True, palette.text_faint)
    chrome.screen.blit(subtitle, subtitle.get_rect(
        midleft=(chrome.space.lg + title.get_width() + 8,
                 toolbar.centery + 1)))

    transition_tool = tool == "transition"
    pan_tool = tool == "pan"

    transition = layout.transition_button
    button(chrome, transition, "", active=transition_tool,
           hovered=transition.collidepoint(mouse_pos),
           pressed=pressed_rect == transition)
    arrow(chrome, transition, palette.text_on_accent if transition_tool
          else palette.text_muted)

    pan = layout.pan_button
    button(chrome, pan, "", active=pan_tool,
           hovered=pan.collidepoint(mouse_pos),
           pressed=pressed_rect == pan)
    hand(chrome, pan, palette.text_on_accent if pan_tool
         else palette.text_muted)

    theme_button = layout.theme_button
    button(chrome, theme_button, "Light" if chrome.theme.is_dark else "Dark",
           hovered=theme_button.collidepoint(mouse_pos),
           pressed=pressed_rect == theme_button)
    load = layout.load_button
    button(chrome, load, "Load", hovered=load.collidepoint(mouse_pos),
           pressed=pressed_rect == load)
    save = layout.save_button
    button(chrome, save, "Save", hovered=save.collidepoint(mouse_pos),
           pressed=pressed_rect == save)
    help_button = layout.help_button
    button(chrome, help_button, "Help", active=show_help,
           hovered=help_button.collidepoint(mouse_pos),
           pressed=pressed_rect == help_button)
