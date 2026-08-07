"""
Renderer module for drawing automaton components.

This module handles all rendering operations including states, transitions,
and UI elements with proper resource management and optimization.
"""

import math
from typing import Tuple

import pygame

from core.camera import Camera
from core.state import State, StateType


class Renderer:
    """
    Handles all rendering operations with proper resource management.
    
    The renderer manages drawing of states, transitions, and provides
    efficient rendering with camera transformations.
    """
    
    def __init__(self, screen: pygame.Surface):
        """
        Initialize the renderer.
        
        Args:
            screen: Pygame surface to render to
        """
        self.screen = screen
        self.camera = Camera(screen.get_width(), screen.get_height())
        
        # Load and cache fonts to avoid repeated loading
        self.fonts = {
            'small': pygame.font.Font(None, 16),
            'medium': pygame.font.Font(None, 24),
            'large': pygame.font.Font(None, 32)
        }
        
        # Color palette for consistent theming
        self.colors = {
            'background': (240, 240, 240),      # Light gray background
            'state_normal': (100, 150, 255),    # Blue for normal states
            'state_accept': (100, 255, 100),    # Green for accept states
            'state_dead_end': (255, 100, 100),  # Red for dead end states
            'state_selected': (255, 255, 100),  # Yellow for selected states
            'state_hover': (200, 200, 255),     # Light blue for hover
            'state_executing': (255, 150, 0),   # Orange for execution highlight
            'transition_a': (255, 50, 50),      # Red for 'a' transitions
            'transition_b': (150, 50, 255),     # Purple for 'b' transitions
            'transition_other': (50, 50, 50),   # Dark gray for other symbols
            'transition_creating': (100, 100, 100),  # Gray for transition being created
            'text': (0, 0, 0),                  # Black text
            'ui_background': (220, 220, 220),   # UI panel background
            'ui_border': (100, 100, 100)        # UI border color
        }
        
    def clear(self):
        """Clear the screen with background color."""
        self.screen.fill(self.colors['background'])
    
    def draw_state(self, state: State, is_executing: bool = False):
        """
        Draw a single state with appropriate styling.
        
        Args:
            state: State object to draw
            is_executing: Whether this state is currently being executed
        """
        screen_pos = self.camera.world_to_screen(state.position)
        radius = int(state.radius * self.camera.zoom)
        
        # Don't draw very small states (performance optimization)
        if radius < 2:
            return
            
        # Choose color based on state type and status
        if is_executing:
            color = self.colors['state_executing']
        elif state.selected:
            color = self.colors['state_selected']
        elif state.hover:
            color = self.colors['state_hover']
        elif state.state_type == StateType.ACCEPT:
            color = self.colors['state_accept']
        elif state.state_type == StateType.DEAD_END:
            color = self.colors['state_dead_end']
        else:
            color = self.colors['state_normal']
        
        # Draw main circle
        center = (int(screen_pos[0]), int(screen_pos[1]))
        pygame.draw.circle(self.screen, color, center, radius)
        pygame.draw.circle(self.screen, (0, 0, 0), center, radius, 2)
        
        # Draw accept state indicator (double circle)
        if state.state_type == StateType.ACCEPT:
            inner_radius = max(2, radius - 8)
            pygame.draw.circle(self.screen, (0, 0, 0), center, inner_radius, 2)
        
        # Draw state label if state is large enough
        if radius > 10:
            text_surface = self.fonts['small'].render(state.id, True, self.colors['text'])
            text_rect = text_surface.get_rect(center=center)
            self.screen.blit(text_surface, text_rect)

    def draw_current_state_indicator(self, state_position: Tuple[float, float]):
        """Draw an arrow pointing to the current state during execution."""
        screen_pos = self.camera.world_to_screen(state_position)

        # Draw arrow pointing down to the state
        arrow_tip = (screen_pos[0], screen_pos[1] - 60)
        arrow_base = (screen_pos[0], screen_pos[1] - 80)
        arrow_left = (screen_pos[0] - 10, screen_pos[1] - 70)
        arrow_right = (screen_pos[0] + 10, screen_pos[1] - 70)

        # Draw arrow
        pygame.draw.line(self.screen, (255, 255, 0), arrow_base, arrow_tip, 3)
        pygame.draw.line(self.screen, (255, 255, 0), arrow_tip, arrow_left, 3)
        pygame.draw.line(self.screen, (255, 255, 0), arrow_tip, arrow_right, 3)

        # Draw "Current" label
        label_surface = self.fonts['small'].render("Current", True, (255, 255, 0))
        label_rect = label_surface.get_rect(center=(screen_pos[0], screen_pos[1] - 95))

        # Draw background for label
        bg_rect = label_rect.inflate(10, 4)
        pygame.draw.rect(self.screen, (0, 0, 0), bg_rect)
        pygame.draw.rect(self.screen, (255, 255, 0), bg_rect, 1)

        self.screen.blit(label_surface, label_rect)
    
    def draw_arrow(self, start_pos: Tuple[float, float], end_pos: Tuple[float, float], 
                   color: Tuple[int, int, int], label: str = "", is_self_loop: bool = False,
                   arc_offset: float = 0.0):
        """
        Draw an arrow between two points with optional label and arc.
        
        Args:
            start_pos: Starting position in world coordinates
            end_pos: Ending position in world coordinates
            color: RGB color tuple for the arrow
            label: Text label to display on the arrow
            is_self_loop: Whether this is a self-referencing transition
            arc_offset: Offset for creating curved arrows (positive = curve up)
        """
        start_screen = self.camera.world_to_screen(start_pos)
        end_screen = self.camera.world_to_screen(end_pos)
        
        if is_self_loop:
            self._draw_self_loop(start_screen, color, label)
            return
        
        # Calculate arrow properties
        dx = end_screen[0] - start_screen[0]
        dy = end_screen[1] - start_screen[1]
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance < 1:  # Too short to draw
            return
            
        # Normalize direction
        dx /= distance
        dy /= distance
        
        # Adjust start and end points to be on circle edges
        radius = 30 * self.camera.zoom
        start_adjusted = (start_screen[0] + dx * radius, start_screen[1] + dy * radius)
        end_adjusted = (end_screen[0] - dx * radius, end_screen[1] - dy * radius)
        
        # Apply arc offset for curved arrows
        if abs(arc_offset) > 0.01:
            self._draw_curved_arrow(start_adjusted, end_adjusted, color, label, arc_offset)
        else:
            self._draw_straight_arrow(start_adjusted, end_adjusted, color, label)
    
    def _draw_straight_arrow(self, start: Tuple[float, float], end: Tuple[float, float],
                           color: Tuple[int, int, int], label: str):
        """Draw a straight arrow between two points."""
        # Draw line
        pygame.draw.line(self.screen, color, start, end, 2)
        
        # Draw arrowhead
        self._draw_arrowhead(start, end, color)
        
        # Draw label at midpoint
        if label and self.camera.zoom > 0.5:
            self._draw_arrow_label(start, end, label)
    
    def _draw_curved_arrow(self, start: Tuple[float, float], end: Tuple[float, float],
                          color: Tuple[int, int, int], label: str, arc_offset: float):
        """Draw a curved arrow using quadratic Bezier curve."""
        # Calculate control point for the curve
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        # Perpendicular offset for curve
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx * dx + dy * dy)
        
        if length > 0:
            # Perpendicular vector
            perp_x = -dy / length
            perp_y = dx / length
            
            # Control point with arc offset
            control_x = mid_x + perp_x * arc_offset * self.camera.zoom
            control_y = mid_y + perp_y * arc_offset * self.camera.zoom
            
            # Draw curve using multiple line segments
            points = []
            num_segments = max(10, int(length / 10))
            
            for i in range(num_segments + 1):
                t = i / num_segments
                # Quadratic Bezier formula
                x = (1-t)**2 * start[0] + 2*(1-t)*t * control_x + t**2 * end[0]
                y = (1-t)**2 * start[1] + 2*(1-t)*t * control_y + t**2 * end[1]
                points.append((x, y))
            
            # Draw the curve
            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 2)
                
                # Draw arrowhead at the end
                if len(points) >= 2:
                    second_last = points[-2]
                    last = points[-1]
                    self._draw_arrowhead(second_last, last, color)
                
                # Draw label at curve midpoint
                if label and self.camera.zoom > 0.5:
                    mid_point = points[len(points) // 2]
                    self._draw_label_at_point(mid_point, label)
    
    def _draw_arrowhead(self, start: Tuple[float, float], end: Tuple[float, float],
                       color: Tuple[int, int, int]):
        """Draw an arrowhead at the end point."""
        arrow_size = 10 * self.camera.zoom
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx * dx + dy * dy)
        
        if length > 0:
            dx /= length
            dy /= length
            
            angle = math.atan2(dy, dx)
            
            arrow_points = [
                end,
                (end[0] - arrow_size * math.cos(angle - 0.5),
                 end[1] - arrow_size * math.sin(angle - 0.5)),
                (end[0] - arrow_size * math.cos(angle + 0.5),
                 end[1] - arrow_size * math.sin(angle + 0.5))
            ]
            pygame.draw.polygon(self.screen, color, arrow_points)

    def _draw_arrow_label(self, start: Tuple[float, float], end: Tuple[float, float], label: str):
        """Draw a label at the midpoint of an arrow."""
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2

        # Offset label slightly from line
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx * dx + dy * dy)

        if length > 0:
            # Perpendicular offset
            offset_x = -dy / length * 15
            offset_y = dx / length * 15

            label_pos = (mid_x + offset_x, mid_y + offset_y)
            self._draw_label_at_point(label_pos, label)

    def _draw_label_at_point(self, pos: Tuple[float, float], label: str):
        """Draw a text label at a specific point with background."""
        text_surface = self.fonts['small'].render(label, True, self.colors['text'])
        text_rect = text_surface.get_rect(center=(int(pos[0]), int(pos[1])))

        # Draw background for better readability
        bg_rect = text_rect.inflate(4, 2)
        pygame.draw.rect(self.screen, (255, 255, 255, 200), bg_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), bg_rect, 1)

        self.screen.blit(text_surface, text_rect)

    def _draw_self_loop(self, pos: Tuple[float, float], color: Tuple[int, int, int], label: str):
        """Draw a self-loop arrow above a state."""
        radius = 30 * self.camera.zoom
        loop_radius = 20 * self.camera.zoom

        # Draw arc above the state
        center_x = pos[0]
        center_y = pos[1] - radius - loop_radius

        # Draw the loop as a circle
        pygame.draw.circle(self.screen, color, (int(center_x), int(center_y)), int(loop_radius), 2)

        # Draw arrowhead
        arrow_size = 8 * self.camera.zoom
        arrow_pos = (center_x + loop_radius * 0.7, center_y - loop_radius * 0.7)
        arrow_points = [
            arrow_pos,
            (arrow_pos[0] - arrow_size, arrow_pos[1] - arrow_size * 0.5),
            (arrow_pos[0] - arrow_size, arrow_pos[1] + arrow_size * 0.5)
        ]
        pygame.draw.polygon(self.screen, color, arrow_points)

        # Draw label
        if label and self.camera.zoom > 0.5:
            label_pos = (center_x, center_y - loop_radius - 15)
            self._draw_label_at_point(label_pos, label)

    def draw_initial_state_arrow(self, state_pos: Tuple[float, float]):
        """
        Draw the arrow indicating the initial state.

        Args:
            state_pos: Position of the initial state in world coordinates
        """
        start_world = (state_pos[0] - 60, state_pos[1])
        end_world = (state_pos[0] - 35, state_pos[1])

        self.draw_arrow(start_world, end_world, (0, 0, 255), "start")

    def update_screen_size(self, width: int, height: int):
        """
        Update the renderer when screen size changes.

        Args:
            width: New screen width
            height: New screen height
        """
        self.camera.screen_width = width
        self.camera.screen_height = height
