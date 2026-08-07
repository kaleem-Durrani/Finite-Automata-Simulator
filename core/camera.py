"""
Camera module for viewport management.

This module provides camera functionality for panning and zooming
the view of the automaton workspace.
"""

from typing import Tuple


class Camera:
    """
    Handles viewport transformations for pan and zoom functionality.
    
    The camera system allows users to navigate large automata by
    panning (moving the view) and zooming (scaling the view).
    """
    
    def __init__(self, screen_width: int, screen_height: int):
        """
        Initialize the camera.
        
        Args:
            screen_width: Width of the screen/window
            screen_height: Height of the screen/window
        """
        # Camera position and zoom
        self.offset_x = 0.0  # Horizontal offset (pan)
        self.offset_y = 0.0  # Vertical offset (pan)
        self.zoom = 1.0      # Zoom level (1.0 = normal, >1.0 = zoomed in)
        
        # Screen dimensions
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Zoom constraints
        self.min_zoom = 0.1  # Maximum zoom out
        self.max_zoom = 5.0  # Maximum zoom in
        
    def world_to_screen(self, world_pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        Convert world coordinates to screen coordinates.
        
        World coordinates are the actual positions of objects in the automaton.
        Screen coordinates are where they appear on the display after camera
        transformations.
        
        Args:
            world_pos: (x, y) position in world space
            
        Returns:
            (x, y) position in screen space
        """
        x, y = world_pos
        screen_x = (x + self.offset_x) * self.zoom
        screen_y = (y + self.offset_y) * self.zoom
        return screen_x, screen_y
    
    def screen_to_world(self, screen_pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        Convert screen coordinates to world coordinates.
        
        This is the inverse of world_to_screen, used for determining
        where the user clicked in world space.
        
        Args:
            screen_pos: (x, y) position in screen space
            
        Returns:
            (x, y) position in world space
        """
        x, y = screen_pos
        world_x = x / self.zoom - self.offset_x
        world_y = y / self.zoom - self.offset_y
        return world_x, world_y
    
    def pan(self, dx: float, dy: float):
        """
        Pan the camera by the given offset.
        
        Panning moves the view without changing the zoom level.
        
        Args:
            dx: Horizontal movement in screen pixels
            dy: Vertical movement in screen pixels
        """
        self.offset_x += dx / self.zoom
        self.offset_y += dy / self.zoom
    
    def zoom_at(self, screen_pos: Tuple[float, float], zoom_factor: float):
        """
        Zoom at a specific screen position.
        
        This ensures that the point under the mouse cursor stays
        in the same place when zooming, providing intuitive zoom behavior.
        
        Args:
            screen_pos: (x, y) position to zoom towards
            zoom_factor: Multiplier for zoom (>1.0 = zoom in, <1.0 = zoom out)
        """
        # Convert screen position to world coordinates before zoom
        world_pos = self.screen_to_world(screen_pos)
        
        # Apply zoom with constraints
        new_zoom = max(self.min_zoom, min(self.max_zoom, self.zoom * zoom_factor))
        
        if new_zoom != self.zoom:
            self.zoom = new_zoom
            
            # Adjust offset to keep the world position under the mouse
            new_screen_pos = self.world_to_screen(world_pos)
            dx = screen_pos[0] - new_screen_pos[0]
            dy = screen_pos[1] - new_screen_pos[1]
            self.offset_x += dx / self.zoom
            self.offset_y += dy / self.zoom
    
    def reset(self):
        """
        Reset camera to default position and zoom.
        
        This centers the view and sets zoom to 1.0.
        """
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.zoom = 1.0
    
    def center_on_point(self, world_pos: Tuple[float, float]):
        """
        Center the camera on a specific world position.
        
        Args:
            world_pos: (x, y) position in world space to center on
        """
        # Calculate offset needed to center the point
        center_screen_x = self.screen_width / 2
        center_screen_y = self.screen_height / 2
        
        # Convert desired center to world coordinates
        desired_world_center = self.screen_to_world((center_screen_x, center_screen_y))
        
        # Adjust offset
        self.offset_x += world_pos[0] - desired_world_center[0]
        self.offset_y += world_pos[1] - desired_world_center[1]
    
    def get_visible_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get the world coordinates of the visible area.
        
        Returns:
            Tuple of (min_x, min_y, max_x, max_y) in world coordinates
        """
        top_left = self.screen_to_world((0, 0))
        bottom_right = self.screen_to_world((self.screen_width, self.screen_height))
        
        return (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
    
    def is_point_visible(self, world_pos: Tuple[float, float], margin: float = 50) -> bool:
        """
        Check if a world position is visible on screen.
        
        Args:
            world_pos: (x, y) position in world space
            margin: Extra margin around screen edges
            
        Returns:
            True if the point is visible, False otherwise
        """
        screen_pos = self.world_to_screen(world_pos)
        return (-margin <= screen_pos[0] <= self.screen_width + margin and
                -margin <= screen_pos[1] <= self.screen_height + margin)
