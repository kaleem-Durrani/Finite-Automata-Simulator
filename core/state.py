"""
State module for the Finite Automata Simulator.

This module contains the State class and related enums that represent
individual states in a finite automaton.
"""

from enum import Enum
from typing import Tuple


class StateType(Enum):
    """Enumeration for different types of states in the automaton."""
    NORMAL = "normal"      # Regular state
    ACCEPT = "accept"      # Accepting/final state
    DEAD_END = "dead_end"  # Dead end/trap state


class State:
    """
    Represents a single state in the finite automaton.
    
    Each state has a position, type, and various visual properties
    for rendering and interaction.
    """
    
    def __init__(self, state_id: str, position: Tuple[float, float]):
        """
        Initialize a new state.
        
        Args:
            state_id: Unique identifier for this state (e.g., "q0", "q1")
            position: (x, y) coordinates for the state's position
        """
        self.id = state_id
        self.position = list(position)  # [x, y] - mutable for dragging
        self.state_type = StateType.NORMAL
        self.radius = 30  # Visual radius for rendering
        
        # Visual state flags
        self.selected = False    # Whether this state is currently selected
        self.being_dragged = False  # Whether this state is being dragged
        self.hover = False       # Whether mouse is hovering over this state
        
    def contains_point(self, point: Tuple[float, float]) -> bool:
        """
        Check if a given point is inside this state's circular boundary.
        
        Args:
            point: (x, y) coordinates to test
            
        Returns:
            True if the point is inside the state, False otherwise
        """
        dx = point[0] - self.position[0]
        dy = point[1] - self.position[1]
        return (dx * dx + dy * dy) <= (self.radius * self.radius)
    
    def distance_to(self, other: 'State') -> float:
        """
        Calculate the distance to another state.
        
        Args:
            other: Another State object
            
        Returns:
            Euclidean distance between the two states
        """
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        return (dx * dx + dy * dy) ** 0.5
    
    def __str__(self) -> str:
        """String representation of the state."""
        return f"State({self.id}, {self.state_type.value}, {self.position})"
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return (f"State(id='{self.id}', type={self.state_type}, "
                f"pos={self.position}, selected={self.selected})")
