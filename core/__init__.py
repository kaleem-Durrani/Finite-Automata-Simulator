"""
Core module for the Finite Automata Simulator.

This module contains the fundamental classes for representing
and manipulating finite automata.
"""

from .camera import Camera
from .dfa import DFA
from .state import State, StateType

__all__ = ['State', 'StateType', 'DFA', 'Camera']
