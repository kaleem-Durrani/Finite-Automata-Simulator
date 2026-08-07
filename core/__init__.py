"""
Core module for the Finite Automata Simulator.

This module contains the fundamental classes for representing
and manipulating finite automata.
"""

from .state import State, StateType
from .dfa import DFA
from .camera import Camera

__all__ = ['State', 'StateType', 'DFA', 'Camera']
