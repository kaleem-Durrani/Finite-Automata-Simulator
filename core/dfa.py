"""
DFA (Deterministic Finite Automaton) module.

This module contains the DFA class that represents the complete
finite automaton with states, transitions, and processing logic.
"""

import json
from typing import Dict, List, Optional, Tuple, Set
from .state import State, StateType


class DFA:
    """
    Represents a Deterministic Finite Automaton.
    
    The DFA manages states, transitions, alphabet, and provides
    string processing functionality with path tracking.
    """
    
    def __init__(self):
        """Initialize an empty DFA."""
        # Core automaton components
        self.states: Dict[str, State] = {}  # state_id -> State object
        self.transitions: Dict[str, Dict[str, str]] = {}  # state_id -> {symbol -> target_state_id}
        # New structure for grouped transitions: (from_state, to_state) -> {symbols, arc_offset}
        self.transition_groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.alphabet: Set[str] = set()  # Set of valid input symbols
        self.initial_state: Optional[str] = None  # ID of the initial state
        self.accept_states: Set[str] = set()  # Set of accepting state IDs
        self.dead_end_states: Set[str] = set()  # Set of dead end state IDs
        
        # Internal state management
        self._next_state_id = 0  # Counter for generating unique state IDs
    
    def add_state(self, position: Tuple[float, float]) -> str:
        """
        Add a new state to the automaton.
        
        Args:
            position: (x, y) coordinates for the new state
            
        Returns:
            The ID of the newly created state
        """
        state_id = f"q{self._next_state_id}"
        self._next_state_id += 1
        
        state = State(state_id, position)
        self.states[state_id] = state
        self.transitions[state_id] = {}
        
        # Set as initial state if it's the first state
        if self.initial_state is None:
            self.initial_state = state_id
            
        return state_id
    
    def remove_state(self, state_id: str) -> bool:
        """
        Remove a state from the automaton.
        
        Args:
            state_id: ID of the state to remove
            
        Returns:
            True if the state was removed, False if it didn't exist
        """
        if state_id not in self.states:
            return False
            
        # Remove the state
        del self.states[state_id]
        del self.transitions[state_id]
        
        # Remove transitions to this state from other states
        for from_state in self.transitions:
            to_remove = []
            for symbol, target in self.transitions[from_state].items():
                if target == state_id:
                    to_remove.append(symbol)
            for symbol in to_remove:
                del self.transitions[from_state][symbol]

        # Remove from transition groups
        groups_to_remove = []
        for (from_state, to_state) in self.transition_groups:
            if from_state == state_id or to_state == state_id:
                groups_to_remove.append((from_state, to_state))

        for group_key in groups_to_remove:
            del self.transition_groups[group_key]

        # Update special state sets
        self.accept_states.discard(state_id)
        self.dead_end_states.discard(state_id)
        
        # Update initial state if necessary
        if self.initial_state == state_id:
            self.initial_state = next(iter(self.states.keys())) if self.states else None
            
        return True
    
    def add_transition(self, from_state: str, to_state: str, symbol: str, arc_offset: float = 0.0) -> bool:
        """
        Add a transition between two states.

        Args:
            from_state: Source state ID
            to_state: Target state ID
            symbol: Input symbol that triggers this transition
            arc_offset: Arc offset for curved arrows

        Returns:
            True if the transition was added, False if states don't exist
        """
        if from_state not in self.states or to_state not in self.states:
            return False

        # Remove existing transition with same symbol if it exists
        if symbol in self.transitions[from_state]:
            old_to_state = self.transitions[from_state][symbol]
            # Remove from old transition group
            old_transition_key = (from_state, old_to_state)
            if old_transition_key in self.transition_groups:
                self.transition_groups[old_transition_key]['symbols'].discard(symbol)
                # Remove group if no symbols left
                if not self.transition_groups[old_transition_key]['symbols']:
                    del self.transition_groups[old_transition_key]

        self.transitions[from_state][symbol] = to_state
        self.alphabet.add(symbol)

        # Update transition groups
        transition_key = (from_state, to_state)
        if transition_key not in self.transition_groups:
            self.transition_groups[transition_key] = {
                'symbols': set(),
                'arc_offset': arc_offset
            }

        self.transition_groups[transition_key]['symbols'].add(symbol)
        # Update arc offset if provided
        if arc_offset != 0.0:
            self.transition_groups[transition_key]['arc_offset'] = arc_offset

        return True
    
    def remove_transition(self, from_state: str, symbol: str) -> bool:
        """
        Remove a transition.

        Args:
            from_state: Source state ID
            symbol: Input symbol of the transition to remove

        Returns:
            True if the transition was removed, False if it didn't exist
        """
        if from_state in self.transitions and symbol in self.transitions[from_state]:
            to_state = self.transitions[from_state][symbol]
            del self.transitions[from_state][symbol]

            # Update transition groups
            transition_key = (from_state, to_state)
            if transition_key in self.transition_groups:
                self.transition_groups[transition_key]['symbols'].discard(symbol)
                # Remove group if no symbols left
                if not self.transition_groups[transition_key]['symbols']:
                    del self.transition_groups[transition_key]

            return True
        return False
    
    def set_state_type(self, state_id: str, state_type: StateType) -> bool:
        """
        Set the type of a state (normal, accept, or dead end).
        
        Args:
            state_id: ID of the state to modify
            state_type: New type for the state
            
        Returns:
            True if the state type was set, False if state doesn't exist
        """
        if state_id not in self.states:
            return False
            
        self.states[state_id].state_type = state_type
        
        # Update special state sets
        if state_type == StateType.ACCEPT:
            self.accept_states.add(state_id)
            self.dead_end_states.discard(state_id)
        elif state_type == StateType.DEAD_END:
            self.dead_end_states.add(state_id)
            self.accept_states.discard(state_id)
        else:
            self.accept_states.discard(state_id)
            self.dead_end_states.discard(state_id)
            
        return True
    
    def process_string(self, input_string: str) -> Tuple[bool, List[str]]:
        """
        Process an input string through the automaton.
        
        Args:
            input_string: String to process
            
        Returns:
            Tuple of (accepted: bool, path: List[str])
            - accepted: True if string is accepted, False otherwise
            - path: List of state IDs visited during processing
        """
        if not self.initial_state or self.initial_state not in self.states:
            return False, []
            
        current_state = self.initial_state
        path = [current_state]
        
        for symbol in input_string:
            if symbol not in self.alphabet:
                return False, path
                
            if current_state not in self.transitions or symbol not in self.transitions[current_state]:
                return False, path
                
            current_state = self.transitions[current_state][symbol]
            path.append(current_state)
            
            # Check for dead end
            if current_state in self.dead_end_states:
                return False, path
        
        # Check if final state is accepting
        accepted = current_state in self.accept_states
        return accepted, path

    def to_dict(self) -> Dict:
        """
        Serialize the DFA to a dictionary for saving.

        Returns:
            Dictionary representation of the DFA
        """
        return {
            'states': {
                state_id: {
                    'position': state.position,
                    'state_type': state.state_type.value
                }
                for state_id, state in self.states.items()
            },
            'transitions': self.transitions,
            'alphabet': list(self.alphabet),
            'initial_state': self.initial_state,
            'accept_states': list(self.accept_states),
            'dead_end_states': list(self.dead_end_states),
            'next_state_id': self._next_state_id
        }

    def from_dict(self, data: Dict):
        """
        Load the DFA from a dictionary.

        Args:
            data: Dictionary containing DFA data
        """
        # Clear current data
        self.states.clear()
        self.transitions.clear()
        self.alphabet.clear()
        self.accept_states.clear()
        self.dead_end_states.clear()

        # Load states
        for state_id, state_data in data.get('states', {}).items():
            state = State(state_id, tuple(state_data['position']))
            state.state_type = StateType(state_data['state_type'])
            self.states[state_id] = state

        # Load other data
        self.transitions = data.get('transitions', {})
        self.alphabet = set(data.get('alphabet', []))
        self.initial_state = data.get('initial_state')
        self.accept_states = set(data.get('accept_states', []))
        self.dead_end_states = set(data.get('dead_end_states', []))
        self._next_state_id = data.get('next_state_id', 0)

    def save_to_file(self, filename: str) -> bool:
        """
        Save the DFA to a JSON file.

        Args:
            filename: Path to save the file

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filename, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving file: {e}")
            return False

    def load_from_file(self, filename: str) -> bool:
        """
        Load the DFA from a JSON file.

        Args:
            filename: Path to load the file from

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            self.from_dict(data)
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
