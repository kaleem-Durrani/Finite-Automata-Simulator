"""
DFA (Deterministic Finite Automaton) module.

This module contains the DFA class that represents the complete
finite automaton with states, transitions, and processing logic.
"""

import json
from typing import Any, Dict, List, Optional, Set, Tuple

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
            
        # Remove the state. `transitions` may not have an entry for this state
        # if the automaton was loaded from a file whose "transitions" object
        # omitted states with no outgoing edges, so pop defensively.
        del self.states[state_id]
        self.transitions.pop(state_id, None)
        
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

        # A state loaded from file may have no entry here (see remove_state).
        self.transitions.setdefault(from_state, {})

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

        The result is an independent snapshot: `position` and `transitions` are
        copied rather than shared. Returning the live objects meant a caller
        holding the dict watched it change underneath them -- loading a file
        emptied a previously-taken snapshot, because from_dict clears the very
        dict the snapshot was pointing at.

        Returns:
            Dictionary representation of the DFA
        """
        return {
            'states': {
                state_id: {
                    'position': list(state.position),
                    'state_type': state.state_type.value
                }
                for state_id, state in self.states.items()
            },
            'transitions': {
                state_id: dict(symbol_map)
                for state_id, symbol_map in self.transitions.items()
            },
            # Sets are serialized sorted. Python randomises string hashing per
            # process, so list(some_set) has no stable order: without sorting,
            # saving the same automaton twice produces different bytes and the
            # file is useless to diff or to compare in a test.
            'alphabet': sorted(self.alphabet),
            'initial_state': self.initial_state,
            'accept_states': sorted(self.accept_states),
            'dead_end_states': sorted(self.dead_end_states),
            # Arc offsets are presentation data, but they are the user's work:
            # without them a saved automaton reloads with all its curves flat.
            'arc_offsets': {
                f"{from_state}|{to_state}": group['arc_offset']
                for (from_state, to_state), group in sorted(self.transition_groups.items())
                if group.get('arc_offset', 0.0) != 0.0
            },
            'next_state_id': self._next_state_id
        }

    def from_dict(self, data: Dict):
        """
        Load the DFA from a dictionary.

        Args:
            data: Dictionary containing DFA data
        """
        # Clear current data. transition_groups must be cleared too -- it is a
        # second representation of the transition function, and leaving it
        # populated means the renderer draws the *previous* automaton's edges
        # over the newly loaded states.
        self.states.clear()
        self.transitions.clear()
        self.transition_groups.clear()
        self.alphabet.clear()
        self.accept_states.clear()
        self.dead_end_states.clear()

        # Load states
        for state_id, state_data in data.get('states', {}).items():
            state = State(state_id, tuple(state_data['position']))
            state.state_type = StateType(state_data['state_type'])
            self.states[state_id] = state

        # Load transitions, guaranteeing one entry per state and discarding any
        # edge that names a state the file does not define.
        raw_transitions = data.get('transitions', {})
        self.transitions = {
            state_id: {
                symbol: target
                for symbol, target in raw_transitions.get(state_id, {}).items()
                if target in self.states
            }
            for state_id in self.states
        }

        # Load other data
        self.alphabet = set(data.get('alphabet', []))
        self.initial_state = data.get('initial_state')
        if self.initial_state is not None and self.initial_state not in self.states:
            self.initial_state = None
        self.accept_states = {s for s in data.get('accept_states', []) if s in self.states}
        self.dead_end_states = {s for s in data.get('dead_end_states', []) if s in self.states}
        self._next_state_id = data.get('next_state_id', 0)

        self._rebuild_transition_groups(data.get('arc_offsets', {}))

    def _rebuild_transition_groups(self, arc_offsets: Optional[Dict[str, float]] = None):
        """
        Regenerate transition_groups from the transition function.

        transition_groups is a rendering-oriented view of `transitions`: it maps
        a (from, to) pair to the set of symbols on that edge, plus the curve
        offset. It is derived data, so it can always be rebuilt from the
        authoritative `transitions` mapping.

        Args:
            arc_offsets: Optional "from|to" -> offset mapping to restore.
        """
        arc_offsets = arc_offsets or {}
        self.transition_groups.clear()

        for from_state, symbol_map in self.transitions.items():
            for symbol, to_state in symbol_map.items():
                key = (from_state, to_state)
                if key not in self.transition_groups:
                    self.transition_groups[key] = {
                        'symbols': set(),
                        'arc_offset': float(arc_offsets.get(f"{from_state}|{to_state}", 0.0))
                    }
                self.transition_groups[key]['symbols'].add(symbol)

    def save_to_file(self, filename: str) -> Tuple[bool, str]:
        """
        Save the DFA to a JSON file.

        Args:
            filename: Path to save the file

        Returns:
            (True, "") on success, or (False, reason) on failure. The reason is
            returned rather than printed so the caller can show it to the user;
            nobody running a windowed application reads stdout.
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True, ""
        except OSError as e:
            return False, e.strerror or str(e)

    def load_from_file(self, filename: str) -> Tuple[bool, str]:
        """
        Load the DFA from a JSON file.

        The automaton is only replaced if the file parses; a malformed file
        leaves the current automaton untouched rather than half-loaded.

        Args:
            filename: Path to load the file from

        Returns:
            (True, "") on success, or (False, reason) on failure.
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except OSError as e:
            return False, e.strerror or str(e)
        except json.JSONDecodeError as e:
            return False, f"not valid JSON (line {e.lineno})"

        if not isinstance(data, dict):
            return False, "file does not contain an automaton"

        try:
            self.from_dict(data)
        except (KeyError, TypeError, ValueError) as e:
            return False, f"malformed automaton ({e})"
        return True, ""
