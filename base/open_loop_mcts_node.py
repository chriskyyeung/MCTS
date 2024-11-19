from typing import Self, Any

import numpy as np

from base.game_state import GameState
from base.mcts_node import MCTSNode

class OpenLoopMCTSNode(MCTSNode):
    """Base class for a Open Loop MCTS node for non-deterministic game
    """
    def __init__(
            self, 
            state: GameState, 
            parent: Self = None,
            parent_action: Any = None,
            discrete_states: np.ndarray = np.array([1.]),
            log_config: dict = dict(),
        ) -> None:
        """Construct a node to perform MCTS on the input game state

        Args:
            state (GameState): 
                Current game state
            parent (Self, optional): 
                Parent node of the current state. 
                Defaults to None.
            parent_action (Any, optional):
                Action taken by the parent node to reach here. 
                Defaults to None.
            discrete_states (np.ndarray, optional):
                Probability for each discrete states. 
                Defaults to np.array([1.]).
            log_config (dict, optional): 
                Configuration for `logging` library. 
                Defaults to dict().
        """
        super().__init__(state, parent, parent_action, log_config)

        # Handling random state
        self._set_random_state(discrete_states)
        
        # Currently designed for numeric "action" values
        self._id_to_move = state.initialize_actions()
        self._move_to_id = {move: i for i, move in enumerate(self._id_to_move)}
        self.children = [None] * len(self._id_to_move)
        pass

    @property
    def child_weights(self) -> list:
        # Dummy random state for validity checking
        random_state = self._get_random_state_index(1)
        valid_children = np.array([self.state._is_valid_move(random_state+1, self._id_to_move[i]) for i in range(len(self.children))])

        # Only count visit of children that are valid in current state
        total_visit = np.dot(valid_children, [c._N if c else 0 for c in self.children])

        weights = np.repeat(-np.inf, repeats=len(self.children))
        for inode, node in enumerate(self.children):
            if valid_children[inode]:
                weights[inode] = node._W / node._N + self._c * np.sqrt(2 * np.log(total_visit)/ node._N)
            
        self.logger.debug(total_visit)
        self.logger.debug(weights)
        return weights

    @property
    def _random_state(self) -> Any:
        """Return a random state based on the random_state_config"""
        raise NotImplementedError("")

    def get_child_by_action(self, action) -> Self:
        return self.children[self._move_to_id[action[1]]]

    def _set_random_state(self, discrete_states) -> None:
        # Handling random state
        self._random_state_config = {
            'a': range(len(discrete_states)),
            'p': discrete_states,
        }
        return

    def _get_action(self) -> list:
        return self._untried_actions.pop()
    
    def best_child(self) -> Self:
        # Decide the best move independent of the random state
        w = self.child_weights
        i = np.random.choice(np.where(w == max(w))[0])
        move = self._id_to_move[i]

        # Generate the random state and update the best child
        dice = self._random_state
        self.children[i].state = self.state.update((dice, move))
        self.children[i]._untried_actions = [j for j in self.children[i].state.get_legal_actions() if self.children[i].children[j] is None]
        self.children[i].parent_action = (dice, move)

        return self.children[i]

    def _get_random_state_index(self, size: int) -> int:
        return np.random.choice(size=size, **self._random_state_config)[0]

    def expand(self) -> Self:
        """Append an expanded node as children and return it

        Returns:
            Self: The expanded child node
        """
        # For open loop, actions only contains deterministic part
        action_id = self._get_action()
        action = (self._random_state, self._move_to_id[action_id])
        self.children[action_id] = self.__class__(
                self.state.update(action),
                parent=self,
                parent_action=action,
                discrete_states=np.repeat(1/6, repeats=6),
        )
        return self.children[action_id]

    def selection(self, c: float) -> Self:
        """Selection step

        Args:
            c (float): Exploration parameter

        Returns:
            Self: Selected node
        """
        self._c = c
        current_node = self
        while (current_node.is_fully_expanded) and (not current_node.is_terminal):
            current_node = current_node.best_child()
            current_node._c = c

        self.logger.debug("End selection")
        return current_node

    @staticmethod
    def simulation_step(state: GameState, random_state: Any) -> Any:
        """Simulation 1 step forward from the input game state

        Args:
            state (GameState): Current game state
            random_state (Any): Generated random state

        Returns:
            Any: 1 simulated move forward from the current game state
        """
        possible_moves = state.get_legal_actions()
        assigned_move = np.random.randint(len(possible_moves))
        return random_state, possible_moves[assigned_move]

    def simulation(self) -> tuple[GameState, Any]:
        """Simulation, involving multiple simulation steps

        Returns:
            - tuple[GameState, Any]: A tuple containing
                1. Final state of the simulated game
                2. Result, in terms of the score
        """
        current_state = self.state
        while not current_state.is_game_over:
            action = self.simulation_step(current_state, self._random_state)
            current_state = current_state.update(action)
        
        return current_state, current_state.game_result