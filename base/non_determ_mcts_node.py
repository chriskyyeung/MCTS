from typing import Self, Any

import numpy as np

from base.game_state import GameState
from base.mcts_node import MCTSNode

class NonDetermMCTSNode(MCTSNode):
    """Base class for a MCTS node for non-deterministic game
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
                Probability for each discrete sta
                tes. 
                Defaults to np.array([1.]).
            log_config (dict, optional): 
                Configuration for `logging` library. 
                Defaults to dict().
        """
        super().__init__(state, parent, parent_action, log_config)

        # Handling random state
        self._random_state_config = {
            'a': range(len(discrete_states)),
            'p': discrete_states,
        }
        # self.children = [[None] * len(discrete_states) for _ in range(len(self._untried_actions))]
        self.children = [[None] * len(discrete_states) for _ in range(3)]
        pass

    @property
    def child_weights(self) -> list:
        weights = np.repeat(-np.inf, repeats=len(self.children))
        for inode, chance_node in enumerate(self.children):
            node_weight, node_N = 0, 0
            for child in chance_node:
                if child is not None:
                    node_weight += child._W
                    node_N += child._N
            if node_N > 0:
                weights[inode] = node_weight / node_N + self._c * np.sqrt(2 * np.log(self._N)/ node_N)
        return weights

    @property
    def _random_state(self) -> Any:
        """Return a random state based on the random_state_config"""
        raise NotImplementedError("")

    def _get_action(self) -> list:
        return (self._untried_actions.pop(), self._random_state)
        
    def best_child(self) -> Self:
        irow = np.argmax(self.child_weights)
        dice = self._random_state
        child = self.children[irow][dice-1]
        if child is None:
            child = self.__class__(
                self.state.update((irow, dice)),
                parent=self,
                parent_action=(irow, dice),
                discrete_states=np.repeat(1/6, repeats=6),
        )
        return child

    def _get_random_state_index(self, size: int) -> int:
        return np.random.choice(size=size, **self._random_state_config)[0]

    def expand(self) -> Self:
        """Append an expanded node as children and return it

        Returns:
            Self: The expanded child node
        """
        action = self._get_action()
        self.children[action[0]][action[1]-1] =self.__class__(
                self.state.update(action),
                parent=self,
                parent_action=action,
                discrete_states=np.repeat(1/6, repeats=6),
        )
        return self.children[action[0]][action[1]-1]

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
        return (possible_moves[np.random.randint(len(possible_moves))], random_state)

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