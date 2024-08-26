from collections import defaultdict
import logging
from typing import Self, Any

import numpy as np

from base.game_state import GameState
from base.mcts_node import MCTSNode

class DetermMCTSNode(MCTSNode):
    """Base class for a Deterministic MCTS node
    """
    def __init__(
            self, 
            state: GameState, 
            parent: Self = None,
            parent_action: Any = None,
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
            log_config (dict, optional): 
                Configuration for `logging` library. 
                Defaults to dict().
        """
        super().__init__(state, parent, parent_action, log_config)
        pass

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
            current_node = current_node.best_child

        return current_node

    @staticmethod
    def simulation_step(state: GameState) -> Any:
        """Simulation 1 step forward from the input game state

        Args:
            state (GameState): Current game state

        Returns:
            Any: 1 simulated move forward from the current game state
        """
        possible_moves = state.get_legal_actions()
        return possible_moves[np.random.randint(len(possible_moves))]

    def simulation(self) -> tuple[GameState, Any]:
        """Simulation, involving multiple simulation steps

        Returns:
            - tuple[GameState, Any]: A tuple containing
                1. Final state of the simulated game
                2. Result, in terms of the score
        """
        current_state = self.state
        while not current_state.is_game_over:
            action = self.simulation_step(current_state)
            current_state = current_state.update(action)
        
        return current_state, current_state.game_result

    def backpropagate(self, score: int) -> None:
        """Backpropagation, triggering parent's one if any

        Args:
            score (int): Score of the current state
        """
        self._N += 1
        self._score[score] += 1
        if self.parent:
            self.parent.backpropagate(score*-1)
        return 