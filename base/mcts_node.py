from collections import defaultdict
import logging
from typing import Self, Any

import numpy as np

from base.game_state import GameState

class MCTSNode:
    """Base class for a MCTS node
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
        self.state = state
        self.parent = parent
        self.parent_action = parent_action

        # Set logging configuration
        logging.basicConfig(**log_config)
        self.logger = logging.getLogger()

        # Define necessary variables
        self.children = []
        self._N = 0
        self._c = 0
        self._score = defaultdict(int)
        self._untried_actions = self.get_untried_actions()
        pass

    @property
    def N(self) -> int:
        # Total game played
        return self._N

    @property
    def W(self) -> float:
        # Winning counts
        return self._score[1]

    @property
    def is_terminal(self) -> bool:
        # Check if the simulation ends
        return self.state.is_game_over
    
    @property
    def is_fully_expanded(self) -> bool:
        # Check if the expansion is done
        return len(self._untried_actions) == 0

    @property
    def child_weights(self) -> list:
        # Weights of every children
        return  np.array([
            child.W / child.N + self._c * np.sqrt(2 * np.log(self.N) / child.N)
            for child in self.children
        ])

    @property
    def best_child(self) -> Self:
        # Select the best child
        return self.children[np.argmax(self.child_weights)]

    def get_untried_actions(self) -> list:
        """Retrieve all un-tried actions for expansion

        Returns:
            list: List of untried legal actions
        """
        return self.state.get_legal_actions()
    
    def expand(self) -> Self:
        """Append an expanded node as children and return it

        Returns:
            Self: The expanded child node
        """
        action = self._untried_actions.pop()
        self.children.append(
            self.__class__(
                self.state.update(action),
                parent=self,
                parent_action=action
            )
        )
        return self.children[-1]

    def selection(self, c: float) -> Self:
        """Defining the selection step"""
        raise NotImplementedError("Defining the selection step")

    @staticmethod
    def simulation_step(state: GameState) -> Any:
        """Defining 1 single simulation step"""
        raise NotImplementedError("Define 1 single simulation step")

    def simulation(self) -> tuple[GameState, Any]:
        """Defining the whole simulation process"""
        raise NotImplementedError("Define the whole simulation process")

    def backpropagate(self, score: Any) -> None:
        """Defining the backpropagation step"""
        raise NotImplementedError("Define the backpropagate step")

    def best_action(self, c:float=0.1, n_simulation:int=100) -> Self:
        """Return the best child as the suggested action

        Args:
            c (float, optional): Exploration parameter. Defaults to 0.1.
            n_simulation (int, optional): No. of iterations. Defaults to 100.

        Returns:
            Self: AI move based on the score of all children
        """
        for _ in range(n_simulation):
            self.logger.debug("Simulation starts")
            node = self.selection(c)
            

            self.logger.debug("Expansion")
            if not node.is_terminal:
                node = node.expand()

            final_state, result = node.simulation()
            node.backpropagate(result)

        self._c = 0
        return self.best_child