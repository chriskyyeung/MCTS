import logging
from time import time
from typing import Self, Any

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
        self._W = 0
        self._untried_actions = self.get_untried_actions()
        pass

    @property
    def is_terminal(self) -> bool:
        # Check if the simulation ends
        return self.state.is_game_over
    
    @property
    def is_fully_expanded(self) -> bool:
        # Check if the expansion is done
        return len(self._untried_actions) == 0
    
    def _get_action(self) -> Any:
        # Get a action for updating the game state
        raise NotImplementedError

    def update_node_N(self) -> None:
        pass

    def best_child(self) -> Self:
        raise NotImplementedError
    
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
        action = self._get_action()
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
        """Backpropagation, triggering parent's one if any

        Args:
            score (int): Score of the current state
        """
        self._N += 1
        self._W += score
        if self.parent:
            self.parent.backpropagate(score*-1)
        return 

    def best_action(self, c:float=0.1, simulation_time:float=1, n_simulation:float=100) -> Self:
        """Return the best child as the suggested action

        Args:
            c (float, optional): Exploration parameter. Defaults to 0.1.
            simulation_time (float, optional): Max. time for MCTS. Defaults to 1
            n_simulation (int, optional): Max no. of MCTS simulations to be performed. Defaults to 100

        Returns:
            Self: AI move based on the score of all children
        """
        t0, i_simulation = time(), 0
        while time() - t0 <= simulation_time and i_simulation < n_simulation:
            node = self.selection(c)

            if not node.is_terminal:
                node = node.expand()

            final_state, result = node.simulation()
            node.backpropagate(result)
            i_simulation += 1

        self._c = 0
        self.logger.debug(f"{i_simulation} iteration in {time()-t0} seconds")
        return self.best_child()