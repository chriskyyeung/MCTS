from collections import defaultdict
import logging
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
                Probability for each discrete states. 
                Defaults to np.array([1.]).
            log_config (dict, optional): 
                Configuration for `logging` library. 
                Defaults to dict().
        """
        super().__init__(state, parent, parent_action, log_config)
        self.random_state_config = {
            'a': range(len(discrete_states)),
            'p': discrete_states,
        }
        pass

    def _get_state(self, idx: int) -> Any:
        """Map the random generated index to state"""
        raise NotImplementedError("")

    def _get_state_index(self, size: int) -> int:
        return np.random.choice(size=size, **self.random_state_config)

    def selection(self, c: float) -> Self:
        return super().selection(c)