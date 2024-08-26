from typing import Self

import numpy as np

from base.mcts_node import MCTSNode

class KnucklebonesNode(MCTSNode):
    """MCTS node for a KnuckleBones Game
    """
    _check = [1, -1]
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass

    @property
    def W(self) -> float:
        """Numerator of the exploitation part

        Returns:
            float: Sum of score of all simulated situations
        """
        return np.sum([k*v for k,v in self._score.items()])

    def simulation(self) -> tuple[Self, int]:
        """Simulation process in one MCTS iteraction

        Returns:
            tuple[Self, int]: Output of the simulation step
            - Self: Final step of the simulated game state 
            - int : The score of the final result
        """
        state, result = super().simulation()
        return state,  result * KnucklebonesNode._check[self.state._moveID]

    def selection(self, c: float) -> Self:
        """Override the parent class method for debugging

        Args:
            c (float): The exploration parameter

        Returns:
            Self: Current node fater the process
        """
        current_node = super().selection(c)

        node = current_node
        msg = f"\n{node.state._board}]\nScore: {node.W}"
        while node.parent:
            node = node.parent
            msg = f"\n{node.state._board}]\nScore: {node.W}" + msg
        
        self.logger.debug(msg)
        return current_node