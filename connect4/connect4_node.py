from typing import Self

import numpy as np

from base.determ_mcts_node import DetermMCTSNode

class Connet4Node(DetermMCTSNode):
    """MCTS node for a Connect-4 Game
    """
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
        return state,  result * -self.state._turnID