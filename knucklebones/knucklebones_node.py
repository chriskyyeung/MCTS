from typing import Self

import numpy as np

from base.game_state import GameState
from base.non_determ_mcts_node import NonDetermMCTSNode

class KnucklebonesNode(NonDetermMCTSNode):
    """MCTS node for a KnuckleBones Game
    """
    _check = [1, -1]
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dice = [1,2,3,4,5,6]
        pass

    @property
    def W(self) -> float:
        """Numerator of the exploitation part

        Returns:
            float: Sum of score of all simulated situations
        """
        return np.sum([k*v for k,v in self._score.items()])

    def _get_state(self, idx: int) -> np.Any:
        return self._dice[idx]
 
    @staticmethod
    def simulation_step(state: GameState) -> list[tuple[int, int]]:
        """_summary_

        Args:
            state (GameState): Current game state

        Returns:
            list[tuple[int, int]]: List of moves in the form of (row, dice)
        """
        return [(super().simulation_step(state), dice) for dice in range(1,7)]
            

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
        return super().selection(c)