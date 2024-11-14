from typing import Self

from base.open_loop_mcts_node import OpenLoopMCTSNode

class KnucklebonesOpenLoopNode(OpenLoopMCTSNode):
    """MCTS node for a KnuckleBones Game
    """
    _check = [1, -1]
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dice = [1,2,3,4,5,6]
        pass

    @property
    def _random_state(self) -> int:
        return self._dice[self._get_random_state_index(1)]
            

    def simulation(self) -> tuple[Self, int]:
        """Simulation process in one MCTS iteraction

        Returns:
            tuple[Self, int]: Output of the simulation step
            - Self: Final step of the simulated game state 
            - int : The score of the final result
        """
        state, result = super().simulation()
        return state,  result * KnucklebonesOpenLoopNode._check[self.state._moveID]