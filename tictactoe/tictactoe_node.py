from typing import Self

from base.determ_mcts_node import DetermMCTSNode

class TicTacToeNode(DetermMCTSNode):
    """MCTS node for a TicTacToe Game
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass

    def simulation(self) -> tuple[Self, int]:
        """Simulation process in one MCTS iteraction

        Returns:
            tuple[Self, int]: Output of the simulation step
            - Self: Final step of the simulated game state 
            - int : The score of the final result
        """
        state, result = super().simulation()
        return state,  result * -self.state._turnID

    def selection(self, c: float) -> Self:
        """Override the parent class method for debugging

        Args:
            c (float): The exploration parameter

        Returns:
            Self: Current node fater the process
        """
        current_node = super().selection(c)

        node = current_node
        msg = f"\n{node.state._board}]\nScore: {node._W}"
        while node.parent:
            node = node.parent
            msg = f"\n{node.state._board}]\nScore: {node._W}" + msg
        
        self.logger.debug(msg)
        return current_node